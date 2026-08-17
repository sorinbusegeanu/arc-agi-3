from __future__ import annotations

"""v8.29 progress-aware exploration control for unsolved DISCOVERY games."""

import threading

_INSTALLED = False
_BASE_DISCOVERY_ACTOR = None
_BASE_PLAN_CHAIN = None
_BASE_SCORE_ACTIONS = None
_BASE_ENV_STEP = None
_BASE_ENV_RESET = None
_CONTROL_STATE = threading.local()
_TESTED = {}
_ATTEMPTS = {}
_NO_PROGRESS = {}
_PROGRESS = {}
_PROGRESS_ACTION = {}
_TRANSFER_ACTION = {}
_SOURCE_COUNTS = {}


def _discovery_mode():
    from v8 import decision_point_sampling_v821 as sampling
    return bool(sampling._decision_mode_enabled())


def _state_key(context):
    game = str(getattr(_CONTROL_STATE, "game_id", ""))
    if not game:
        return None
    return (game, int(getattr(_CONTROL_STATE, "level", 0)), int(context))


def _action_key(state, action):
    return (*state, int(action))


def _set_selection(context, source, actions=()):
    _CONTROL_STATE.context = int(context)
    _CONTROL_STATE.selection_source = str(source)
    _CONTROL_STATE.planned_actions = frozenset(int(value) for value in actions)


def _exact_strategy_uids(view, context):
    from v8.model import stable_u64

    bucket = stable_u64(int(context), person=b"v8-context")
    return {
        row.strategy_uid
        for row in tuple(getattr(view, "_strategy_by_context", {}).get(bucket, ()))
    }


def _plan_chain_v829(self, context_signature, action_ids, **kwargs):
    """During a bound DISCOVERY actor, coverage outranks autonomous M7 control."""
    if not _discovery_mode():
        return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)

    available = tuple(sorted({int(value) for value in action_ids}))
    state = _state_key(context_signature)
    # Planner calls made outside an active actor (tests, inspection, evaluation) must
    # retain the pre-v8.29 semantics exactly.
    if state is None:
        return _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)

    _set_selection(context_signature, "DISCOVERY")
    if not available:
        return ()

    progress_action = _PROGRESS_ACTION.get(state)
    if progress_action in available:
        _set_selection(context_signature, "PROGRESS_REPLAY", (progress_action,))
        return ()

    tested = _TESTED.setdefault(state, set())
    if any(action not in tested for action in available):
        return ()

    rows = tuple(_BASE_PLAN_CHAIN(self, context_signature, available, **kwargs))
    if not rows:
        return ()

    # Cross-context M7 rows can suggest transfer in explicit TRANSFER mode, but they
    # cannot autonomously control an unsolved DISCOVERY state.
    exact = _exact_strategy_uids(self, context_signature)
    rows = tuple(row for row in rows if row.strategy_uid in exact)
    if not rows:
        return ()

    # Once local coverage is complete, keep planner use fair: a repeatedly
    # non-progressing action cannot monopolize the state while another tested action
    # has fewer no-progress executions.
    minimum = min(
        int(_NO_PROGRESS.get(_action_key(state, action), 0)) for action in available
    )
    rows = tuple(
        row
        for row in rows
        if int(_NO_PROGRESS.get(_action_key(state, row.action_id), 0)) <= minimum
    )
    _set_selection(context_signature, "PLANNER", (row.action_id for row in rows))
    return rows


def _score_actions_v829(view, context_signature, action_ids):
    """Systematically cover actions, replay true progress, then use fair fallback."""
    if not _discovery_mode():
        return _BASE_SCORE_ACTIONS(view, context_signature, action_ids)

    from v8.publication import ActionScore

    available = tuple(sorted({int(value) for value in action_ids}))
    state = _state_key(context_signature)
    if state is None or not available:
        return _BASE_SCORE_ACTIONS(view, context_signature, available)

    progress_action = _PROGRESS_ACTION.get(state)
    if progress_action in available:
        _set_selection(context_signature, "PROGRESS_REPLAY", (progress_action,))
        return (ActionScore(int(progress_action), 1, 1.0, 0),)

    tested = _TESTED.setdefault(state, set())
    untested = [action for action in available if action not in tested]
    if untested:
        transfer = _TRANSFER_ACTION.get((state[0], state[1]))
        if transfer in untested:
            action, source = int(transfer), "TRANSFER_PROBE"
        else:
            action, source = int(untested[0]), "DISCOVERY"
        _set_selection(context_signature, source, (action,))
        return (ActionScore(action, 0, 0.0, 0),)

    minimum = min(
        int(_NO_PROGRESS.get(_action_key(state, action), 0)) for action in available
    )
    least_stagnant = tuple(
        action
        for action in available
        if int(_NO_PROGRESS.get(_action_key(state, action), 0)) == minimum
    )
    _set_selection(context_signature, "FALLBACK", least_stagnant)
    return _BASE_SCORE_ACTIONS(view, context_signature, least_stagnant)


def _env_step_v829(self, action):
    """Record every executed action while keeping task progress separate from M7 reliability."""
    before_level = int(
        getattr(self, "last_levels_completed", getattr(_CONTROL_STATE, "level", 0))
    )
    game = str(getattr(_CONTROL_STATE, "game_id", ""))
    context = getattr(_CONTROL_STATE, "context", None)
    source = str(getattr(_CONTROL_STATE, "selection_source", "UNKNOWN"))

    result = _BASE_ENV_STEP(self, action)

    after_level = int(getattr(self, "last_levels_completed", before_level))
    progress = bool(
        after_level > before_level
        or str(getattr(self, "last_outcome_state", "")) == "WIN"
    )
    _CONTROL_STATE.level = after_level

    if game and context is not None:
        state = (game, before_level, int(context))
        key = _action_key(state, action)
        _TESTED.setdefault(state, set()).add(int(action))
        _ATTEMPTS[key] = int(_ATTEMPTS.get(key, 0)) + 1
        if progress:
            _PROGRESS[key] = int(_PROGRESS.get(key, 0)) + 1
            _NO_PROGRESS[key] = 0
            _PROGRESS_ACTION[state] = int(action)
            if after_level > before_level:
                _TRANSFER_ACTION[(game, after_level)] = int(action)
        else:
            _NO_PROGRESS[key] = int(_NO_PROGRESS.get(key, 0)) + 1

        counts = _SOURCE_COUNTS.setdefault(game, {})
        counts[source] = int(counts.get(source, 0)) + 1

    _CONTROL_STATE.selection_source = "UNKNOWN"
    _CONTROL_STATE.planned_actions = frozenset()
    return result


def _env_reset_v829(self, *args, **kwargs):
    result = _BASE_ENV_RESET(self, *args, **kwargs)
    _CONTROL_STATE.level = int(getattr(self, "last_levels_completed", 0))
    _CONTROL_STATE.context = None
    _CONTROL_STATE.selection_source = "UNKNOWN"
    _CONTROL_STATE.planned_actions = frozenset()
    return result


def _discovery_actor_v829(*, job, **kwargs):
    """Bind progress-aware state only for the active continuous DISCOVERY actor."""
    names = ("game_id", "level", "context", "selection_source", "planned_actions")
    prior = tuple(getattr(_CONTROL_STATE, name, None) for name in names)
    _CONTROL_STATE.game_id = str(job.game_id)
    _CONTROL_STATE.level = 0
    _CONTROL_STATE.context = None
    _CONTROL_STATE.selection_source = "UNKNOWN"
    _CONTROL_STATE.planned_actions = frozenset()
    try:
        return _BASE_DISCOVERY_ACTOR(job=job, **kwargs)
    finally:
        for name, value in zip(names, prior):
            if value is None:
                try:
                    delattr(_CONTROL_STATE, name)
                except AttributeError:
                    pass
            else:
                setattr(_CONTROL_STATE, name, value)


def sampling_telemetry_v829(game_id):
    game = str(game_id)
    return {
        "source_counts": dict(_SOURCE_COUNTS.get(game, {})),
        "states": sum(1 for key in _TESTED if key[0] == game),
        "tested_actions": sum(
            len(values) for key, values in _TESTED.items() if key[0] == game
        ),
        "progress_actions": sum(1 for key in _PROGRESS_ACTION if key[0] == game),
        "attempts": sum(value for key, value in _ATTEMPTS.items() if key[0] == game),
        "no_progress_attempts": sum(
            value for key, value in _NO_PROGRESS.items() if key[0] == game
        ),
    }


def _reset_sampling_state_v829():
    _TESTED.clear()
    _ATTEMPTS.clear()
    _NO_PROGRESS.clear()
    _PROGRESS.clear()
    _PROGRESS_ACTION.clear()
    _TRANSFER_ACTION.clear()
    _SOURCE_COUNTS.clear()


def install_sampling_progress_control_v829():
    global _INSTALLED, _BASE_DISCOVERY_ACTOR, _BASE_PLAN_CHAIN
    global _BASE_SCORE_ACTIONS, _BASE_ENV_STEP, _BASE_ENV_RESET
    if _INSTALLED:
        return

    from v8 import behavior_recovery as behavior
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import learning_performance_repair_v824 as v824
    from v8 import solved_game_recovery_v821 as recovery

    # Keep v8.28's continuous DISCOVERY actor and bind only its actor-local state.
    _BASE_DISCOVERY_ACTOR = sampling._BASE_ACTOR_WORKER
    sampling._BASE_ACTOR_WORKER = _discovery_actor_v829

    # Insert beneath the historical planner wrappers so public function identities
    # and non-actor planner semantics remain unchanged.
    _BASE_PLAN_CHAIN = v824._BASE_PLAN_CHAIN
    v824._BASE_PLAN_CHAIN = _plan_chain_v829

    # Insert beneath behavior scoring for the same compatibility reason.
    _BASE_SCORE_ACTIONS = behavior._ORIGINAL_SCORE_ACTIONS
    behavior._ORIGINAL_SCORE_ACTIONS = _score_actions_v829

    # v8.25 requires v8.22 to compose through recovery._tracked_env_step/reset.
    # Hook the lower delegate used by those tracked functions instead of replacing
    # the v8.22 identities themselves.
    _BASE_ENV_STEP = recovery._BASE_ENV_STEP
    _BASE_ENV_RESET = recovery._BASE_ENV_RESET
    recovery._BASE_ENV_STEP = _env_step_v829
    recovery._BASE_ENV_RESET = _env_reset_v829

    _INSTALLED = True

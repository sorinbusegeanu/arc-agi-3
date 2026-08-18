from __future__ import annotations

"""Compatibility composition for v8.37.

Keep historical public hook identities while placing environment-neutral transfer
and validation beneath them.  No legacy raw-action transfer authority is restored.
"""

import os

_INSTALLED = False
_BASE_V824_PLAN_CHAIN = None
_BASE_ENV_VALIDATE = None


def _grounded_plan_chain(self, context_signature, action_ids, **kwargs):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import environment_neutrality_v837 as v837
    from v8.publication import PlannedAction

    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    if mode != v819.SamplingMode.TRANSFER.value:
        return _BASE_V824_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)

    game = v837._current_game_id()
    if not game:
        return ()
    m7, _m1n = v837._grounded_transfer_index(self, game)
    available = tuple(sorted({int(value) for value in action_ids}))
    by_uid = {
        row.strategy_uid: row for row in tuple(getattr(self, "_strategy_fallback", ()))
    }
    rows = []
    for action in available:
        candidates = m7.get(int(action), ())
        if not candidates:
            continue
        score, strategy_uid, _origin = candidates[0]
        strategy = by_uid.get(strategy_uid)
        if strategy is None:
            continue
        rows.append(
            PlannedAction(
                int(action),
                strategy.outcome_uid,
                strategy_uid,
                float(score),
                False,
            )
        )
    return tuple(sorted(rows, key=lambda row: (-float(row.score), int(row.action_id))))


def _environment_validate_compatible(self, candidate):
    from v8 import environment_neutrality_v837 as v837
    from v8 import trajectory_target_minimization_v820 as v820

    if not v837._target_aware_service_v837(self.service):
        raw = self.service.validator(candidate)
        return v820._validation_result_from_legacy(raw)
    return _BASE_ENV_VALIDATE(self, candidate)


def _replay_segments_composed(service, candidate):
    from v8 import environment_neutrality_v837 as v837
    from v8 import trajectory_optimizer_v818 as v818

    if not v837._global_target_source(candidate.source):
        return v837._BASE_V836_REPLAY_LEVELS(service, candidate)
    prefix = tuple(service._v818_prefix_for(candidate))
    if prefix:
        return None
    actions = tuple(int(value) for value in candidate.actions)
    if not actions:
        return None

    # Resolve through the installed v8.18 symbol so tests and alternate adapters
    # can provide their own replay validator without changing cognition.
    validator = v818._GameReplayValidator(
        service,
        str(candidate.source.anchor.source_id),
    )
    replays = []
    for seed in v818._VALIDATION_SEEDS:
        env = validator._environment(seed, candidate.source.anchor.env_root)
        indexer = getattr(env, "cognitive_subepisode_index", None)
        prior_index = (
            int(indexer())
            if indexer is not None
            else max(0, int(getattr(env, "last_levels_completed", 0)))
        )
        segments = []
        current = []
        valid = True
        reached = False
        for action in actions:
            if int(action) not in {int(value) for value in env.available_actions()}:
                valid = False
                break
            env.step(int(action))
            current.append(int(action))
            index = (
                int(indexer())
                if indexer is not None
                else max(0, int(getattr(env, "last_levels_completed", prior_index)))
            )
            if index > prior_index:
                if index != prior_index + 1:
                    valid = False
                    break
                segments.append(tuple(current))
                current = []
                prior_index = index
            if v837._generic_target_reached(env, candidate.source):
                reached = True
                if current:
                    segments.append(tuple(current))
                    current = []
                break
            if v837._generic_failed_boundary(env):
                valid = False
                break
        if not valid or not reached:
            continue
        if tuple(value for segment in segments for value in segment) != actions:
            segments = [actions]
        replays.append(tuple(segments))
    if not replays or any(row != replays[0] for row in replays[1:]):
        return None
    return replays[0]


def install_environment_neutrality_v837_fixups() -> None:
    global _INSTALLED, _BASE_V824_PLAN_CHAIN, _BASE_ENV_VALIDATE
    if _INSTALLED:
        return

    from v8 import environment_neutrality_v837 as v837
    from v8 import learning_control_continuity_v826 as v826
    from v8 import learning_performance_repair_v824 as v824
    from v8 import trajectory_optimizer_convergence_v836 as v836

    # Restore the historical v8.24 -> v8.26 public composition.  Structural
    # grounding is inserted under v8.24's planner, so its foreign-provenance gate
    # still owns transfer labeling while raw foreign actions never reach it.
    _BASE_V824_PLAN_CHAIN = v824._BASE_PLAN_CHAIN
    v824._BASE_PLAN_CHAIN = _grounded_plan_chain
    v826._BASE_PLAN_CANDIDATES = v824._plan_candidates_v824

    # Custom validators remain custom. Environment factories opt into the generic
    # target-aware replay path; ARC does so through its adapter validator.
    _BASE_ENV_VALIDATE = v837._EnvironmentReplayValidator.validate
    v837._EnvironmentReplayValidator.validate = _environment_validate_compatible

    # Compose replay-boundary reconstruction through the installed validator symbol.
    v836._replay_full_win_levels = _replay_segments_composed

    _INSTALLED = True

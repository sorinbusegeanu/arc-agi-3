from __future__ import annotations

"""v8.37 environment-neutral cognition and structural transfer.

ARC-specific terminal/level/grid semantics are adapted at the environment boundary.
Sampling, transfer and trajectory optimization consume generic boundary, valence,
outcome and structural-transition contracts.  Legacy serialized fields remain readable
for snapshot compatibility.
"""

import os
from dataclasses import dataclass

from v8.environment_contract import (
    BoundaryEvent,
    BoundaryScope,
    OptimizationScopeKind,
    TransitionSemantics,
    is_complete_positive_episode,
    optimization_scope_for,
    target_boundary,
)


_INSTALLED = False
_BASE_V826_PLAN = None
_BASE_V832_OBSERVE = None
_BASE_V833_CROSS_GAME = None
_BASE_V834_PREFIX = None
_BASE_V835_CANDIDATE_SCOPE = None
_BASE_V835_STATUS = None
_BASE_V835_SCOPE_OVERRIDE = None
_BASE_V835_PROCESS = None
_BASE_V835_SUBMIT_NEXT = None
_BASE_V836_GENERATE = None
_BASE_V836_REPLAY_LEVELS = None
_BASE_V836_PUBLISH = None
_SCOPE_LABELS: dict[tuple[str, int], str] = {}


# ---------------------------------------------------------------------------
# Phase 1/7/9: generic transition semantics, ARC confined to adapter glue.
# ---------------------------------------------------------------------------


def _arc_boundary_event(env) -> BoundaryEvent:
    state = str(getattr(env, "last_outcome_state", ""))
    if state == "WIN":
        return BoundaryEvent(BoundaryScope.EPISODE, +1, False)
    if state == "GAME_OVER":
        return BoundaryEvent(BoundaryScope.EPISODE, -1, False)
    if bool(getattr(env, "level_completed_event", False)):
        return BoundaryEvent(BoundaryScope.SUBEPISODE, +1, True)
    return BoundaryEvent(BoundaryScope.NONE, 0, True)


def _arc_target_reached(env, target, outcome_uid=None) -> bool:
    if outcome_uid is not None and not bool(getattr(outcome_uid, "is_zero", True)):
        matcher = getattr(env, "matches_outcome_uid", None)
        if matcher is not None:
            matched = matcher(outcome_uid)
            if matched is not None:
                return bool(matched)

    boundary = target_boundary(target)
    if boundary.scope is BoundaryScope.EPISODE:
        current = _arc_boundary_event(env)
        return bool(
            current.scope is BoundaryScope.EPISODE
            and int(current.primary_valence) == int(boundary.primary_valence)
        )
    if boundary.scope is BoundaryScope.SUBEPISODE:
        return int(getattr(env, "last_levels_completed", 0)) >= max(
            1, int(getattr(target, "levels_completed", 1))
        )
    return False


def _arc_context_signature(env) -> int:
    from v7.environment.encoding import structural_grid_signature

    return int(structural_grid_signature(env.observe()))


def _arc_transition_signature(env, before, after) -> int:
    del env
    from v7.environment.encoding import transition_signature

    return int(transition_signature(before, after))


def _arc_subepisode_index(env) -> int:
    return max(0, int(getattr(env, "last_levels_completed", 0)))


def _boundary_from_kwargs(kwargs) -> BoundaryEvent:
    supplied = kwargs.get("boundary_event")
    if isinstance(supplied, BoundaryEvent):
        return supplied

    raw_scope = kwargs.get("boundary_scope")
    if raw_scope is not None:
        try:
            scope = BoundaryScope(str(raw_scope))
        except ValueError:
            scope = BoundaryScope.NONE
        return BoundaryEvent(
            scope,
            max(-1, min(1, int(kwargs.get("primary_valence", 0)))),
            bool(kwargs.get("continuation", scope is not BoundaryScope.EPISODE)),
        )

    # Legacy ARC actor kwargs are translated here, never interpreted by the
    # environment-neutral policy functions below.
    terminal = str(kwargs.get("terminal_state", ""))
    if terminal == "WIN":
        return BoundaryEvent(BoundaryScope.EPISODE, +1, False)
    if terminal == "GAME_OVER":
        return BoundaryEvent(BoundaryScope.EPISODE, -1, False)
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    if bool(kwargs.get("level_advanced", False)) or after_level > before_level:
        return BoundaryEvent(BoundaryScope.SUBEPISODE, +1, True)
    return BoundaryEvent(BoundaryScope.NONE, 0, True)


def _transition_semantics(kwargs) -> TransitionSemantics:
    before_context = int(kwargs.get("before_context", 0))
    after_context = int(kwargs.get("after_context", before_context))
    if "structural_changed" in kwargs:
        structural_changed = bool(kwargs.get("structural_changed"))
    else:
        structural_changed = int(kwargs.get("changed_cells", 0)) > 0
    return TransitionSemantics(
        _boundary_from_kwargs(kwargs),
        structural_changed=structural_changed,
        context_changed=after_context != before_context,
    )


# ---------------------------------------------------------------------------
# Phase 2/9: compatible generic trajectory target.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V837TrajectoryTarget:
    # The first two fields are retained exactly for old snapshots and callers.
    levels_completed: int = 0
    terminal_state: str = "LEVEL"
    boundary_scope: str = ""
    primary_valence: int = 0
    continuation: bool = True

    def __post_init__(self) -> None:
        levels = max(0, int(self.levels_completed))
        object.__setattr__(self, "levels_completed", levels)
        terminal = str(self.terminal_state)
        object.__setattr__(self, "terminal_state", terminal)

        scope = str(self.boundary_scope).strip().upper()
        valence = max(-1, min(1, int(self.primary_valence)))
        continuation = bool(self.continuation)
        if not scope:
            # Legacy ARC labels are decoded only by this compatibility adapter.
            if terminal == "WIN":
                scope, valence, continuation = BoundaryScope.EPISODE.value, +1, False
            elif terminal == "GAME_OVER":
                scope, valence, continuation = BoundaryScope.EPISODE.value, -1, False
            elif terminal == "LEVEL":
                scope, valence, continuation = BoundaryScope.SUBEPISODE.value, +1, True
            else:
                scope = BoundaryScope.NONE.value
        if scope not in {item.value for item in BoundaryScope}:
            scope = BoundaryScope.NONE.value
        object.__setattr__(self, "boundary_scope", scope)
        object.__setattr__(self, "primary_valence", valence)
        object.__setattr__(self, "continuation", continuation)

    def to_dict(self) -> dict[str, object]:
        return {
            "levels_completed": int(self.levels_completed),
            "terminal_state": str(self.terminal_state),
            "boundary_scope": str(self.boundary_scope),
            "primary_valence": int(self.primary_valence),
            "continuation": bool(self.continuation),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "V837TrajectoryTarget":
        return cls(
            int(raw.get("levels_completed", 0)),
            str(raw.get("terminal_state", "LEVEL")),
            str(raw.get("boundary_scope", "")),
            int(raw.get("primary_valence", 0)),
            bool(raw.get("continuation", True)),
        )


def _generic_target_reached(env, source) -> bool:
    matcher = getattr(env, "cognitive_target_reached", None)
    if matcher is not None:
        return bool(matcher(source.target, getattr(source, "target_outcome_uid", None)))

    outcome = getattr(source, "target_outcome_uid", None)
    if outcome is not None and not bool(getattr(outcome, "is_zero", True)):
        outcome_matcher = getattr(env, "matches_outcome_uid", None)
        if outcome_matcher is not None:
            matched = outcome_matcher(outcome)
            if matched is not None:
                return bool(matched)

    event_getter = getattr(env, "cognitive_boundary_event", None)
    if event_getter is None:
        return False
    current = event_getter()
    target = target_boundary(source.target)
    return bool(
        target.crossed
        and current.scope is target.scope
        and int(current.primary_valence) == int(target.primary_valence)
    )


def _generic_failed_boundary(env) -> bool:
    getter = getattr(env, "cognitive_boundary_event", None)
    if getter is None:
        return False
    event = getter()
    return bool(
        (event.scope is BoundaryScope.EPISODE and int(event.primary_valence) < 0)
        or not bool(event.continuation)
        and int(event.primary_valence) <= 0
    )


def _generic_context_signature(env) -> int:
    method = getattr(env, "cognitive_context_signature", None)
    return 0 if method is None else int(method())


def _generic_transition_signature(env, before, after) -> int:
    method = getattr(env, "cognitive_transition_signature", None)
    return 0 if method is None else int(method(before, after))


# ---------------------------------------------------------------------------
# Phase 4/5/6: structural correspondence -> target-local grounded action.
# ---------------------------------------------------------------------------


def _current_game_id() -> str:
    from v8 import optimizer_budget_control_v830 as _unused  # noqa: F401
    from v8 import sampling_progress_control_v829 as v829
    from v8 import trajectory_optimizer_v814 as optimizer

    return str(getattr(v829._CONTROL_STATE, "game_id", "")) or str(
        getattr(optimizer, "_CAPTURE_SOURCE_ID", "")
    )


def _grounded_transfer_index(view, game_id: str):
    from v8 import behavior_recovery as behavior
    from v8 import normalized_memory_v086 as normalized
    from v8.model import MemoryLevel, RelationType, signed_u64, stable_u64

    view._refresh_strategy_cache()
    current_game = int(stable_u64(str(game_id), person=b"v8-game"))
    version = tuple(getattr(view, "_strategy_version", ()))
    cache_key = (version, current_game)
    if getattr(view, "_v837_transfer_index_key", None) == cache_key:
        return getattr(view, "_v837_transfer_index", ({}, {}))

    nodes = dict(getattr(view, "_node_by_uid", {}))
    parents = getattr(view, "_parents", {})
    direct_games: dict[object, set[int]] = {}
    formal: dict[object, list[tuple[object, float]]] = {}
    for edge in view.edge_records():
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            direct_games.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
        elif relation == int(RelationType.TRANSFER_CORRESPONDENCE):
            score = max(0.0, float(getattr(edge, "score", 0.0)))
            formal.setdefault(edge.source_uid, []).append((edge.target_uid, score))
            formal.setdefault(edge.target_uid, []).append((edge.source_uid, score))

    lineage_cache: dict[object, tuple[frozenset[object], frozenset[int]]] = {}

    def lineage(uid):
        cached = lineage_cache.get(uid)
        if cached is not None:
            return cached
        visited = {uid}
        frontier = {uid}
        games = set(direct_games.get(uid, ()))
        for _ in range(8):
            following = set()
            for current in frontier:
                for parent in parents.get(current, ()):
                    games.update(direct_games.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        result = frozenset(visited), frozenset(games)
        lineage_cache[uid] = result
        return result

    def grounded_actions(uid) -> tuple[int, ...]:
        ancestors, _games = lineage(uid)
        actions = set()
        for ancestor in ancestors:
            row = nodes.get(ancestor)
            if row is None or not normalized.is_grounded_contingency(row):
                continue
            _lineage, games = lineage(ancestor)
            if current_game not in games or len(row.key_parts) < 2:
                continue
            actions.add(int(signed_u64(int(row.key_parts[1]))))
        return tuple(sorted(actions))

    m7: dict[int, list[tuple[float, object, str]]] = {}
    for strategy in tuple(getattr(view, "_strategy_fallback", ())):
        ancestors, formation_games = lineage(strategy.strategy_uid)
        if not any(int(game) != current_game for game in formation_games):
            continue
        if not (
            behavior.strategy_can_control(view, strategy.strategy_uid, strategy.outcome_uid)
            or behavior._strategy_can_probe(view, strategy.strategy_uid, strategy.outcome_uid)
        ):
            continue

        matches: dict[int, float] = {}
        for ancestor in ancestors:
            source_row = nodes.get(ancestor)
            if source_row is None or int(source_row.level) not in {
                int(MemoryLevel.M3), int(MemoryLevel.M4)
            }:
                continue
            for target_uid, corr_score in formal.get(ancestor, ()):
                target_row = nodes.get(target_uid)
                if target_row is None or int(target_row.level) not in {
                    int(MemoryLevel.M3), int(MemoryLevel.M4)
                }:
                    continue
                _target_lineage, target_games = lineage(target_uid)
                if current_game not in target_games:
                    continue
                for action in grounded_actions(target_uid):
                    matches[action] = max(matches.get(action, 0.0), float(corr_score))

        if not matches:
            continue
        node = nodes.get(strategy.strategy_uid)
        transfer_prior = max(0.0, float(getattr(node, "transfer_prior", 0.0))) if node else 0.0
        reliability = max(0.0, float(strategy.reliability))
        efficiency = 1.0 / max(1e-9, float(strategy.mean_cost))
        support = max(0, int(strategy.support))
        probation_penalty = 0.20 if bool(strategy.probationary) else 0.0
        for action, corr_score in matches.items():
            score = (
                reliability
                + 0.10 * efficiency
                + 0.05 * __import__("math").log1p(support)
                + 0.50 * transfer_prior
                + 0.75 * float(corr_score)
                - probation_penalty
            )
            m7.setdefault(int(action), []).append(
                (float(score), strategy.strategy_uid, "M7_CORRESPONDENCE")
            )

    # M1N reuse is permitted only by grounding the SAME normalized structural fact
    # into a current-environment M1G action.  No source action identifier is reused.
    m1n: dict[int, list[tuple[float, object | None, str]]] = {}
    for row in nodes.values():
        if not normalized.is_normalized_contingency(row):
            continue
        grounded = []
        for parent in parents.get(row.uid, ()):
            parent_row = nodes.get(parent)
            if parent_row is None or not normalized.is_grounded_contingency(parent_row):
                continue
            _parents, games = lineage(parent)
            grounded.append((parent_row, games))
        if not grounded:
            continue
        if not any(any(int(game) != current_game for game in games) for _r, games in grounded):
            continue
        current_rows = [parent for parent, games in grounded if current_game in games]
        for parent in current_rows:
            if len(parent.key_parts) < 2:
                continue
            action = int(signed_u64(int(parent.key_parts[1])))
            score = max(
                0.0,
                float(getattr(row, "significance", 0.0)),
                float(getattr(row, "learning_value", 0.0)),
            ) + 0.01 * max(1, int(getattr(row, "support_count", 1)))
            m1n.setdefault(action, []).append((score, None, "M1N_GROUNDED"))

    m7_result = {
        action: tuple(sorted(rows, key=lambda value: (-value[0], int(value[1].hi), int(value[1].lo))))
        for action, rows in m7.items()
    }
    m1n_result = {
        action: tuple(sorted(rows, key=lambda value: -value[0]))
        for action, rows in m1n.items()
    }
    result = (m7_result, m1n_result)
    view._v837_transfer_index_key = cache_key
    view._v837_transfer_index = result
    return result


def _grounded_m7_index_v837(view, game_id: str):
    return _grounded_transfer_index(view, game_id)[0]


def _cross_game_transfer_action_v837(sampler, actions: tuple[int, ...]):
    from v8 import behavior_recovery as behavior

    view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    available = tuple(sorted({int(value) for value in actions}))
    if view is None or not available:
        return None
    m7, m1n = _grounded_transfer_index(view, sampler.game_id)

    choices = []
    for action in available:
        rows = m7.get(action, ())
        if rows:
            score, uid, origin = rows[0]
            choices.append((float(score), int(action), str(origin), uid))
    if choices:
        score, action, origin, uid = min(
            choices,
            key=lambda value: (-value[0], value[1], int(value[3].hi), int(value[3].lo)),
        )
        return action, origin, uid

    fallback = []
    for action in available:
        rows = m1n.get(action, ())
        if rows:
            score, uid, origin = rows[0]
            fallback.append((float(score), int(action), str(origin), uid))
    if not fallback:
        return None
    _score, action, origin, uid = min(fallback, key=lambda value: (-value[0], value[1]))
    return action, origin, uid


def _plan_candidates_grounded_v837(self, context_signature, action_ids, **kwargs):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import learning_performance_repair_v824 as v824
    from v8.publication import PlannedAction

    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    if mode != v819.SamplingMode.TRANSFER.value:
        return _BASE_V826_PLAN(self, context_signature, action_ids, **kwargs)

    game = _current_game_id()
    if not game:
        return ()
    available = tuple(sorted({int(value) for value in action_ids}))
    m7, _m1n = _grounded_transfer_index(self, game)
    rows = []
    for action in available:
        candidates = m7.get(action, ())
        if not candidates:
            continue
        score, strategy_uid, _origin = candidates[0]
        outcome = getattr(self, "_node_by_uid", {}).get(strategy_uid)
        # Strategy rows in the view retain the canonical M6 outcome directly.
        strategy_rows = [
            row for row in tuple(getattr(self, "_strategy_fallback", ()))
            if row.strategy_uid == strategy_uid
        ]
        if not strategy_rows:
            continue
        strategy = strategy_rows[0]
        v824._FOREIGN_TRANSFER_STRATEGIES.add(v824._foreign_key(game, strategy_uid))
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


# ---------------------------------------------------------------------------
# Phase 7/8: generic rollout termination while preserving random floor/budget.
# ---------------------------------------------------------------------------


def _observe_transition_v831_generic(self, **kwargs) -> None:
    from v8 import sampling_portfolio_v831 as portfolio

    intervention = self.base.current
    semantics = _transition_semantics(kwargs)
    if intervention is None or str(intervention.kind) != "SEQUENCE":
        if semantics.successful_boundary:
            self.saw_progress = True
        self.base.observe_transition(**kwargs)
        return

    self.base.current = None
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    after_context = int(kwargs.get("after_context", kwargs.get("before_context", 0)))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    history_after = tuple(int(value) for value in kwargs.get("history_after", ()))

    if not semantics.terminal_failure and after_actions:
        self.base.register_point(
            level=after_level,
            context=after_context,
            anchor=history_after,
            actions=after_actions,
            priority=6 if semantics.successful_boundary else (4 if semantics.productive else 1),
        )

    if semantics.successful_boundary:
        self.saw_progress = True
        self.base.transfer_action = int(kwargs.get("action", intervention.action))
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        self.active_sequence.clear()
        self.active_sequence_full = ()
        self.active_point = None
        self.active_anchor = ()
        portfolio._set_mode(None)
        return

    point = self.active_point or intervention.point_key
    if not semantics.terminal_failure and self.active_sequence:
        return
    self.active_sequence.clear()
    self.active_sequence_full = ()
    self.active_anchor = ()
    self.active_point = None
    if point is not None:
        self._schedule_next_sequence(point)
    portfolio._set_mode(None)


def _observe_persist_v837(self, intervention, **kwargs) -> None:
    from v8 import sampling_persistence_v832 as persistence
    from v8 import sampling_portfolio_v831 as portfolio

    self.base.current = None
    semantics = _transition_semantics(kwargs)
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    after_context = int(kwargs.get("after_context", 0))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    history_after = tuple(int(value) for value in kwargs.get("history_after", ()))
    action = int(kwargs.get("action", intervention.action))

    if not semantics.terminal_failure and after_actions:
        self.base.register_point(
            level=after_level,
            context=after_context,
            anchor=history_after,
            actions=after_actions,
            priority=6 if semantics.successful_boundary else (4 if semantics.productive else 1),
        )
    if semantics.successful_boundary:
        self.saw_progress = True
        self.base.transfer_action = action
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        persistence._clear_persistence_v832(self)
        portfolio._set_mode(None)
        return
    if not semantics.terminal_failure and semantics.productive and action in after_actions:
        portfolio._set_mode(None)
        return
    origin = getattr(self, "_v832_persist_origin", None)
    persistence._clear_persistence_v832(self)
    if not semantics.terminal_failure and origin is not None:
        self._schedule_next_sequence(origin)
    portfolio._set_mode(None)


def _observe_transition_v832_generic(self, **kwargs) -> None:
    from v8 import sampling_persistence_v832 as persistence

    intervention = self.base.current
    if intervention is not None and str(intervention.kind) == "PERSIST":
        _observe_persist_v837(self, intervention, **kwargs)
        return

    sequence = bool(intervention is not None and str(intervention.kind) == "SEQUENCE")
    remaining = len(self.active_sequence) if sequence else -1
    origin = self.active_point or intervention.point_key if sequence and intervention is not None else None
    action = int(kwargs.get("action", intervention.action if intervention is not None else 0))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    semantics = _transition_semantics(kwargs)

    result = persistence._BASE_OBSERVE_TRANSITION(self, **kwargs)
    if sequence:
        if semantics.successful_boundary:
            persistence._clear_persistence_v832(self)
        elif (
            not semantics.terminal_failure
            and remaining == 0
            and semantics.productive
            and action in after_actions
        ):
            self.pending_sequence = None
            persistence._arm_persistence_v832(self, action, origin)
    elif semantics.successful_boundary:
        persistence._clear_persistence_v832(self)
    return result


def _observe_random_v837(self, intervention, **kwargs) -> None:
    from v8 import sampling_portfolio_v831 as portfolio

    self.base.current = None
    semantics = _transition_semantics(kwargs)
    before_level = int(kwargs.get("before_level", 0))
    action = int(kwargs.get("action", intervention.action))
    from v8 import sampling_transfer_v833 as transfer

    transfer._register_destination(
        self,
        kwargs=kwargs,
        priority=6 if semantics.successful_boundary else (3 if semantics.productive else 1),
    )
    if semantics.successful_boundary:
        self.saw_progress = True
        self.base.transfer_action = action
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        self._v833_random_rollout = False
    elif semantics.terminal_failure or not tuple(kwargs.get("after_actions", ())):
        self._v833_random_rollout = False
    else:
        self._v833_random_rollout = True
    portfolio._set_mode(None)


def _observe_transfer_v837(self, intervention, **kwargs) -> None:
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_transfer_v833 as transfer

    self.base.current = None
    semantics = _transition_semantics(kwargs)
    before_level = int(kwargs.get("before_level", 0))
    action = int(kwargs.get("action", intervention.action))
    transfer._register_destination(
        self,
        kwargs=kwargs,
        priority=6 if semantics.successful_boundary else (4 if semantics.productive else 1),
    )
    if semantics.successful_boundary:
        self.saw_progress = True
        self.base.transfer_action = action
        self.base.transfer_from_level = max(int(self.base.transfer_from_level), before_level)
        self._v833_transfer_rollout = False
        self._v833_transfer_origin = ""
    elif semantics.terminal_failure or not semantics.productive or not tuple(kwargs.get("after_actions", ())):
        self._v833_transfer_rollout = False
        self._v833_transfer_origin = ""
    else:
        self._v833_transfer_rollout = True
    portfolio._set_mode(None)


# ---------------------------------------------------------------------------
# Phase 2/3/10: generic replay target, optimization scope and convergence.
# ---------------------------------------------------------------------------


class _EnvironmentReplayValidator:
    def __init__(self, service, environment_scope: str) -> None:
        self.service = service
        self.environment_scope = str(environment_scope)
        self._envs: dict[tuple[int, str | None], object] = {}

    def _environment(self, execution_seed: int, env_root: str | None):
        key = (int(execution_seed), env_root)
        env = self._envs.get(key)
        if env is not None:
            env.reset()
            return env
        factory = getattr(self.service, "_v837_environment_factory", None)
        if factory is None:
            from v7.environment.arc_adapter import ArcGridEnvironment

            env = ArcGridEnvironment(
                game_id=self.environment_scope,
                seed=int(execution_seed),
                env_root=env_root,
            )
            env.game_wait_seconds = 0.0
        else:
            env = factory(
                environment_scope=self.environment_scope,
                seed=int(execution_seed),
                env_root=env_root,
            )
        self._envs[key] = env
        return env

    def _target_reached(self, env, source) -> bool:
        return _generic_target_reached(env, source)

    def _trial(self, candidate, execution_seed: int, prefix: tuple[int, ...]):
        env = self._environment(execution_seed, candidate.source.anchor.env_root)
        prefix_executed = 0
        for action in prefix:
            available = {int(value) for value in env.available_actions()}
            if int(action) not in available:
                return False, 0, "prefix_action_unavailable", 0, 0, 0, prefix_executed
            env.step(int(action))
            prefix_executed += 1
            if _generic_failed_boundary(env):
                return False, 0, "anchor_failed", 0, 0, 0, prefix_executed
        if self._target_reached(env, candidate.source):
            return False, 0, "anchor_already_reaches_target", 0, 0, 0, prefix_executed

        candidate_steps = 0
        for action in candidate.actions:
            available = {int(value) for value in env.available_actions()}
            if int(action) not in available:
                return False, candidate_steps, "candidate_action_unavailable", 0, 0, 0, prefix_executed
            before = env.observe()
            context = _generic_context_signature(env)
            after = env.step(int(action))
            candidate_steps += 1
            outcome = _generic_transition_signature(env, before, after)
            if self._target_reached(env, candidate.source):
                return True, candidate_steps, "target_preserved", context, int(action), outcome, prefix_executed
            if _generic_failed_boundary(env):
                return False, candidate_steps, "candidate_failed", 0, 0, 0, prefix_executed
        return False, candidate_steps, "target_not_reached", 0, 0, 0, prefix_executed

    def validate(self, candidate):
        from v8 import trajectory_optimizer_v818 as v818

        prefix = tuple(self.service._v818_prefix_for(candidate))
        successes = attempts = total_actions = 0
        lengths = []
        last_reason = "target_not_reached"
        terminal_context = terminal_action = outcome_signature = 0
        for seed in v818._VALIDATION_SEEDS:
            attempts += 1
            try:
                ok, steps, reason, context, action, outcome, _prefix_steps = self._trial(candidate, seed, prefix)
            except BaseException as exc:
                ok, steps, reason = False, 0, f"{type(exc).__name__}: {exc}"
                context = action = outcome = 0
            total_actions += int(steps)
            last_reason = str(reason)
            if ok:
                successes += 1
                lengths.append(int(steps))
                terminal_context, terminal_action, outcome_signature = int(context), int(action), int(outcome)
        required = max(1, (len(v818._VALIDATION_SEEDS) + 1) // 2)
        accepted = successes >= required
        from v8.trajectory_target_minimization_v820 import V820ValidationResult

        return V820ValidationResult(
            accepted,
            total_actions,
            "target_preserved" if accepted else last_reason,
            str(getattr(candidate.source.target, "terminal_state", "")),
            int(getattr(candidate.source.target, "levels_completed", 0)),
            attempts,
            successes,
            prefix,
            terminal_context,
            terminal_action,
            outcome_signature,
            tuple(lengths),
        )


def _target_aware_service_v837(service) -> bool:
    if getattr(service, "_v837_environment_factory", None) is not None:
        return True
    try:
        from v8.trajectory_validation_v814 import validate_arc_candidate

        return getattr(service, "validator", None) is validate_arc_candidate
    except BaseException:
        return False


def _global_target_source(source) -> bool:
    if tuple(getattr(source.anchor, "prefix_actions", ())):
        return False
    scope = optimization_scope_for(source)
    return bool(
        scope.kind is OptimizationScopeKind.OUTCOME
        or (
            scope.kind is OptimizationScopeKind.BOUNDARY
            and scope.boundary_scope is BoundaryScope.EPISODE
        )
    )


def _candidate_scope_v837(candidate):
    if not _global_target_source(candidate.source):
        return _BASE_V835_CANDIDATE_SCOPE(candidate)
    scope = optimization_scope_for(candidate.source)
    key = int(scope.legacy_budget_key())
    game = str(candidate.source.anchor.source_id)
    _SCOPE_LABELS[(game, key)] = scope.label()
    return game, key, max(1, int(candidate.source.cost))


def _status_message_v837(coordinator, game_id: str, level: int, status: str) -> str:
    key = (str(game_id), max(1, int(level)))
    label = _SCOPE_LABELS.get(key)
    if not label:
        return _BASE_V835_STATUS(coordinator, game_id, level, status)
    from v8 import optimizer_budget_control_v830 as v830

    with coordinator._lock:
        stats = v830._stats_for(coordinator, game_id, level)
        record = coordinator._record(game_id, level)
        budget, stall, potential = v830._budget_limits(coordinator, game_id, level)
        best = max(0, int(stats.best_cost))
        source = max(0, int(stats.source_cost))
        return (
            f"optimizer environment={str(game_id)} target={label} "
            f"status={str(status)} cost={best or source} potential={potential} "
            f"environment_potential={v830._game_potential(coordinator, game_id)} "
            f"validations={int(stats.validations)} successes={int(stats.successes)} "
            f"saved={int(stats.saved_actions)} "
            f"budget={int(record.consumed_optimization_budget)}/{budget} "
            f"no_progress={int(stats.validations_since_improvement)}/{stall}"
        )


def _scope_override_v837(game_id: str, level: int) -> int:
    from v8 import optimizer_budget_control_v830 as v830
    from v8 import runtime_win_scope_v835 as v835

    game = str(game_id)
    budget_key = getattr(v830._BUDGET_CONTEXT, "key", None)
    if isinstance(budget_key, tuple) and len(budget_key) >= 2 and str(budget_key[0]) == game:
        if (game, int(budget_key[1])) in _SCOPE_LABELS:
            return int(budget_key[1])
    process_key = getattr(v835._FULL_WIN_CONTEXT, "key", None)
    if isinstance(process_key, tuple) and len(process_key) >= 2 and str(process_key[0]) == game:
        if (game, int(process_key[1])) in _SCOPE_LABELS:
            return int(process_key[1])
    return max(1, int(level))


def _process_candidate_v837(service, validator, candidate):
    if not _global_target_source(candidate.source):
        return _BASE_V835_PROCESS(service, validator, candidate)
    from v8 import runtime_win_scope_v835 as v835

    game, key, _cost = _candidate_scope_v837(candidate)
    present = hasattr(v835._FULL_WIN_CONTEXT, "key")
    prior = getattr(v835._FULL_WIN_CONTEXT, "key", None)
    v835._FULL_WIN_CONTEXT.key = (game, key)
    try:
        return _BASE_V835_PROCESS(service, validator, candidate)
    finally:
        if present:
            v835._FULL_WIN_CONTEXT.key = prior
        else:
            try:
                delattr(v835._FULL_WIN_CONTEXT, "key")
            except AttributeError:
                pass


def _submit_next_source_v837(service, candidate, validated) -> None:
    if not _global_target_source(candidate.source):
        return _BASE_V835_SUBMIT_NEXT(service, candidate, validated)
    if validated is None or int(validated.cost) <= 1:
        return
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer

    next_source = optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(validated.anchor, validated.target, validated.actions),
        validated.anchor,
        validated.target,
        validated.actions,
        validated.strategy_uid,
        validated.target_outcome_uid,
        int(candidate.source.round_index) + 1,
    )
    v819._BASE_SERVICE_SUBMIT(service, next_source)


def _prefix_for_v837(service, candidate) -> tuple[int, ...]:
    if _global_target_source(candidate.source) and not tuple(candidate.source.anchor.prefix_actions):
        return ()
    return _BASE_V834_PREFIX(service, candidate)


def _replay_segments_v837(service, candidate):
    if not _global_target_source(candidate.source):
        return _BASE_V836_REPLAY_LEVELS(service, candidate)
    from v8 import trajectory_optimizer_v818 as v818

    prefix = tuple(service._v818_prefix_for(candidate))
    if prefix:
        return None
    actions = tuple(int(value) for value in candidate.actions)
    if not actions:
        return None
    validator = _EnvironmentReplayValidator(service, str(candidate.source.anchor.source_id))
    replays = []
    for seed in v818._VALIDATION_SEEDS:
        env = validator._environment(seed, candidate.source.anchor.env_root)
        indexer = getattr(env, "cognitive_subepisode_index", None)
        prior_index = 0 if indexer is None else int(indexer())
        segments = []
        current = []
        valid = True
        reached = False
        for action in actions:
            if int(action) not in {int(v) for v in env.available_actions()}:
                valid = False
                break
            env.step(int(action))
            current.append(int(action))
            if indexer is not None:
                index = int(indexer())
                if index > prior_index:
                    if index != prior_index + 1:
                        valid = False
                        break
                    segments.append(tuple(current))
                    current = []
                    prior_index = index
            if _generic_target_reached(env, candidate.source):
                reached = True
                if current:
                    segments.append(tuple(current))
                    current = []
                break
            if _generic_failed_boundary(env):
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


# ---------------------------------------------------------------------------
# Runtime observed positive-episode compatibility.
# ---------------------------------------------------------------------------


def _positive_episode_count(result) -> int:
    explicit = getattr(result, "positive_episode_count", None)
    if explicit is not None:
        return max(0, int(explicit))
    return max(0, int(getattr(result, "wins", 0)))


def _write_runtime_positive_episode_marker(game_id: str, result) -> None:
    if _positive_episode_count(result) <= 0:
        return
    from v8 import runtime_win_optimization_v834 as v834
    from v8 import runtime_win_scope_v835 as v835
    from v8 import trajectory_optimizer_v814 as optimizer

    path = v834._marker_path(game_id)
    if path is None:
        return
    optimizer._atomic_json(
        path,
        {
            "environment_scope": str(game_id),
            "game_id": str(game_id),
            "run_session": v835._ensure_run_session(),
            "boundary_scope": BoundaryScope.EPISODE.value,
            "primary_valence": +1,
            "observed_levels": max(0, int(getattr(result, "levels_completed", 0))),
            "steps": max(1, int(getattr(result, "steps", 0))),
        },
    )


def _promote_positive_episode_v837(coordinator, game_id: str) -> bool:
    import json
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import runtime_win_optimization_v834 as v834
    from v8 import runtime_win_scope_v835 as v835

    session = str(os.environ.get(v835._RUN_SESSION_ENV, "")).strip()
    path = v834._marker_path(game_id)
    if not session or path is None or not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    game = str(game_id)
    if str(raw.get("environment_scope", raw.get("game_id", ""))) != game:
        return False
    if str(raw.get("run_session", "")) != session:
        return False
    if str(raw.get("boundary_scope", "")) != BoundaryScope.EPISODE.value:
        return False
    if int(raw.get("primary_valence", 0)) <= 0:
        return False

    from v8.environment_contract import OptimizationScope, OptimizationScopeKind

    scope = OptimizationScope(
        OptimizationScopeKind.BOUNDARY,
        game,
        BoundaryScope.EPISODE,
        +1,
    )
    scope_key = int(scope.legacy_budget_key())
    _SCOPE_LABELS[(game, scope_key)] = scope.label()
    runtime = getattr(coordinator, "_v834_runtime", None)
    generation = max(1, int(getattr(runtime, "generation", 1)))
    with coordinator._lock:
        coordinator.register_games((game,))
        row = coordinator._record(game, scope_key)
        previous = row.state
        coordinator._game_won[game] = True
        if row.first_success_generation <= 0:
            row.first_success_generation = generation
        row.last_success_generation = generation
        row.last_frontier_improvement_generation = max(row.last_frontier_improvement_generation, generation)
        row.optimizer_exhausted_version = -1
        if row.state == v819.GameLearningState.UNSOLVED:
            row.state = v819.GameLearningState.SOLVED_OPTIMIZING
        if previous != row.state:
            coordinator._emit(
                f"learning state environment={game} target={scope.label()} "
                f"{previous.value}->{row.state.value} observed_runtime_boundary=1"
            )
    return True


# ---------------------------------------------------------------------------
# Installation composition.
# ---------------------------------------------------------------------------


def install_environment_neutrality_v837() -> None:
    global _INSTALLED
    global _BASE_V826_PLAN, _BASE_V832_OBSERVE, _BASE_V833_CROSS_GAME
    global _BASE_V834_PREFIX, _BASE_V835_CANDIDATE_SCOPE, _BASE_V835_STATUS
    global _BASE_V835_SCOPE_OVERRIDE, _BASE_V835_PROCESS, _BASE_V835_SUBMIT_NEXT
    global _BASE_V836_GENERATE, _BASE_V836_REPLAY_LEVELS, _BASE_V836_PUBLISH
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import learning_control_continuity_v826 as v826
    from v8 import runtime_win_optimization_v834 as v834
    from v8 import runtime_win_scope_v835 as v835
    from v8 import sampling_persistence_v832 as persistence
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_transfer_v833 as transfer
    from v8 import trajectory_optimizer_convergence_v836 as v836
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818
    from v8 import trajectory_target_minimization_v820 as v820

    # ARC is now one configured adapter implementing the generic cognition boundary.
    ArcGridEnvironment.cognitive_boundary_event = _arc_boundary_event
    ArcGridEnvironment.cognitive_target_reached = _arc_target_reached
    ArcGridEnvironment.cognitive_context_signature = _arc_context_signature
    ArcGridEnvironment.cognitive_transition_signature = _arc_transition_signature
    ArcGridEnvironment.cognitive_subepisode_index = _arc_subepisode_index

    # Generic target with old constructor/serialization compatibility.
    optimizer.TrajectoryTarget = V837TrajectoryTarget

    # Transfer: foreign raw action IDs and normalized global action priors are no
    # longer executable evidence.  Formal correspondence must ground into target M1G.
    _BASE_V833_CROSS_GAME = transfer._cross_game_transfer_action
    transfer._lineage_transfer_index = _grounded_m7_index_v837
    transfer._cross_game_transfer_action = _cross_game_transfer_action_v837

    # Keep v8.26 public planner authority; replace only its TRANSFER delegate.
    _BASE_V826_PLAN = v826._BASE_PLAN_CANDIDATES
    v826._BASE_PLAN_CANDIDATES = _plan_candidates_grounded_v837

    # Generic rollout semantics beneath the historical v8.32 public authority.
    transfer._BASE_OBSERVE_TRANSITION = _observe_transition_v831_generic
    transfer._productive_transition = lambda kwargs: _transition_semantics(kwargs).productive
    transfer._observe_random_v833 = _observe_random_v837
    transfer._observe_transfer_v833 = _observe_transfer_v837
    persistence._productive_transition = lambda kwargs: _transition_semantics(kwargs).productive
    persistence._observe_persist_v832 = _observe_persist_v837
    _BASE_V832_OBSERVE = persistence._observe_transition_v832
    persistence._observe_transition_v832 = _observe_transition_v832_generic
    portfolio.PortfolioSampler.observe_transition = _observe_transition_v832_generic

    # Generic environment replay validator. Existing ARC validator remains an adapter.
    v818._GameReplayValidator = _EnvironmentReplayValidator
    v820._is_arc_validator = _target_aware_service_v837

    # Generic complete target classification and optimizer scope.
    v836._full_win_source = _global_target_source
    v835._is_full_win_source = _global_target_source
    _BASE_V835_CANDIDATE_SCOPE = v835._candidate_scope_v835
    _BASE_V835_STATUS = v835._status_message_v835
    _BASE_V835_SCOPE_OVERRIDE = v835._scope_override
    _BASE_V835_PROCESS = v835._process_candidate_base_v835
    _BASE_V835_SUBMIT_NEXT = v835._submit_next_source_v835
    v835._candidate_scope_v835 = _candidate_scope_v837
    v835._status_message_v835 = _status_message_v837
    v835._scope_override = _scope_override_v837
    v835._process_candidate_base_v835 = _process_candidate_v837
    v835._submit_next_source_v835 = _submit_next_source_v837

    # v8.30 calls these module attributes dynamically.
    from v8 import optimizer_budget_control_v830 as v830
    v830._candidate_scope = _candidate_scope_v837
    v830._status_message = _status_message_v837
    v830._BASE_PROCESS_CANDIDATE = _process_candidate_v837
    v820._submit_next_source = _submit_next_source_v837

    # Generic no-prefix global target semantics underneath v8.34.
    _BASE_V834_PREFIX = v834._prefix_for_v834
    v834._prefix_for_v834 = _prefix_for_v837
    v818._prefix_for = _prefix_for_v837

    # Runtime positive-episode marker remains compatible with old result fields.
    v834._write_runtime_win_marker = _write_runtime_positive_episode_marker
    v834._promote_runtime_win_if_present = _promote_positive_episode_v837

    # Keep the historical compatibility constant as a storage-key alias only.
    from v8.environment_contract import OptimizationScope, OptimizationScopeKind
    episode_scope = OptimizationScope(
        OptimizationScopeKind.BOUNDARY,
        "",
        BoundaryScope.EPISODE,
        +1,
    )
    v835._FULL_WIN_SCOPE_LEVEL = int(episode_scope.legacy_budget_key())

    # v8.36 convergence applies to any global comparable target, not ARC WIN only.
    _BASE_V836_GENERATE = v836._generate_v836
    _BASE_V836_REPLAY_LEVELS = v836._replay_full_win_levels
    _BASE_V836_PUBLISH = v836._publish_optimized_solution_v836
    v836._replay_full_win_levels = _replay_segments_v837

    _INSTALLED = True

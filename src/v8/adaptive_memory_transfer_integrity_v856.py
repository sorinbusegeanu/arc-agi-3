from __future__ import annotations

"""v8.56 M7 credit, exploration-floor and cross-game transfer integrity.

This layer closes the remaining v8.55 control-loop gaps without changing the arena
schema.  It keeps the portfolio's hard RANDOM floor, makes the adaptive floor an
additional conditional floor rather than double-counting RANDOM, preserves an
already selected composite procedure across portfolio slots, publishes every
executed M7 plan into the canonical behavior-credit state, uses recent strategy
failures instead of lifetime reliability as failure evidence, and restores the
compact transfer subgraph required by cross-game M7/M1N transfer.
"""

import time
from dataclasses import replace

from v8.arena import EdgeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.publication import PlannedAction


_INSTALLED = False
_BASE_PLAN_CHAIN = None
_BASE_FORCED_DELEGATE = None
_BASE_CROSS_GAME = None
_BASE_TRANSFER_INDEX = None
_BASE_ORDERED_SEQUENCES = None
_BASE_OBSERVED = None
_BASE_RESET_CONTROL = None
_BASE_ARM_COMPOSITE = None
_BASE_ACTOR_REFRESH = None

_PORTFOLIO_RANDOM_FLOOR = 0.10
_TOTAL_COLD_EXPLORATION_FLOOR = 0.25
_TOTAL_WARM_EXPLORATION_FLOOR = 0.15
_CONDITIONAL_COLD_FLOOR = (
    _TOTAL_COLD_EXPLORATION_FLOOR - _PORTFOLIO_RANDOM_FLOOR
) / (1.0 - _PORTFOLIO_RANDOM_FLOOR)
_CONDITIONAL_WARM_FLOOR = (
    _TOTAL_WARM_EXPLORATION_FLOOR - _PORTFOLIO_RANDOM_FLOOR
) / (1.0 - _PORTFOLIO_RANDOM_FLOOR)
_MAX_RECENT_FAILURES = 6
_MAX_RECENT_FAILURE_KEYS = 8192
_TRANSFER_BACKOFF_LIMIT = 4
_RECENT_FAILURES: dict[tuple[str, int, int, int], int] = {}
_TARGET_TRANSFER_FAILURES: dict[tuple[str, int, int], int] = {}

_TRANSFER_EDGE_RELATIONS = {
    int(RelationType.SIMILAR_TO),
    int(RelationType.TRANSFER_CORRESPONDENCE),
}


def combined_exploration_floor_v856(*, warm: bool) -> float:
    conditional = _CONDITIONAL_WARM_FLOOR if bool(warm) else _CONDITIONAL_COLD_FLOOR
    return _PORTFOLIO_RANDOM_FLOOR + (1.0 - _PORTFOLIO_RANDOM_FLOOR) * conditional


def _failure_key(game: str, context: int, strategy_uid) -> tuple[str, int, int, int]:
    return (
        str(game),
        int(context),
        int(strategy_uid.hi),
        int(strategy_uid.lo),
    )


def _transfer_failure_key(game: str, strategy_uid) -> tuple[str, int, int]:
    return str(game), int(strategy_uid.hi), int(strategy_uid.lo)


def _bounded_set(mapping: dict, key, value: int, *, limit: int) -> int:
    value = max(0, int(value))
    if value <= 0:
        mapping.pop(key, None)
        return 0
    if key not in mapping and len(mapping) >= int(limit):
        mapping.pop(next(iter(mapping)), None)
    mapping[key] = value
    return value


def _record_strategy_result(game: str, context: int, strategy_uid, *, success: bool, foreign: bool) -> None:
    key = _failure_key(game, context, strategy_uid)
    current = max(0, int(_RECENT_FAILURES.get(key, 0)))
    _bounded_set(
        _RECENT_FAILURES,
        key,
        0 if success else min(_MAX_RECENT_FAILURES, current + 1),
        limit=_MAX_RECENT_FAILURE_KEYS,
    )
    if foreign:
        transfer_key = _transfer_failure_key(game, strategy_uid)
        transfer_current = max(0, int(_TARGET_TRANSFER_FAILURES.get(transfer_key, 0)))
        _bounded_set(
            _TARGET_TRANSFER_FAILURES,
            transfer_key,
            0 if success else min(_MAX_RECENT_FAILURES, transfer_current + 1),
            limit=_MAX_RECENT_FAILURE_KEYS,
        )


def _strategy_failure_evidence_v856(node) -> int:
    if node is None or not hasattr(node, "uid"):
        return 0
    try:
        from v8 import sampling_progress_control_v829 as v829

        game = str(getattr(v829._CONTROL_STATE, "game_id", ""))
        context = getattr(v829._CONTROL_STATE, "context", None)
    except (AttributeError, ImportError):
        return 0
    if not game or context is None:
        return 0
    return max(0, int(_RECENT_FAILURES.get(_failure_key(game, int(context), node.uid), 0)))


def _current_view():
    try:
        from v8 import behavior_recovery as behavior

        return getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    except (AttributeError, ImportError):
        return None


def _publish_plans(view, plans) -> tuple:
    rows = tuple(plans)
    if view is not None:
        view._behavior_last_plans = rows
    return rows


def _clear_current_plans() -> None:
    view = _current_view()
    if view is not None:
        view._behavior_last_plans = ()


def _forced_delegate_v856(self, *args, **kwargs):
    """Every decision starts with no stale M7 credit; transfer may republish one."""
    _clear_current_plans()
    return _BASE_FORCED_DELEGATE(self, *args, **kwargs)


def _arm_composite_v856(self, context_signature: int, plan) -> None:
    _BASE_ARM_COMPOSITE(self, context_signature, plan)
    active = getattr(self, "_v055_active_sequence", None)
    if active is None or active[0] != plan.strategy_uid:
        return
    try:
        from v8 import sampling_progress_control_v829 as v829

        game = str(getattr(v829._CONTROL_STATE, "game_id", ""))
    except (AttributeError, ImportError):
        game = ""
    self._v856_composite_origin = (game, int(context_signature), plan.strategy_uid)


def _plan_chain_v856(self, context_signature, action_ids, **kwargs):
    """Publish canonical plan state and let an active composite cross RANDOM slots."""
    from v8 import adaptive_memory_control_v855_fixups as fixups
    from v8 import sampling_progress_control_v829 as v829

    available = tuple(sorted({int(value) for value in action_ids}))
    _publish_plans(self, ())

    if v829._discovery_mode() and available:
        state = v829._state_key(context_signature)
        progress_action = None if state is None else v829._PROGRESS_ACTION.get(state)
        active_before = getattr(self, "_v055_active_sequence", None)
        if progress_action not in available and active_before is not None:
            rows = tuple(fixups._continue_composite(self, int(context_signature), available))
            if rows:
                v829._set_selection(
                    context_signature,
                    "M7_ADAPTIVE",
                    (row.action_id for row in rows),
                )
                return _publish_plans(self, rows)
            if getattr(self, "_v055_active_sequence", None) is None:
                origin = getattr(self, "_v856_composite_origin", None)
                if origin is not None:
                    game, origin_context, strategy_uid = origin
                    _record_strategy_result(
                        game,
                        int(origin_context),
                        strategy_uid,
                        success=False,
                        foreign=False,
                    )
                    self._v856_composite_origin = None

    rows = tuple(_BASE_PLAN_CHAIN(self, context_signature, available, **kwargs))
    return _publish_plans(self, rows)


def _plan_for_strategy(view, action: int, strategy_uid) -> PlannedAction | None:
    if view is None or strategy_uid is None:
        return None
    row = getattr(view, "_node_by_uid", {}).get(strategy_uid)
    if row is None or len(getattr(row, "key_parts", ())) < 3:
        return None
    outcome_uid = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
    reliability = max(0.0, min(1.0, float(getattr(row, "strategy_reliability", 0.0))))
    return PlannedAction(int(action), outcome_uid, strategy_uid, reliability, False)


def _cross_game_v856(sampler, actions):
    """Expose sampler-driven foreign M7 use to the same credit/trace path as planner M7."""
    context = None
    try:
        import inspect

        caller = inspect.currentframe().f_back
        if caller is not None:
            context = caller.f_locals.get("context")
    except (AttributeError, ValueError):
        context = None

    selected = _BASE_CROSS_GAME(sampler, actions)
    view = _current_view()
    if selected is None:
        _publish_plans(view, ())
        return None
    action, _origin, strategy_uid = selected
    if strategy_uid is None:
        _publish_plans(view, ())
        return selected
    plan = _plan_for_strategy(view, int(action), strategy_uid)
    _publish_plans(view, () if plan is None else (plan,))
    if context is not None and view is not None:
        view._v856_transfer_context = int(context)
    return selected


def _transfer_index_v856(view, game_id: str):
    """Use target-local runtime failure only as operational backoff, never validation."""
    raw = _BASE_TRANSFER_INDEX(view, game_id)
    result = {}
    for action, rows in raw.items():
        adjusted = []
        for score, strategy_uid, origin in rows:
            failures = max(
                0,
                int(
                    _TARGET_TRANSFER_FAILURES.get(
                        _transfer_failure_key(str(game_id), strategy_uid),
                        0,
                    )
                ),
            )
            if failures >= _TRANSFER_BACKOFF_LIMIT:
                continue
            adjusted.append((float(score) - 0.15 * failures, strategy_uid, origin))
        if adjusted:
            adjusted.sort(key=lambda item: (-float(item[0]), item[1]))
            result[int(action)] = tuple(adjusted)
    return result


def _ordered_sequences_v856(view, game_id: str):
    rows = tuple(_BASE_ORDERED_SEQUENCES(view, game_id))
    adjusted = []
    for row in rows:
        failures = max(
            0,
            int(
                _TARGET_TRANSFER_FAILURES.get(
                    _transfer_failure_key(str(game_id), row.strategy_uid),
                    0,
                )
            ),
        )
        if failures >= _TRANSFER_BACKOFF_LIMIT:
            continue
        adjusted.append(replace(row, score=float(row.score) - 0.15 * failures))
    adjusted.sort(key=lambda row: (-float(row.score), row.strategy_uid))
    return tuple(adjusted)


def _observed_outcomes_v856(**kwargs):
    result = _BASE_OBSERVED(**kwargs)
    view = _current_view()
    if view is None:
        return result
    last_action = getattr(view, "_behavior_last_action", None)
    plans = tuple(getattr(view, "_behavior_last_plans", ()))
    if last_action is None or not plans:
        return result
    action = int(last_action[1])
    plan = next((row for row in plans if int(row.action_id) == action), None)
    if plan is None:
        return result

    try:
        from v8 import learning_blockers_v055 as blockers
        from v8 import sampling_progress_control_v829 as v829

        game = str(getattr(v829._CONTROL_STATE, "game_id", ""))
    except (AttributeError, ImportError):
        return result
    if not game:
        return result

    context = int(last_action[0])
    observed = {uid for uid in result if not uid.is_zero}
    success = bool(plan.outcome_uid in observed or int(kwargs.get("terminal_polarity", 0)) > 0)
    row = getattr(view, "_node_by_uid", {}).get(plan.strategy_uid)
    composite = bool(row is not None and blockers.is_composite_strategy(row))

    if composite and success:
        if getattr(view, "_v055_active_sequence", None) is not None:
            view._v055_active_sequence = None
        view._v856_composite_origin = None
    elif composite:
        active_local = getattr(view, "_v055_active_sequence", None)
        if (
            active_local is not None
            and active_local[0] == plan.strategy_uid
            and bool(active_local[2])
        ):
            return result
        transfer_state = getattr(view, "_v854_transfer_active", {}).get(game)
        if transfer_state is not None:
            sequence, index = transfer_state
            if sequence.strategy_uid == plan.strategy_uid and int(index) < len(sequence.path):
                return result

    try:
        formation_games = set(int(value) for value in view.source_games(plan.strategy_uid))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        formation_games = set()
    from v8.model import stable_u64

    current_game = int(stable_u64(game, person=b"v8-game"))
    foreign = bool(formation_games and current_game not in formation_games)
    _record_strategy_result(
        game,
        context,
        plan.strategy_uid,
        success=success,
        foreign=foreign,
    )
    return result


def _stable_edge_rows(arena, wanted) -> tuple[EdgeRecord, ...]:
    for _attempt in range(20):
        before = int(arena.sequence)
        if before & 1:
            time.sleep(0.0005)
            continue
        rows = []
        count = int(arena.count)
        for index in range(count):
            edge = arena.read(index)
            if wanted(edge):
                rows.append(edge)
        after = int(arena.sequence)
        if before == after and not (after & 1):
            return tuple(rows)
    raise RuntimeError("actor transfer edge scan could not obtain stable arena")


def _augment_actor_transfer_cut(view) -> None:
    """Restore only the low-level/correspondence records needed by transfer."""
    version = tuple(getattr(view, "_strategy_version", ()))
    if getattr(view, "_v856_transfer_cut_version", None) == version:
        return

    by_uid = getattr(view, "_node_by_uid", {})
    grounded = {
        uid
        for uid, row in by_uid.items()
        if int(getattr(row, "level", -1)) == int(MemoryLevel.M1)
        and int(getattr(row, "memory_type", -1)) == int(MemoryType.CONTINGENCY)
        and len(getattr(row, "key_parts", ())) >= 4
    }
    for values in getattr(view, "_behavior_strategy_dependencies", {}).values():
        grounded.update(values)
    relevant = set(by_uid)

    normalized_uids = set()
    correspondence = []
    for arena in getattr(view, "_edges", ()):
        rows = _stable_edge_rows(
            arena,
            lambda edge: int(edge.relation_type) == int(RelationType.EXPLAINS)
            or int(edge.relation_type) in _TRANSFER_EDGE_RELATIONS,
        )
        for edge in rows:
            relation = int(edge.relation_type)
            if relation == int(RelationType.EXPLAINS) and edge.target_uid in grounded:
                normalized_uids.add(edge.source_uid)
            elif relation in _TRANSFER_EDGE_RELATIONS and (
                edge.source_uid in relevant or edge.target_uid in relevant
            ):
                correspondence.append(edge)

    grounding_edges = []
    if normalized_uids:
        for arena in getattr(view, "_edges", ()):
            grounding_edges.extend(
                _stable_edge_rows(
                    arena,
                    lambda edge: int(edge.relation_type) == int(RelationType.EXPLAINS)
                    and edge.source_uid in normalized_uids,
                )
            )

    needed = set(normalized_uids)
    for edge in grounding_edges:
        needed.add(edge.source_uid)
        needed.add(edge.target_uid)
    for edge in correspondence:
        needed.add(edge.source_uid)
        needed.add(edge.target_uid)

    loader = getattr(type(view), "_load_needed_low", None)
    if callable(loader) and needed:
        loaded = loader(getattr(view, "_nodes", ()), needed - set(by_uid))
        by_uid.update(loaded)

    provenance_loader = getattr(type(view), "_load_provenance_games", None)
    if callable(provenance_loader) and needed:
        extra_games = provenance_loader(getattr(view, "_edges", ()), needed)
        direct = getattr(view, "_source_games_direct", {})
        for uid, games in extra_games.items():
            direct.setdefault(uid, set()).update(int(value) for value in games)
        view._source_games_direct = direct
        view._source_games_cache = {}

    # Actor transfer code historically consumed edge_records(), so expose exact
    # provenance there without retaining the entire raw GAME_PROVENANCE arena.
    provenance_edges = []
    for uid, games in getattr(view, "_source_games_direct", {}).items():
        if uid not in by_uid:
            continue
        for game_hash in games:
            provenance_edges.append(
                EdgeRecord(
                    uid,
                    int(RelationType.GAME_PROVENANCE),
                    MemoryUid(0, int(game_hash)),
                    1,
                    0,
                )
            )

    extras = tuple(grounding_edges) + tuple(correspondence) + tuple(provenance_edges)
    existing = list(getattr(view, "_v851_compact_edges", ()))
    seen = {
        (edge.source_uid, int(edge.relation_type), edge.target_uid)
        for edge in existing
    }
    for edge in extras:
        key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
        if key in seen:
            continue
        seen.add(key)
        existing.append(edge)
        if int(edge.relation_type) in {
            int(RelationType.PROVENANCE),
            int(RelationType.EXPLAINS),
            int(RelationType.LEADS_TO),
            int(RelationType.CONTEXT_REFINES),
            int(RelationType.DEPENDS_ON),
        }:
            getattr(view, "_parents", {}).setdefault(edge.source_uid, set()).add(edge.target_uid)

    view._node_by_uid = by_uid
    view._v851_compact_nodes = tuple(by_uid.values())
    view._v851_compact_edges = tuple(existing)
    view._v856_transfer_cut_version = version
    publish = getattr(view, "_publish_compact_cut", None)
    if callable(publish):
        publish()


def _actor_refresh_v856(self) -> None:
    if bool(getattr(self, "_v856_refreshing", False)):
        return
    self._v856_refreshing = True
    try:
        _BASE_ACTOR_REFRESH(self)
        _augment_actor_transfer_cut(self)
    finally:
        self._v856_refreshing = False


def _reset_v856() -> None:
    _BASE_RESET_CONTROL()
    _RECENT_FAILURES.clear()
    _TARGET_TRANSFER_FAILURES.clear()


def install_adaptive_memory_transfer_integrity_v856() -> None:
    global _INSTALLED, _BASE_PLAN_CHAIN, _BASE_FORCED_DELEGATE, _BASE_CROSS_GAME
    global _BASE_TRANSFER_INDEX, _BASE_ORDERED_SEQUENCES, _BASE_OBSERVED
    global _BASE_RESET_CONTROL, _BASE_ARM_COMPOSITE, _BASE_ACTOR_REFRESH
    if _INSTALLED:
        return

    from v8 import adaptive_memory_control_v855 as v855
    from v8 import adaptive_memory_control_v855_fixups as fixups
    from v8 import actor as actor_module
    from v8 import learning_performance_repair_v824 as v824
    from v8 import learning_transfer_correctness_v854 as v854
    from v8 import sampling_persistence_v832 as persistence
    from v8 import sampling_progress_control_v829 as v829
    from v8 import sampling_transfer_v833 as transfer
    from v8.actor_read_view_v851 import ActorReadView

    # The 10% portfolio RANDOM floor is already unconditional.  These conditional
    # floors make the combined minimum exactly 25% cold / 15% warm.
    v855._COLD_EXPLORATION_FLOOR = _CONDITIONAL_COLD_FLOOR
    v855._WARM_EXPLORATION_FLOOR = _CONDITIONAL_WARM_FLOOR

    _BASE_RESET_CONTROL = v855._reset_adaptive_memory_control_v855
    v855._reset_adaptive_memory_control_v855 = _reset_v856

    # Recent, target-local execution evidence is the only adaptive failure term.
    fixups._strategy_failure_evidence = _strategy_failure_evidence_v856

    _BASE_ARM_COMPOSITE = fixups._arm_composite
    fixups._arm_composite = _arm_composite_v856

    _BASE_PLAN_CHAIN = v829._plan_chain_v829
    v829._plan_chain_v829 = _plan_chain_v856
    v824._BASE_PLAN_CHAIN = v829._plan_chain_v829

    # Preserve the public v8.32 sampler authority while clearing stale credit at
    # its lower forced-action delegate.
    _BASE_FORCED_DELEGATE = persistence._BASE_FORCED_ACTION
    persistence._BASE_FORCED_ACTION = _forced_delegate_v856

    # Cross-game sampler actions now publish their originating M7 plan.
    _BASE_CROSS_GAME = transfer._cross_game_transfer_action
    transfer._cross_game_transfer_action = _cross_game_v856

    # Runtime transfer backoff is target-specific and does not alter formal
    # transfer validation state.
    _BASE_TRANSFER_INDEX = transfer._lineage_transfer_index
    transfer._lineage_transfer_index = _transfer_index_v856
    _BASE_ORDERED_SEQUENCES = v854._ordered_sequences
    v854._ordered_sequences = _ordered_sequences_v856

    _BASE_OBSERVED = actor_module._observed_outcome_uids
    actor_module._observed_outcome_uids = _observed_outcomes_v856

    # Compact actors regain only the transfer-relevant low-level subgraph.
    _BASE_ACTOR_REFRESH = ActorReadView._refresh_strategy_cache
    ActorReadView._refresh_strategy_cache = _actor_refresh_v856

    _INSTALLED = True

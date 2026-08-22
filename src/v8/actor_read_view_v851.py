from __future__ import annotations

"""v8.51 compact actor-only graph read view.

Actors retain the M7/M6/control lineage and aggregated M1 prediction information they
actually use, instead of caching every NodeRecord and EdgeRecord in every process.
"""

import time

from v8.arena import EdgeRecord, NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, signed_u64
from v8.publication import LiveReadView, _StrategyRow


_LINEAGE_RELATIONS = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
}
_ACTIVE_STATES = {
    int(CognitiveState.ACTIVE),
    int(CognitiveState.VALIDATED),
    int(CognitiveState.REACTIVATED),
}
_PROBE_STATES = _ACTIVE_STATES | {
    int(CognitiveState.CANDIDATE),
    int(CognitiveState.PROBATION),
}
_COMPACT_CUT_KEY = ("v851", "actor-compact-cut")
_COMPACT_CUT_SCHEMA = 1


class ActorReadView(LiveReadView):
    """LiveReadView-compatible compact actor representation."""

    def __init__(self, *args, **kwargs) -> None:
        # Calling the fully patched LiveReadView initializer is important because
        # behavior/action layers add actor state there. Dynamic dispatch to our
        # bootstrap method prevents that initializer from decoding the full graph.
        self._v851_bootstrapping = True
        self._v851_ready = False
        super().__init__(*args, **kwargs)
        self._record_cache.clear()
        self._v839_node_query_cache = {}
        self._v851_compact_nodes: tuple[NodeRecord, ...] = ()
        self._v851_compact_edges: tuple[EdgeRecord, ...] = ()
        self._v851_ready = True
        self._v851_bootstrapping = False
        self._strategy_cache_stale = True
        self._strategy_version = ()
        self._restore_compact_cut()

    def _restore_compact_cut(self) -> bool:
        cuts = getattr(self, "_record_cuts", None)
        if cuts is None:
            return False
        raw = cuts.get(_COMPACT_CUT_KEY)
        if not isinstance(raw, tuple) or len(raw) != 2:
            return False
        payload, schema = raw
        if schema != _COMPACT_CUT_SCHEMA or not isinstance(payload, tuple):
            return False
        if len(payload) != 11:
            return False
        (
            self._parents,
            self._node_by_uid,
            self._strategy_by_context,
            self._strategy_fallback,
            self._preferred_outcomes,
            self._suppressed_outcomes,
            self._refined_action_scores,
            self._outcome_counts,
            self._outcome_totals,
            self._v851_compact_nodes,
            edge_state,
        ) = payload
        self._v851_compact_edges, self._strategy_version = edge_state
        self._strategy_cache_stale = False
        if self._refresh_interval_seconds is not None:
            self._next_strategy_refresh = time.monotonic() + self._refresh_interval_seconds
        try:
            from v8 import behavior_recovery

            behavior_recovery._refresh_behavior_indexes(self)
        except (AttributeError, ImportError):
            pass
        return True

    def _publish_compact_cut(self) -> None:
        cuts = getattr(self, "_record_cuts", None)
        if cuts is None:
            return
        payload = (
            self._parents,
            self._node_by_uid,
            self._strategy_by_context,
            self._strategy_fallback,
            self._preferred_outcomes,
            self._suppressed_outcomes,
            self._refined_action_scores,
            self._outcome_counts,
            self._outcome_totals,
            self._v851_compact_nodes,
            (self._v851_compact_edges, self._strategy_version),
        )
        cuts.clear()
        cuts[_COMPACT_CUT_KEY] = (payload, _COMPACT_CUT_SCHEMA)

    def _warm_compact_cut(self) -> None:
        """Build the reusable coherent actor cut before writers resume."""
        self._strategy_cache_stale = True
        self._refresh_strategy_cache()

    def _stable_records_with_version(self, arena, *, timeout: float = 1.0):
        if getattr(self, "_v851_bootstrapping", False) or not getattr(self, "_v851_ready", False):
            return (), int(arena.sequence)
        # Compatibility path only: return a coherent cut but never retain it in the
        # actor's record cache.
        return arena.snapshot_records(timeout=timeout)

    def node_records(self, *, level=None):
        self._refresh_strategy_cache()
        wanted = None if level is None else int(level)
        if wanted is None:
            return self._v851_compact_nodes
        return tuple(row for row in self._v851_compact_nodes if int(row.level) == wanted)

    def edge_records(self):
        self._refresh_strategy_cache()
        return self._v851_compact_edges

    @staticmethod
    def _scan_node_arena(arena):
        try:
            from v8 import action_targeting_v810

            legacy_payload = action_targeting_v810._legacy_coordinate_payload
        except (ImportError, AttributeError):
            legacy_payload = None
        for _attempt in range(20):
            before = int(arena.sequence)
            if before & 1:
                time.sleep(0.0005)
                continue
            count = int(arena.count)
            node_by_uid: dict[MemoryUid, NodeRecord] = {}
            lineage_uids: set[MemoryUid] = set()
            active_uids: set[MemoryUid] = set()
            outcome_counts: dict[tuple[int, int], dict[int, int]] = {}
            outcome_totals: dict[tuple[int, int], int] = {}
            for index in range(count):
                row = arena.read(index)
                level = int(row.level)
                if level >= int(MemoryLevel.M2):
                    lineage_uids.add(row.uid)
                if level >= int(MemoryLevel.M3):
                    node_by_uid[row.uid] = row
                    if int(row.cognitive_state) in _ACTIVE_STATES:
                        active_uids.add(row.uid)
                if level == int(MemoryLevel.M1) and len(row.key_parts) >= 2:
                    # v8.10 may still consult exact legacy click contingencies. Keep
                    # only those M1 rows; ordinary M1 is aggregated below.
                    try:
                        if legacy_payload is not None and legacy_payload(
                            signed_u64(int(row.key_parts[1]))
                        ) is not None:
                            node_by_uid[row.uid] = row
                    except (TypeError, ValueError):
                        pass
                if level == int(MemoryLevel.M1) and len(row.key_parts) >= 3:
                    key = (int(row.key_parts[0]), signed_u64(int(row.key_parts[1])))
                    support = max(0, int(row.support_count))
                    outcome = int(row.key_parts[2])
                    bucket = outcome_counts.setdefault(key, {})
                    bucket[outcome] = bucket.get(outcome, 0) + support
                    outcome_totals[key] = outcome_totals.get(key, 0) + support
            after = int(arena.sequence)
            if before == after and not (after & 1):
                return (
                    before,
                    node_by_uid,
                    lineage_uids,
                    active_uids,
                    outcome_counts,
                    outcome_totals,
                )
        raise RuntimeError("actor compact node scan could not obtain stable arena")

    @staticmethod
    def _scan_edge_arena(arena, *, lineage_uids, active_uids, node_by_uid):
        for _attempt in range(20):
            before = int(arena.sequence)
            if before & 1:
                time.sleep(0.0005)
                continue
            count = int(arena.count)
            parents: dict[MemoryUid, set[MemoryUid]] = {}
            preferred: set[MemoryUid] = set()
            suppressed: set[MemoryUid] = set()
            context_edges: list[EdgeRecord] = []
            compact_edges: list[EdgeRecord] = []
            needed_low: set[MemoryUid] = set()
            for index in range(count):
                edge = arena.read(index)
                relation = int(edge.relation_type)
                keep = False
                if relation in _LINEAGE_RELATIONS and edge.source_uid in lineage_uids:
                    parents.setdefault(edge.source_uid, set()).add(edge.target_uid)
                    if edge.target_uid not in node_by_uid:
                        needed_low.add(edge.target_uid)
                    keep = True
                if relation == int(RelationType.PREFERENCE) and edge.source_uid in active_uids:
                    preferred.add(edge.source_uid)
                    keep = True
                if relation == int(RelationType.SUPERSEDES):
                    if edge.source_uid in active_uids:
                        suppressed.add(edge.target_uid)
                    if edge.source_uid in node_by_uid or edge.target_uid in node_by_uid:
                        keep = True
                        if edge.target_uid not in node_by_uid:
                            needed_low.add(edge.target_uid)
                if relation == int(RelationType.CONTEXT_REFINES) and edge.source_uid in node_by_uid:
                    context_edges.append(edge)
                    needed_low.add(edge.target_uid)
                    keep = True
                if relation == int(RelationType.DEPENDS_ON) and edge.source_uid in node_by_uid:
                    needed_low.add(edge.target_uid)
                    keep = True
                if keep:
                    compact_edges.append(edge)
            after = int(arena.sequence)
            if before == after and not (after & 1):
                return (
                    before,
                    parents,
                    preferred,
                    suppressed,
                    context_edges,
                    compact_edges,
                    needed_low,
                )
        raise RuntimeError("actor compact edge scan could not obtain stable arena")

    @staticmethod
    def _load_needed_low(nodes, needed: set[MemoryUid]) -> dict[MemoryUid, NodeRecord]:
        if not needed:
            return {}
        result: dict[MemoryUid, NodeRecord] = {}
        remaining = set(needed)
        for arena in nodes:
            for _attempt in range(20):
                before = int(arena.sequence)
                if before & 1:
                    time.sleep(0.0005)
                    continue
                local: dict[MemoryUid, NodeRecord] = {}
                count = int(arena.count)
                for index in range(count):
                    row = arena.read(index)
                    if row.uid in remaining:
                        local[row.uid] = row
                after = int(arena.sequence)
                if before == after and not (after & 1):
                    result.update(local)
                    remaining.difference_update(local)
                    break
            if not remaining:
                break
        return result

    def _refresh_strategy_cache(self) -> None:
        now = time.monotonic()
        current = tuple(int(arena.sequence) for arena in (*self._nodes, *self._edges))
        if (
            not self._strategy_cache_stale
            and current == tuple(self._strategy_version)
            and not any(value & 1 for value in current)
        ):
            return

        node_by_uid: dict[MemoryUid, NodeRecord] = {}
        lineage_uids: set[MemoryUid] = set()
        active_uids: set[MemoryUid] = set()
        outcome_counts: dict[tuple[int, int], dict[int, int]] = {}
        outcome_totals: dict[tuple[int, int], int] = {}
        node_versions = []
        for arena in self._nodes:
            (
                version,
                local_nodes,
                local_lineage,
                local_active,
                local_counts,
                local_totals,
            ) = self._scan_node_arena(arena)
            node_versions.append(int(version))
            node_by_uid.update(local_nodes)
            lineage_uids.update(local_lineage)
            active_uids.update(local_active)
            for key, counts in local_counts.items():
                bucket = outcome_counts.setdefault(key, {})
                for outcome, support in counts.items():
                    bucket[outcome] = bucket.get(outcome, 0) + int(support)
            for key, total in local_totals.items():
                outcome_totals[key] = outcome_totals.get(key, 0) + int(total)

        parents: dict[MemoryUid, set[MemoryUid]] = {}
        preferred: set[MemoryUid] = set()
        suppressed: set[MemoryUid] = set()
        context_edges: list[EdgeRecord] = []
        compact_edges: list[EdgeRecord] = []
        needed_low: set[MemoryUid] = set()
        edge_versions = []
        for arena in self._edges:
            (
                version,
                local_parents,
                local_preferred,
                local_suppressed,
                local_context,
                local_edges,
                local_needed,
            ) = self._scan_edge_arena(
                arena,
                lineage_uids=lineage_uids,
                active_uids=active_uids,
                node_by_uid=node_by_uid,
            )
            edge_versions.append(int(version))
            for uid, values in local_parents.items():
                parents.setdefault(uid, set()).update(values)
            preferred.update(local_preferred)
            suppressed.update(local_suppressed)
            context_edges.extend(local_context)
            compact_edges.extend(local_edges)
            needed_low.update(local_needed)

        node_by_uid.update(self._load_needed_low(self._nodes, needed_low - set(node_by_uid)))

        refined: dict[tuple[int, int], list[float]] = {}
        for edge in context_edges:
            role = node_by_uid.get(edge.source_uid)
            source = node_by_uid.get(edge.target_uid)
            if (
                role is not None
                and source is not None
                and int(role.memory_type) == int(MemoryType.CONTEXTUAL_ROLE)
                and len(source.key_parts) >= 2
                and int(role.cognitive_state) in _PROBE_STATES
                and int(role.support_count) > 0
            ):
                key = (int(source.key_parts[0]), signed_u64(int(source.key_parts[1])))
                value = max(0.0, float(role.learning_value), float(role.significance))
                weight = max(1, int(role.support_count))
                bucket = refined.setdefault(key, [0.0, 0.0])
                bucket[0] += weight
                bucket[1] += value * weight

        self._parents = parents
        self._node_by_uid = node_by_uid
        by_context: dict[int, list[_StrategyRow]] = {}
        fallback: list[_StrategyRow] = []
        for row in node_by_uid.values():
            if (
                int(row.level) != int(MemoryLevel.M7)
                or int(row.memory_type) != int(MemoryType.STRATEGY)
                or len(row.key_parts) < 4
                or int(row.cognitive_state) not in _PROBE_STATES
            ):
                continue
            action = signed_u64(int(row.key_parts[0]))
            outcome = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
            outcome_row = node_by_uid.get(outcome)
            if outcome_row is None or int(outcome_row.cognitive_state) not in _PROBE_STATES:
                continue
            if outcome in suppressed:
                continue
            context_bucket = int(row.key_parts[3])
            if float(row.attempt_weight) > 0.0:
                reliability = max(0.0, min(1.0, float(row.strategy_reliability)))
                mean_cost = max(1e-9, float(row.strategy_mean_cost))
            else:
                reliability = min(0.55, max(0, int(row.support_count)) / 10.0)
                mean_cost = 1.0
            probationary = int(row.cognitive_state) not in _ACTIVE_STATES
            transferable = self._has_transferable_ancestor(row.uid)
            strategy = _StrategyRow(
                action,
                outcome,
                row.uid,
                int(row.support_count),
                reliability,
                mean_cost,
                context_bucket,
                probationary,
                transferable,
            )
            by_context.setdefault(context_bucket, []).append(strategy)
            if transferable:
                fallback.append(strategy)

        self._strategy_by_context = by_context
        self._strategy_fallback = fallback
        self._preferred_outcomes = preferred
        self._suppressed_outcomes = suppressed
        self._refined_action_scores = {
            key: (int(values[0]), float(values[1]) / max(1.0, float(values[0])))
            for key, values in refined.items()
        }
        self._outcome_counts = outcome_counts
        self._outcome_totals = outcome_totals
        self._v851_compact_nodes = tuple(node_by_uid.values())
        self._v851_compact_edges = tuple(compact_edges)
        self._strategy_version = tuple((*node_versions, *edge_versions))
        self._strategy_cache_stale = False
        if self._refresh_interval_seconds is not None:
            self._next_strategy_refresh = now + self._refresh_interval_seconds

        # behavior_recovery wraps the full graph refresh. Rebuild only its causal
        # control indexes from the compact actor cut.
        try:
            from v8 import behavior_recovery

            behavior_recovery._refresh_behavior_indexes(self)
        except (AttributeError, ImportError):
            pass
        self._publish_compact_cut()

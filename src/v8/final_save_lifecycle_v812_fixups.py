from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64


_INSTALLED = False


def _run_generation_lifecycle(supervisor, nodes) -> int:
    from v8 import final_save_lifecycle_v812 as base

    lifecycle = supervisor.lifecycle
    pause = getattr(supervisor, "_pause", None)
    stop_event = getattr(supervisor, "_stop", None)

    def cancelled() -> bool:
        return bool(
            (pause is not None and pause.is_set())
            or (stop_event is not None and stop_event.is_set())
        )

    if cancelled():
        return 0
    if not bool(getattr(lifecycle, "_v812_enforce_generation_sweep", False)):
        return 0
    global_window = max(0, int(supervisor.current_generation())) // int(
        base._LIFECYCLE_GENERATION_SPAN
    )
    active_window = int(getattr(lifecycle, "_v812_active_window", -1))
    last_completed = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    if active_window < 0:
        if global_window <= last_completed:
            return 0
        active_window = global_window
        lifecycle._v812_active_window = active_window
        lifecycle._v812_next_bucket = 0

    previous = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    delta = 1 if previous < 0 else max(1, active_window - previous)
    lifecycle._v812_window_delta = delta
    lifecycle._v812_sweep_mode = True
    evaluated = 0
    try:
        buckets_per_cycle = max(1, min(8, int(supervisor.candidate_budget) // 64))
        start = int(getattr(lifecycle, "_v812_next_bucket", 0))
        stop = min(int(base._LIFECYCLE_BUCKETS), start + buckets_per_cycle)
        for bucket in range(start, stop):
            if cancelled():
                return evaluated
            for row_index, row in enumerate(nodes):
                if row_index % 256 == 0 and cancelled():
                    return evaluated
                if (
                    (int(row.uid.hi) ^ int(row.uid.lo))
                    & (base._LIFECYCLE_BUCKETS - 1)
                ) != bucket:
                    continue
                evaluated += 1
                decision = lifecycle.decide(row)
                if decision is None:
                    continue
                freshness = f"lifecycle-window:{active_window}"
                if hasattr(supervisor, "_fresh") and not supervisor._fresh(
                    freshness, row.uid, active_window
                ):
                    continue
                supervisor._submit(
                    supervisor._existing_proposal(
                        row,
                        cognitive_state=int(decision.cognitive_state),
                        validation_state=int(decision.validation_state),
                    )
                )
        if cancelled():
            return evaluated
        lifecycle._v812_next_bucket = stop
        if stop >= int(base._LIFECYCLE_BUCKETS):
            lifecycle._v812_last_completed_window = active_window
            lifecycle._v812_active_window = -1
            lifecycle._v812_next_bucket = 0
    finally:
        lifecycle._v812_sweep_mode = False
        lifecycle._v812_window_delta = 1
    return evaluated


def _indexed_relational_roles(self, rows, edges):
    """Build all carrier-role descriptors in one edge pass instead of C*E scans."""
    from v8.roles import RoleCandidate

    rows = tuple(rows)
    edges = tuple(edges)
    by_uid = {row.uid: row for row in rows}
    carriers = {
        row.uid: row
        for row in rows
        if int(row.level) == int(MemoryLevel.M3)
        and int(row.memory_type) == int(MemoryType.CARRIER)
        and len(row.key_parts) >= 3
    }
    if not carriers:
        return ()

    relation_counts: dict[MemoryUid, Counter] = {
        uid: Counter() for uid in carriers
    }
    dependency_parts: dict[MemoryUid, list[int]] = defaultdict(list)
    consequence_parts: dict[MemoryUid, list[int]] = defaultdict(list)
    lower_support: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    dependency_relations = {
        int(RelationType.DEPENDS_ON),
        int(RelationType.ENABLES),
        int(RelationType.BLOCKS),
    }
    admitted_relations = set(int(value) for value in self._RELATIONS)

    def observe(carrier_uid, neighbor_uid, direction, relation, support) -> None:
        neighbor = by_uid.get(neighbor_uid)
        if neighbor is None:
            return
        relation_counts[carrier_uid][
            (direction, relation, int(neighbor.level), int(neighbor.memory_type))
        ] += max(1, int(support))
        if relation in dependency_relations:
            dependency_parts[carrier_uid].extend(
                (direction, relation, int(neighbor.level), int(neighbor.memory_type))
            )
        if int(neighbor.level) >= int(MemoryLevel.M5):
            consequence_parts[carrier_uid].extend(
                (int(neighbor.uid.hi), int(neighbor.uid.lo))
            )

    for edge in edges:
        relation = int(edge.relation_type)
        source_is_carrier = edge.source_uid in carriers
        target_is_carrier = edge.target_uid in carriers
        if relation in admitted_relations:
            if source_is_carrier:
                observe(
                    edge.source_uid,
                    edge.target_uid,
                    1,
                    relation,
                    edge.support_count,
                )
            if target_is_carrier:
                observe(
                    edge.target_uid,
                    edge.source_uid,
                    -1,
                    relation,
                    edge.support_count,
                )
        if source_is_carrier and relation == int(RelationType.EXPLAINS):
            target = by_uid.get(edge.target_uid)
            if target is not None and int(target.level) < int(MemoryLevel.M3):
                lower_support[edge.source_uid].add(target.uid)

    grouped = defaultdict(list)
    for uid, row in carriers.items():
        relation_parts = []
        for key, count in sorted(relation_counts[uid].items()):
            relation_parts.extend((*key, min(15, int(count))))
        relation_signature = (
            stable_u64(*relation_parts, person=b"v8.7-role-rel")
            if relation_parts
            else 0
        )
        dependency = dependency_parts.get(uid, [])
        dependency_signature = (
            stable_u64(*dependency, person=b"v8.7-role-dep") if dependency else 0
        )
        consequence = consequence_parts.get(uid, [])
        consequence_signature = (
            stable_u64(*sorted(consequence), person=b"v8.7-role-conseq")
            if consequence
            else 0
        )
        descriptor = (
            int(relation_signature),
            int(dependency_signature),
            int(self._future_bucket(row.future_option_delta)),
            int(consequence_signature),
        )
        grouped[descriptor].append(row)

    result = []
    for descriptor, members in sorted(grouped.items()):
        carrier_ids = {int(row.key_parts[1]) for row in members}
        support_uids = set()
        for row in members:
            support_uids.update(lower_support.get(row.uid, ()))
        if len(carrier_ids) < self.min_carriers or len(support_uids) < 2:
            continue
        key = tuple(int(value) for value in descriptor)
        uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, key)
        mask = 0
        for row in members:
            mask |= int(row.game_mask)
        result.append(
            RoleCandidate(
                uid,
                key,
                tuple(sorted(row.uid for row in members)),
                int(mask).bit_count(),
            )
        )
    return tuple(result)


def _parallel_peer_analyses(self, nodes, edges):
    """Run each independent peer analysis once, including relational role induction."""
    relational = getattr(self.roles, "propose_relational", None)
    role_fn = relational if callable(relational) else self.roles.propose
    role_args = (nodes, edges) if callable(relational) else (nodes,)
    with ThreadPoolExecutor(max_workers=9, thread_name_prefix="v8-peer") as pool:
        futures = {
            "prediction": pool.submit(self.prediction.evaluate, nodes),
            "context": pool.submit(self.context.propose, nodes),
            "roles": pool.submit(role_fn, *role_args),
            "future": pool.submit(self.future_options.evaluate, nodes),
            "compression": pool.submit(self.compression.evaluate, nodes, edges),
            "similarity": pool.submit(self.similarity.evaluate, nodes, edges),
            "transfer": pool.submit(
                self.transfer.candidates,
                nodes,
                provenance=self.read_view.source_games,
            ),
            "world": pool.submit(self.world_model.propose, nodes),
            "replay": pool.submit(
                self.replay.candidates,
                nodes,
                budget=self.candidate_budget,
            ),
        }
        return {name: future.result() for name, future in futures.items()}


def install_final_save_lifecycle_v812_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v8 import final_save_lifecycle_v812 as base
    from v8 import intelligence_loop_v087 as intelligence_module
    from v8 import peers_v82

    base._run_generation_lifecycle = _run_generation_lifecycle
    intelligence_module.V087RelationalRoleEstimator.propose_relational = (
        _indexed_relational_roles
    )
    peers_v82.V82DevelopmentalPeerSupervisor._parallel_analyses = (
        _parallel_peer_analyses
    )
    _INSTALLED = True

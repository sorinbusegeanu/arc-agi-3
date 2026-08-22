from __future__ import annotations

"""Final v8.56 cross-game transfer correctness audit fixes.

This layer keeps exact provenance fail-closed, prevents target-world execution from
rewriting source-intrinsic learning, excludes transfer bookkeeping from structural
similarity, and makes structural correspondence depend on graph structure rather
than observation frequency.
"""

from collections import Counter
from dataclasses import is_dataclass, replace

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64
from v8.transfer import TransferTrial


_INSTALLED = False
_BASE_SCOPE_FILTER = None
_BASE_RUNTIME_RECORD_RESULTS = None
_BASE_TRANSFER_CANDIDATES = None
_BASE_SIMILARITY_DESCRIPTORS = None
_BASE_CORRESPONDENCE_EVALUATE = None

_CANONICAL_PROVENANCE_LINEAGE = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
}


def _allowed_transfer_abstraction(row) -> bool:
    if row is None:
        return False
    level = int(getattr(row, "level", -1))
    memory_type = int(getattr(row, "memory_type", -1))
    if level == int(MemoryLevel.M4):
        return memory_type == int(MemoryType.CONCEPT)
    if level == int(MemoryLevel.M3):
        return memory_type in {
            int(MemoryType.ROLE),
            int(MemoryType.CONTEXTUAL_ROLE),
        }
    return False


def _provenance_status(runtime, game_hash: int, uid, cache: dict) -> str:
    cached = cache.get(uid)
    if cached is not None:
        return str(cached)
    try:
        games = frozenset(int(value) for value in runtime.read_view.source_games(uid))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        games = frozenset()
    if not games:
        status = "UNKNOWN"
    elif int(game_hash) in games:
        status = "LOCAL"
    else:
        status = "FOREIGN"
    cache[uid] = status
    return status


def _unsafe_intrinsic_strategy(runtime, game_hash: int, uid, cache: dict) -> bool:
    """Unknown provenance is not permission to rewrite intrinsic source quality."""
    return _provenance_status(runtime, game_hash, uid, cache) != "LOCAL"


def _replace_row(row, **changes):
    if not changes:
        return row
    if is_dataclass(row):
        return replace(row, **changes)
    try:
        from types import SimpleNamespace

        values = dict(vars(row))
        values.update(changes)
        return SimpleNamespace(**values)
    except (TypeError, ValueError):
        return row


def _filter_target_scoped_learning_v856(runtime, row):
    filtered = _BASE_SCOPE_FILTER(runtime, row)
    game_id = str(getattr(filtered, "game_id", ""))
    probes = getattr(filtered, "preference_probes", None)
    if not game_id or probes is None:
        return filtered

    game_hash = int(stable_u64(game_id, person=b"v8-game"))
    cache = {}
    kept = tuple(
        probe
        for probe in probes
        if _provenance_status(runtime, game_hash, probe.outcome_a, cache) == "LOCAL"
        and _provenance_status(runtime, game_hash, probe.outcome_b, cache) == "LOCAL"
    )
    if kept == tuple(probes):
        return filtered
    return _replace_row(filtered, preference_probes=kept)


def _record_actor_results_v856(self, results) -> None:
    rows = tuple(_filter_target_scoped_learning_v856(self, row) for row in results)
    return _BASE_RUNTIME_RECORD_RESULTS(self, rows)


def _provenance_from_edges_v856(
    uids: tuple[MemoryUid, ...],
    edges,
    *,
    max_depth: int = 8,
):
    direct: dict[MemoryUid, set[int]] = {}
    parents: dict[MemoryUid, set[MemoryUid]] = {}
    for edge in edges:
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            direct.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
        elif relation in _CANONICAL_PROVENANCE_LINEAGE:
            parents.setdefault(edge.source_uid, set()).add(edge.target_uid)

    result: dict[MemoryUid, tuple[int, ...]] = {}
    for uid in uids:
        found = set(direct.get(uid, ()))
        frontier = {uid}
        visited = {uid}
        for _depth in range(max(0, int(max_depth))):
            following: set[MemoryUid] = set()
            for current in frontier:
                for parent in parents.get(current, ()):
                    found.update(direct.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        result[uid] = tuple(sorted(found))
    return result


def _record_transfer_trial_v856(
    self,
    uid: MemoryUid,
    *,
    target_game_hash: int,
    metric_on: float,
    metric_off: float,
    formation_games: tuple[int, ...] = (),
    intervention: str = "matched_memory_ablation",
) -> TransferTrial:
    formation = tuple(sorted(set(int(value) for value in formation_games)))
    target = int(target_game_hash)
    effect = float(metric_on) - float(metric_off)
    # Exact formation provenance is mandatory for a scientific held-out claim.
    held_out = bool(formation) and target not in formation
    trial = TransferTrial(
        uid,
        target,
        float(metric_on),
        float(metric_off),
        effect,
        bool(held_out and effect > float(self.effect_threshold)),
        formation,
        str(intervention),
    )
    self._trials.setdefault(uid, []).append(trial)
    return trial


def _transfer_candidates_v856(self, rows, edges=(), *, provenance=None):
    filtered = tuple(row for row in rows if _allowed_transfer_abstraction(row))
    return _BASE_TRANSFER_CANDIDATES(
        self,
        filtered,
        tuple(edges),
        provenance=provenance,
    )


def _similarity_descriptors_v856(nodes, edges):
    rows = tuple(nodes)
    raw = _BASE_SIMILARITY_DESCRIPTORS(rows, tuple(edges))
    allowed = {row.uid for row in rows if _allowed_transfer_abstraction(row)}
    return {uid: descriptor for uid, descriptor in raw.items() if uid in allowed}


def _structural_descriptor_v856(cls, uid, edges, by_uid):
    descriptor: Counter[tuple[int, int, int, int]] = Counter()
    for edge in edges:
        relation = int(edge.relation_type)
        if relation not in cls._STRUCTURAL_RELATIONS:
            continue
        if edge.source_uid == uid:
            neighbor = by_uid.get(edge.target_uid)
            if neighbor is not None:
                descriptor[(1, relation, int(neighbor.level), int(neighbor.memory_type))] += 1
        elif edge.target_uid == uid:
            neighbor = by_uid.get(edge.source_uid)
            if neighbor is not None:
                descriptor[(-1, relation, int(neighbor.level), int(neighbor.memory_type))] += 1
    return descriptor


def _structural_descriptors_v856(cls, uids, edges, by_uid):
    descriptors = {uid: Counter() for uid in uids}
    for edge in edges:
        relation = int(edge.relation_type)
        if relation not in cls._STRUCTURAL_RELATIONS:
            continue
        source_descriptor = descriptors.get(edge.source_uid)
        if source_descriptor is not None:
            neighbor = by_uid.get(edge.target_uid)
            if neighbor is not None:
                source_descriptor[
                    (1, relation, int(neighbor.level), int(neighbor.memory_type))
                ] += 1
        target_descriptor = descriptors.get(edge.target_uid)
        if target_descriptor is not None:
            neighbor = by_uid.get(edge.source_uid)
            if neighbor is not None:
                target_descriptor[
                    (-1, relation, int(neighbor.level), int(neighbor.memory_type))
                ] += 1
    return descriptors


def _correspondence_evaluate_v856(self, nodes, edges, *, budget: int = 256):
    rows = tuple(nodes)
    by_uid = {row.uid: row for row in rows}
    filtered_edges = tuple(
        edge
        for edge in edges
        if int(edge.relation_type) != int(RelationType.SIMILAR_TO)
        or (
            _allowed_transfer_abstraction(by_uid.get(edge.source_uid))
            and _allowed_transfer_abstraction(by_uid.get(edge.target_uid))
        )
    )
    return _BASE_CORRESPONDENCE_EVALUATE(
        self,
        rows,
        filtered_edges,
        budget=budget,
    )


def install_adaptive_memory_transfer_audit_v856() -> None:
    global _INSTALLED, _BASE_SCOPE_FILTER, _BASE_RUNTIME_RECORD_RESULTS
    global _BASE_TRANSFER_CANDIDATES, _BASE_SIMILARITY_DESCRIPTORS
    global _BASE_CORRESPONDENCE_EVALUATE
    if _INSTALLED:
        return

    from v8 import adaptive_memory_transfer_scope_v856 as scope
    from v8 import learning_transfer_correctness_v854 as v854
    from v8 import peers_v82
    from v8.runtime_v82 import V82ContinuousMemoryRuntime
    from v8.similarity import BoundedNeighborhoodSimilarity
    from v8.structural_correspondence import StructuralCorrespondenceEstimator
    from v8.transfer import TransferValidator

    # One exact provenance definition for scientific transfer decisions.
    v854._LINEAGE = set(_CANONICAL_PROVENANCE_LINEAGE)
    peers_v82._LINEAGE = set(_CANONICAL_PROVENANCE_LINEAGE)
    TransferValidator._provenance_from_edges = staticmethod(_provenance_from_edges_v856)
    TransferValidator.record_trial = _record_transfer_trial_v856

    # Transfer validation/similarity applies to semantic roles/concepts, never to
    # v8.54 TRANSFER_EVIDENCE bookkeeping nodes.
    _BASE_TRANSFER_CANDIDATES = TransferValidator.candidates
    TransferValidator.candidates = _transfer_candidates_v856
    _BASE_SIMILARITY_DESCRIPTORS = BoundedNeighborhoodSimilarity.descriptors
    BoundedNeighborhoodSimilarity.descriptors = staticmethod(_similarity_descriptors_v856)

    # Structural admissibility is topology evidence. Repeated observations increase
    # confidence elsewhere; they must not change the structural mapping itself.
    StructuralCorrespondenceEstimator._descriptor = classmethod(_structural_descriptor_v856)
    StructuralCorrespondenceEstimator._descriptors = classmethod(_structural_descriptors_v856)
    _BASE_CORRESPONDENCE_EVALUATE = StructuralCorrespondenceEstimator.evaluate
    StructuralCorrespondenceEstimator.evaluate = _correspondence_evaluate_v856

    # The real runtime authority is the v8.2 subclass after v8.39/v8.41 wrapping.
    # Filter before those asynchronous feedback queues receive the batch.
    _BASE_SCOPE_FILTER = scope._filter_target_scoped_learning
    scope._foreign_strategy = _unsafe_intrinsic_strategy
    scope._filter_target_scoped_learning = _filter_target_scoped_learning_v856
    _BASE_RUNTIME_RECORD_RESULTS = V82ContinuousMemoryRuntime.record_actor_results
    V82ContinuousMemoryRuntime.record_actor_results = _record_actor_results_v856

    _INSTALLED = True

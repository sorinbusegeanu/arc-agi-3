from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from v8.arena import EdgeRecord, NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState


@dataclass(frozen=True, slots=True)
class IntelligenceChainAudit:
    m1_normalized: bool
    m2_family: bool
    m3_role: bool
    similarity: bool
    transfer_correspondence: bool
    validated_concept: bool
    consequence: bool
    persistent_outcome: bool
    strategy: bool
    auditable_strategy_paths: int

    @property
    def complete(self) -> bool:
        return all((self.m1_normalized, self.m2_family, self.m3_role, self.similarity, self.transfer_correspondence, self.validated_concept, self.consequence, self.persistent_outcome, self.strategy, self.auditable_strategy_paths > 0))


@dataclass(frozen=True, slots=True)
class AblationControls:
    memory_enabled: bool = True
    transfer_enabled: bool = True
    concepts_enabled: bool = True
    replay_enabled: bool = True
    restored: bool = False
    fresh: bool = False


@dataclass(frozen=True, slots=True)
class BehavioralMetrics:
    first_useful_action_latency: float
    m7_usage: float
    level_progress: float
    positive_valence_rate: float
    steps_to_outcome: float
    held_out_transfer_effect: float

    @property
    def utility(self) -> float:
        return (2.0 * float(self.level_progress) + float(self.positive_valence_rate) + float(self.held_out_transfer_effect) + 0.25 * float(self.m7_usage) - 0.01 * float(self.first_useful_action_latency) - 0.01 * float(self.steps_to_outcome))


@dataclass(frozen=True, slots=True)
class IntelligenceLoopExperiment:
    audit: IntelligenceChainAudit
    results: Mapping[str, BehavioralMetrics]

    @property
    def memory_effect(self) -> float:
        return self.results["normal"].utility - self.results["memory_off"].utility

    @property
    def restored_delta(self) -> float:
        return self.results["restored"].utility - self.results["normal"].utility


_ABLATIONS: tuple[tuple[str, AblationControls], ...] = (
    ("normal", AblationControls()),
    ("memory_off", AblationControls(memory_enabled=False)),
    ("transfer_off", AblationControls(transfer_enabled=False)),
    ("concepts_off", AblationControls(concepts_enabled=False)),
    ("replay_off", AblationControls(replay_enabled=False)),
    ("restored", AblationControls(restored=True)),
    ("fresh", AblationControls(memory_enabled=False, transfer_enabled=False, concepts_enabled=False, replay_enabled=False, fresh=True)),
)


def _parents(edges: Iterable[EdgeRecord]) -> dict[MemoryUid, set[MemoryUid]]:
    result: dict[MemoryUid, set[MemoryUid]] = {}
    lineage = {int(RelationType.PROVENANCE), int(RelationType.EXPLAINS), int(RelationType.LEADS_TO), int(RelationType.DEPENDS_ON), int(RelationType.CONTEXT_REFINES), int(RelationType.SUPERSEDES)}
    for edge in edges:
        if int(edge.relation_type) in lineage:
            result.setdefault(edge.source_uid, set()).add(edge.target_uid)
    return result


def _has_ancestor(uid: MemoryUid, *, target_level: int, parents: dict[MemoryUid, set[MemoryUid]], by_uid: dict[MemoryUid, NodeRecord], max_depth: int = 12) -> bool:
    frontier, visited = {uid}, {uid}
    for _ in range(max(0, int(max_depth))):
        following: set[MemoryUid] = set()
        for current in frontier:
            for parent in parents.get(current, ()):
                row = by_uid.get(parent)
                if row is not None and int(row.level) == int(target_level):
                    return True
                if parent not in visited:
                    visited.add(parent)
                    following.add(parent)
        if not following:
            break
        frontier = following
    return False


def audit_intelligence_chain(nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> IntelligenceChainAudit:
    nodes, edges = tuple(nodes), tuple(edges)
    by_uid = {row.uid: row for row in nodes}
    parents = _parents(edges)
    m1n = [row for row in nodes if int(row.level) == int(MemoryLevel.M1) and int(row.memory_type) == int(MemoryType.CONTINGENCY) and len(row.key_parts) == 1]
    m2 = [row for row in nodes if int(row.level) == int(MemoryLevel.M2)]
    m3 = [row for row in nodes if int(row.level) == int(MemoryLevel.M3) and int(row.memory_type) in {int(MemoryType.ROLE), int(MemoryType.CONTEXTUAL_ROLE)}]
    m4 = [row for row in nodes if int(row.level) == int(MemoryLevel.M4) and int(row.memory_type) == int(MemoryType.CONCEPT) and int(row.validation_state) == int(ValidationState.VALIDATED) and int(row.cognitive_state) in {int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)}]
    m5 = [row for row in nodes if int(row.level) == int(MemoryLevel.M5)]
    m6 = [row for row in nodes if int(row.level) == int(MemoryLevel.M6) and int(row.memory_type) == int(MemoryType.OUTCOME) and int(row.cognitive_state) in {int(CognitiveState.ACTIVE), int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)}]
    m7 = [row for row in nodes if int(row.level) == int(MemoryLevel.M7) and int(row.memory_type) == int(MemoryType.STRATEGY)]
    similar = any(int(edge.relation_type) == int(RelationType.SIMILAR_TO) for edge in edges)
    correspondence = any(int(edge.relation_type) == int(RelationType.TRANSFER_CORRESPONDENCE) and float(edge.score) > 0.0 for edge in edges)
    auditable = sum(1 for row in m7 if _has_ancestor(row.uid, target_level=int(MemoryLevel.M6), parents=parents, by_uid=by_uid) and _has_ancestor(row.uid, target_level=int(MemoryLevel.M4), parents=parents, by_uid=by_uid) and _has_ancestor(row.uid, target_level=int(MemoryLevel.M1), parents=parents, by_uid=by_uid))
    return IntelligenceChainAudit(bool(m1n), bool(m2), bool(m3), bool(similar), bool(correspondence), bool(m4), bool(m5), bool(m6), bool(m7), int(auditable))


def run_intelligence_loop_experiment(*, nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord], episode_runner: Callable[[AblationControls], BehavioralMetrics]) -> IntelligenceLoopExperiment:
    """Run one fixed causal ablation matrix through an environment-owned episode runner."""
    results = {name: episode_runner(controls) for name, controls in _ABLATIONS}
    return IntelligenceLoopExperiment(audit_intelligence_chain(nodes, edges), results)

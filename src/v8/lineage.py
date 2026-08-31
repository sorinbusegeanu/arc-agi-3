from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import IntEnum

from v8.model import CognitiveState, MemoryUid, ValidationState, stable_u64


@dataclass(frozen=True, slots=True, order=True)
class LineageUid:
    value: int = 0


@dataclass(frozen=True, slots=True, order=True)
class ContextScopeId:
    value: int = 0


class AuthorityState(IntEnum):
    ACTIVE = 1
    SUSPENDED = 2
    REJECTED = 3


@dataclass(frozen=True, slots=True)
class NodeOverlay:
    node_uid: MemoryUid
    lineage_uid: LineageUid
    context_scope_id: ContextScopeId
    regime_state: int = int(CognitiveState.ACTIVE)
    authority_state: AuthorityState = AuthorityState.ACTIVE
    validation_state: int = int(ValidationState.UNTESTED)
    independent_support: float = 1.0
    probation_start_watermark: int = 0
    relevant_evidence_opportunities: int = 0
    pending_validation_count: int = 0
    last_mutation_watermark: int = 0
    object_version: int = 0


@dataclass(frozen=True, slots=True)
class EdgeOverlay:
    edge_uid: int
    lineage_uid: LineageUid
    context_scope_id: ContextScopeId
    authority_state: AuthorityState = AuthorityState.ACTIVE
    last_mutation_watermark: int = 0
    object_version: int = 0


class LineageOverlayStore:
    STATE_VERSION = 1

    def __init__(self, *, restore_support_threshold: float = 0.60, evidence_opportunity_budget: int = 32, developmental_age_budget: int = 256) -> None:
        self.restore_support_threshold = float(restore_support_threshold)
        self.evidence_opportunity_budget = int(evidence_opportunity_budget)
        self.developmental_age_budget = int(developmental_age_budget)
        self.nodes: dict[tuple[MemoryUid, int, int], NodeOverlay] = {}
        self.edges: dict[tuple[int, int, int], EdgeOverlay] = {}

    @staticmethod
    def edge_uid(source_uid: MemoryUid, relation_type: int, target_uid: MemoryUid) -> int:
        return stable_u64(source_uid.hi, source_uid.lo, int(relation_type), target_uid.hi, target_uid.lo, person=b"v9-edge-uid")

    def effective_state(self, node_uid: MemoryUid, lineage_uid: LineageUid = LineageUid(), context_scope_id: ContextScopeId = ContextScopeId()) -> NodeOverlay:
        key = (node_uid, int(lineage_uid.value), int(context_scope_id.value))
        return self.nodes.get(key, NodeOverlay(node_uid, lineage_uid, context_scope_id))

    def mutate_node(self, overlay: NodeOverlay) -> NodeOverlay:
        key = (overlay.node_uid, overlay.lineage_uid.value, overlay.context_scope_id.value)
        current = self.nodes.get(key)
        version = 1 if current is None else current.object_version + 1
        row = replace(overlay, object_version=version)
        self.nodes[key] = row
        return row

    def suspend_dependency(self, source_uid: MemoryUid, relation_type: int, target_uid: MemoryUid, *, lineage_uid: LineageUid, context_scope_id: ContextScopeId, watermark: int, descendant_uid: MemoryUid | None = None, independent_support: float = 0.0) -> None:
        edge_uid = self.edge_uid(source_uid, relation_type, target_uid)
        key = (edge_uid, lineage_uid.value, context_scope_id.value)
        prior = self.edges.get(key)
        self.edges[key] = EdgeOverlay(edge_uid, lineage_uid, context_scope_id, AuthorityState.SUSPENDED, int(watermark), 1 if prior is None else prior.object_version + 1)
        if descendant_uid is not None and float(independent_support) < self.restore_support_threshold:
            current = self.effective_state(descendant_uid, lineage_uid, context_scope_id)
            self.mutate_node(replace(
                current,
                regime_state=int(CognitiveState.PROBATION),
                independent_support=float(independent_support),
                probation_start_watermark=int(watermark),
                relevant_evidence_opportunities=0,
                last_mutation_watermark=int(watermark),
            ))

    def note_evidence_opportunity(self, node_uid: MemoryUid, *, lineage_uid: LineageUid, context_scope_id: ContextScopeId, watermark: int, independent_support: float, pending_validation_count: int = 0) -> NodeOverlay:
        current = self.effective_state(node_uid, lineage_uid, context_scope_id)
        opportunities = current.relevant_evidence_opportunities + 1
        regime = current.regime_state
        authority = current.authority_state
        if float(independent_support) >= self.restore_support_threshold:
            regime = int(CognitiveState.ACTIVE)
            authority = AuthorityState.ACTIVE
            opportunities = 0
        return self.mutate_node(replace(
            current,
            regime_state=regime,
            authority_state=authority,
            independent_support=float(independent_support),
            relevant_evidence_opportunities=opportunities,
            pending_validation_count=int(pending_validation_count),
            last_mutation_watermark=int(watermark),
        ))

    def eligible_for_retirement(self, overlay: NodeOverlay, *, watermark: int) -> bool:
        if overlay.regime_state != int(CognitiveState.PROBATION):
            return False
        age = max(0, int(watermark) - int(overlay.probation_start_watermark))
        return bool(
            age >= self.developmental_age_budget
            and overlay.relevant_evidence_opportunities >= self.evidence_opportunity_budget
            and overlay.pending_validation_count == 0
            and overlay.independent_support < self.restore_support_threshold
        )

    def state_dict(self) -> dict[str, object]:
        def uid(uid: MemoryUid) -> list[int]: return [uid.hi, uid.lo]
        return {
            "version": self.STATE_VERSION,
            "restore_support_threshold": self.restore_support_threshold,
            "evidence_opportunity_budget": self.evidence_opportunity_budget,
            "developmental_age_budget": self.developmental_age_budget,
            "nodes": [{**asdict(row), "node_uid": uid(row.node_uid), "lineage_uid": row.lineage_uid.value, "context_scope_id": row.context_scope_id.value, "authority_state": int(row.authority_state)} for row in self.nodes.values()],
            "edges": [{**asdict(row), "lineage_uid": row.lineage_uid.value, "context_scope_id": row.context_scope_id.value, "authority_state": int(row.authority_state)} for row in self.edges.values()],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LineageOverlayStore":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported lineage state")
        obj = cls(
            restore_support_threshold=float(state.get("restore_support_threshold", 0.60)),
            evidence_opportunity_budget=int(state.get("evidence_opportunity_budget", 32)),
            developmental_age_budget=int(state.get("developmental_age_budget", 256)),
        )
        for raw in state.get("nodes", []):
            if not isinstance(raw, dict): continue
            data = dict(raw); pair = data.pop("node_uid"); data["node_uid"] = MemoryUid(int(pair[0]), int(pair[1])); data["lineage_uid"] = LineageUid(int(data["lineage_uid"])); data["context_scope_id"] = ContextScopeId(int(data["context_scope_id"])); data["authority_state"] = AuthorityState(int(data["authority_state"])); row = NodeOverlay(**data); obj.nodes[(row.node_uid, row.lineage_uid.value, row.context_scope_id.value)] = row
        for raw in state.get("edges", []):
            if not isinstance(raw, dict): continue
            data = dict(raw); data["lineage_uid"] = LineageUid(int(data["lineage_uid"])); data["context_scope_id"] = ContextScopeId(int(data["context_scope_id"])); data["authority_state"] = AuthorityState(int(data["authority_state"])); row = EdgeOverlay(**data); obj.edges[(row.edge_uid, row.lineage_uid.value, row.context_scope_id.value)] = row
        return obj

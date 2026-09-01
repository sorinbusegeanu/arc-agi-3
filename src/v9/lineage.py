from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum

from v8.model import stable_u64


class RegimeState(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    PROBATION = "PROBATION"
    RETIRED = "RETIRED"


class AuthorityState(str, Enum):
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"


class ValidationState(str, Enum):
    UNKNOWN = "UNKNOWN"
    VALID = "VALID"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class LineageContext:
    lineage_uid: int
    context_scope_id: int

    @classmethod
    def root(cls) -> "LineageContext":
        return cls(0, 0)


@dataclass(frozen=True, slots=True)
class NodeOverlay:
    node_uid: int
    lineage_uid: int
    context_scope_id: int
    regime_state: RegimeState = RegimeState.ACTIVE
    authority_state: AuthorityState = AuthorityState.ACTIVE
    validation_state: ValidationState = ValidationState.UNKNOWN
    support_summary: int = 0
    last_mutation_watermark: int = 0
    object_version: int = 0
    probation_start_watermark: int = 0
    relevant_evidence_opportunities: int = 0
    independent_support: int = 0
    pending_validation_count: int = 0

    @property
    def overlay_uid(self) -> int:
        return stable_u64(self.node_uid, self.lineage_uid, self.context_scope_id, person=b"v9-overlay")


class LineageOverlayStore:
    STATE_VERSION = 1

    def __init__(self) -> None:
        self._overlays: dict[tuple[int, int, int], NodeOverlay] = {}

    @staticmethod
    def lineage_uid(parent_lineage: int, trigger_uid: int, watermark: int) -> int:
        return stable_u64(parent_lineage, trigger_uid, watermark, person=b"v9-lineage")

    def effective_state(self, node_uid: int, lineage_uid: int = 0, context_scope_id: int = 0) -> NodeOverlay:
        exact = self._overlays.get((int(node_uid), int(lineage_uid), int(context_scope_id)))
        if exact is not None:
            return exact
        root = self._overlays.get((int(node_uid), 0, 0))
        if root is not None:
            return root
        return NodeOverlay(int(node_uid), int(lineage_uid), int(context_scope_id))

    def put(self, overlay: NodeOverlay) -> NodeOverlay:
        key = (int(overlay.node_uid), int(overlay.lineage_uid), int(overlay.context_scope_id))
        current = self._overlays.get(key)
        version = 1 if current is None else int(current.object_version) + 1
        stored = replace(overlay, object_version=version)
        self._overlays[key] = stored
        return stored

    def suspend(self, node_uid: int, lineage_uid: int, context_scope_id: int, *, watermark: int) -> NodeOverlay:
        current = self.effective_state(node_uid, lineage_uid, context_scope_id)
        return self.put(replace(current, node_uid=int(node_uid), lineage_uid=int(lineage_uid), context_scope_id=int(context_scope_id), regime_state=RegimeState.SUSPENDED, authority_state=AuthorityState.BLOCKED, last_mutation_watermark=int(watermark)))

    def audit_descendant(self, node_uid: int, lineage_uid: int, context_scope_id: int, *, independent_support: int, watermark: int) -> NodeOverlay:
        current = self.effective_state(node_uid, lineage_uid, context_scope_id)
        if int(independent_support) > 0:
            state = RegimeState.ACTIVE
            authority = AuthorityState.ACTIVE
            probation_start = 0
        else:
            state = RegimeState.PROBATION
            authority = AuthorityState.BLOCKED
            probation_start = int(watermark)
        return self.put(replace(current, node_uid=int(node_uid), lineage_uid=int(lineage_uid), context_scope_id=int(context_scope_id), regime_state=state, authority_state=authority, independent_support=int(independent_support), probation_start_watermark=probation_start, relevant_evidence_opportunities=0, last_mutation_watermark=int(watermark)))

    def observe_relevant_evidence(self, node_uid: int, lineage_uid: int, context_scope_id: int, *, positive_independent_support: bool = False) -> NodeOverlay:
        current = self.effective_state(node_uid, lineage_uid, context_scope_id)
        opportunities = int(current.relevant_evidence_opportunities) + 1
        independent = int(current.independent_support) + (1 if positive_independent_support else 0)
        state = RegimeState.ACTIVE if independent > 0 else current.regime_state
        authority = AuthorityState.ACTIVE if independent > 0 else current.authority_state
        return self.put(replace(current, relevant_evidence_opportunities=opportunities, independent_support=independent, regime_state=state, authority_state=authority))

    def retire_if_exhausted(self, node_uid: int, lineage_uid: int, context_scope_id: int, *, required_opportunities: int) -> NodeOverlay:
        current = self.effective_state(node_uid, lineage_uid, context_scope_id)
        if current.regime_state is RegimeState.PROBATION and current.independent_support <= 0 and current.relevant_evidence_opportunities >= int(required_opportunities):
            return self.put(replace(current, regime_state=RegimeState.RETIRED, authority_state=AuthorityState.BLOCKED))
        return current

    @staticmethod
    def canonical_identity_fork_required(old_structural_signature: int, new_structural_signature: int) -> bool:
        return int(old_structural_signature) != int(new_structural_signature)

    def state_dict(self) -> dict[str, object]:
        return {"version": self.STATE_VERSION, "overlays": [{**asdict(row), "regime_state": row.regime_state.value, "authority_state": row.authority_state.value, "validation_state": row.validation_state.value} for row in sorted(self._overlays.values(), key=lambda x: (x.node_uid, x.lineage_uid, x.context_scope_id))]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LineageOverlayStore":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported lineage state")
        store = cls()
        rows = state.get("overlays", [])
        if not isinstance(rows, list):
            raise ValueError("invalid lineage overlays")
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            data = dict(raw)
            data["regime_state"] = RegimeState(str(data["regime_state"]))
            data["authority_state"] = AuthorityState(str(data["authority_state"]))
            data["validation_state"] = ValidationState(str(data["validation_state"]))
            row = NodeOverlay(**data)
            store._overlays[(row.node_uid, row.lineage_uid, row.context_scope_id)] = row
        return store

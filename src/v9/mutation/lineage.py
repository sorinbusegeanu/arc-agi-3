from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from v9.memory.identity import ContextScopeId, LineageUid, MemoryUid, stable_u64
from v9.memory.model import CognitiveState
from v9.memory.relations import EdgeAuthority

from .context import ContextScope


class RegimeState(str, Enum):
    PRE_PREDICTIVE = "PRE_PREDICTIVE"
    PREDICTIVE = "PREDICTIVE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class LineageContextOverlay:
    canonical_uid: MemoryUid
    lineage_uid: LineageUid
    context_scope: ContextScope
    regime: RegimeState = RegimeState.PRE_PREDICTIVE
    lifecycle: CognitiveState = CognitiveState.ACTIVE
    support: int = 0
    evidence_opportunities: int = 0
    object_version: int = 0


@dataclass(frozen=True, slots=True)
class LineageAwareDependencyEdge:
    source: MemoryUid
    target: MemoryUid
    lineage_uid: LineageUid
    context_scope: ContextScope
    authority: EdgeAuthority = EdgeAuthority.ACTIVE
    independent_support: int = 0


class LineageStore:
    def __init__(self) -> None:
        self.overlays: dict[tuple[MemoryUid, LineageUid, ContextScope], LineageContextOverlay] = {}
        self.dependencies: dict[tuple[MemoryUid, MemoryUid, LineageUid, ContextScope], LineageAwareDependencyEdge] = {}

    @staticmethod
    def derive(parent: LineageUid, trigger: MemoryUid, watermark: int) -> LineageUid:
        return LineageUid(stable_u64(parent.value, trigger.hi, trigger.lo, watermark, person=b"v9-lineage"))

    def put_overlay(self, overlay: LineageContextOverlay) -> LineageContextOverlay:
        key = (overlay.canonical_uid, overlay.lineage_uid, overlay.context_scope)
        current = self.overlays.get(key)
        stored = replace(overlay, object_version=1 if current is None else current.object_version + 1)
        self.overlays[key] = stored
        return stored

    def suspend_parent(self, parent: MemoryUid, lineage: LineageUid, context: ContextScope) -> None:
        key = (parent, lineage, context)
        current = self.overlays.get(key, LineageContextOverlay(parent, lineage, context))
        self.put_overlay(replace(current, regime=RegimeState.UNCERTAIN))
        for edge_key, edge in tuple(self.dependencies.items()):
            if edge.source == parent and edge.lineage_uid == lineage and edge.context_scope == context:
                self.dependencies[edge_key] = replace(edge, authority=EdgeAuthority.SUSPENDED)
                if edge.independent_support <= 0:
                    child_key = (edge.target, lineage, context)
                    child = self.overlays.get(child_key, LineageContextOverlay(edge.target, lineage, context))
                    self.put_overlay(replace(child, lifecycle=CognitiveState.PROBATION))

    @staticmethod
    def canonical_fork_required(old_structural_identity: tuple[int, ...], new_structural_identity: tuple[int, ...]) -> bool:
        return old_structural_identity != new_structural_identity

    def state_dict(self) -> dict[str, object]:
        return {
            "overlays": [{"canonical_uid": [row.canonical_uid.hi, row.canonical_uid.lo], "lineage_uid": row.lineage_uid.value, "context_scope": row.context_scope.value, "regime": row.regime.value, "lifecycle": row.lifecycle.value, "support": row.support, "evidence_opportunities": row.evidence_opportunities, "object_version": row.object_version} for row in sorted(self.overlays.values(), key=lambda row: (row.canonical_uid, row.lineage_uid, row.context_scope))],
            "dependencies": [{"source": [row.source.hi, row.source.lo], "target": [row.target.hi, row.target.lo], "lineage_uid": row.lineage_uid.value, "context_scope": row.context_scope.value, "authority": row.authority.value, "independent_support": row.independent_support} for row in sorted(self.dependencies.values(), key=lambda row: (row.source, row.target, row.lineage_uid, row.context_scope))],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LineageStore":
        result = cls()
        for raw in state.get("overlays", []):
            raw_lifecycle = raw["lifecycle"]
            lifecycle = CognitiveState[str(raw_lifecycle)] if isinstance(raw_lifecycle, str) else CognitiveState(int(raw_lifecycle))
            row = LineageContextOverlay(MemoryUid(*map(int, raw["canonical_uid"])), LineageUid(int(raw["lineage_uid"])), ContextScopeId(int(raw["context_scope"])), RegimeState(str(raw["regime"])), lifecycle, int(raw["support"]), int(raw["evidence_opportunities"]), int(raw["object_version"]))
            result.overlays[(row.canonical_uid, row.lineage_uid, row.context_scope)] = row
        for raw in state.get("dependencies", []):
            row = LineageAwareDependencyEdge(MemoryUid(*map(int, raw["source"])), MemoryUid(*map(int, raw["target"])), LineageUid(int(raw["lineage_uid"])), ContextScopeId(int(raw["context_scope"])), EdgeAuthority(str(raw["authority"])), int(raw["independent_support"]))
            result.dependencies[(row.source, row.target, row.lineage_uid, row.context_scope)] = row
        return result

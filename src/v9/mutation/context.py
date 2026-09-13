from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from v9.memory.identity import ContextScopeId, LineageUid, MemoryUid, stable_u64
from v9.memory.model import CanonicalNode, CognitiveState
from v9.memory.relations import EdgeAuthority


# Kept as the public spelling used by existing v9 callers. There is one
# underlying identity type and one context registry.
ContextScope = ContextScopeId


class ContextState(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    RETIRED = "RETIRED"


@dataclass(frozen=True, slots=True)
class ContextRecord:
    scope_id: ContextScopeId
    parent_scope_id: ContextScopeId | None
    partition_descriptor: tuple[int, ...]
    evidence_refs: tuple[MemoryUid, ...]
    formation_watermark: int
    support: int
    improvement: float
    state: ContextState = ContextState.ACTIVE


class ContextRegistry:
    """Owns empirical, evidence-supported structural context partitions."""

    def __init__(self, *, limit: int = 4096, descriptor_limit: int = 64) -> None:
        if min(limit, descriptor_limit) <= 0:
            raise ValueError("context limits must be positive")
        self.limit = int(limit)
        self.descriptor_limit = int(descriptor_limit)
        self.records: dict[ContextScopeId, ContextRecord] = {}

    def form(
        self,
        partition_descriptor: tuple[int, ...],
        *,
        evidence_refs: tuple[MemoryUid, ...],
        formation_watermark: int,
        support: int,
        improvement: float,
        parent_scope_id: ContextScopeId | None = None,
    ) -> ContextRecord:
        descriptor = tuple(int(value) for value in partition_descriptor)
        if not descriptor or len(descriptor) > self.descriptor_limit or not evidence_refs or support <= 0 or improvement <= 0:
            raise ValueError("context formation requires a bounded descriptor and positive empirical improvement")
        scope = ContextScopeId(stable_u64(*(descriptor + ((parent_scope_id.value if parent_scope_id else 0),)), person=b"v9-context"))
        if scope not in self.records and len(self.records) >= self.limit:
            raise OverflowError("context scope budget exhausted")
        candidate = ContextRecord(scope, parent_scope_id, descriptor, tuple(sorted(set(evidence_refs))), int(formation_watermark), int(support), float(improvement))
        current = self.records.get(scope)
        if current is not None and current.partition_descriptor != descriptor:
            raise RuntimeError("ContextScopeId collision detected")
        self.records[scope] = candidate if current is None else current
        return self.records[scope]

    def state_dict(self) -> dict[str, object]:
        return {
            "limit": self.limit,
            "descriptor_limit": self.descriptor_limit,
            "records": [
                {
                    "scope_id": row.scope_id.value,
                    "parent_scope_id": None if row.parent_scope_id is None else row.parent_scope_id.value,
                    "partition_descriptor": list(row.partition_descriptor),
                    "evidence_refs": [[uid.hi, uid.lo] for uid in row.evidence_refs],
                    "formation_watermark": row.formation_watermark,
                    "support": row.support,
                    "improvement": row.improvement,
                    "state": row.state.value,
                }
                for row in sorted(self.records.values(), key=lambda value: value.scope_id)
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ContextRegistry":
        result = cls(limit=int(state.get("limit", 4096)), descriptor_limit=int(state.get("descriptor_limit", 64)))
        for raw in state.get("records", []):
            scope = ContextScopeId(int(raw["scope_id"]))
            parent = None if raw.get("parent_scope_id") is None else ContextScopeId(int(raw["parent_scope_id"]))
            row = ContextRecord(
                scope,
                parent,
                tuple(int(value) for value in raw["partition_descriptor"]),
                tuple(MemoryUid(int(value[0]), int(value[1])) for value in raw["evidence_refs"]),
                int(raw["formation_watermark"]),
                int(raw["support"]),
                float(raw["improvement"]),
                ContextState(str(raw["state"])),
            )
            result.records[scope] = row
        return result


@dataclass(frozen=True, slots=True)
class EffectiveCognitiveState:
    canonical_node: CanonicalNode
    context: ContextRecord | None
    lineage_uid: LineageUid | None
    lifecycle: CognitiveState
    dependency_authority: tuple[EdgeAuthority, ...]
    target_trust: float | None
    usable: bool


class EffectiveStateResolver:
    """The single authority for combining canonical and scoped state."""

    def __init__(self, contexts: ContextRegistry, lineage_store: object) -> None:
        self.contexts = contexts
        self.lineage_store = lineage_store

    def resolve(
        self,
        node: CanonicalNode,
        *,
        lineage_uid: LineageUid | None = None,
        context_scope_id: ContextScopeId | None = None,
        target_environment_id: int | None = None,
        target_trust: Mapping[tuple[MemoryUid, int, int], float] | None = None,
    ) -> EffectiveCognitiveState:
        context = self.contexts.records.get(context_scope_id) if context_scope_id is not None else None
        overlay = None
        authorities: list[EdgeAuthority] = []
        if lineage_uid is not None and context_scope_id is not None:
            key = (node.uid, lineage_uid, context_scope_id)
            overlay = self.lineage_store.overlays.get(key)
            authorities = [
                edge.authority
                for edge in self.lineage_store.dependencies.values()
                if edge.lineage_uid == lineage_uid
                and edge.context_scope == context_scope_id
                and (edge.source == node.uid or edge.target == node.uid)
            ]
        lifecycle = CognitiveState.ACTIVE if overlay is None else overlay.lifecycle
        trust = None
        if target_environment_id is not None and target_trust is not None:
            trust = target_trust.get((node.uid, int(target_environment_id), 0 if context_scope_id is None else context_scope_id.value))
        usable = lifecycle not in {CognitiveState.QUARANTINED, CognitiveState.RETIRE_PENDING, CognitiveState.RETIRED}
        usable = usable and (context is None or context.state is ContextState.ACTIVE)
        usable = usable and all(authority is not EdgeAuthority.REJECTED for authority in authorities)
        return EffectiveCognitiveState(node, context, lineage_uid, lifecycle, tuple(sorted(authorities, key=lambda value: value.value)), trust, usable)

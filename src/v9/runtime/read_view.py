from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.relations import RelationEdge
from v9.mutation.versions import ObjectRef


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ReadView:
    generation: int
    nodes: Mapping[MemoryUid, CanonicalNode]
    payloads: Mapping[MemoryUid, Mapping[str, Any]]
    edges: tuple[RelationEdge, ...]
    versions: Mapping[ObjectRef, int]
    normalized_action_supports: Mapping[int, float]

    @classmethod
    def build(cls, generation: int, nodes: dict[MemoryUid, CanonicalNode], payloads: dict[MemoryUid, dict[str, Any]], edges: dict[tuple[MemoryUid, str, MemoryUid], RelationEdge], versions: dict[ObjectRef, int]) -> "ReadView":
        frozen_payloads = {key: _freeze(value) for key, value in payloads.items()}
        action_supports: dict[int, float] = {}
        for uid, node in nodes.items():
            if node.memory_type is not MemoryType.NORMALIZED_RELATION:
                continue
            observable = str(payloads[uid].get("observable_relation", ""))
            prefix, separator, remainder = observable.partition(":")
            action_text, action_separator, _ = remainder.partition(":")
            if prefix != "ACTION" or not separator or not action_separator:
                continue
            try:
                action = int(action_text)
            except ValueError:
                continue
            action_supports[action] = action_supports.get(action, 0.0) + float(payloads[uid].get("support", 0))
        return cls(int(generation), MappingProxyType(dict(nodes)), MappingProxyType(frozen_payloads), tuple(edges[key] for key in sorted(edges)), MappingProxyType(dict(versions)), MappingProxyType(action_supports))

    def memory_count(self, level: MemoryLevel | None = None) -> int:
        return len(self.nodes) if level is None else sum(row.level is level for row in self.nodes.values())

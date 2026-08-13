from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
from typing import Iterator, Mapping

from v7.memory.arenas.mapped import MappedCompactMemoryArena, MappedNodeColumns, MappedPackedAdjacency, MappedScoreColumns
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import ActionAggregate, CognitionIndexes
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.base import ReadViewHandle

_FORMAT_VERSION = 1


def _write_atomic(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return digest


def _segment_payload(*arrays) -> tuple[bytes, list[dict[str, int | str]]]:
    payload = bytearray()
    layout: list[dict[str, int | str]] = []
    for values in arrays:
        raw = values.tobytes()
        layout.append({"offset": len(payload), "count": len(values), "typecode": values.typecode})
        payload.extend(raw)
    return bytes(payload), layout


def _cast(mapped: mmap.mmap, spec: Mapping[str, int | str]) -> memoryview:
    offset = int(spec["offset"])
    count = int(spec["count"])
    typecode = str(spec["typecode"])
    itemsize = {"B": 1, "I": 4, "Q": 8, "q": 8, "d": 8}[typecode]
    return memoryview(mapped)[offset : offset + count * itemsize].cast(typecode)


class _NodeMap(Mapping[MemoryId, MemoryNode]):
    def __init__(self, columns: MappedNodeColumns) -> None:
        self.columns = columns
    def __len__(self) -> int:
        return self.columns.count
    def __iter__(self) -> Iterator[MemoryId]:
        return (MemoryId(int(value)) for value in self.columns.memory_ids)
    def __getitem__(self, key: MemoryId) -> MemoryNode:
        value = self.columns.get(key)
        if value is None:
            raise KeyError(key)
        return value


class _ScoreMap(Mapping[MemoryId, MemoryScore]):
    def __init__(self, columns: MappedScoreColumns) -> None:
        self.columns = columns
    def __len__(self) -> int:
        return self.columns.count
    def __iter__(self) -> Iterator[MemoryId]:
        return (MemoryId(int(value)) for value in self.columns.memory_ids)
    def __getitem__(self, key: MemoryId) -> MemoryScore:
        value = self.columns.get(key)
        if value is None:
            raise KeyError(key)
        return value


class _AdjacencyMap(Mapping[tuple[MemoryId, int], tuple[MemoryId, ...]]):
    def __init__(self, adjacency: MappedPackedAdjacency) -> None:
        self.adjacency = adjacency
    def __len__(self) -> int:
        return len(self.adjacency.source_ids)
    def __iter__(self) -> Iterator[tuple[MemoryId, int]]:
        return ((MemoryId(int(self.adjacency.source_ids[i])), int(self.adjacency.relation_types[i])) for i in range(len(self.adjacency.source_ids)))
    def __getitem__(self, key: tuple[MemoryId, int]) -> tuple[MemoryId, ...]:
        for candidate in self:
            if candidate == key:
                return self.adjacency.neighbors(*key)
        raise KeyError(key)


class SegmentedMmapReadViewTransport:
    """Direct numeric mmap transport for node, score and packed-adjacency arenas."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def publish(self, view: MemoryReadView) -> ReadViewHandle:
        arena = view.compact_arena
        node_payload, node_layout = _segment_payload(
            arena.nodes.memory_ids, arena.nodes.levels, arena.nodes.type_ids,
            arena.nodes.created_generations, arena.nodes.updated_generations,
            arena.nodes.status_flags, arena.nodes.support_counts,
        )
        score_payload, score_layout = _segment_payload(
            arena.scores.memory_ids, arena.scores.significance, arena.scores.prediction_error,
            arena.scores.learning_value, arena.scores.transfer_prior,
            arena.scores.explanatory_potential, arena.scores.future_option_delta,
        )
        adjacency_payload, adjacency_layout = _segment_payload(
            arena.adjacency.source_ids, arena.adjacency.relation_types, arena.adjacency.offsets,
            arena.adjacency.lengths, arena.adjacency.targets,
        )
        prefix = f"generation-{int(view.generation_id)}"
        node_name, score_name, adjacency_name = f"{prefix}.nodes", f"{prefix}.scores", f"{prefix}.adj"
        node_digest = _write_atomic(self.directory / node_name, node_payload)
        score_digest = _write_atomic(self.directory / score_name, score_payload)
        adjacency_digest = _write_atomic(self.directory / adjacency_name, adjacency_payload)
        indexes = view.cognition_indexes
        manifest = {
            "format_version": _FORMAT_VERSION,
            "generation_id": int(view.generation_id),
            "nodes": {"file": node_name, "digest": node_digest, "layout": node_layout},
            "scores": {"file": score_name, "digest": score_digest, "layout": score_layout},
            "adjacency": {"file": adjacency_name, "digest": adjacency_digest, "layout": adjacency_layout},
            "contingencies": [[c, a, [int(v) for v in values]] for (c, a), values in sorted(indexes.contingency_by_context_action.items())],
            "roles_exact": [[c, a, int(f), [int(v) for v in values]] for (c, a, f), values in sorted(indexes.role_by_context_action_family.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2])))],
            "roles_fallback": [[c, a, [int(v) for v in values]] for (c, a), values in sorted(indexes.role_by_context_action.items())],
            "concepts": [[int(role), [int(v) for v in values]] for role, values in sorted(indexes.concepts_by_role.items(), key=lambda item: int(item[0]))],
            "aggregates": [[a, x.future_option_sum, x.future_option_count, x.positive_count, x.negative_count, x.failure_count, x.contradiction_count] for a, x in sorted(indexes.action_aggregates.items())],
        }
        manifest_payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        manifest_name = f"{prefix}.manifest"
        manifest_digest = _write_atomic(self.directory / manifest_name, manifest_payload)
        return ReadViewHandle(view.generation_id, f"{manifest_name}:{manifest_digest}")

    def _open_segment(self, spec: Mapping[str, object]) -> mmap.mmap:
        path = self.directory / str(spec["file"])
        stream = path.open("rb")
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        stream.close()
        digest = hashlib.sha256(mapped[:]).hexdigest()
        if digest != spec["digest"]:
            mapped.close()
            raise ValueError("mmap numeric segment digest mismatch")
        return mapped

    def attach(self, handle: ReadViewHandle) -> MemoryReadView:
        manifest_name, sep, expected_digest = handle.transport_key.partition(":")
        if not sep:
            raise ValueError("invalid segmented mmap handle")
        payload = (self.directory / manifest_name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise ValueError("mmap manifest digest mismatch")
        raw = json.loads(payload)
        if raw["format_version"] != _FORMAT_VERSION or int(raw["generation_id"]) != int(handle.generation_id):
            raise ValueError("segmented mmap generation mismatch")
        node_map = self._open_segment(raw["nodes"])
        score_map = self._open_segment(raw["scores"])
        adjacency_map = self._open_segment(raw["adjacency"])
        n = [_cast(node_map, spec) for spec in raw["nodes"]["layout"]]
        s = [_cast(score_map, spec) for spec in raw["scores"]["layout"]]
        a = [_cast(adjacency_map, spec) for spec in raw["adjacency"]["layout"]]
        arena = MappedCompactMemoryArena(
            generation_id=handle.generation_id,
            nodes=MappedNodeColumns(*n),
            scores=MappedScoreColumns(*s),
            adjacency=MappedPackedAdjacency(*a),
            owners=(node_map, score_map, adjacency_map),
        )
        indexes = CognitionIndexes.freeze(
            contingency_by_context_action={(int(c), int(action)): tuple(MemoryId(v) for v in values) for c, action, values in raw["contingencies"]},
            role_by_context_action_family={(int(c), int(action), MemoryId(f)): tuple(MemoryId(v) for v in values) for c, action, f, values in raw["roles_exact"]},
            role_by_context_action={(int(c), int(action)): tuple(MemoryId(v) for v in values) for c, action, values in raw["roles_fallback"]},
            concepts_by_role={MemoryId(role): tuple(MemoryId(v) for v in values) for role, values in raw["concepts"]},
            action_aggregates={int(row[0]): ActionAggregate(*row[1:]) for row in raw["aggregates"]},
        )
        return MemoryReadView.from_compact_arena(
            generation_id=handle.generation_id,
            nodes=_NodeMap(arena.nodes),
            scores=_ScoreMap(arena.scores),
            adjacency=_AdjacencyMap(arena.adjacency),
            cognition_indexes=indexes,
            compact_arena=arena,
        )

    def release(self, handle: ReadViewHandle) -> None:
        manifest_name = handle.transport_key.partition(":")[0]
        manifest_path = self.directory / manifest_name
        if not manifest_path.exists():
            return
        raw = json.loads(manifest_path.read_bytes())
        for key in ("nodes", "scores", "adjacency"):
            (self.directory / raw[key]["file"]).unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

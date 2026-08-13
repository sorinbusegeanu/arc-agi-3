from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
from typing import Iterator, Mapping

from v7.memory.arenas.mapped import MappedCompactMemoryArena, MappedNodeColumns, MappedPackedAdjacency, MappedScoreColumns
from v7.memory.ids import MemoryId
from v7.memory.indexes.cognition import CognitionIndexes
from v7.memory.indexes.packed import PackedActionAggregates, PackedCognitionIndexes, PackedPairIndex, PackedRoleExactIndex
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.base import ReadViewHandle

_FORMAT_VERSION = 2
_ITEMSIZE = {"B": 1, "I": 4, "Q": 8, "q": 8, "d": 8}


def _typecode(values: object) -> str:
    code = getattr(values, "typecode", None) or getattr(values, "format", None)
    if code not in _ITEMSIZE:
        raise ValueError(f"unsupported numeric typecode: {code}")
    return str(code)


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
        code = _typecode(values)
        layout.append({"offset": len(payload), "count": len(values), "typecode": code})
        payload.extend(raw)
    return (bytes(payload) if payload else b"\0"), layout


def _cast(mapped: mmap.mmap, spec: Mapping[str, int | str]) -> memoryview:
    offset, count, typecode = int(spec["offset"]), int(spec["count"]), str(spec["typecode"])
    return memoryview(mapped)[offset : offset + count * _ITEMSIZE[typecode]].cast(typecode).toreadonly()


def _content_segment(directory: Path, kind: str, payload: bytes, layout: list[dict[str, int | str]]) -> dict[str, object]:
    digest = hashlib.sha256(payload).hexdigest()
    name = f"segment-{kind}-{digest[:24]}.bin"
    _write_atomic(directory / name, payload)
    return {"file": name, "digest": digest, "layout": layout}


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
        value = self.adjacency.neighbors(*key)
        if not value and key not in tuple(self):
            raise KeyError(key)
        return value


class SegmentedMmapReadViewTransport:
    """Content-addressed low-copy mmap transport for numeric cognition generations."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def publish(self, view: MemoryReadView) -> ReadViewHandle:
        arena = view.compact_arena
        nodes_payload, nodes_layout = _segment_payload(arena.nodes.memory_ids, arena.nodes.levels, arena.nodes.type_ids, arena.nodes.created_generations, arena.nodes.updated_generations, arena.nodes.status_flags, arena.nodes.support_counts)
        scores_payload, scores_layout = _segment_payload(arena.scores.memory_ids, arena.scores.significance, arena.scores.prediction_error, arena.scores.learning_value, arena.scores.transfer_prior, arena.scores.explanatory_potential, arena.scores.future_option_delta)
        adj_payload, adj_layout = _segment_payload(arena.adjacency.source_ids, arena.adjacency.relation_types, arena.adjacency.offsets, arena.adjacency.lengths, arena.adjacency.targets)
        packed = view.packed_cognition
        cognition_arrays = (
            packed.contingencies.key_a, packed.contingencies.key_b, packed.contingencies.offsets, packed.contingencies.lengths, packed.contingencies.values,
            packed.roles_fallback.key_a, packed.roles_fallback.key_b, packed.roles_fallback.offsets, packed.roles_fallback.lengths, packed.roles_fallback.values,
            packed.roles_exact.contexts, packed.roles_exact.actions, packed.roles_exact.families, packed.roles_exact.offsets, packed.roles_exact.lengths, packed.roles_exact.values,
            packed.concepts_by_role.key_a, packed.concepts_by_role.key_b, packed.concepts_by_role.offsets, packed.concepts_by_role.lengths, packed.concepts_by_role.values,
            packed.action_aggregates.action_ids, packed.action_aggregates.values,
        )
        cognition_payload, cognition_layout = _segment_payload(*cognition_arrays)
        manifest = {
            "format_version": _FORMAT_VERSION,
            "generation_id": int(view.generation_id),
            "nodes": _content_segment(self.directory, "nodes", nodes_payload, nodes_layout),
            "scores": _content_segment(self.directory, "scores", scores_payload, scores_layout),
            "adjacency": _content_segment(self.directory, "adj", adj_payload, adj_layout),
            "cognition": _content_segment(self.directory, "cog", cognition_payload, cognition_layout),
        }
        payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        manifest_name = f"generation-{int(view.generation_id)}.manifest"
        digest = hashlib.sha256(payload).hexdigest()
        path = self.directory / manifest_name
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return ReadViewHandle(view.generation_id, f"{manifest_name}:{digest}")

    def _open_segment(self, spec: Mapping[str, object]) -> mmap.mmap:
        path = self.directory / str(spec["file"])
        if not path.exists():
            raise KeyError(str(spec["file"]))
        stream = path.open("rb")
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        stream.close()
        if hashlib.sha256(mapped[:]).hexdigest() != spec["digest"]:
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
        node_map, score_map, adjacency_map, cognition_map = (self._open_segment(raw[key]) for key in ("nodes", "scores", "adjacency", "cognition"))
        n = [_cast(node_map, spec) for spec in raw["nodes"]["layout"]]
        s = [_cast(score_map, spec) for spec in raw["scores"]["layout"]]
        a = [_cast(adjacency_map, spec) for spec in raw["adjacency"]["layout"]]
        c = [_cast(cognition_map, spec) for spec in raw["cognition"]["layout"]]
        arena = MappedCompactMemoryArena(handle.generation_id, MappedNodeColumns(*n), MappedScoreColumns(*s), MappedPackedAdjacency(*a), owners=(node_map, score_map, adjacency_map, cognition_map))
        packed = PackedCognitionIndexes(
            PackedPairIndex(*c[0:5]),
            PackedPairIndex(*c[5:10]),
            PackedRoleExactIndex(*c[10:16]),
            PackedPairIndex(*c[16:21]),
            PackedActionAggregates(*c[21:23]),
        )
        return MemoryReadView.from_compact_arena(
            generation_id=handle.generation_id,
            nodes=_NodeMap(arena.nodes),
            scores=_ScoreMap(arena.scores),
            adjacency=_AdjacencyMap(arena.adjacency),
            cognition_indexes=CognitionIndexes.empty(),
            packed_cognition=packed,
            compact_arena=arena,  # type: ignore[arg-type]
        )

    def release(self, handle: ReadViewHandle) -> None:
        manifest_name = handle.transport_key.partition(":")[0]
        path = self.directory / manifest_name
        if not path.exists():
            return
        raw = json.loads(path.read_bytes())
        segment_names = {str(raw[key]["file"]) for key in ("nodes", "scores", "adjacency", "cognition")}
        path.unlink(missing_ok=True)
        referenced: set[str] = set()
        for other in self.directory.glob("generation-*.manifest"):
            try:
                other_raw = json.loads(other.read_bytes())
                referenced.update(str(other_raw[key]["file"]) for key in ("nodes", "scores", "adjacency", "cognition"))
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
        for name in segment_names - referenced:
            (self.directory / name).unlink(missing_ok=True)

    @property
    def retained_generations(self) -> tuple[int, ...]:
        generations: list[int] = []
        for path in self.directory.glob("generation-*.manifest"):
            try:
                generations.append(int(path.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return tuple(sorted(generations))

from __future__ import annotations

import json
import os
import pickle
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v9.memory.identity import MemoryUid

from .snapshot_chunks import CHUNK_BYTES, read_chunks, sha256, write_chunks

NATIVE_SCHEMA = "arc-agi3-hydra-v9"
LEGACY_SNAPSHOT_VERSION = 3
SNAPSHOT_VERSION = 4
STATE_FORMAT = "pickle5-fixed-graph-shards-v1"
_GRAPH_SHARD_MAGIC = b"V9GSHD01"
_GRAPH_SHARD_VERSION = 1
_GRAPH_SHARD_HEADER = struct.Struct("<8sIIQQQ")
_NODE_RECORD = struct.Struct("<QQiiqQIQI")
_EDGE_RECORD = struct.Struct("<QQQQQIQIQIq")
_UID_PAIR = struct.Struct("<QQ")


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    path: Path
    snapshot_id: int
    watermark: int
    graph_generation: int


def latest_snapshot(root: Path) -> Path | None:
    directory = root / "snapshots"
    if not directory.is_dir():
        return None
    rows = [path for path in directory.glob("snapshot-*") if path.is_dir() and (path / "manifest.json").is_file() and (path / "COMPLETE").is_file()]
    return max(rows, default=None, key=lambda path: path.name)


def _checked_chunks(root: Path, chunks: list[dict[str, Any]], expected_sha256: str, label: str) -> bytes:
    payload = read_chunks(root, chunks)
    if sha256(payload) != str(expected_sha256):
        raise RuntimeError(f"snapshot {label} checksum mismatch")
    return payload


def _append_blob(blob: bytearray, payload: bytes) -> tuple[int, int]:
    offset = len(blob)
    blob.extend(payload)
    return offset, len(payload)


def _encode_graph_shard(partition: int, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> bytes:
    blob = bytearray()
    node_records = bytearray()
    edge_records = bytearray()

    for raw in sorted(nodes, key=lambda row: (int(row["hi"]), int(row["lo"]))):
        structural_bytes = pickle.dumps(tuple(int(value) for value in raw.get("structural_key", ())), protocol=5)
        structural_offset, structural_length = _append_blob(blob, structural_bytes)
        payload_bytes = pickle.dumps(dict(raw.get("payload", {})), protocol=5)
        payload_offset, payload_length = _append_blob(blob, payload_bytes)
        node_records.extend(
            _NODE_RECORD.pack(
                int(raw["hi"]),
                int(raw["lo"]),
                int(raw["level"]),
                int(raw["memory_type"]),
                int(raw.get("created_watermark", 0)),
                structural_offset,
                structural_length,
                payload_offset,
                payload_length,
            )
        )

    for raw in sorted(
        edges,
        key=lambda row: (
            int(row["source_hi"]),
            int(row["source_lo"]),
            str(row["relation"]),
            int(row["target_hi"]),
            int(row["target_lo"]),
        ),
    ):
        relation_offset, relation_length = _append_blob(blob, str(raw["relation"]).encode("utf-8"))
        authority_offset, authority_length = _append_blob(blob, str(raw["authority"]).encode("utf-8"))
        evidence_bytes = bytearray()
        for evidence_uid in raw.get("evidence", ()):
            evidence_bytes.extend(_UID_PAIR.pack(int(evidence_uid[0]), int(evidence_uid[1])))
        evidence_offset, evidence_length = _append_blob(blob, bytes(evidence_bytes))
        edge_records.extend(
            _EDGE_RECORD.pack(
                int(raw["source_hi"]),
                int(raw["source_lo"]),
                int(raw["target_hi"]),
                int(raw["target_lo"]),
                relation_offset,
                relation_length,
                authority_offset,
                authority_length,
                evidence_offset,
                evidence_length,
                int(raw.get("object_version", 0)),
            )
        )

    header = _GRAPH_SHARD_HEADER.pack(
        _GRAPH_SHARD_MAGIC,
        _GRAPH_SHARD_VERSION,
        int(partition),
        len(nodes),
        len(edges),
        len(blob),
    )
    return bytes(header + node_records + edge_records + blob)


def _decode_graph_shard(payload: bytes, *, expected_partition: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(payload) < _GRAPH_SHARD_HEADER.size:
        raise RuntimeError("snapshot graph shard is truncated")
    magic, version, partition, node_count, edge_count, blob_length = _GRAPH_SHARD_HEADER.unpack_from(payload, 0)
    if magic != _GRAPH_SHARD_MAGIC or int(version) != _GRAPH_SHARD_VERSION:
        raise RuntimeError("snapshot graph shard format is unsupported")
    if int(partition) != int(expected_partition):
        raise RuntimeError("snapshot graph shard partition mismatch")

    node_bytes = int(node_count) * _NODE_RECORD.size
    edge_bytes = int(edge_count) * _EDGE_RECORD.size
    blob_start = _GRAPH_SHARD_HEADER.size + node_bytes + edge_bytes
    if blob_start + int(blob_length) != len(payload):
        raise RuntimeError("snapshot graph shard length mismatch")
    blob = memoryview(payload)[blob_start:]

    def blob_slice(offset: int, length: int) -> bytes:
        start = int(offset)
        end = start + int(length)
        if start < 0 or end < start or end > len(blob):
            raise RuntimeError("snapshot graph shard blob reference is invalid")
        return bytes(blob[start:end])

    nodes: list[dict[str, Any]] = []
    cursor = _GRAPH_SHARD_HEADER.size
    for _ in range(int(node_count)):
        hi, lo, level, memory_type, created_watermark, structural_offset, structural_length, payload_offset, payload_length = _NODE_RECORD.unpack_from(payload, cursor)
        cursor += _NODE_RECORD.size
        structural_key = pickle.loads(blob_slice(structural_offset, structural_length))
        node_payload = pickle.loads(blob_slice(payload_offset, payload_length))
        nodes.append(
            {
                "hi": int(hi),
                "lo": int(lo),
                "level": int(level),
                "memory_type": int(memory_type),
                "structural_key": list(structural_key),
                "created_watermark": int(created_watermark),
                "payload": dict(node_payload),
            }
        )

    edges: list[dict[str, Any]] = []
    cursor = _GRAPH_SHARD_HEADER.size + node_bytes
    for _ in range(int(edge_count)):
        source_hi, source_lo, target_hi, target_lo, relation_offset, relation_length, authority_offset, authority_length, evidence_offset, evidence_length, object_version = _EDGE_RECORD.unpack_from(payload, cursor)
        cursor += _EDGE_RECORD.size
        evidence_payload = blob_slice(evidence_offset, evidence_length)
        if len(evidence_payload) % _UID_PAIR.size:
            raise RuntimeError("snapshot graph shard evidence payload is invalid")
        evidence = [list(_UID_PAIR.unpack_from(evidence_payload, offset)) for offset in range(0, len(evidence_payload), _UID_PAIR.size)]
        edges.append(
            {
                "source_hi": int(source_hi),
                "source_lo": int(source_lo),
                "relation": blob_slice(relation_offset, relation_length).decode("utf-8"),
                "target_hi": int(target_hi),
                "target_lo": int(target_lo),
                "evidence": evidence,
                "authority": blob_slice(authority_offset, authority_length).decode("utf-8"),
                "object_version": int(object_version),
            }
        )
    return nodes, edges


def _load_v3(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = path.parent.parent
    state_bytes = _checked_chunks(root, list(manifest.get("state_chunks", [])), str(manifest.get("state_sha256", "")), "state")
    return {
        "schema": NATIVE_SCHEMA,
        "snapshot_version": LEGACY_SNAPSHOT_VERSION,
        "scientific_config_id": manifest["scientific_config_id"],
        "snapshot_id": int(manifest["snapshot_id"]),
        "watermark": int(manifest["watermark"]),
        "graph_generation": int(manifest["graph_generation"]),
        "state": json.loads(state_bytes.decode("utf-8")),
    }


def _load_v4(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if str(manifest.get("state_format", "")) != STATE_FORMAT:
        raise RuntimeError("snapshot binary state format is unsupported")
    root = path.parent.parent
    runtime_bytes = _checked_chunks(
        root,
        list(manifest.get("runtime_state_chunks", [])),
        str(manifest.get("runtime_state_sha256", "")),
        "runtime state",
    )
    graph_header_bytes = _checked_chunks(
        root,
        list(manifest.get("graph_header_chunks", [])),
        str(manifest.get("graph_header_sha256", "")),
        "graph header",
    )
    state = dict(pickle.loads(runtime_bytes))
    graph = dict(pickle.loads(graph_header_bytes))
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    shard_specs = sorted(list(manifest.get("graph_shards", [])), key=lambda row: int(row["partition"]))
    expected_partitions = int(graph.get("partition_count", 0))
    if len(shard_specs) != expected_partitions:
        raise RuntimeError("snapshot graph shard count mismatch")
    for expected_partition, spec in enumerate(shard_specs):
        if int(spec["partition"]) != expected_partition:
            raise RuntimeError("snapshot graph shard sequence mismatch")
        shard_bytes = _checked_chunks(root, list(spec.get("chunks", [])), str(spec.get("sha256", "")), f"graph shard {expected_partition}")
        shard_nodes, shard_edges = _decode_graph_shard(shard_bytes, expected_partition=expected_partition)
        nodes.extend(shard_nodes)
        edges.extend(shard_edges)
    graph["nodes"] = nodes
    graph["edges"] = edges
    state["graph"] = graph
    return {
        "schema": NATIVE_SCHEMA,
        "snapshot_version": SNAPSHOT_VERSION,
        "scientific_config_id": manifest["scientific_config_id"],
        "snapshot_id": int(manifest["snapshot_id"]),
        "watermark": int(manifest["watermark"]),
        "graph_generation": int(manifest["graph_generation"]),
        "state": state,
    }


def load_snapshot(path: Path, *, expected_config_id: str) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    version = int(manifest.get("snapshot_version", 0))
    if manifest.get("schema") != NATIVE_SCHEMA or version not in {LEGACY_SNAPSHOT_VERSION, SNAPSHOT_VERSION}:
        raise RuntimeError("snapshot is not a supported native v9 snapshot")
    if manifest.get("scientific_config_id") != expected_config_id:
        raise RuntimeError("snapshot ScientificConfigId does not match this run")
    if version == LEGACY_SNAPSHOT_VERSION:
        return _load_v3(path, manifest)
    return _load_v4(path, manifest)


def write_snapshot(root: Path, payload: dict[str, Any], *, snapshot_id: int, watermark: int, graph_generation: int, scientific_config_id: str) -> SnapshotResult:
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    state = dict(payload)
    graph = dict(state.pop("graph"))
    nodes = list(graph.pop("nodes", []))
    edges = list(graph.pop("edges", []))
    partition_count = int(graph.get("partition_count", 0))
    if partition_count <= 0:
        raise RuntimeError("snapshot graph has invalid partition count")

    node_shards: list[list[dict[str, Any]]] = [[] for _ in range(partition_count)]
    edge_shards: list[list[dict[str, Any]]] = [[] for _ in range(partition_count)]
    for raw in nodes:
        owner = MemoryUid(int(raw["hi"]), int(raw["lo"])).shard(partition_count)
        node_shards[owner].append(raw)
    for raw in edges:
        owner = MemoryUid(int(raw["source_hi"]), int(raw["source_lo"])).shard(partition_count)
        edge_shards[owner].append(raw)

    runtime_bytes = pickle.dumps(state, protocol=5)
    graph_header_bytes = pickle.dumps(graph, protocol=5)
    graph_shards: list[dict[str, Any]] = []
    for partition in range(partition_count):
        shard_bytes = _encode_graph_shard(partition, node_shards[partition], edge_shards[partition])
        graph_shards.append(
            {
                "partition": partition,
                "nodes": len(node_shards[partition]),
                "edges": len(edge_shards[partition]),
                "bytes": len(shard_bytes),
                "sha256": sha256(shard_bytes),
                "chunks": write_chunks(root, shard_bytes),
            }
        )

    target = snapshots / f"snapshot-{int(snapshot_id):020d}"
    temporary = snapshots / f".{target.name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    manifest = {
        "schema": NATIVE_SCHEMA,
        "snapshot_version": SNAPSHOT_VERSION,
        "state_format": STATE_FORMAT,
        "scientific_config_id": scientific_config_id,
        "snapshot_id": int(snapshot_id),
        "watermark": int(watermark),
        "graph_generation": int(graph_generation),
        "chunk_bytes": CHUNK_BYTES,
        "runtime_state_bytes": len(runtime_bytes),
        "runtime_state_sha256": sha256(runtime_bytes),
        "runtime_state_chunks": write_chunks(root, runtime_bytes),
        "graph_header_bytes": len(graph_header_bytes),
        "graph_header_sha256": sha256(graph_header_bytes),
        "graph_header_chunks": write_chunks(root, graph_header_bytes),
        "graph_shards": graph_shards,
    }
    payload_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    try:
        (temporary / "manifest.json").write_bytes(payload_bytes)
        (temporary / "COMPLETE").write_text(sha256(payload_bytes) + "\n", encoding="ascii")
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return SnapshotResult(target, int(snapshot_id), int(watermark), int(graph_generation))

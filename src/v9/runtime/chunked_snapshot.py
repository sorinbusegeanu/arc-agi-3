from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .snapshot_chunks import CHUNK_BYTES, read_chunks, sha256, write_chunks

NATIVE_SCHEMA = "arc-agi3-hydra-v9"
SNAPSHOT_VERSION = 3


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
    rows = [
        path for path in directory.glob("snapshot-*")
        if path.is_dir()
        and (path / "manifest.json").is_file()
        and (path / "COMPLETE").is_file()
    ]
    return max(rows, default=None, key=lambda path: path.name)


def load_snapshot(path: Path, *, expected_config_id: str) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != NATIVE_SCHEMA or int(manifest.get("snapshot_version", 0)) != SNAPSHOT_VERSION:
        raise RuntimeError("snapshot is not a supported native v9 snapshot")
    if manifest.get("scientific_config_id") != expected_config_id:
        raise RuntimeError("snapshot ScientificConfigId does not match this run")
    root = path.parent.parent
    state_bytes = read_chunks(root, list(manifest.get("state_chunks", [])))
    if sha256(state_bytes) != str(manifest.get("state_sha256", "")):
        raise RuntimeError("snapshot state checksum mismatch")
    return {
        "schema": NATIVE_SCHEMA,
        "snapshot_version": SNAPSHOT_VERSION,
        "scientific_config_id": manifest["scientific_config_id"],
        "snapshot_id": int(manifest["snapshot_id"]),
        "watermark": int(manifest["watermark"]),
        "graph_generation": int(manifest["graph_generation"]),
        "state": json.loads(state_bytes.decode("utf-8")),
    }


def write_snapshot(
    root: Path,
    payload: dict[str, Any],
    *,
    snapshot_id: int,
    watermark: int,
    graph_generation: int,
    scientific_config_id: str,
) -> SnapshotResult:
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    state_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    target = snapshots / f"snapshot-{int(snapshot_id):020d}"
    temporary = snapshots / f".{target.name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    manifest = {
        "schema": NATIVE_SCHEMA,
        "snapshot_version": SNAPSHOT_VERSION,
        "scientific_config_id": scientific_config_id,
        "snapshot_id": int(snapshot_id),
        "watermark": int(watermark),
        "graph_generation": int(graph_generation),
        "chunk_bytes": CHUNK_BYTES,
        "state_bytes": len(state_bytes),
        "state_sha256": sha256(state_bytes),
        "state_chunks": write_chunks(root, state_bytes),
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

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

NATIVE_SCHEMA = "arc-agi3-hydra-v9"
SNAPSHOT_VERSION = 2


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    path: Path
    snapshot_id: int
    watermark: int
    graph_generation: int


def assert_native_root(root: Path) -> None:
    if not root.exists():
        return
    manifest = root / "scientific_config.json"
    if manifest.exists():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if str(raw.get("design_version")) != "9.5":
            raise RuntimeError("predecessor run root is not a native v9.5 root; use a fresh v9 root")
    predecessor_markers = (root / "v8_run_summary.json", root / "v9_auxiliary_state.json")
    if any(path.exists() for path in predecessor_markers):
        raise RuntimeError("predecessor run roots are not migrated; use a fresh v9 root")


def latest_snapshot(root: Path) -> Path | None:
    paths = sorted((root / "snapshots").glob("snapshot-*.json")) if (root / "snapshots").is_dir() else []
    return paths[-1] if paths else None


def load_snapshot(path: Path, *, expected_config_id: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != NATIVE_SCHEMA or int(raw.get("snapshot_version", 0)) != SNAPSHOT_VERSION:
        raise RuntimeError("snapshot is not a supported native v9 snapshot")
    if raw.get("scientific_config_id") != expected_config_id:
        raise RuntimeError("snapshot ScientificConfigId does not match this run")
    return raw


def write_snapshot(root: Path, payload: dict[str, Any], *, snapshot_id: int, watermark: int, graph_generation: int, scientific_config_id: str) -> SnapshotResult:
    directory = root / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"snapshot-{int(snapshot_id):020d}.json"
    document = {
        "schema": NATIVE_SCHEMA, "snapshot_version": SNAPSHOT_VERSION,
        "scientific_config_id": scientific_config_id, "snapshot_id": int(snapshot_id),
        "watermark": int(watermark), "graph_generation": int(graph_generation),
        "state": payload,
    }
    with NamedTemporaryFile("w", encoding="utf-8", dir=directory, prefix=".snapshot-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return SnapshotResult(target, int(snapshot_id), int(watermark), int(graph_generation))

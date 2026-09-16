from __future__ import annotations

import json
from pathlib import Path

from . import chunked_snapshot
from . import snapshot as legacy

SnapshotResult = chunked_snapshot.SnapshotResult
assert_native_root = legacy.assert_native_root
SYMBOL_GRAPH_SCHEMA_VERSION = 3


def latest_snapshot(root: Path):
    old = legacy.latest_snapshot(root)
    new = chunked_snapshot.latest_snapshot(root)
    if old is None:
        return new
    if new is None:
        return old
    old_id = int(old.stem.rsplit("-", 1)[1])
    new_id = int(new.name.rsplit("-", 1)[1])
    return new if new_id >= old_id else old


def _validate_symbol_graph_schema(path: Path) -> None:
    if path.is_file():
        return
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = int(manifest.get("symbol_graph_schema_version", 0))
    if version > SYMBOL_GRAPH_SCHEMA_VERSION:
        raise RuntimeError("snapshot symbolic graph schema is newer than this runtime")


def _snapshot_config_id(path: Path) -> str:
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return str(raw.get("scientific_config_id", ""))


def _validated_snapshot_config_id(path: Path, expected_config_id: str) -> str:
    actual = _snapshot_config_id(path)
    if actual == expected_config_id:
        return expected_config_id
    root = path.parent.parent
    migration_path = root / "scientific_config.migration.json"
    if migration_path.exists():
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        if (
            str(migration.get("source_design_version", "")) == "9.7.8"
            and str(migration.get("target_design_version", "")) == "9.7.9"
            and str(migration.get("source_scientific_config_id", "")) == actual
            and str(migration.get("target_scientific_config_id", "")) == expected_config_id
        ):
            return actual
    raise RuntimeError("snapshot ScientificConfigId does not match this run")


def load_snapshot(path: Path, *, expected_config_id: str):
    accepted_config_id = _validated_snapshot_config_id(path, expected_config_id)
    if path.is_file():
        return legacy.load_snapshot(path, expected_config_id=accepted_config_id)
    _validate_symbol_graph_schema(path)
    return chunked_snapshot.load_snapshot(path, expected_config_id=accepted_config_id)


def write_snapshot(root: Path, payload: dict, *, snapshot_id: int, watermark: int, graph_generation: int, scientific_config_id: str) -> SnapshotResult:
    result = chunked_snapshot.write_snapshot(
        root,
        payload,
        snapshot_id=snapshot_id,
        watermark=watermark,
        graph_generation=graph_generation,
        scientific_config_id=scientific_config_id,
    )
    manifest_path = result.path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["symbol_graph_schema_version"] = SYMBOL_GRAPH_SCHEMA_VERSION
        manifest["symbol_grounding_design_version"] = "9.7.9"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    return result


def load_snapshot_direct(path: Path, *, expected_config_id: str):
    if path.is_file():
        return None
    _validate_symbol_graph_schema(path)
    accepted_config_id = _validated_snapshot_config_id(path, expected_config_id)
    return chunked_snapshot.load_snapshot_parts(path, expected_config_id=accepted_config_id)


decode_graph_shard = chunked_snapshot._decode_graph_shard
load_graph_shard = chunked_snapshot.load_graph_shard

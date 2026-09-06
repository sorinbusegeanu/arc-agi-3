from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from v8.model import stable_u64


PERSISTENT_IDENTITY_SCHEMA_VERSION = 2
PERSISTENT_IDENTITY_SCHEMA_NAME = "environment-scoped-seed-free"
PERSISTENT_IDENTITY_MARKER = "persistent_identity.json"

_PERSISTED_STATE_NAMES = frozenset(
    {
        "RUN_COMPLETE.json",
        "action_learning_events_v849",
        "evidence",
        "maintenance",
        "snapshot_chunks",
        "snapshots",
        "trajectory_optimizer",
        "verified_success",
        "v8_run_summary.json",
    }
)


def environment_world_id(family: str, environment_type: str, config: str = "default") -> int:
    """Persistent generic-world identity; config, instance, and seed are metadata."""
    del config
    return stable_u64(
        str(family),
        str(environment_type),
        person=b"v8-env-world-v2",
    )


def arc_world_id(game_id: str) -> int:
    """Preserve the established ARC game-scoped provenance identity exactly."""
    return stable_u64(str(game_id), person=b"v8-game")


def world_id(source_id: str) -> int:
    """Resolve normal v8 source IDs through one persistent provenance scheme."""
    source = str(source_id)
    if source == "FrozenLake-v1":
        return environment_world_id("gymnasium", source)
    if source == "ArcAgi/Chess-v0":
        return environment_world_id("chess", source)
    if source == "ArcAgi/Sudoku-v0":
        return environment_world_id("puzzle", source)
    return arc_world_id(source)


def trajectory_identity(
    source_world_id: int,
    *,
    producer_id: int,
    episode_ordinal: int,
    sequence_base: int,
    namespace: bytes,
) -> int:
    """Seed-free persistent trajectory/deduplication identity."""
    return stable_u64(
        int(source_world_id),
        int(producer_id),
        int(episode_ordinal),
        int(sequence_base),
        person=bytes(namespace),
    )


def _marker_payload() -> dict[str, object]:
    return {
        "schema": PERSISTENT_IDENTITY_SCHEMA_NAME,
        "version": PERSISTENT_IDENTITY_SCHEMA_VERSION,
        "world_scope": "environment_or_arc_game",
        "seed_in_identity": False,
    }


def _read_marker(root: Path) -> dict[str, object] | None:
    path = root / PERSISTENT_IDENTITY_MARKER
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid v8 persistent identity marker: {path}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid v8 persistent identity marker: {path}")
    return raw


def _marker_is_current(marker: dict[str, object]) -> bool:
    try:
        version = int(marker.get("version", 0))
    except (TypeError, ValueError):
        return False
    return bool(
        marker.get("schema") == PERSISTENT_IDENTITY_SCHEMA_NAME
        and version == PERSISTENT_IDENTITY_SCHEMA_VERSION
        and marker.get("seed_in_identity") is False
    )


def _has_persisted_state(root: Path) -> bool:
    return any((root / name).exists() for name in _PERSISTED_STATE_NAMES)


def _archive_path(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = root.with_name(f"{root.name}.seed-scoped-identity-v1.{stamp}")
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root.with_name(
            f"{root.name}.seed-scoped-identity-v1.{stamp}.{suffix}"
        )
    return candidate


def _write_marker(root: Path) -> None:
    path = root / PERSISTENT_IDENTITY_MARKER
    temp = root / f".{PERSISTENT_IDENTITY_MARKER}.{os.getpid()}.tmp"
    temp.write_text(
        json.dumps(_marker_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def prepare_persistent_identity_root(
    root: str | Path,
    *,
    reset_legacy: bool = False,
) -> Path | None:
    """Gate persisted state and optionally archive a legacy seed-scoped store."""
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    marker = _read_marker(path)
    incompatible = marker is not None and not _marker_is_current(marker)
    legacy = marker is None and _has_persisted_state(path)
    if incompatible or legacy:
        if not reset_legacy:
            raise RuntimeError(
                "v8 persistent memory uses legacy seed-scoped provenance; rerun with "
                "--reset-persistent-identity to archive it and start one consistent "
                "environment-scoped store"
            )
        archive = _archive_path(path)
        os.replace(path, archive)
        path.mkdir(parents=True, exist_ok=False)
        _write_marker(path)
        return archive
    if marker is None:
        _write_marker(path)
    return None

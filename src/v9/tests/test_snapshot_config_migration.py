from __future__ import annotations

import json
from pathlib import Path

import pytest

from v9.runtime.snapshot_backend import _validated_snapshot_config_id


def _snapshot(root: Path, config_id: str) -> Path:
    path = root / "snapshots" / "snapshot-00000000000000000003"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"scientific_config_id": config_id}) + "\n",
        encoding="utf-8",
    )
    return path


def test_snapshot_restore_accepts_manifest_approved_same_version_migration(tmp_path: Path) -> None:
    source_id = "a" * 64
    target_id = "b" * 64
    path = _snapshot(tmp_path, source_id)
    (tmp_path / "scientific_config.migration.json").write_text(
        json.dumps(
            {
                "source_design_version": "9.7.9",
                "source_scientific_config_id": source_id,
                "target_design_version": "9.7.9",
                "target_scientific_config_id": target_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _validated_snapshot_config_id(path, target_id) == source_id


def test_snapshot_restore_rejects_unapproved_config_mismatch(tmp_path: Path) -> None:
    source_id = "a" * 64
    target_id = "b" * 64
    path = _snapshot(tmp_path, source_id)
    (tmp_path / "scientific_config.migration.json").write_text(
        json.dumps(
            {
                "source_design_version": "9.7.9",
                "source_scientific_config_id": "c" * 64,
                "target_design_version": "9.7.9",
                "target_scientific_config_id": target_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ScientificConfigId"):
        _validated_snapshot_config_id(path, target_id)

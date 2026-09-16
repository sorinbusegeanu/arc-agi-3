from __future__ import annotations

import json
from pathlib import Path

from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.reset_memory import RESET_MEMORY_ENV


def _write_accepted_model(root: Path) -> None:
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "hgt-000007.pt").write_bytes(b"preserved-checkpoint")
    (model_dir / "hgt_manifest.json").write_text(
        json.dumps(
            {
                "model_schema_version": 1,
                "version_index": 7,
                "current_model_version": "hgt-000007",
                "current_checkpoint": "models/hgt-000007.pt",
                "accepted_model_version": "hgt-000007",
                "last_accepted_model_version": "hgt-000007",
                "accepted_checkpoint": "models/hgt-000007.pt",
                "candidate_model_version": None,
                "candidate_status": "PROMOTED",
                "parent_model_version": None,
                "action_scores": {"9": {"2": 0.75}},
                "context_action_scores": {"9": {"12": {"2": 0.9}}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_reset_memory_starts_empty_and_keeps_last_accepted_hgt(tmp_path: Path, monkeypatch) -> None:
    first = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    first.submit(
        first.make_experience(
            producer_id=1,
            producer_sequence=1,
            environment_instance_id=7,
            global_step=0,
            context_signature=3,
            action_id=2,
            outcome_signature=4,
            family_signature=5,
        )
    )
    first.close()
    assert first.graph.memory_count() > 0
    assert (tmp_path / "snapshots").exists()

    _write_accepted_model(tmp_path)
    monkeypatch.setenv(RESET_MEMORY_ENV, "1")

    reset = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    assert reset.graph.memory_count() == 0
    assert len(reset.graph.edges) == 0
    assert not (tmp_path / "snapshots").exists()
    assert not (tmp_path / "snapshot_chunks").exists()
    assert (tmp_path / "models" / "hgt-000007.pt").read_bytes() == b"preserved-checkpoint"
    assert reset._hgt_action_scores == {9: {2: 0.75}}
    assert reset._hgt_context_action_scores == {9: {12: {2: 0.9}}}
    assert reset.unified_telemetry.model_version == "hgt-000007"
    assert reset.unified_telemetry.gauges["reset_memory"] == 1
    assert reset.unified_telemetry.gauges["reset_memory_hgt_model"] == "hgt-000007"

    manifest = json.loads((tmp_path / "models" / "hgt_manifest.json").read_text(encoding="utf-8"))
    assert manifest["current_model_version"] == "hgt-000007"
    assert manifest["candidate_model_version"] is None
    assert manifest["candidate_status"] == "RESET_TO_ACCEPTED"
    reset.close()


def test_reset_memory_requires_an_accepted_hgt_before_deleting_snapshots(tmp_path: Path, monkeypatch) -> None:
    first = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    first.close()
    snapshot_paths = tuple((tmp_path / "snapshots").iterdir())
    assert snapshot_paths

    monkeypatch.setenv(RESET_MEMORY_ENV, "1")
    try:
        ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    except RuntimeError as exc:
        assert "accepted HGT" in str(exc)
    else:
        raise AssertionError("reset-memory accepted an empty model registry")

    assert tuple((tmp_path / "snapshots").iterdir()) == snapshot_paths


def test_module_entrypoint_consumes_reset_memory_flag(monkeypatch) -> None:
    import sys
    from v9.__main__ import _consume_reset_memory_flag

    monkeypatch.delenv(RESET_MEMORY_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["python", "continuous-run", "--reset-memory", "--games", "step1"])
    _consume_reset_memory_flag()
    assert "--reset-memory" not in sys.argv
    assert sys.argv[-2:] == ["--games", "step1"]
    assert __import__("os").environ[RESET_MEMORY_ENV] == "1"

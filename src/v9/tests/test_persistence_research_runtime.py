from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from v9.research.grounding_h16 import GroundingCondition, evaluate_h16, run_matched_controls, run_synthetic_h16_controls
from v9.research.evidence import EvidenceLedger
from v9.research.hypotheses import HypothesisStatus
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig


def _learn(root: Path, events: int, *, restore: bool) -> ContinuousMemoryRuntime:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(root, restore=restore))
    runtime.start()
    for index in range(events):
        runtime.submit(runtime.make_experience(producer_id=1, producer_sequence=index + 1, environment_instance_id=7, global_step=index, context_signature=3, action_id=2, outcome_signature=4, family_signature=5))
    return runtime


def test_snapshot_restart_preserves_graph_and_continues_identity(tmp_path: Path) -> None:
    first = _learn(tmp_path, 3, restore=False)
    before = first.metrics()
    first.close()
    restored = _learn(tmp_path, 1, restore=True)
    after = restored.metrics()
    assert after["watermark"] > before["watermark"]
    assert after["memories"] > before["memories"]
    restored.close()


def test_predecessor_root_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "v8_run_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="predecessor"):
        ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))


def test_evidence_is_append_only_across_restart(tmp_path: Path) -> None:
    first = _learn(tmp_path, 2, restore=False)
    count = len(first.evidence.records)
    first.close()
    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    assert len(restored.evidence.records) == count


def test_restore_preserves_durable_evidence_ahead_of_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "ledger.jsonl"
    ledger = EvidenceLedger(path, "config")
    first = ledger.append("INGESTION", 1, {"event": 1})
    snapshot_state = ledger.state_dict()
    second = ledger.append("ISF_DECISION", 2, {"score": 0.5})

    restored = EvidenceLedger(path, "config")
    restored.load_state(snapshot_state)

    assert restored.records == [first, second]


def test_h16_requires_matched_c0_through_c3_and_aligned_advantage() -> None:
    def runner(condition: GroundingCondition, seed: int, budget: int):
        del seed, budget
        return (1.0, 1.0, 0.5) if condition is GroundingCondition.C2_ALIGNED else (0.1, 0.1, 0.0)

    trials = run_matched_controls(runner, seeds=(1, 2), environment_config_id=9, interaction_budget=10)
    report = evaluate_h16(trials)
    assert report.matched
    assert report.result.status is HypothesisStatus.SUPPORTED


def test_synthetic_h16_executes_all_matched_controls() -> None:
    trials = run_synthetic_h16_controls(seeds=(3, 4), environment_config_id=11, interaction_budget=24)
    assert {trial.condition for trial in trials} == set(GroundingCondition)
    assert evaluate_h16(trials).matched


def test_cli_smoke_uses_only_v9_named_artifacts(tmp_path: Path) -> None:
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run([sys.executable, "-m", "v9", "smoke", "--root", str(tmp_path), "--events", "4", "--no-restore"], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "smoke done" in result.stdout
    assert "v9 smoke done" not in result.stdout
    assert (tmp_path / "snapshots").is_dir()
    assert not (tmp_path / "v8_run_summary.json").exists()
    report = json.loads((tmp_path / "reports" / "reporting_cut.json").read_text())
    assert report["scientific_config"]["design_version"] == "9.7.6"


def test_native_snapshot_uses_content_addressed_chunks(tmp_path: Path) -> None:
    runtime = _learn(tmp_path, 3, restore=False)
    result = runtime.close()
    assert result is not None
    assert result.path.is_dir()
    assert (result.path / "manifest.json").is_file()
    assert (result.path / "COMPLETE").is_file()
    chunks = list((tmp_path / "snapshot_chunks").glob("*.bin"))
    assert chunks

    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    assert restored.metrics()["memory_levels"]["M0"] >= 3


def test_snapshot_chunks_are_content_addressed_and_reused(tmp_path: Path) -> None:
    from v9.runtime.snapshot_chunks import write_chunks

    payload = b"x" * (4 * 1024 * 1024 + 17)
    first = write_chunks(tmp_path, payload)
    before = {path.name for path in (tmp_path / "snapshot_chunks").glob("*.bin")}
    second = write_chunks(tmp_path, payload)
    after = {path.name for path in (tmp_path / "snapshot_chunks").glob("*.bin")}

    assert first == second
    assert before == after
    assert len(after) == 2

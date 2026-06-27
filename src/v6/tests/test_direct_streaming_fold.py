from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.continuous_research import ContinuousResearchConfig, run_continuous_research
from v6.evaluation.interaction_sampling import InteractionSamplingConfig, run_interaction_sampling_v05c
from v6.memory.direct_streaming_fold import (
    DirectStreamingFoldConfig,
    DirectStreamingFoldJob,
    DirectStreamingFoldWriter,
    direct_streaming_manifest_exists,
)


def _make_fake_raw_job(tmp_path: Path, name: str = "seed_0.sqlite") -> Path:
    db_path = tmp_path / name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"fake-db")
    for sidecar in (
        "live_graph_compact.json",
        "carrier_candidates.json",
        "context_contradictions.json",
        "memory_lifecycle_summary.json",
        "memory_replay_candidates.json",
        "efficiency_summary.json",
    ):
        db_path.with_name(sidecar).write_text("{}", encoding="utf-8")
    return db_path


def test_direct_streaming_fold_creates_no_temp_dirs_and_deletes_raw(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory",
        lambda *args, **kwargs: {"db_files_folded": 1},
    )

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir)),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
    )
    writer.start()
    writer.submit(
        DirectStreamingFoldJob(
            job_id="tt01:random_baseline:seed0:steps5",
            db_path=str(db_path),
            game="tt01",
            sampler="random_baseline",
            seed=0,
            steps=5,
            horizon=2,
            context_depth=1,
            global_step_start=1,
            global_step_end=5,
            memory_dir=str(memory_dir),
        )
    )
    summary = writer.close()

    assert summary["direct_streaming_fold_success_count"] == 1
    assert summary["direct_streaming_fold_deleted_raw_count"] == 1
    assert direct_streaming_manifest_exists(memory_dir)
    assert not any(memory_dir.glob("sampling_sidecar_fold_*"))
    assert not any(memory_dir.glob("compact_merge_*"))
    assert not any(memory_dir.glob("compact_fold_*"))
    assert not any(memory_dir.glob("streaming_fold_shards"))
    assert not db_path.exists()
    with sqlite3.connect(memory_dir / "direct_streaming_fold_manifest.sqlite") as conn:
        row = conn.execute("SELECT status, deleted_raw FROM folded_jobs WHERE job_id = ?", ("tt01:random_baseline:seed0:steps5",)).fetchone()
    assert row == ("folded", 1)


def test_direct_streaming_fold_failure_keeps_raw(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("broken fold")

    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", _boom)

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir)),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
    )
    writer.start()
    writer.submit(
        DirectStreamingFoldJob(
            job_id="tt01:random_baseline:seed0:steps5",
            db_path=str(db_path),
            game="tt01",
            sampler="random_baseline",
            seed=0,
            steps=5,
            horizon=2,
            context_depth=1,
            global_step_start=1,
            global_step_end=5,
            memory_dir=str(memory_dir),
        )
    )
    summary = writer.close()

    assert summary["direct_streaming_fold_failed_count"] == 1
    assert db_path.exists()
    with sqlite3.connect(memory_dir / "direct_streaming_fold_manifest.sqlite") as conn:
        row = conn.execute("SELECT status, deleted_raw, error FROM folded_jobs WHERE job_id = ?", ("tt01:random_baseline:seed0:steps5",)).fetchone()
    assert row[0] == "failed"
    assert row[1] == 0
    assert "broken fold" in str(row[2])


def test_reports_work_after_raw_deletion(tmp_path: Path) -> None:
    output_dir = tmp_path / "sampling"
    memory_dir = output_dir / "memory"
    rows = run_interaction_sampling_v05c(
        InteractionSamplingConfig(
            games=("tt01",),
            samplers=("random_baseline",),
            seeds=(0,),
            train_seeds=(0,),
            test_seed=0,
            steps=10,
            horizon=2,
            context_depth=1,
            workers=1,
            output_dir=str(output_dir),
            memory_output_dir=str(memory_dir),
            direct_streaming_fold_enabled=True,
            delete_raw_after_direct_streaming_fold=True,
        )
    )
    assert rows
    report = json.loads((output_dir / "interaction_sampling_v05c_report.json").read_text(encoding="utf-8"))
    assert "temporal_milestones" in report
    assert report["temporal_milestones"]["by_game_sampler_seed"]
    assert list((output_dir / "sampling_v05c").rglob("*.sqlite")) == []


def test_continuous_run_skips_final_raw_fold(tmp_path: Path) -> None:
    output_dir = tmp_path / "continuous"
    manifest = run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="direct_fold_smoke",
            games="tt01",
            samplers="random_baseline",
            seeds="0",
            steps_per_epoch=10,
            max_epochs=1,
            horizon=2,
            context_depth=1,
            output_dir=str(output_dir),
            cleanup=False,
            workers=1,
            initial_workers=1,
            fast_postprocessing=True,
            direct_streaming_fold=True,
            delete_raw_after_direct_streaming_fold=True,
        )
    )
    epoch = manifest["epochs"][0]
    worker_execution = epoch["worker_execution"]
    assert worker_execution["direct_streaming_fold_enabled"] is True
    assert worker_execution["legacy_sidecar_fold_removed"] is True
    assert epoch["final_raw_epoch_fold_skipped"] is True
    assert not any((output_dir / "memory").glob("sampling_sidecar_fold_*"))
    assert not any((output_dir / "memory").glob("compact_merge_*"))


def test_cli_rejects_legacy_sidecar_flags() -> None:
    from v6.cli import main

    try:
        main(["interaction-sampling-v05c", "--sidecar-fold"])
    except SystemExit as exc:
        assert "Direct streaming fold is now the only normal fold mode." in str(exc)
    else:
        raise AssertionError("expected SystemExit for removed legacy sidecar flags")

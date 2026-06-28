from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from v6.cli import build_parser
from v6.continuous_research import ContinuousResearchConfig, run_continuous_research
from v6.evaluation.interaction_sampling import InteractionSamplingConfig, _generate_sampling_dbs, _run_sampling_jobs, run_interaction_sampling_v05c
from v6.memory.compact_memory import (
    CompactMemoryFoldConfig,
    checkpoint_compact_memory,
    ensure_memory_layout,
    finalize_main_compact_memory,
)
from v6.memory.direct_streaming_fold import (
    DirectStreamingFoldConfig,
    DirectStreamingFoldJob,
    DirectStreamingFoldWriter,
    direct_streaming_manifest_exists,
    ensure_direct_streaming_fold_manifest,
    fold_one_completed_job_to_shard,
    is_retryable_fold_error,
    retry_direct_streaming_fold_failures,
)
from v6.storage.migration import migrate_sqlite_to_parquet


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


class _ThreadPoolCompat(ThreadPoolExecutor):
    def __init__(self, max_workers=None, max_tasks_per_child=None):
        super().__init__(max_workers=max_workers)


def _patch_writer_parallelism(monkeypatch) -> None:
    monkeypatch.setattr("v6.memory.direct_streaming_fold.ProcessPoolExecutor", _ThreadPoolCompat)


def test_direct_streaming_fold_creates_no_temp_dirs_and_deletes_raw(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")
    _patch_writer_parallelism(monkeypatch)

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": [{"contingency_id": 1, "contingency_key": [1, ["ctx"], 1, 1], "features": {"context_level": 1.0}, "label": "PRESERVE"}]},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory",
        lambda *args, **kwargs: {"db_files_folded": 1},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.merge_direct_fold_shards",
        lambda *args, **kwargs: {"merged": True},
    )
    monkeypatch.setattr("v6.memory.direct_streaming_fold.finalize_main_compact_memory", lambda *args, **kwargs: {"finalized": True})

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=2),
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
    assert not any(memory_dir.glob("direct_streaming_fold_shards"))
    assert not db_path.exists()
    assert not db_path.with_name("live_graph_compact.json").exists()
    assert not db_path.with_name("carrier_candidates.json").exists()
    assert not db_path.with_name("context_contradictions.json").exists()
    assert not db_path.with_name("memory_lifecycle_summary.json").exists()
    assert not db_path.with_name("memory_replay_candidates.json").exists()
    assert not db_path.with_name("efficiency_summary.json").exists()
    with sqlite3.connect(memory_dir / "direct_streaming_fold_manifest.sqlite") as conn:
        row = conn.execute("SELECT status, deleted_raw FROM folded_jobs WHERE job_id = ?", ("tt01:random_baseline:seed0:steps5",)).fetchone()
        validation_count = conn.execute("SELECT COUNT(*) FROM validation_payloads").fetchone()[0]
    assert row == ("folded", 1)
    assert validation_count == 1
    assert summary["direct_streaming_fold_worker_count"] == 2
    assert summary["direct_streaming_fold_shard_count"] == 1
    assert summary["direct_streaming_fold_shards_deleted"] is True
    assert summary["direct_streaming_fold_finalized_main_memory"] is True
    assert summary["direct_streaming_fold_shards_merged_incrementally"] is True
    assert summary["direct_streaming_fold_job_shards_deleted_count"] == 1
    assert summary["direct_streaming_fold_raw_deleted_after_main_merge_count"] == 1


def test_direct_streaming_fold_failure_keeps_raw(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")
    _patch_writer_parallelism(monkeypatch)

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": []},
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("broken fold")

    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", _boom)
    monkeypatch.setattr("v6.memory.direct_streaming_fold.merge_direct_fold_shards", lambda *args, **kwargs: {"merged": True})
    monkeypatch.setattr("v6.memory.direct_streaming_fold.finalize_main_compact_memory", lambda *args, **kwargs: {"finalized": True})

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=2),
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


def test_direct_streaming_merge_failure_keeps_raw_and_job_shard(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")
    _patch_writer_parallelism(monkeypatch)

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": []},
    )
    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", lambda *args, **kwargs: {"db_files_folded": 1})
    monkeypatch.setattr("v6.memory.direct_streaming_fold.merge_direct_fold_shards", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom merge")))
    monkeypatch.setattr("v6.memory.direct_streaming_fold.finalize_main_compact_memory", lambda *args, **kwargs: {"finalized": True})

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=2),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
    )
    writer.start()
    job = DirectStreamingFoldJob(
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
    writer.submit(job)
    summary = writer.close()

    job_shard = memory_dir / "direct_streaming_fold_shards" / "job_tt01_random_baseline_seed0_steps5"
    assert summary["direct_streaming_fold_failed_count"] == 1
    assert summary["direct_streaming_fold_success_count"] == 0
    assert db_path.exists()
    assert job_shard.exists()
    assert db_path.with_name("live_graph_compact.json").exists()
    assert db_path.with_name("carrier_candidates.json").exists()
    assert db_path.with_name("context_contradictions.json").exists()
    assert db_path.with_name("memory_lifecycle_summary.json").exists()
    with sqlite3.connect(memory_dir / "direct_streaming_fold_manifest.sqlite") as conn:
        row = conn.execute("SELECT status, deleted_raw, error FROM folded_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
    assert row[0] == "failed"
    assert row[1] == 0
    assert "merge_failed:" in str(row[2])


def test_direct_streaming_fold_workers_cli_and_defaults() -> None:
    parser = build_parser()
    sampling_args = parser.parse_args(["interaction-sampling-v05c", "--direct-streaming-fold-workers", "4"])
    continuous_args = parser.parse_args(["continuous-research-run", "--experiment-name", "x", "--games", "tt01", "--samplers", "random_baseline", "--seeds", "0", "--steps-per-epoch", "10", "--max-epochs", "1", "--horizon", "2", "--context-depth", "1", "--output-dir", "runs/tmp", "--direct-streaming-fold-workers", "4"])
    assert sampling_args.direct_streaming_fold_workers == 4
    assert continuous_args.direct_streaming_fold_workers == 4
    assert InteractionSamplingConfig().direct_streaming_fold_workers == 8
    assert ContinuousResearchConfig(
        experiment_name="x",
        games="tt01",
        samplers="random_baseline",
        seeds="0",
        steps_per_epoch=10,
        max_epochs=1,
        horizon=2,
        context_depth=1,
        output_dir="runs/tmp",
    ).direct_streaming_fold_workers == 8


def test_direct_streaming_fold_job_dict_includes_worker_count(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _capture_jobs(jobs, **kwargs):
        captured["jobs"] = list(jobs)
        return {"requested_workers": kwargs.get("workers", 0), "initial_workers": kwargs.get("initial_workers", 0), "peak_workers": 0}

    import v6.evaluation.interaction_sampling as interaction_sampling_module

    original = interaction_sampling_module._invoke_run_sampling_jobs
    interaction_sampling_module._invoke_run_sampling_jobs = _capture_jobs
    try:
        _generate_sampling_dbs(
        InteractionSamplingConfig(
            games=("tt01",),
            samplers=("random_baseline",),
            seeds=(0,),
            steps=5,
            horizon=2,
            context_depth=1,
            output_dir=str(tmp_path / "out"),
            direct_streaming_fold_workers=4,
        ),
        tmp_path / "sampling_v05c",
        )
    finally:
        interaction_sampling_module._invoke_run_sampling_jobs = original
    jobs = captured["jobs"]
    assert isinstance(jobs, list) and jobs
    assert jobs[0]["direct_streaming_fold_workers"] == 4


def test_bounded_shard_creation_and_summary(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    raw_dir = tmp_path / "raw"
    _patch_writer_parallelism(monkeypatch)
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": []},
    )
    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", lambda *args, **kwargs: {"db_files_folded": 1})
    monkeypatch.setattr("v6.memory.direct_streaming_fold.merge_direct_fold_shards", lambda *args, **kwargs: {"merged": True})
    monkeypatch.setattr("v6.memory.direct_streaming_fold.finalize_main_compact_memory", lambda *args, **kwargs: {"finalized": True})

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=3),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
    )
    writer.start()
    assert writer._shard_root.exists()
    for idx in range(10):
        db_path = _make_fake_raw_job(raw_dir / f"job_{idx}", name=f"seed_{idx}.sqlite")
        writer.submit(
            DirectStreamingFoldJob(
                job_id=f"tt01:random_baseline:seed{idx}:steps5",
                db_path=str(db_path),
                game="tt01",
                sampler="random_baseline",
                seed=idx,
                steps=5,
                horizon=2,
                context_depth=1,
                global_step_start=idx * 5 + 1,
                global_step_end=idx * 5 + 5,
                memory_dir=str(memory_dir),
            )
        )
    summary = writer.close()
    assert summary["direct_streaming_fold_worker_count"] == 3
    assert summary["direct_streaming_fold_shard_count"] == 10
    assert summary["direct_streaming_fold_jobs_submitted"] == 10
    assert summary["direct_streaming_fold_jobs_completed"] == 10
    assert summary["direct_streaming_fold_success_count"] == 10
    assert summary["direct_streaming_fold_shards_deleted"] is True
    assert summary["direct_streaming_fold_job_shards_deleted_count"] == 10


def test_direct_streaming_fold_preserves_validation(tmp_path: Path) -> None:
    output_dir = tmp_path / "sampling"
    memory_dir = output_dir / "memory"
    rows = run_interaction_sampling_v05c(
        InteractionSamplingConfig(
            games=("tt01",),
            samplers=("random_baseline",),
            seeds=(0, 1, 2),
            train_seeds=(0, 1),
            test_seed=2,
            steps=50,
            horizon=2,
            context_depth=1,
            workers=1,
            output_dir=str(output_dir),
            memory_output_dir=str(memory_dir),
            direct_streaming_fold_enabled=True,
            direct_streaming_fold_workers=2,
            delete_raw_after_direct_streaming_fold=True,
        )
    )
    assert rows
    report = json.loads((output_dir / "interaction_sampling_v05c_report.json").read_text(encoding="utf-8"))
    assert "temporal_milestones" in report
    assert report["temporal_milestones"]["by_game_sampler_seed"]
    run_row = report["runs"][0]
    assert run_row["feature_set"] != "manifest_only"
    assert run_row["classifier"] != "manifest_only"
    assert run_row["id_free_accuracy"] is not None
    assert list((output_dir / "sampling_v05c").rglob("*.sqlite")) == []


def test_collect_only_parquet_exports_before_delete(tmp_path: Path) -> None:
    output_dir = tmp_path / "collect_only"
    parquet_root = tmp_path / "parquet"
    rows = run_interaction_sampling_v05c(
        InteractionSamplingConfig(
            games=("tt01",),
            samplers=("random_baseline",),
            seeds=(0,),
            train_seeds=(0,),
            test_seed=0,
            steps=20,
            horizon=2,
            context_depth=1,
            workers=1,
            collect_only=True,
            storage_backend="parquet",
            parquet_root=str(parquet_root),
            output_dir=str(output_dir),
            memory_output_dir=str(output_dir / "memory"),
            direct_streaming_fold_enabled=True,
            delete_raw_after_direct_streaming_fold=True,
        )
    )
    assert rows == []
    assert list(parquet_root.rglob("*.parquet"))
    with sqlite3.connect(output_dir / "memory" / "direct_streaming_fold_manifest.sqlite") as conn:
        row = conn.execute("SELECT status, deleted_raw, parquet_exported FROM folded_jobs").fetchone()
    assert row == ("folded", 1, 1)
    assert list(output_dir.rglob("seed_*.sqlite")) == []


def test_direct_streaming_fold_finalizes_once(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path_a = _make_fake_raw_job(tmp_path / "raw_a", "seed_0.sqlite")
    db_path_b = _make_fake_raw_job(tmp_path / "raw_b", "seed_1.sqlite")
    calls = {"fold": 0, "merge": 0, "finalize": 0}

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": []},
    )

    _patch_writer_parallelism(monkeypatch)

    def _fold(*args, **kwargs):
        calls["fold"] += 1
        return {"db_files_folded": 1}

    def _merge(*args, **kwargs):
        calls["merge"] += 1
        return {"merged": True}

    def _finalize(*args, **kwargs):
        calls["finalize"] += 1
        return {"finalized": True}

    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", _fold)
    monkeypatch.setattr("v6.memory.direct_streaming_fold.merge_direct_fold_shards", _merge)
    monkeypatch.setattr("v6.memory.direct_streaming_fold.finalize_main_compact_memory", _finalize)

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=2),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
    )
    writer.start()
    for idx, db_path in enumerate((db_path_a, db_path_b)):
        writer.submit(
            DirectStreamingFoldJob(
                job_id=f"tt01:random_baseline:seed{idx}:steps5",
                db_path=str(db_path),
                game="tt01",
                sampler="random_baseline",
                seed=idx,
                steps=5,
                horizon=2,
                context_depth=1,
                global_step_start=idx * 5 + 1,
                global_step_end=idx * 5 + 5,
                memory_dir=str(memory_dir),
            )
        )
    summary = writer.close()
    assert calls["fold"] == 2
    assert calls["merge"] == 2
    assert calls["finalize"] == 1
    assert summary["direct_streaming_fold_finalized_main_memory"] is True


def test_retry_direct_streaming_fold_rebuilds_reports_after_success(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    manifest_path = ensure_direct_streaming_fold_manifest(memory_dir)
    db_path = _make_fake_raw_job(
        tmp_path / "epochs" / "epoch_0001" / "raw" / "sampling_v05c" / "tt01" / "random_baseline" / "steps_5",
        "seed_0.sqlite",
    )
    with sqlite3.connect(manifest_path) as conn:
        conn.execute(
            """
            INSERT INTO folded_jobs (
                job_id, db_path, game, sampler, seed, steps, horizon, context_depth,
                global_step_start, global_step_end, status, fold_started_at, fold_finished_at,
                deleted_raw, parquet_exported, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tt01:random_baseline:seed0:steps5",
                str(db_path),
                "tt01",
                "random_baseline",
                0,
                5,
                2,
                1,
                1,
                5,
                "failed",
                0.0,
                None,
                0,
                0,
                "database is locked",
            ),
        )
        conn.commit()

    _patch_writer_parallelism(monkeypatch)
    def _fake_fold(**kwargs):
        with sqlite3.connect(manifest_path) as conn:
            conn.execute(
                "UPDATE folded_jobs SET status = 'folded', fold_finished_at = 1.0, deleted_raw = 1, error = NULL WHERE job_id = ?",
                (kwargs["job"].job_id,),
            )
            conn.commit()
        return type(
            "Result",
            (),
            {
                "job_id": kwargs["job"].job_id,
                "db_path": kwargs["job"].db_path,
                "status": "folded",
                "fold_started_at": 0.0,
                "fold_finished_at": 1.0,
                "deleted_raw": True,
            },
        )()

    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_one_completed_job_to_shard", _fake_fold)
    monkeypatch.setattr("v6.memory.direct_streaming_fold.merge_direct_fold_shards", lambda **kwargs: {"merged": True})
    monkeypatch.setattr("v6.memory.direct_streaming_fold.finalize_main_compact_memory", lambda **kwargs: {"finalized": True})
    rebuilt: list[str] = []

    def _fake_rebuild(*, memory_dir, jobs):
        rebuilt.append(str(memory_dir))
        return [str(tmp_path / "epochs" / "epoch_0001")]

    monkeypatch.setattr("v6.memory.direct_streaming_fold._rerun_reports_for_retried_epochs", _fake_rebuild)

    summary = retry_direct_streaming_fold_failures(
        manifest_path=manifest_path,
        memory_dir=memory_dir,
        workers=1,
        delete_raw_after_fold=False,
        finalize_after_success=True,
    )

    assert summary["direct_streaming_fold_finalized_main_memory"] is True
    assert summary["direct_streaming_fold_reports_rebuilt"] == [str(tmp_path / "epochs" / "epoch_0001")]
    assert rebuilt == [str(memory_dir)]


def test_direct_streaming_fold_start_removes_stale_shard_root_when_manifest_clean(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    manifest_path = ensure_direct_streaming_fold_manifest(memory_dir)
    shard_root = memory_dir / "direct_streaming_fold_shards"
    (shard_root / "shard_0000").mkdir(parents=True, exist_ok=False)

    with sqlite3.connect(manifest_path) as conn:
        conn.execute(
            """
            INSERT INTO folded_jobs (
                job_id, db_path, game, sampler, seed, steps, horizon, context_depth,
                global_step_start, global_step_end, status, fold_started_at, fold_finished_at,
                deleted_raw, parquet_exported, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tt01:random_baseline:seed0:steps5",
                str(tmp_path / "seed_0.sqlite"),
                "tt01",
                "random_baseline",
                0,
                5,
                2,
                1,
                1,
                5,
                "folded",
                0.0,
                1.0,
                1,
                0,
                None,
            ),
        )
        conn.commit()

    _patch_writer_parallelism(monkeypatch)
    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=2),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
    )
    writer.start()
    try:
        assert writer._summary["direct_streaming_fold_stale_shard_root_removed"] is True
        assert shard_root.exists()
        assert writer._shard_root == shard_root
        assert list(shard_root.iterdir()) == []
    finally:
        if writer._executor is not None:
            writer._executor.shutdown(wait=True, cancel_futures=True)


def test_sampling_pool_refills_before_direct_fold_submit(monkeypatch, tmp_path: Path) -> None:
    class FakeFuture:
        def __init__(self, job):
            self.job = job

        def result(self):
            return {"legacy_future_effects_removed": True}

    class FakeExecutor:
        last_instance = None

        def __init__(self, max_workers=None, max_tasks_per_child=None):
            self.submit_count = 0
            FakeExecutor.last_instance = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, job):
            self.submit_count += 1
            return FakeFuture(job)

    class FakeWriter:
        last_instance = None

        def __init__(self, *args, **kwargs):
            self.submit_counts: list[int] = []
            FakeWriter.last_instance = self

        def start(self) -> None:
            pass

        def submit(self, job) -> None:
            self.submit_counts.append(int(FakeExecutor.last_instance.submit_count))

        def close(self) -> dict[str, object]:
            return {
                "direct_streaming_fold_worker_count": 1,
                "direct_streaming_fold_shard_count": 1,
                "direct_streaming_fold_job_count": 64,
                "direct_streaming_fold_success_count": 64,
                "direct_streaming_fold_failed_count": 0,
                "direct_streaming_fold_deleted_raw_count": 0,
                "direct_streaming_fold_manifest_path": str(tmp_path / "memory" / "direct_streaming_fold_manifest.sqlite"),
                "direct_streaming_fold_shards_deleted": True,
                "direct_streaming_fold_merge_started_at": None,
                "direct_streaming_fold_merge_finished_at": None,
                "direct_streaming_fold_merge_seconds": None,
                "direct_streaming_fold_jobs_submitted": 64,
                "direct_streaming_fold_jobs_completed": 64,
                "direct_streaming_fold_jobs_failed": 0,
                "direct_streaming_fold_raw_deleted_after_shard_fold_count": 0,
                "direct_streaming_fold_finalized_main_memory": True,
                "direct_streaming_fold_total_raw_bytes": 0,
                "direct_streaming_fold_total_shard_bytes_added": 0,
                "direct_streaming_fold_mean_job_seconds": 0.0,
                "direct_streaming_fold_mean_write_mb_per_second": 0.0,
                "direct_streaming_shard_synchronous": "off",
            }

    def _fake_wait(futures, timeout=None, return_when=None):
        return set(list(futures.keys())), set()

    monkeypatch.setattr("v6.evaluation.interaction_sampling.ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr("v6.evaluation.interaction_sampling.wait", _fake_wait)
    monkeypatch.setattr("v6.evaluation.interaction_sampling.DirectStreamingFoldWriter", FakeWriter)

    jobs = [
        {
            "game": "tt01",
            "sampler_name": "random_baseline",
            "seed": idx,
            "steps": 5,
            "horizon": 2,
            "context_depth": 1,
            "global_step_offset": idx * 5,
            "memory_output_dir": str(tmp_path / "memory"),
            "direct_streaming_fold_enabled": True,
            "delete_raw_after_direct_streaming_fold": False,
            "direct_streaming_fold_workers": 2,
            "direct_streaming_fold_submit_delay_seconds": 10.0,
            "max_tasks_per_child": 1,
            "db_path": str(tmp_path / f"seed_{idx}.sqlite"),
        }
        for idx in range(64)
    ]
    stats = _run_sampling_jobs(jobs, workers=32, initial_workers=32)
    writer = FakeWriter.last_instance
    assert writer is not None
    assert len(writer.submit_counts) == 64
    assert all(count == 64 for count in writer.submit_counts[:32])
    assert stats["sampling_refill_count"] >= 2
    assert stats["max_done_batch_size"] == 32
    assert stats["seconds_spent_in_fold_submit_delay"] == 0.0


def test_compact_sqlite_busy_timeout_wal(tmp_path: Path) -> None:
    output_dir = tmp_path / "sampling"
    memory_dir = output_dir / "memory"
    run_interaction_sampling_v05c(
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
            direct_streaming_fold_workers=2,
            delete_raw_after_direct_streaming_fold=True,
        )
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert journal_mode == "wal"


def test_checkpoint_compact_memory_truncates_wal(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        for idx in range(500):
            conn.execute(
                "INSERT OR REPLACE INTO memory_summary (key, value_json) VALUES (?, ?)",
                (f"k{idx}", json.dumps({"value": idx})),
            )
        conn.commit()
    wal_path = memory_dir / "current_state.sqlite-wal"
    before_wal = wal_path.stat().st_size if wal_path.exists() else 0
    result = checkpoint_compact_memory(memory_dir, truncate=True)
    after_wal = wal_path.stat().st_size if wal_path.exists() else 0

    assert "current_state.sqlite" in result["databases"]
    assert before_wal >= after_wal


def test_finalize_main_compact_memory_respects_example_caps(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        for idx in range(3):
            conn.execute(
                """
                INSERT INTO representative_examples (
                    example_id, owner_type, owner_id, game, sampler, seed, global_step,
                    example_kind, compact_payload_json, priority_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"example:{idx}",
                    "contingency",
                    "contingency:1",
                    "tt01",
                    "random_baseline",
                    0,
                    idx + 1,
                    "priority",
                    json.dumps({"idx": idx}),
                    float(idx),
                ),
            )
        conn.commit()

    finalize_main_compact_memory(
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(
            global_step_start=1,
            global_step_end=3,
            max_examples_per_contingency=1,
        ),
    )

    with sqlite3.connect(paths.current_state) as conn:
        retained = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM representative_examples
                WHERE owner_type = 'contingency' AND owner_id = 'contingency:1'
                """
            ).fetchone()[0]
        )
    assert retained == 1


def test_parquet_migration_streams_batches_without_fetchall(tmp_path: Path, monkeypatch) -> None:
    sqlite_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("CREATE TABLE interactions (game TEXT, seed INTEGER, global_step INTEGER, action INTEGER, terminated INTEGER, success INTEGER)")
        for idx in range(5):
            conn.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?)", ("toy", 0, idx + 1, idx % 4, 0, 0))
        conn.commit()

    real_connect = sqlite3.connect

    class NoFetchAllCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        @property
        def description(self):
            return self._cursor.description

        def fetchmany(self, size):
            return self._cursor.fetchmany(size)

        def fetchall(self):
            raise AssertionError("fetchall should not be used for parquet migration")

        def fetchone(self):
            return self._cursor.fetchone()

        def __iter__(self):
            return iter(self._cursor)

    class NoFetchAllConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, *args, **kwargs):
            return NoFetchAllCursor(self._conn.execute(*args, **kwargs))

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(
        "v6.storage.migration.sqlite3.connect",
        lambda *args, **kwargs: NoFetchAllConnection(real_connect(*args, **kwargs)),
    )

    migrate_sqlite_to_parquet(
        sqlite_path=sqlite_path,
        parquet_root=tmp_path / "parquet",
        game="toy",
        sampler="mixed",
        seed=0,
        steps=5,
        batch_size=2,
    )

    assert list((tmp_path / "parquet").rglob("*.parquet"))


def test_direct_streaming_replay_queue_payload_is_minimal(tmp_path: Path) -> None:
    output_dir = tmp_path / "sampling"
    memory_dir = output_dir / "memory"
    run_interaction_sampling_v05c(
        InteractionSamplingConfig(
            games=("tt01",),
            samplers=("random_baseline",),
            seeds=(0,),
            steps=30,
            horizon=2,
            context_depth=1,
            workers=1,
            output_dir=str(output_dir),
            memory_output_dir=str(memory_dir),
            direct_streaming_fold_enabled=True,
            direct_streaming_fold_workers=1,
            delete_raw_after_direct_streaming_fold=True,
        )
    )
    with sqlite3.connect(memory_dir / "replay_queue.sqlite") as conn:
        row = conn.execute(
            "SELECT compact_payload_json FROM replay_queue ORDER BY priority_score DESC, replay_id ASC LIMIT 1"
        ).fetchone()
    if row is None:
        raise AssertionError("expected replay queue rows for minimal payload check")
    payload = json.loads(row[0])
    assert set(payload).issubset(
        {
            "replay_id",
            "game",
            "sampler",
            "seed",
            "global_step",
            "action",
            "context_signature",
            "context_signature_hash",
            "family_signature",
            "carrier_signature",
            "contradiction_key",
            "prediction_error",
            "isf_prediction_error",
            "memory_replay_priority",
            "future_option_delta",
            "terminal",
            "success",
        }
    )
    assert "grid_before" not in payload
    assert "grid_after" not in payload
    assert "prediction_payload" not in payload


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
            direct_streaming_fold_workers=2,
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


def test_failed_validation_payload_keeps_raw(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")
    _patch_writer_parallelism(monkeypatch)

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "steps": 5, "horizon": 2},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )

    def _validation_boom(*args, **kwargs):
        raise RuntimeError("bad validation payload")

    monkeypatch.setattr("v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload", _validation_boom)

    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=2),
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
    assert db_path.with_name("carrier_candidates.json").exists()


def test_is_retryable_fold_error() -> None:
    assert is_retryable_fold_error(sqlite3.OperationalError("database is locked")) is True
    assert is_retryable_fold_error(sqlite3.OperationalError("database is busy")) is True
    assert is_retryable_fold_error(sqlite3.OperationalError("other sqlite issue")) is False
    assert is_retryable_fold_error(RuntimeError("database is locked")) is False


def test_retryable_locked_fold_is_retried_and_succeeds(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")
    attempts = {"fold": 0}
    monkeypatch.setattr("v6.memory.direct_streaming_fold.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": []},
    )

    def _fold(*args, **kwargs):
        attempts["fold"] += 1
        if attempts["fold"] < 2:
            raise sqlite3.OperationalError("database is locked")
        return {"db_files_folded": 1}

    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", _fold)
    result = fold_one_completed_job_to_shard(
        job=DirectStreamingFoldJob(
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
        ),
        config=DirectStreamingFoldConfig(memory_dir=str(memory_dir), retry_attempts=5),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
        shard_dir=str(memory_dir / "retry_shard"),
    )
    assert result.status == "folded"
    assert attempts["fold"] == 2


def test_retryable_locked_fold_exhausts_and_fails(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")
    attempts = {"fold": 0}
    monkeypatch.setattr("v6.memory.direct_streaming_fold.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": []},
    )

    def _fold(*args, **kwargs):
        attempts["fold"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", _fold)
    result = fold_one_completed_job_to_shard(
        job=DirectStreamingFoldJob(
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
        ),
        config=DirectStreamingFoldConfig(memory_dir=str(memory_dir), retry_attempts=3),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
        shard_dir=str(memory_dir / "retry_shard"),
    )
    assert result.status == "failed"
    assert attempts["fold"] == 3


def test_non_retryable_fold_error_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    db_path = _make_fake_raw_job(tmp_path / "raw")
    attempts = {"fold": 0}
    monkeypatch.setattr("v6.memory.direct_streaming_fold.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: {"game": "tt01", "sampler": "random_baseline", "seed": 0},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: {"game": "tt01", "sampler_name": "random_baseline", "seed": 0, "examples": []},
    )

    def _fold(*args, **kwargs):
        attempts["fold"] += 1
        raise RuntimeError("broken fold")

    monkeypatch.setattr("v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory", _fold)
    result = fold_one_completed_job_to_shard(
        job=DirectStreamingFoldJob(
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
        ),
        config=DirectStreamingFoldConfig(memory_dir=str(memory_dir), retry_attempts=5),
        sampling_config=type("Cfg", (), {"steps": 5, "horizon": 2})(),
        shard_dir=str(memory_dir / "retry_shard"),
    )
    assert result.status == "failed"
    assert attempts["fold"] == 1


def test_cli_rejects_legacy_sidecar_flags() -> None:
    from v6.cli import main

    try:
        main(["interaction-sampling-v05c", "--sidecar-fold"])
    except SystemExit as exc:
        assert "Direct streaming fold is now the only normal fold mode." in str(exc)
    else:
        raise AssertionError("expected SystemExit for removed legacy sidecar flags")

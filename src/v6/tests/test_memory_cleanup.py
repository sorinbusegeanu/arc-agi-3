from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.memory.compact_memory import CompactMemoryFoldConfig, ensure_memory_layout, fold_epoch_raw_into_compact_memory
from v6.memory.memory_cleanup import cleanup_epoch_artifacts, stop_due_to_disk


def _write_epoch_db(path: Path, *, replay_rows: int = 8, contingency_support: int = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY,
                game_id TEXT,
                level_id TEXT,
                sampler_name TEXT,
                episode_id INTEGER,
                global_step INTEGER,
                outcome_state TEXT,
                level_completed_event INTEGER,
                state_hash_before TEXT,
                state_hash_after TEXT,
                action INTEGER,
                efficiency_no_effect_action INTEGER,
                efficiency_future_option_gain_per_cost REAL,
                memory_replay_priority REAL,
                memory_replay_candidate INTEGER,
                carrier_signature TEXT,
                context_depth_used INTEGER
            );
            CREATE TABLE contingencies (
                id INTEGER PRIMARY KEY,
                context_signature TEXT,
                action INTEGER,
                transformation_family INTEGER,
                support_count INTEGER
            );
            CREATE TABLE prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER,
                global_step INTEGER,
                context_signature TEXT,
                action INTEGER,
                predicted_family INTEGER,
                actual_family INTEGER,
                prediction_error INTEGER,
                isf_prediction_error REAL,
                memory_replay_priority REAL,
                context_contradiction INTEGER
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO interactions (
                id, game_id, level_id, sampler_name, episode_id, global_step,
                outcome_state, level_completed_event, state_hash_before, state_hash_after,
                action, efficiency_no_effect_action, efficiency_future_option_gain_per_cost,
                memory_replay_priority, memory_replay_candidate, carrier_signature, context_depth_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    index,
                    "tt01",
                    "lvl1",
                    "mixed",
                    0,
                    index,
                    "RUNNING",
                    0,
                    f"s{index}",
                    f"s{index+1}",
                    index % 3,
                    0,
                    0.0,
                    1.0 - index * 0.05,
                    1,
                    "carrier-a" if index % 2 == 0 else "",
                    1,
                )
                for index in range(1, replay_rows + 1)
            ],
        )
        connection.execute(
            "INSERT INTO contingencies (id, context_signature, action, transformation_family, support_count) VALUES (1, '[1,2]', 0, 10, ?)",
            (contingency_support,),
        )
        connection.executemany(
            """
            INSERT INTO prediction_results (
                interaction_id, global_step, context_signature, action, predicted_family, actual_family,
                prediction_error, isf_prediction_error, memory_replay_priority, context_contradiction
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, "ctx-a", 0, 10, 10, 0, 0.0, 0.9, 0),
                (2, 2, "ctx-a", 0, 9, 10, 1, 1.0, 0.95, 1),
                (3, 3, "ctx-b", 1, 10, 10, 0, 0.0, 0.3, 0),
            ],
        )
        connection.commit()
    path.with_name("carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "carrier-a",
                    "carrier_signature": "carrier-a",
                    "carrier_source": "object",
                    "support_count": 3,
                    "distinct_family_count": 1,
                    "prediction_lift": 0.4,
                    "status": "candidate",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def test_replay_queue_limit_and_representative_example_limits_are_enforced(tmp_path: Path) -> None:
    raw_dir = tmp_path / "epoch_0001" / "raw"
    raw_dir.mkdir(parents=True)
    _write_epoch_db(raw_dir / "seed_0.sqlite", replay_rows=12, contingency_support=30)
    (raw_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps(
            {
                "temporal_milestones": {
                    "by_game_sampler_seed": [
                        {
                            "game": "tt01",
                            "sampler": "mixed",
                            "seed": 0,
                            "first_interaction_step": 1,
                            "first_contingency_candidate_step": None,
                            "first_stable_contingency_step": None,
                            "first_prediction_violation_step": 2,
                            "first_high_replay_priority_step": 2,
                            "first_transformation_family_step": 1,
                            "first_stable_transformation_family_step": None,
                            "first_carrier_candidate_step": None,
                            "first_emergent_carrier_step": None,
                        }
                    ]
                },
                "validation": {"memory_record_count": 12},
            }
        ),
        encoding="utf-8",
    )
    memory_paths = ensure_memory_layout(tmp_path / "memory")

    fold_epoch_raw_into_compact_memory(
        epoch_raw_dir=raw_dir,
        memory_dir=memory_paths.root,
        fold_config=CompactMemoryFoldConfig(
            global_step_start=1,
            global_step_end=5000,
            max_replay_queue_size=5,
            max_examples_per_contingency=4,
        ),
    )

    with sqlite3.connect(memory_paths.replay_queue) as connection:
        assert connection.execute("SELECT COUNT(*) FROM replay_queue").fetchone()[0] == 5
    with sqlite3.connect(memory_paths.current_state) as connection:
        assert connection.execute("SELECT COUNT(*) FROM representative_examples WHERE owner_type = 'contingency'").fetchone()[0] <= 4
        row = connection.execute("SELECT first_prediction_violation_step FROM temporal_milestones WHERE game = 'tt01' AND sampler = 'mixed' AND seed = 0").fetchone()
        assert row[0] == 2


def test_cleanup_deletes_raw_files_but_keeps_reports_and_memory(tmp_path: Path) -> None:
    epoch_dir = tmp_path / "epoch_0001"
    raw_dir = epoch_dir / "raw"
    reports_dir = epoch_dir / "reports"
    status_dir = epoch_dir / "status"
    cleanup_dir = epoch_dir / "cleanup"
    raw_dir.mkdir(parents=True)
    reports_dir.mkdir()
    status_dir.mkdir()
    cleanup_dir.mkdir()
    (raw_dir / "seed_0.sqlite").write_text("raw", encoding="utf-8")
    (reports_dir / "hypothesis_suite_summary.json").write_text("{}", encoding="utf-8")
    (status_dir / "epoch_status.json").write_text("{}", encoding="utf-8")
    memory_paths = ensure_memory_layout(tmp_path / "memory")
    (memory_paths.summary_json).write_text(json.dumps({"fold_summary": {"stable_contingencies_added": 1}}, indent=2), encoding="utf-8")

    summary = cleanup_epoch_artifacts(epoch_dir=epoch_dir, memory_dir=memory_paths.root)

    assert summary["raw_files_deleted_count"] == 1
    assert not raw_dir.exists()
    assert reports_dir.exists()
    assert status_dir.exists()
    assert memory_paths.current_state.exists()
    assert (cleanup_dir / "cleanup_summary.json").exists()
    assert summary["wal_checkpoint_mode"] == "TRUNCATE"
    assert summary["vacuum_performed"] is False


def test_cleanup_refuses_without_compact_memory_update(tmp_path: Path) -> None:
    epoch_dir = tmp_path / "epoch_0001"
    (epoch_dir / "raw").mkdir(parents=True)
    (epoch_dir / "reports").mkdir()
    (epoch_dir / "reports" / "hypothesis_suite_summary.json").write_text("{}", encoding="utf-8")
    memory_paths = ensure_memory_layout(tmp_path / "memory")

    try:
        cleanup_epoch_artifacts(epoch_dir=epoch_dir, memory_dir=memory_paths.root)
    except RuntimeError as exc:
        assert "fold summary" in str(exc)
    else:
        raise AssertionError("cleanup should have refused without fold summary")


def test_stop_due_to_disk_triggers_at_threshold(tmp_path: Path, monkeypatch) -> None:
    class _Usage:
        total = 100
        used = 90
        free = 10

    monkeypatch.setattr("v6.memory.memory_cleanup.shutil.disk_usage", lambda path: _Usage())
    triggered, snapshot = stop_due_to_disk(tmp_path, threshold_percent=90)

    assert triggered is True
    assert snapshot["disk_stop_triggered"] is True


def test_cleanup_runs_vacuum_only_every_tenth_epoch(tmp_path: Path, monkeypatch) -> None:
    epoch_dir = tmp_path / "epoch_0010"
    raw_dir = epoch_dir / "raw"
    reports_dir = epoch_dir / "reports"
    raw_dir.mkdir(parents=True)
    reports_dir.mkdir()
    (raw_dir / "seed_0.sqlite").write_text("raw", encoding="utf-8")
    (reports_dir / "hypothesis_suite_summary.json").write_text("{}", encoding="utf-8")
    memory_paths = ensure_memory_layout(tmp_path / "memory_vacuum")
    memory_paths.summary_json.write_text(json.dumps({"fold_summary": {"stable_contingencies_added": 1}}, indent=2), encoding="utf-8")

    calls: list[tuple[str, bool]] = []

    def _record_cleanup(path: Path, *, vacuum: bool = False) -> None:
        calls.append((path.name, vacuum))

    monkeypatch.setattr("v6.memory.memory_cleanup._sqlite_cleanup", _record_cleanup)
    summary = cleanup_epoch_artifacts(epoch_dir=epoch_dir, memory_dir=memory_paths.root)

    assert summary["vacuum_performed"] is True
    assert calls
    assert all(vacuum is True for _name, vacuum in calls)

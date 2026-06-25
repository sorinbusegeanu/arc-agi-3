from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import v6.continuous_research as continuous_research
from v6.continuous_research import ContinuousResearchConfig, run_continuous_research


def _write_sampling_fixture(output_dir: Path, *, global_step_offset: int, stable_support: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "sampling_v05c" / "tt01" / "mixed" / "steps_5000" / "seed_0.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY,
                global_step INTEGER,
                isf_prediction_error REAL,
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
                context_contradiction INTEGER,
                context_expansion_suggested INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO interactions (id, global_step, isf_prediction_error, memory_replay_priority, memory_replay_candidate, carrier_signature, context_depth_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, global_step_offset + 1, 0.0, 0.20, 1, "", 1),
                (2, global_step_offset + 2, 1.0, 0.95, 1, "", 1),
                (3, global_step_offset + 3, 0.0, 0.25, 1, "", 1),
                (4, global_step_offset + 4, 1.0, 0.90, 1, "", 1),
            ],
        )
        connection.execute(
            "INSERT INTO contingencies (id, context_signature, action, transformation_family, support_count) VALUES (1, '[1,2]', 0, 10, ?)",
            (stable_support,),
        )
        connection.executemany(
            """
            INSERT INTO prediction_results (
                interaction_id, global_step, context_signature, action, predicted_family, actual_family,
                prediction_error, isf_prediction_error, memory_replay_priority, context_contradiction, context_expansion_suggested
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, global_step_offset + 1, "ctx-a", 0, 10, 10, 0, 0.0, 0.2, 0, 0),
                (2, global_step_offset + 2, "ctx-a", 0, 9, 10, 1, 1.0, 0.95, 1, 1),
                (3, global_step_offset + 3, "ctx-b", 1, 10, 10, 0, 0.0, 0.25, 0, 0),
                (4, global_step_offset + 4, "ctx-b", 1, 9, 10, 1, 1.0, 0.90, 1, 1),
            ],
        )
        connection.commit()
    db_path.with_name("carrier_candidates.json").write_text("[]", encoding="utf-8")
    (output_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps(
            {
                "games": ["tt01"],
                "samplers": ["mixed"],
                "seeds": [0],
                "runs": [
                    {
                        "game": "tt01",
                        "sampler_name": "mixed",
                        "run_status": "ok",
                        "total_interactions": 4,
                        "stable_contingency_count": 1 if stable_support >= 20 else 0,
                        "prediction_accuracy": 0.5,
                        "unique_transformation_families": 1,
                        "mean_isf_prediction_error": 0.5,
                        "high_priority_replay_count": 2,
                    }
                ],
                "validation": {
                    "memory_record_count": 4,
                    "mean_isf_total": 1.0,
                    "max_isf_total": 1.0,
                    "mean_isf_prediction_error": 0.5,
                    "mean_isf_learning_value": 0.1,
                    "mean_isf_transfer_potential": 0.1,
                    "mean_isf_explanatory_potential": 0.1,
                    "high_isf_interaction_count": 2,
                    "context_contradiction_count": 2,
                    "contradicted_context_count": 2,
                    "contradicted_context_action_count": 2,
                    "repeated_contradiction_count": 1,
                    "context_expansion_suggested_count": 2,
                    "memory_replay_candidate_count": 4,
                    "memory_mean_replay_priority": 0.58,
                    "memory_max_replay_priority": 0.95,
                    "high_priority_replay_count": 2,
                    "carrier_candidate_count": 0,
                    "emergent_carrier_count": 0,
                    "emergent_object_carrier_count": 0,
                    "emergent_context_action_fallback_count": 0,
                    "carrier_object_candidate_count": 0,
                    "carrier_context_action_fallback_candidate_count": 0,
                },
                "temporal_milestones": {
                    "by_game_sampler_seed": [
                        {
                            "game": "tt01",
                            "sampler": "mixed",
                            "seed": 0,
                            "first_interaction_step": global_step_offset + 1,
                            "first_contingency_candidate_step": global_step_offset + 1,
                            "first_stable_contingency_step": None if stable_support < 20 else global_step_offset + 2,
                            "first_prediction_violation_step": global_step_offset + 2,
                            "first_high_replay_priority_step": global_step_offset + 2,
                            "first_transformation_family_step": global_step_offset + 1,
                            "first_stable_transformation_family_step": None,
                            "first_carrier_candidate_step": None,
                            "first_emergent_carrier_step": None,
                        }
                    ]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "interaction_sampling_v05c_report.txt").write_text("stub\n", encoding="utf-8")
    return [{"game": "tt01", "total_interactions": 4}]


def test_continuous_command_creates_manifest_epoch_and_memory(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_run_sampling(config):
        calls.append(int(config.global_step_offset))
        return _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25)

    monkeypatch.setattr(continuous_research, "run_interaction_sampling_v05c", fake_run_sampling)
    manifest = run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=5000,
            max_epochs=1,
            horizon=10,
            context_depth=1,
            output_dir=str(tmp_path / "continuous"),
        )
    )

    root = tmp_path / "continuous"
    assert (root / "manifest.json").exists()
    assert (root / "memory").is_dir()
    assert (root / "epochs" / "epoch_0001").is_dir()
    assert manifest["completed_epochs"] == 1
    assert calls == [0]


def test_epoch_2_loads_existing_memory_and_global_step_offset_increments(tmp_path: Path, monkeypatch) -> None:
    seen_offsets: list[int] = []

    def fake_run_sampling(config):
        seen_offsets.append(int(config.global_step_offset))
        return _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25)

    monkeypatch.setattr(continuous_research, "run_interaction_sampling_v05c", fake_run_sampling)
    root = tmp_path / "continuous"
    first = ContinuousResearchConfig(
        experiment_name="exp",
        games="tt01",
        samplers="mixed",
        seeds="0",
        steps_per_epoch=5000,
        max_epochs=1,
        horizon=10,
        context_depth=1,
        output_dir=str(root),
    )
    second = ContinuousResearchConfig(
        experiment_name="exp",
        games="tt01",
        samplers="mixed",
        seeds="0",
        steps_per_epoch=5000,
        max_epochs=2,
        horizon=10,
        context_depth=1,
        output_dir=str(root),
        resume=True,
    )

    run_continuous_research(first)
    manifest = run_continuous_research(second)

    assert seen_offsets == [0, 5000]
    assert manifest["current_epoch"] == 2


def test_no_new_contingencies_stop_works(tmp_path: Path, monkeypatch) -> None:
    stable_supports = [25, 10]

    def fake_run_sampling(config):
        support = stable_supports.pop(0)
        return _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=support)

    monkeypatch.setattr(continuous_research, "run_interaction_sampling_v05c", fake_run_sampling)
    manifest = run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=5000,
            max_epochs=5,
            horizon=10,
            context_depth=1,
            output_dir=str(tmp_path / "continuous"),
            stop_if_no_new_stable_contingencies_for=1,
        )
    )

    assert manifest["stopped"] is True
    assert manifest["stop_reason"] == "no new stable contingencies for configured consecutive epochs"


def test_epoch_status_and_suite_summary_exist_and_memory_continuity_is_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    root = tmp_path / "continuous"
    run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=5000,
            max_epochs=1,
            horizon=10,
            context_depth=1,
            output_dir=str(root),
        )
    )

    status = json.loads((root / "epochs" / "epoch_0001" / "status" / "epoch_status.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "epochs" / "epoch_0001" / "reports" / "hypothesis_suite_summary.json").read_text(encoding="utf-8"))
    continuity = json.loads((root / "epochs" / "epoch_0001" / "reports" / "epoch_memory_continuity.json").read_text(encoding="utf-8"))
    assert status["H01"] is not None
    assert status["H02"] is not None
    assert status["H03"] is not None
    assert summary["H04 decision"] in {"INCONCLUSIVE", "PARTIALLY_VALID", "VALID", "INVALID"}
    assert continuity["continuity_valid"] is True


def test_disk_stop_triggers_and_default_is_90(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        continuous_research,
        "stop_due_to_disk",
        lambda path, threshold_percent: (True, {"disk_used_percent": 95.0, "stop_if_disk_above_percent": threshold_percent, "disk_stop_triggered": True}),
    )
    manifest = run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=5000,
            max_epochs=1,
            horizon=10,
            context_depth=1,
            output_dir=str(tmp_path / "continuous"),
        )
    )

    assert manifest["stop_if_disk_above_percent"] == 90.0
    assert manifest["stop_reason"] == "disk usage exceeded configured limit"


def test_resume_true_fails_if_memory_files_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    root = tmp_path / "continuous"
    run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=5000,
            max_epochs=1,
            horizon=10,
            context_depth=1,
            output_dir=str(root),
        )
    )
    (root / "memory" / "current_state.sqlite").unlink()
    try:
        run_continuous_research(
            ContinuousResearchConfig(
                experiment_name="exp",
                games="tt01",
                samplers="mixed",
                seeds="0",
                steps_per_epoch=5000,
                max_epochs=2,
                horizon=10,
                context_depth=1,
                output_dir=str(root),
                resume=True,
            )
        )
    except RuntimeError as exc:
        assert "resume requested" in str(exc)
    else:
        raise AssertionError("expected resume failure when memory files are missing")

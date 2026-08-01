from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import v6.continuous_research as continuous_research
import v6.evaluation.interaction_sampling as interaction_sampling
from v6.continuous_research import ContinuousResearchConfig, run_continuous_research
from v6.memory.compact_memory import ensure_memory_layout


def _write_sampling_fixture(
    output_dir: Path,
    *,
    global_step_offset: int,
    stable_support: int,
    worker_execution: dict | None = None,
) -> list[dict]:
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
                "worker_execution": worker_execution or {
                    "requested_workers": 4,
                    "initial_workers": 2,
                    "peak_workers": 3,
                "worker_ramp_enabled": True,
                "ram_ramp_threshold_percent": 85.0,
                "initial_worker_ramp_delay_seconds": 20.0,
                "per_worker_ramp_delay_seconds": 5.0,
                "ram_used_percent_at_start": 42.0,
                "ramp_event_count": 1,
                "ramp_events": [{"target_workers": 3, "ram_used_percent": 43.0, "seconds_since_start": 20.0}],
            },
                "level_completion_records": [
                    {
                        "game_id": "tt01",
                        "level_id": "level_0001",
                        "level_name": "level_0001",
                        "completed": True,
                        "success": True,
                        "seed": 0,
                        "sampler": "mixed",
                        "steps_used": None,
                    }
                ],
                "levels_successfully_completed_per_epoch": 1,
                "games_solved_per_epoch": 0,
                "solved_games": [],
                "completed_levels_by_game": {"tt01": 1},
        },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "interaction_sampling_v05c_report.txt").write_text("stub\n", encoding="utf-8")
    return [{"game": "tt01", "total_interactions": 4}]


def test_log_epoch_phase_prints_without_jsonl_side_effect(tmp_path: Path, capsys) -> None:
    continuous_research._log_epoch_phase("epoch_0001", "hypothesis_suite", "done", {"H01": "VALID"})
    captured = capsys.readouterr()
    assert " hypothesis_suite: done" in captured.out
    assert " E0001]" in captured.out
    assert "H01" in captured.out
    assert not any(tmp_path.rglob("epoch_phase_log.jsonl"))


def test_log_epoch_phase_done_prints_one_timed_compact_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr(continuous_research.time, "perf_counter", lambda: 12.345)

    continuous_research._log_epoch_phase_done(
        "epoch_0001",
        "artifact_cleanup",
        10.0,
        {"disk_after_cleanup_bytes": 50},
    )

    output = capsys.readouterr().out.strip()
    assert " artifact_cleanup: done seconds=2.35 disk_after_cleanup_bytes=50" in output
    assert "starting" not in output
    assert "{" not in output


def test_hypothesis_suite_phase_log_prints_all_hypotheses_in_order(capsys) -> None:
    summary = " ".join(f"H{i:02d}=DECISION_{i:02d}" for i in range(1, 13))
    continuous_research._log_epoch_phase("epoch_0003", "hypothesis_suite", "done", summary)
    output = capsys.readouterr().out.strip()
    assert output.endswith(summary)
    assert "{" not in output
    assert '"' not in output
    assert ":" not in output.split(" done ", 1)[1]
    assert [output.index(f"H{i:02d}=") for i in range(1, 13)] == sorted(
        output.index(f"H{i:02d}=") for i in range(1, 13)
    )


def test_continuous_stdout_is_written_to_overwritten_log_file(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "continuous_log"
    root.mkdir(parents=True, exist_ok=True)
    (root / "log.txt").write_text("OLD CONTENT\n", encoding="utf-8")

    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    monkeypatch.setattr(continuous_research, "direct_streaming_manifest_has_failures", lambda memory_dir: False)
    monkeypatch.setattr(
        continuous_research,
        "run_hypothesis_suite_report",
        lambda **kwargs: {
            "game_count": 1,
            "interactions_this_epoch": 4,
            "levels_successfully_completed_per_epoch": 1,
            "games_solved_per_epoch": 0,
            "solved_games": [],
            "completed_levels_by_game": {"tt01": 1},
            "H01 decision": "PARTIALLY_VALID",
            "H02 decision": "PARTIALLY_VALID",
            "H03 decision": "PARTIALLY_VALID",
            "H04 decision": "INCONCLUSIVE",
            "H05 decision": "INCONCLUSIVE",
            "H06 decision": "INCONCLUSIVE",
            "H07 decision": "INCONCLUSIVE",
            "H08 decision": "INCONCLUSIVE",
            "H09 decision": "INCONCLUSIVE",
            "H10 decision": "INCONCLUSIVE",
            "H11 decision": "INCONCLUSIVE",
            "H12 decision": "INSUFFICIENT_EVIDENCE",
            "H01 core metrics": {},
            "H02 core metrics": {},
            "H03 core metrics": {},
            "H04 core metrics": {},
            "H05 core metrics": {},
            "H06 core metrics": {},
            "H07 core metrics": {},
            "H08 core metrics": {},
            "H09 core metrics": {},
            "H10 core metrics": {},
            "H11 core metrics": {},
            "H12 core metrics": {},
            "memory_size_before_bytes": 0,
            "memory_size_after_bytes": 0,
        },
    )
    monkeypatch.setattr(continuous_research, "run_selective_forgetting_pass", lambda **kwargs: {"changed": True})
    monkeypatch.setattr(continuous_research, "evaluate_h10b_selective_forgetting", lambda **kwargs: {"decision": "PARTIALLY_VALID"})
    monkeypatch.setattr(
        continuous_research,
        "build_memory_summary",
        lambda memory_paths: {
            "stable_contingency_count": 1,
            "transformation_family_count": 1,
            "memory_node_count": 2,
            "graph_node_count": 0,
            "graph_edge_count": 0,
            "replay_queue_size": 0,
        },
    )
    monkeypatch.setattr(continuous_research, "_write_memory_continuity_report", lambda **kwargs: {"continuity_valid": True})
    monkeypatch.setattr(continuous_research, "validate_cleanup_safe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuous_research,
        "cleanup_epoch_artifacts",
        lambda **kwargs: {
            "disk_before_cleanup_bytes": 100,
            "disk_after_cleanup_bytes": 50,
            "disk_freed_bytes": 50,
            "raw_files_deleted_count": 1,
            "raw_bytes_deleted": 50,
            "temp_files_deleted_count": 0,
            "temp_bytes_deleted": 0,
            "memory_db_size_bytes": 0,
            "graph_db_size_bytes": 0,
            "replay_queue_db_size_bytes": 0,
            "reports_size_bytes": 0,
            "kept_files": [],
            "deleted_files_sample": [],
            "deletion_errors": [],
        },
    )

    run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=10,
            max_epochs=1,
            horizon=2,
            context_depth=1,
            output_dir=str(root),
        )
    )

    log_text = (root / "log.txt").read_text(encoding="utf-8")
    assert "OLD CONTENT" not in log_text
    assert "Epoch 0001 starting" in log_text


def test_continuous_run_emits_single_timed_post_fold_phase_lines(tmp_path: Path, monkeypatch) -> None:
    phase_events: list[tuple[str, str, str | None]] = []

    def record_phase(epoch_id: str, phase: str, status: str = "starting", extra: dict | str | None = None) -> None:
        phase_events.append((phase, status, None if extra is None else str(extra)))

    monkeypatch.setattr(continuous_research, "_log_epoch_phase", record_phase)
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    monkeypatch.setattr(continuous_research, "direct_streaming_manifest_has_failures", lambda memory_dir: False)
    monkeypatch.setattr(
        continuous_research,
        "run_hypothesis_suite_report",
        lambda **kwargs: {
            "game_count": 1,
            "interactions_this_epoch": 4,
            "levels_successfully_completed_per_epoch": 1,
            "games_solved_per_epoch": 0,
            "solved_games": [],
            "completed_levels_by_game": {"tt01": 1},
            "H01 decision": "PARTIALLY_VALID",
            "H02 decision": "PARTIALLY_VALID",
            "H03 decision": "PARTIALLY_VALID",
            "H04 decision": "INCONCLUSIVE",
            "H05 decision": "INCONCLUSIVE",
            "H06 decision": "INCONCLUSIVE",
            "H07 decision": "INCONCLUSIVE",
            "H08 decision": "INCONCLUSIVE",
            "H09 decision": "INCONCLUSIVE",
            "H10 decision": "INCONCLUSIVE",
            "H11 decision": "INCONCLUSIVE",
            "H12 decision": "INSUFFICIENT_EVIDENCE",
            "H01 core metrics": {"stable_contingency_count": 1, "games_with_stable_contingencies": 1},
            "H02 core metrics": {"direct_replay_lift_available": False, "prediction_violation_replay_lift": None},
            "H03 core metrics": {"compression_ratio": 1.1, "singleton_family_ratio": 0.5, "family_cross_context_count": 1},
            "H04 core metrics": {},
            "H05 core metrics": {},
            "H06 core metrics": {},
            "H07 core metrics": {},
            "H08 core metrics": {},
            "H09 core metrics": {},
            "H10 core metrics": {},
            "H11 core metrics": {},
            "H12 core metrics": {},
            "memory_size_before_bytes": 0,
            "memory_size_after_bytes": 0,
        },
    )
    monkeypatch.setattr(continuous_research, "run_selective_forgetting_pass", lambda **kwargs: {"changed": True})
    monkeypatch.setattr(continuous_research, "evaluate_h10b_selective_forgetting", lambda **kwargs: {"decision": "PARTIALLY_VALID"})
    monkeypatch.setattr(
        continuous_research,
        "build_memory_summary",
        lambda memory_paths: {
            "stable_contingency_count": 1,
            "transformation_family_count": 1,
            "memory_node_count": 2,
            "graph_node_count": 0,
            "graph_edge_count": 0,
            "replay_queue_size": 0,
        },
    )
    monkeypatch.setattr(continuous_research, "_write_memory_continuity_report", lambda **kwargs: {"continuity_valid": True})
    monkeypatch.setattr(continuous_research, "validate_cleanup_safe", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        continuous_research,
        "cleanup_epoch_artifacts",
        lambda **kwargs: {
            "disk_before_cleanup_bytes": 100,
            "disk_after_cleanup_bytes": 50,
            "disk_freed_bytes": 50,
            "raw_files_deleted_count": 1,
            "raw_bytes_deleted": 50,
            "temp_files_deleted_count": 0,
            "temp_bytes_deleted": 0,
            "memory_db_size_bytes": 0,
            "graph_db_size_bytes": 0,
            "replay_queue_db_size_bytes": 0,
            "reports_size_bytes": 0,
            "kept_files": [],
            "deleted_files_sample": [],
            "deletion_errors": [],
        },
    )

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
            output_dir=str(tmp_path / "continuous"),
        )
    )

    assert not any(status == "starting" for _, status, _ in phase_events)
    done_phases = {phase for phase, status, _ in phase_events if status == "done"}
    assert "hypothesis_suite" in done_phases
    assert "selective_forgetting" in done_phases
    assert "h10b_selective_forgetting" in done_phases
    assert "memory_summary" in done_phases
    assert not {
        "memory_continuity_report",
        "cleanup_validation",
        "artifact_cleanup",
        "epoch_status_write",
    } & done_phases
    for phase, status, extra in phase_events:
        if status == "done" and phase != "epoch_results":
            assert extra is not None
            assert "seconds=" in extra


def test_continuous_resolves_full_hypothesis_suite_epoch_mode(tmp_path: Path, monkeypatch) -> None:
    captured_modes: list[str] = []

    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    monkeypatch.setattr(continuous_research, "direct_streaming_manifest_has_failures", lambda memory_dir: False)
    monkeypatch.setattr(
        continuous_research,
        "run_hypothesis_suite_report",
        lambda **kwargs: captured_modes.append(str(kwargs["suite_mode"])) or {
            "game_count": 1,
            "interactions_this_epoch": 4,
            "levels_successfully_completed_per_epoch": 0,
            "games_solved_per_epoch": 0,
            "solved_games": [],
            "completed_levels_by_game": {},
            "H01 decision": "PARTIALLY_VALID",
            "H02 decision": "PARTIALLY_VALID",
            "H03 decision": "PARTIALLY_VALID",
            "H04 decision": "INCONCLUSIVE",
            "H05 decision": "INCONCLUSIVE",
            "H06 decision": "INCONCLUSIVE",
            "H07 decision": "INCONCLUSIVE",
            "H08 decision": "INCONCLUSIVE",
            "H09 decision": "INCONCLUSIVE",
            "H10 decision": "INCONCLUSIVE",
            "H11 decision": "INCONCLUSIVE",
            "H12 decision": "INSUFFICIENT_EVIDENCE",
            "H01 core metrics": {},
            "H02 core metrics": {},
            "H03 core metrics": {},
            "H04 core metrics": {},
            "H05 core metrics": {},
            "H06 core metrics": {},
            "H07 core metrics": {},
            "H08 core metrics": {},
            "H09 core metrics": {},
            "H10 core metrics": {},
            "H11 core metrics": {},
            "H12 core metrics": {},
            "memory_size_before_bytes": 0,
            "memory_size_after_bytes": 0,
        },
    )
    monkeypatch.setattr(continuous_research, "run_selective_forgetting_pass", lambda **kwargs: {})
    monkeypatch.setattr(continuous_research, "evaluate_h10b_selective_forgetting", lambda **kwargs: {"decision": "PARTIALLY_VALID"})
    monkeypatch.setattr(continuous_research, "build_memory_summary", lambda memory_paths: {"stable_contingency_count": 1, "transformation_family_count": 1, "memory_node_count": 1, "graph_node_count": 0, "graph_edge_count": 0, "replay_queue_size": 0})
    monkeypatch.setattr(continuous_research, "_write_memory_continuity_report", lambda **kwargs: {})
    monkeypatch.setattr(continuous_research, "validate_cleanup_safe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuous_research, "cleanup_epoch_artifacts", lambda **kwargs: {"disk_before_cleanup_bytes": 0, "disk_after_cleanup_bytes": 0, "disk_freed_bytes": 0, "raw_files_deleted_count": 0, "raw_bytes_deleted": 0, "temp_files_deleted_count": 0, "temp_bytes_deleted": 0, "memory_db_size_bytes": 0, "graph_db_size_bytes": 0, "replay_queue_db_size_bytes": 0, "reports_size_bytes": 0, "kept_files": [], "deleted_files_sample": [], "deletion_errors": []})

    run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=10,
            max_epochs=2,
            horizon=2,
            context_depth=1,
            output_dir=str(tmp_path / "continuous"),
            hypothesis_suite_mode="fast",
            full_hypothesis_suite_every_epochs=2,
        )
    )
    assert captured_modes == ["fast", "full"]


def test_continuous_passes_hypothesis_suite_limits(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, int] = {}

    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    monkeypatch.setattr(continuous_research, "direct_streaming_manifest_has_failures", lambda memory_dir: False)
    monkeypatch.setattr(
        continuous_research,
        "run_hypothesis_suite_report",
        lambda **kwargs: captured.update(
            {
                "max_role_transfer_attempts": int(kwargs["max_role_transfer_attempts"]),
                "max_future_option_events": int(kwargs["max_future_option_events"]),
                "max_future_option_motifs": int(kwargs["max_future_option_motifs"]),
            }
        ) or {
            "game_count": 1,
            "interactions_this_epoch": 4,
            "levels_successfully_completed_per_epoch": 0,
            "games_solved_per_epoch": 0,
            "solved_games": [],
            "completed_levels_by_game": {},
            "H01 decision": "PARTIALLY_VALID",
            "H02 decision": "PARTIALLY_VALID",
            "H03 decision": "PARTIALLY_VALID",
            "H04 decision": "INCONCLUSIVE",
            "H05 decision": "INCONCLUSIVE",
            "H06 decision": "INCONCLUSIVE",
            "H07 decision": "INCONCLUSIVE",
            "H08 decision": "INCONCLUSIVE",
            "H09 decision": "INCONCLUSIVE",
            "H10 decision": "INCONCLUSIVE",
            "H11 decision": "INCONCLUSIVE",
            "H12 decision": "INSUFFICIENT_EVIDENCE",
            "H01 core metrics": {},
            "H02 core metrics": {},
            "H03 core metrics": {},
            "H04 core metrics": {},
            "H05 core metrics": {},
            "H06 core metrics": {},
            "H07 core metrics": {},
            "H08 core metrics": {},
            "H09 core metrics": {},
            "H10 core metrics": {},
            "H11 core metrics": {},
            "H12 core metrics": {},
            "memory_size_before_bytes": 0,
            "memory_size_after_bytes": 0,
        },
    )
    monkeypatch.setattr(continuous_research, "run_selective_forgetting_pass", lambda **kwargs: {})
    monkeypatch.setattr(continuous_research, "evaluate_h10b_selective_forgetting", lambda **kwargs: {"decision": "PARTIALLY_VALID"})
    monkeypatch.setattr(continuous_research, "build_memory_summary", lambda memory_paths: {"stable_contingency_count": 1, "transformation_family_count": 1, "memory_node_count": 1, "graph_node_count": 0, "graph_edge_count": 0, "replay_queue_size": 0})
    monkeypatch.setattr(continuous_research, "_write_memory_continuity_report", lambda **kwargs: {})
    monkeypatch.setattr(continuous_research, "validate_cleanup_safe", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuous_research, "cleanup_epoch_artifacts", lambda **kwargs: {"disk_before_cleanup_bytes": 0, "disk_after_cleanup_bytes": 0, "disk_freed_bytes": 0, "raw_files_deleted_count": 0, "raw_bytes_deleted": 0, "temp_files_deleted_count": 0, "temp_bytes_deleted": 0, "memory_db_size_bytes": 0, "graph_db_size_bytes": 0, "replay_queue_db_size_bytes": 0, "reports_size_bytes": 0, "kept_files": [], "deleted_files_sample": [], "deletion_errors": []})

    run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=10,
            max_epochs=1,
            horizon=2,
            context_depth=1,
            output_dir=str(tmp_path / "continuous"),
            max_role_transfer_attempts_per_epoch=111,
            max_future_option_events_per_epoch=222,
            max_future_option_motifs_per_epoch=333,
        )
    )
    assert captured == {
        "max_role_transfer_attempts": 111,
        "max_future_option_events": 222,
        "max_future_option_motifs": 333,
    }


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


def test_continuous_prints_epoch_game_and_level_results(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(
            Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25
        ),
    )
    run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=10,
            max_epochs=1,
            horizon=2,
            context_depth=1,
            output_dir=str(tmp_path / "continuous"),
        )
    )
    output = capsys.readouterr().out
    assert " epoch_results: done level_completion_events=" in output
    assert "levels_solved_this_epoch=" in output
    assert "total_unique_games_solved=" in output
    assert "total_unique_levels_solved=" in output


def test_continuous_initial_memory_checkpoint_is_copied_to_new_output_only(tmp_path: Path, monkeypatch, capsys) -> None:
    source = tmp_path / "phase_a" / "memory"
    ensure_memory_layout(source)
    source_marker = source / "checkpoint_marker.txt"
    source_marker.write_text("phase-a", encoding="utf-8")
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(
            Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25
        ),
    )

    root = tmp_path / "phase_b"
    manifest = run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="phase_b",
            games="tt01",
            samplers="mixed",
            seeds="0",
            steps_per_epoch=10,
            max_epochs=1,
            horizon=2,
            context_depth=1,
            output_dir=str(root),
            initial_memory_dir=str(source),
        )
    )

    destination = root / "memory"
    assert (destination / "checkpoint_marker.txt").read_text(encoding="utf-8") == "phase-a"
    assert source_marker.read_text(encoding="utf-8") == "phase-a"
    assert source != destination
    assert manifest["initial_memory_dir"] == str(source)
    assert manifest["memory_checkpoint_restored"] is True
    assert float(manifest["memory_checkpoint_copy_time_seconds"]) >= 0.0
    output = capsys.readouterr().out
    assert f"Initializing memory from:\n  {source}" in output
    assert f"Writing memory to:\n  {destination}" in output
    assert "Memory checkpoint restored successfully." in output


def test_initial_memory_checkpoint_validates_source_and_destination(tmp_path: Path) -> None:
    source = tmp_path / "phase_a" / "memory"
    ensure_memory_layout(source)
    common = dict(
        experiment_name="phase_b", games="tt01", samplers="mixed", seeds="0",
        steps_per_epoch=10, max_epochs=1, horizon=2, context_depth=1,
    )
    with pytest.raises(RuntimeError, match="does not exist"):
        run_continuous_research(
            ContinuousResearchConfig(**common, output_dir=str(tmp_path / "missing"), initial_memory_dir=str(tmp_path / "nope"))
        )
    incomplete = tmp_path / "incomplete_memory"
    incomplete.mkdir()
    with pytest.raises(RuntimeError, match="missing required SQLite"):
        run_continuous_research(
            ContinuousResearchConfig(**common, output_dir=str(tmp_path / "incomplete_output"), initial_memory_dir=str(incomplete))
        )
    with pytest.raises(RuntimeError, match="must not equal"):
        run_continuous_research(
            ContinuousResearchConfig(**common, output_dir=str(tmp_path / "phase_a"), initial_memory_dir=str(source))
        )
    existing_output = tmp_path / "existing"
    (existing_output / "memory").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        run_continuous_research(
            ContinuousResearchConfig(
                **common,
                output_dir=str(existing_output),
                initial_memory_dir=str(source),
                resume=False,
            )
        )


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
    assert status["levels_successfully_completed_per_epoch"] == 1
    assert status["games_solved_per_epoch"] == 1
    assert status["solved_games"] == ["tt01"]
    assert summary["H04 decision"] in {"INCONCLUSIVE", "PARTIALLY_VALID", "VALID", "INVALID"}
    assert summary["levels_successfully_completed_per_epoch"] == 1
    assert summary["games_solved_per_epoch"] == 1
    assert summary["solved_games"] == ["tt01"]
    assert summary["completed_levels_by_game"] == {"tt01": 1}
    assert continuity["continuity_valid"] is True


def test_compute_epoch_completion_counters_case_a_all_levels_completed(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "expected_levels_by_game", lambda: {"ga01": 3})
    result = interaction_sampling.compute_epoch_completion_counters(
        [
            {"game_id": "ga01", "level_id": "level_0001", "completed": True},
            {"game_id": "ga01", "level_id": "level_0002", "completed": True},
            {"game_id": "ga01", "level_id": "level_0003", "completed": True},
        ]
    )
    assert result["levels_successfully_completed_per_epoch"] == 3
    assert result["games_solved_per_epoch"] == 1
    assert result["solved_games"] == ["ga01"]
    assert result["completed_levels_by_game"] == {"ga01": 3}


def test_compute_epoch_completion_counters_case_b_partial_game(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "expected_levels_by_game", lambda: {"ga01": 3})
    result = interaction_sampling.compute_epoch_completion_counters(
        [
            {"game_id": "ga01", "level_id": "level_0001", "completed": True},
            {"game_id": "ga01", "level_id": "level_0002", "completed": True},
        ]
    )
    assert result["levels_successfully_completed_per_epoch"] == 2
    assert result["games_solved_per_epoch"] == 1
    assert result["solved_games"] == ["ga01"]
    assert result["completed_levels_by_game"] == {"ga01": 2}


def test_compute_epoch_completion_counters_case_c_mixed_games(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "expected_levels_by_game", lambda: {"ga01": 2, "gb01": 2})
    result = interaction_sampling.compute_epoch_completion_counters(
        [
            {"game_id": "ga01", "level_id": "level_0001", "completed": True},
            {"game_id": "ga01", "level_id": "level_0002", "completed": True},
            {"game_id": "gb01", "level_id": "level_0001", "completed": True},
        ]
    )
    assert result["levels_successfully_completed_per_epoch"] == 3
    assert result["games_solved_per_epoch"] == 2
    assert result["solved_games"] == ["ga01", "gb01"]
    assert result["completed_levels_by_game"] == {"ga01": 2, "gb01": 1}


def test_compute_epoch_completion_counters_case_d_deduplicates_duplicate_level_records(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "expected_levels_by_game", lambda: {"ga01": 2})
    result = interaction_sampling.compute_epoch_completion_counters(
        [
            {"game_id": "ga01", "level_id": "level_0001", "completed": True, "sampler": "a"},
            {"game_id": "ga01", "level_id": "level_0001", "completed": True, "sampler": "b"},
            {"game_id": "ga01", "level_id": "level_0002", "success": True, "sampler": "a"},
        ]
    )
    assert result["levels_successfully_completed_per_epoch"] == 2
    assert result["games_solved_per_epoch"] == 1
    assert result["solved_games"] == ["ga01"]
    assert result["completed_levels_by_game"] == {"ga01": 2}


def test_manifest_tracks_run_wide_completion_totals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
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
    assert manifest["total_levels_successfully_completed"] == 1
    assert manifest["total_games_solved"] == 1
    assert manifest["games_solved_by_epoch"] == {"epoch_0001": ["tt01"]}


def test_completion_counters_separate_duplicate_events_from_unique_level_solves() -> None:
    result = interaction_sampling.compute_epoch_completion_counters(
        [
            {"game_id": "ga01", "level_id": "01", "completed": True, "sampler": "a", "seed": 0},
            {"game_id": "ga01", "level_id": 1, "completed": True, "sampler": "b", "seed": 0},
            {"game_id": "ga01", "level_id": "1.0", "completed": True, "sampler": "a", "seed": 1},
            {"game_id": "ga01", "level_id": "001", "completed": True, "sampler": "b", "seed": 1},
        ]
    )
    assert result["level_completion_event_count"] == 4
    assert result["levels_solved_this_epoch"] == 1
    assert result["new_levels_solved_this_epoch"] == 1
    assert result["total_unique_levels_solved"] == 1
    assert result["game_completion_event_count"] == 4
    assert result["games_solved_this_epoch"] == 1
    assert result["new_games_solved_this_epoch"] == 1
    assert result["repeated_level_keys_sample"] == [["ga01", "1"]]


def test_completion_counters_keep_repeated_levels_out_of_new_totals() -> None:
    result = interaction_sampling.compute_epoch_completion_counters(
        [
            {"game_id": "ga01", "level_id": "l1", "completed": True},
            {"game_id": "ga01", "level_id": "l2", "completed": True},
        ],
        previous_level_keys=[("ga01", "l1")],
        previous_game_keys=["ga01"],
    )
    assert result["levels_solved_this_epoch"] == 2
    assert result["new_levels_solved_this_epoch"] == 1
    assert result["total_unique_levels_solved"] == 2
    assert result["new_games_solved_this_epoch"] == 0
    assert result["total_unique_games_solved"] == 1
    assert result["repeated_level_keys_sample"] == [["ga01", "l1"]]
    assert result["levels_solved"] == result["levels_solved_this_epoch"]
    assert result["total_levels_solved"] == result["total_unique_levels_solved"]


def test_completion_counters_reject_missing_level_id_from_unique_counts() -> None:
    result = interaction_sampling.compute_epoch_completion_counters(
        [{"game_id": "ga01", "completed": True, "sampler": "mixed", "seed": 0}]
    )
    assert result["level_completion_event_count"] == 1
    assert result["levels_solved_this_epoch"] == 0
    assert result["games_solved_this_epoch"] == 0
    assert result["invalid_completion_records_sample"] == [
        {"game_id": "ga01", "level_id": None, "sampler": "mixed", "seed": 0, "source_run_db": None}
    ]


def test_completion_identity_state_loads_persisted_and_reconstructs_epoch_artifacts(tmp_path: Path) -> None:
    persisted = continuous_research._load_completion_identity_state(
        tmp_path,
        {
            "solved_level_keys": [["ga01", "l1"]],
            "solved_game_ids": ["ga01"],
            "completion_identity_state_complete": True,
            "completion_identity_state_source": "manifest",
        },
    )
    assert persisted["solved_level_keys"] == {("ga01", "l1")}
    assert persisted["source"] == "manifest"

    epoch_dir = tmp_path / "epochs" / "epoch_0001"
    report_path = epoch_dir / "raw" / "interaction_sampling_v05c_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"epoch_level_keys": [["gb01", "l2"]]}), encoding="utf-8")
    reconstructed = continuous_research._load_completion_identity_state(
        tmp_path,
        {
            "completed_epochs": 1,
            "epochs": [{"status": "complete", "epoch_dir": str(epoch_dir)}],
        },
    )
    assert reconstructed["complete"] is True
    assert reconstructed["source"] == "epoch_artifacts"
    assert reconstructed["solved_level_keys"] == {("gb01", "l2")}


def test_completion_identity_state_does_not_change_when_same_epoch_is_recomputed() -> None:
    first = interaction_sampling.compute_epoch_completion_counters(
        [{"game_id": "ga01", "level_id": "l1", "completed": True}]
    )
    rerun = interaction_sampling.compute_epoch_completion_counters(
        [{"game_id": "ga01", "level_id": "l1", "completed": True}],
        previous_level_keys=first["solved_level_keys"],
        previous_game_keys=first["solved_game_ids"],
    )
    assert rerun["new_levels_solved_this_epoch"] == 0
    assert rerun["new_games_solved_this_epoch"] == 0
    assert rerun["total_unique_levels_solved"] == 1
    assert rerun["total_unique_games_solved"] == 1


def test_continuous_completion_totals_do_not_accumulate_repeated_epoch_solves(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    root = tmp_path / "continuous"
    manifest = run_continuous_research(
        ContinuousResearchConfig(
            experiment_name="exp", games="tt01", samplers="mixed", seeds="0",
            steps_per_epoch=2, max_epochs=2, horizon=1, context_depth=1, output_dir=str(root),
        )
    )
    second_status = json.loads((root / "epochs" / "epoch_0002" / "status" / "epoch_status.json").read_text(encoding="utf-8"))
    assert second_status["level_completion_event_count"] == 1
    assert second_status["levels_solved_this_epoch"] == 1
    assert second_status["new_levels_solved_this_epoch"] == 0
    assert second_status["total_unique_levels_solved"] == 1
    assert second_status["new_games_solved_this_epoch"] == 0
    assert second_status["total_unique_games_solved"] == 1
    assert manifest["solved_level_keys"] == [["tt01", "level_0001"]]


def test_completion_identity_state_is_not_persisted_when_epoch_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        continuous_research,
        "run_interaction_sampling_v05c",
        lambda config: _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25),
    )
    monkeypatch.setattr(continuous_research, "run_hypothesis_suite_report", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    root = tmp_path / "continuous"
    with pytest.raises(RuntimeError, match="boom"):
        run_continuous_research(
            ContinuousResearchConfig(
                experiment_name="exp", games="tt01", samplers="mixed", seeds="0",
                steps_per_epoch=1, max_epochs=1, horizon=1, context_depth=1, output_dir=str(root),
            )
        )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["solved_level_keys"] == []
    assert manifest["solved_game_ids"] == []


def test_continuous_run_passes_fast_postprocessing_into_sampling(tmp_path: Path, monkeypatch) -> None:
    seen_flags: list[bool] = []

    def fake_run_sampling(config):
        seen_flags.append(bool(config.fast_postprocessing))
        return _write_sampling_fixture(Path(config.output_dir), global_step_offset=int(config.global_step_offset), stable_support=25)

    monkeypatch.setattr(continuous_research, "run_interaction_sampling_v05c", fake_run_sampling)
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
            output_dir=str(tmp_path / "continuous"),
            fast_postprocessing=True,
        )
    )

    assert seen_flags == [True]


def test_continuous_run_preserves_configured_worker_cap_and_records_ram_start(tmp_path: Path, monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_run_sampling(config):
        captured.append(
            {
                "workers": int(config.workers),
                "initial_workers": int(config.initial_workers or 0),
                "enable_worker_ramp": bool(config.enable_worker_ramp),
                "ram_ramp_threshold_percent": float(config.ram_ramp_threshold_percent),
                "initial_worker_ramp_delay_seconds": float(config.initial_worker_ramp_delay_seconds),
                "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
            }
        )
        return _write_sampling_fixture(
            Path(config.output_dir),
            global_step_offset=int(config.global_step_offset),
            stable_support=25,
            worker_execution={
                "requested_workers": int(config.workers),
                "initial_workers": int(config.initial_workers or 0),
                "peak_workers": int(config.workers),
                "worker_ramp_enabled": bool(config.enable_worker_ramp),
                "ram_ramp_threshold_percent": float(config.ram_ramp_threshold_percent),
                "initial_worker_ramp_delay_seconds": float(config.initial_worker_ramp_delay_seconds),
                "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
                "ram_used_percent_at_start": 44.0,
                "ramp_event_count": 0,
                "ramp_events": [],
            },
        )

    monkeypatch.setattr(continuous_research, "run_interaction_sampling_v05c", fake_run_sampling)
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
            workers=12,
            ram_ramp_threshold_percent=85.0,
        )
    )
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
            workers=12,
            ram_ramp_threshold_percent=85.0,
            resume=True,
        )
    )

    assert captured[0]["workers"] == 12
    assert captured[0]["initial_workers"] == 1
    assert captured[0]["initial_worker_ramp_delay_seconds"] == 20.0
    assert captured[0]["per_worker_ramp_delay_seconds"] == 5.0
    assert captured[1]["workers"] == 12
    assert captured[1]["initial_workers"] == 1
    assert captured[1]["initial_worker_ramp_delay_seconds"] == 20.0
    assert captured[1]["per_worker_ramp_delay_seconds"] == 5.0
    epoch_start = json.loads((root / "epochs" / "epoch_0002" / "status" / "epoch_start.json").read_text(encoding="utf-8"))
    assert epoch_start["requested_workers"] == 12
    assert epoch_start["initial_epoch_workers"] == 1
    assert epoch_start["ram_ramp_threshold_percent"] == 85.0
    assert epoch_start["initial_worker_ramp_delay_seconds"] == 20.0
    assert epoch_start["per_worker_ramp_delay_seconds"] == 5.0
    assert "ram_used_percent" in epoch_start["ram_snapshot_at_epoch_start"]


def test_continuous_run_uses_configured_initial_workers(tmp_path: Path, monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_run_sampling(config):
        captured.append(
            {
                "workers": int(config.workers),
                "initial_workers": int(config.initial_workers or 0),
            }
        )
        return _write_sampling_fixture(
            Path(config.output_dir),
            global_step_offset=int(config.global_step_offset),
            stable_support=25,
            worker_execution={
                "requested_workers": int(config.workers),
                "initial_workers": int(config.initial_workers or 0),
                "peak_workers": int(config.workers),
                "worker_ramp_enabled": bool(config.enable_worker_ramp),
                "ram_ramp_threshold_percent": float(config.ram_ramp_threshold_percent),
                "initial_worker_ramp_delay_seconds": float(config.initial_worker_ramp_delay_seconds),
                "per_worker_ramp_delay_seconds": float(config.per_worker_ramp_delay_seconds),
                "ram_used_percent_at_start": 44.0,
                "ramp_event_count": 0,
                "ramp_events": [],
            },
        )

    monkeypatch.setattr(continuous_research, "run_interaction_sampling_v05c", fake_run_sampling)
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
            workers=12,
            initial_workers=3,
            ram_ramp_threshold_percent=85.0,
        )
    )

    assert captured == [{"workers": 12, "initial_workers": 3}]
    epoch_start = json.loads((root / "epochs" / "epoch_0001" / "status" / "epoch_start.json").read_text(encoding="utf-8"))
    assert epoch_start["requested_workers"] == 12
    assert epoch_start["initial_epoch_workers"] == 3


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

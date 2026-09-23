from __future__ import annotations

import json
import os
import re
import sys

import pytest

from v9.cli import build_parser, run_continuous


def test_actor_output_suppression_preserves_live_python_streams(capfd) -> None:
    from v9.runtime.multiprocess import _silence_actor_output

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with _silence_actor_output():
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        assert not sys.stdout.closed
        assert not sys.stderr.closed
        os.write(sys.stdout.fileno(), b"actor stdout must be silent\n")
        os.write(sys.stderr.fileno(), b"actor stderr must be silent\n")

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert not sys.stdout.closed
    assert not sys.stderr.closed
    captured = capfd.readouterr()
    assert "actor stdout must be silent" not in captured.out
    assert "actor stderr must be silent" not in captured.err


def test_process_topology_is_reported(tmp_path) -> None:
    root = tmp_path / "mp"
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "2",
        "--no-peers",
        "--no-dashboard",
    ])
    assert run_continuous(args) == 0
    summary = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]["telemetry_diagnostics"]
    assert metrics["actor_processes"] == 2
    assert metrics["stage_worker_processes"] == 2
    assert metrics["shard_worker_processes"] == 2
    assert metrics["ingest_worker_processes"] == 4
    assert metrics["derivation_worker_processes"] == 4
    assert metrics["multiprocess_transitions_published"] == 1200
    assert metrics["sampling_backlog"] == 0
    assert metrics["coordinator_action_requests"] == 0
    assert metrics["policy_snapshot_generation"] >= 0


def test_progress_is_written_to_stdout(tmp_path, capsys) -> None:
    root = tmp_path / "progress"
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "1",
        "--progress-interval-seconds", "60",
        "--no-peers",
        "--no-dashboard",
    ])
    assert run_continuous(args) == 0
    output = capsys.readouterr().out
    assert re.search(r"\[\d{2}:\d{2}\]\s+100\.0%", output)
    assert "sampled=1200/1200" in output
    assert "ingested=1200" in output
    assert "backlog=0" in output
    assert "M0=" not in output
    assert "M7=" not in output


def test_restored_parallel_run_appends_unique_m0(tmp_path) -> None:
    root = tmp_path / "append"
    argv = [
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "1",
        "--ingest-workers", "2",
        "--derivation-workers", "2",
        "--no-peers",
        "--no-dashboard",
    ]
    assert run_continuous(build_parser().parse_args(argv)) == 0
    first = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    first_m0 = first["metrics"]["memory_levels"]["M0"]

    assert run_continuous(build_parser().parse_args(argv)) == 0
    second = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    second_m0 = second["metrics"]["memory_levels"]["M0"]

    assert second_m0 > first_m0


def test_epochs_repeat_sampling_and_training(tmp_path) -> None:
    root = tmp_path / "epochs"
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--epochs", "2",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "1",
        "--ingest-workers", "2",
        "--derivation-workers", "2",
        "--no-peers",
        "--no-dashboard",
    ])
    assert run_continuous(args) == 0

    summary = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    assert len(summary["epochs"]) == 2
    assert len(summary["actors"]) == 24
    assert summary["metrics"]["memory_levels"]["M0"] >= 24
    assert summary["epochs"][0]["training"]["status"].startswith(("SKIPPED_", "PROMOTED", "REJECTED", "TESTING_"))
    assert summary["epochs"][1]["training"]["status"].startswith(("SKIPPED_", "PROMOTED", "REJECTED", "TESTING_"))
    assert summary["epochs"][0]["performance"]["optimized_sampling_path"] is True
    assert summary["epochs"][1]["performance"]["optimized_sampling_path"] is True
    assert summary["epochs"][1]["performance"]["coordinator_action_requests"] == 0


def test_behavioral_success_is_game_independent() -> None:
    from v9.runtime.epoch_runner import _scenario_success
    from v9.runtime.parallel_memory_coordinator import ProcessActorResult

    rows = [
        ProcessActorResult(1, "game_a", 10, 8, 1, 10, 0, 0, 8, 1, 1, 3),
        ProcessActorResult(2, "game_b", 10, 2, 3, 4, 0, 0, 2, 1, 1, 1),
    ]
    rates, macro = _scenario_success(rows)

    assert rates == {"game_a": 0.8, "game_b": 0.5}
    assert macro == 0.65


def test_game_level_metrics_report_wins_and_best_level() -> None:
    from v9.runtime.epoch_runner import _game_level_metrics
    from v9.runtime.parallel_memory_coordinator import ProcessActorResult

    rows = [
        ProcessActorResult(1, "arc_a", 20, 3, 0, 1, 0, 0, 1, 0, 0, 3),
        ProcessActorResult(2, "arc_b", 20, 2, 1, 1, 0, 0, 0, 1, 0, 2),
    ]
    metrics = _game_level_metrics(rows)
    assert metrics["current_run_solved_games"] == 1
    assert metrics["current_run_total_games"] == 2
    assert metrics["current_run_levels_completed"] == 5
    assert metrics["current_run_best_level_by_game"] == {"arc_a": 3, "arc_b": 2}


def test_actor_policy_snapshot_is_picklable(tmp_path) -> None:
    import pickle
    from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig

    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    snapshot = runtime.actor_policy_snapshot()
    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored.generation == snapshot.generation
    assert restored.normalized_action_supports == snapshot.normalized_action_supports


def _prepared_transition(sequence: int):
    from v9.runtime.memory_pipeline import IngestionTask, prepare_ingestion
    from v9.runtime.multiprocess import EncodedTransition

    transition = EncodedTransition(
        actor_id=1,
        producer_sequence=sequence,
        environment_identity=("synthetic", "syn_move", "default", "instance-1"),
        episode_id=1,
        global_step=sequence - 1,
        before_signature=1,
        action_id=0,
        after_signature=2,
        available_actions_after=2,
        primary_valence=0,
        observation_schema_id=1,
        action_schema_id=2,
        available_action_set_signature=3,
        boundary_scope="NONE",
        task_success=False,
        task_failure=False,
        task_truncated=False,
        level_index=0,
        levels_completed=0,
        symbols=(),
        symbols_only=False,
        curriculum_step="step1",
        game_scenario="syn_move",
    )
    return prepare_ingestion(IngestionTask(sequence, sequence, transition))


def test_parallel_sampling_publishes_admitted_raw_graph_inline(tmp_path) -> None:
    from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig

    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    runtime.apply_prepared_ingestion(_prepared_transition(1))

    # One-off ordinary novelty advances normalized evidence but is not retained
    # as a concrete episode. Recurrence promotes a representative.
    assert runtime.metrics()["memory_levels"]["M0"] == 0
    runtime.apply_prepared_ingestion(_prepared_transition(2))

    assert runtime.metrics()["memory_levels"]["M0"] == 1
    assert runtime.graph.memory_count() >= 3
    assert runtime._deferred_base_nodes == {}
    before_flush = runtime.graph.memory_count()

    runtime.flush_deferred_memory_updates()

    assert runtime.graph.memory_count() == before_flush
    diagnostics = runtime.unified_telemetry.diagnostic_metrics()
    assert diagnostics["inline_lowlevel_publication_rows"] >= 3
    assert diagnostics["deferred_base_nodes"] == 0


def test_canonical_batch_publishes_lowlevel_inline_and_defers_only_support(tmp_path) -> None:
    from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
    from v9.runtime.canonical_commit import apply_canonical_commit_batch
    from v9.runtime.memory_pipeline import build_commit_plan

    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    first = build_commit_plan(_prepared_transition(1))
    second = build_commit_plan(_prepared_transition(2))

    apply_canonical_commit_batch(runtime, (first, second))

    assert runtime._deferred_base_nodes == {}
    assert runtime.graph.memory_count() >= 3
    assert runtime._m1n_dirty
    diagnostics = runtime.unified_telemetry.diagnostic_metrics()
    assert diagnostics["inline_lowlevel_publication_rows"] >= 3
    assert diagnostics["dirty_m1n_supports"] >= 1


def test_child_queue_flush_closes_and_joins_feeder() -> None:
    from v9.runtime.multiprocess import _flush_child_queue

    events: list[str] = []

    class ProbeQueue:
        def close(self) -> None:
            events.append("close")

        def join_thread(self) -> None:
            events.append("join_thread")

    _flush_child_queue(ProbeQueue())
    assert events == ["close", "join_thread"]


def test_action_set_signature_accepts_and_canonicalizes_action_tuples() -> None:
    from v9.runtime.multiprocess import _action_set_signature

    signature = _action_set_signature(7, (3, 1, 3))

    assert isinstance(signature, int)
    assert signature == _action_set_signature(7, (1, 3))
    assert signature != _action_set_signature(7, (1, 2))


def test_clean_actor_exit_without_done_becomes_bounded_protocol_error() -> None:
    from v9.runtime.parallel_memory_coordinator import _reconcile_actor_liveness

    class CleanExitProcess:
        exitcode = 0
        name = "v9-actor-7"

    active = {7: (0, CleanExitProcess())}
    first_seen: dict[int, float] = {}

    assert _reconcile_actor_liveness(active, first_seen, now=10.0, grace_seconds=1.0) == 1
    assert first_seen == {7: 10.0}
    with pytest.raises(RuntimeError, match="no ActorDone was received"):
        _reconcile_actor_liveness(active, first_seen, now=11.01, grace_seconds=1.0)

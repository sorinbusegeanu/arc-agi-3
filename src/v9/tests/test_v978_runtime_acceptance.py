from __future__ import annotations

from v9 import ContinuousMemoryRuntime
from v9.runtime import RuntimeConfig
from v9.runtime.memory_pipeline import IngestionTask, prepare_ingestion
from v9.runtime.multiprocess import EncodedTransition


def _transition(*, symbols: tuple[object, ...], sequence: int, before: int = 1, after: int = 2) -> EncodedTransition:
    return EncodedTransition(
        actor_id=1,
        producer_sequence=sequence,
        global_step=sequence,
        environment_identity=("synthetic", "symbolic", "test", "instance"),
        episode_id=1,
        observation_schema_id=1,
        before_signature=before,
        action_id=0,
        after_signature=after,
        available_actions_after=2,
        primary_valence=1,
        symbols=symbols,
        curriculum_step=None,
        game_scenario="synthetic",
        boundary_scope="TASK",
        task_success=True,
        levels_completed=1,
    )


def test_symbolic_runtime_smoke_builds_m0_m1n_and_cross_modal_evidence(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path / "symbolic", restore=False, enable_snapshots=False))
    prepared = prepare_ingestion(IngestionTask(1, 10, _transition(symbols=("opaque", "opaque"), sequence=1)))
    signatures = runtime.apply_prepared_ingestion(prepared)
    runtime.flush_deferred_memory_updates()
    metrics = runtime.metrics()
    assert signatures
    assert metrics["symbol_occurrences"] == 2
    assert metrics["symbolic_M0_count"] == 2
    assert metrics["symbolic_M1N_count"] > 0
    assert metrics["cross_modal_correspondences"] > 0
    relation_names = {str(payload.get("symbol_relation")) for payload in runtime.graph.payloads.values() if payload.get("symbol_relation")}
    assert "SYMBOL_RECURS_WITHIN_WINDOW" in relation_names
    assert "SYMBOL_PRECEDES_ACTION" in relation_names
    assert "SYMBOL_COINCIDENT_WITH_OUTCOME" in relation_names


def test_non_symbolic_regression_does_not_create_symbol_grounding(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path / "world", restore=False, enable_snapshots=False))
    prepared = prepare_ingestion(IngestionTask(1, 10, _transition(symbols=(), sequence=1)))
    signatures = runtime.apply_prepared_ingestion(prepared)
    runtime.flush_deferred_memory_updates()
    metrics = runtime.metrics()
    assert signatures
    assert metrics["symbol_occurrences"] == 0
    assert metrics["symbolic_M0_count"] == 0
    assert metrics["symbolic_M1N_count"] == 0

from __future__ import annotations

from v9 import ContinuousMemoryRuntime
from v9.runtime import RuntimeConfig, ScientificConfig
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
    symbols = (
        {"token": "opaque", "phase": "BEFORE_ACTION", "micro_step": 0, "macro_step": 1},
        {"token": "opaque", "phase": "AFTER_OUTCOME", "micro_step": 1, "macro_step": 1},
    )
    prepared = prepare_ingestion(IngestionTask(1, 10, _transition(symbols=symbols, sequence=1)))
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
    assert "CROSS_MODAL_CORRESPONDENCE" in relation_names
    assert runtime.symbol_occurrences(temporal_phase="BEFORE_ACTION", nearby_action_id=0)
    assert runtime.symbol_occurrences(temporal_phase="AFTER_OUTCOME", progress=True, outcome=1)


def test_symbol_ingestion_enforces_budget_dedup_and_window(tmp_path) -> None:
    scientific = ScientificConfig(
        symbol_budget_per_window=3,
        max_symbol_facts_per_window=2,
        symbol_deduplication_policy="token",
        symbol_window_time_span=2,
    )
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path / "bounded", restore=False, enable_snapshots=False, scientific=scientific))
    transition = _transition(
        symbols=(
            {"token": "x", "phase": "BEFORE_ACTION", "macro_step": 1, "micro_step": 0},
            {"token": "x", "phase": "BEFORE_ACTION", "macro_step": 1, "micro_step": 1},
            {"token": "y", "phase": "BEFORE_ACTION", "macro_step": 99, "micro_step": 2},
        ),
        sequence=1,
    )
    prepared = prepare_ingestion(
        IngestionTask(
            1,
            10,
            transition,
            min(scientific.symbol_budget_per_window, scientific.max_symbol_facts_per_window),
            scientific.symbol_payload_bytes,
            scientific.max_cross_modal_facts_per_macro_event,
            scientific.symbol_deduplication_policy,
            scientific.symbol_window_time_span,
            scientific.symbol_codec_name,
            scientific.symbol_codec_version,
        )
    )
    assert len(prepared.symbol_occurrences) == 1


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

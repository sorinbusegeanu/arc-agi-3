from pathlib import Path
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.config import ScientificConfig
from v9.runtime.memory_pipeline import IngestionTask, prepare_ingestion
from v9.runtime.multiprocess import EncodedTransition
from v9.runtime.parallel_memory_coordinator import _adaptive_canonical_batch_size

def _prepared(count: int = 320):
    rows = []
    watermark = 0
    for index in range(count):
        symbols = (f"token-{index % 3}",) if index % 19 == 0 else ()
        transition = EncodedTransition(actor_id=1, producer_sequence=index + 1, global_step=index, environment_identity=("synthetic", "batch", "default", "seed=0"), episode_id=7, observation_schema_id=11, before_signature=index % 5, action_id=index % 3, after_signature=(index + 1) % 5, available_actions_after=4, primary_valence=1 if index % 101 == 0 else 0, symbols=symbols, curriculum_step="broad", game_scenario="batch")
        watermark += 1
        rows.append(prepare_ingestion(IngestionTask(index + 1, watermark, transition)))
        watermark += len(symbols)
    return tuple(rows)

def _stage_semantics(runtime):
    state = runtime.stage_tracker.state_dict()
    return {
        **state,
        "history": [
            {key: value for key, value in row.items() if key != "evidence_watermark"}
            for row in state["history"]
        ],
    }

def test_true_batch_matches_ordered_single_event_state(tmp_path: Path, monkeypatch) -> None:
    rows = _prepared()
    scientific = ScientificConfig(concrete_admission_enabled=False)
    single = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path / "single",
            restore=False,
            enable_snapshots=False,
            scientific=scientific,
        )
    )
    batched = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path / "batched",
            restore=False,
            enable_snapshots=False,
            scientific=scientific,
        )
    )
    for row in rows:
        single.apply_prepared_ingestion(row)
    def no_single_event_fallback(_row):
        raise AssertionError("batch path delegated to per-event apply")
    monkeypatch.setattr(batched, "apply_prepared_ingestion", no_single_event_fallback)
    batch_signatures = batched.apply_prepared_ingestion_batch(rows)
    assert len(batch_signatures) == len(rows)
    single.flush_deferred_memory_updates()
    batched.flush_deferred_memory_updates()
    assert batched._m1n_supports == single._m1n_supports
    assert batched._actor_action_supports == single._actor_action_supports
    assert _stage_semantics(batched) == _stage_semantics(single)
    assert batched.isf.state_dict() == single.isf.state_dict()
    assert batched.timeline.state_dict() == single.timeline.state_dict()
    assert batched.environments.state_dict() == single.environments.state_dict()
    assert batched.graph.state_dict() == single.graph.state_dict()
    assert batched.telemetry == single.telemetry

def test_adaptive_canonical_batch_scales_with_backlog() -> None:
    assert _adaptive_canonical_batch_size(256, 10_000) == 512
    assert _adaptive_canonical_batch_size(512, 10_000) == 1024
    assert _adaptive_canonical_batch_size(4096, 100_000) == 4096
    assert _adaptive_canonical_batch_size(1024, 10) == 512
    assert _adaptive_canonical_batch_size(256, 0) == 256

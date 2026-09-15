from pathlib import Path

from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.canonical_commit import apply_canonical_commit_batch
from v9.runtime.memory_pipeline import IngestionTask, prepare_ingestion
from v9.runtime.memory_pipeline_v2 import build_commit_plan
from v9.runtime.multiprocess import EncodedTransition


def _rows(count: int = 320):
    prepared = []
    watermark = 0
    for index in range(count):
        symbols = (f"token-{index % 3}",) if index % 19 == 0 else ()
        transition = EncodedTransition(
            actor_id=1,
            producer_sequence=index + 1,
            global_step=index,
            environment_identity=("synthetic", "batch-v2", "default", "seed=0"),
            episode_id=7,
            observation_schema_id=11,
            before_signature=index % 5,
            action_id=index % 3,
            after_signature=(index + 1) % 5,
            available_actions_after=4,
            primary_valence=1 if index % 101 == 0 else 0,
            symbols=symbols,
            curriculum_step="broad",
            game_scenario="batch-v2",
        )
        watermark += 1
        row = prepare_ingestion(IngestionTask(index + 1, watermark, transition))
        prepared.append(row)
        watermark += len(symbols)
    return tuple(prepared)


def test_fast_canonical_commit_matches_reference_batch(tmp_path: Path) -> None:
    rows = _rows()
    reference = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path / "reference", restore=False, enable_snapshots=False)
    )
    fast = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path / "fast", restore=False, enable_snapshots=False)
    )

    expected_signatures = reference.apply_prepared_ingestion_batch(rows)
    reference.record_curriculum_events_batch(rows)
    result = apply_canonical_commit_batch(fast, tuple(build_commit_plan(row) for row in rows))

    assert result.signature_rows == expected_signatures
    assert fast._m1n_supports == reference._m1n_supports
    assert fast._actor_action_supports == reference._actor_action_supports
    assert fast.stage_tracker.state_dict() == reference.stage_tracker.state_dict()
    assert fast.isf.state_dict() == reference.isf.state_dict()
    assert fast.timeline.state_dict() == reference.timeline.state_dict()
    assert fast.environments.state_dict() == reference.environments.state_dict()
    assert fast.unified_telemetry.curriculum_counts == reference.unified_telemetry.curriculum_counts

    reference.flush_deferred_memory_updates()
    fast.flush_deferred_memory_updates()
    # Evidence confidence is a post-ingestion annotation maintained by the
    # continuous runtime. Normalize it before comparing canonical graph identity.
    for runtime in (reference, fast):
        for payload in runtime.graph.payloads.values():
            if payload.get("evidence_confidence") == 1.0:
                payload.pop("evidence_confidence", None)
    assert fast.graph.state_dict() == reference.graph.state_dict()
    assert fast.telemetry == reference.telemetry

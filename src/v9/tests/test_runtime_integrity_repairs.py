from pathlib import Path

from v9 import ContinuousMemoryRuntime
from v9.cognition.grounding import GroundingEvidence, GroundingRegistry
from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.runtime import RuntimeConfig
from v9.runtime.memory_pipeline import IngestionTask, prepare_ingestion
from v9.runtime.multiprocess import EncodedTransition
from v9.runtime.publication import node_ref


def _runtime(root: Path) -> ContinuousMemoryRuntime:
    return ContinuousMemoryRuntime(
        RuntimeConfig.from_path(root, restore=False, enable_snapshots=False)
    )


def test_matched_experiment_state_is_disk_backed_and_clears_branch_caches(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "matched")
    try:
        captured = runtime.capture_experiment_state()
        assert captured.path.exists()
        fake = MemoryUid(123, 456)
        runtime._m2[fake] = object()
        runtime._m7[fake] = object()
        runtime._transfer_trials[fake] = [{"branch": "on"}]
        runtime.restore_experiment_state(captured)
        assert fake not in runtime._m2
        assert fake not in runtime._m7
        assert fake not in runtime._transfer_trials
    finally:
        runtime.close()


def test_viability_observed_once_at_canonical_apply(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "viability")
    try:
        transition = EncodedTransition(
            actor_id=1,
            producer_sequence=1,
            global_step=0,
            environment_identity=("synthetic", "integrity", "default", "seed=1"),
            episode_id=1,
            observation_schema_id=1,
            before_signature=10,
            action_id=0,
            after_signature=11,
            available_actions_after=2,
            game_scenario="integrity",
        )
        prepared = prepare_ingestion(IngestionTask(1, 1, transition))
        runtime.apply_prepared_ingestion_batch((prepared,))
        profiles = tuple(runtime._environment_viability.profiles.values())
        assert len(profiles) == 1
        assert profiles[0].observations == 1
    finally:
        runtime.close()


def test_environment_confidence_update_bumps_authoritative_version(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "confidence")
    try:
        uid = MemoryUid(10, 20)
        node = CanonicalNode(uid, MemoryLevel.M0, MemoryType.EPISODE, (1,), 1)
        runtime.graph.nodes[uid] = node
        runtime.graph.payloads[uid] = {"environment_instance_id": 77, "evidence_confidence": 1.0}
        runtime.graph._uids_by_level[MemoryLevel.M0].add(uid)
        before_version = runtime.graph.versions.get(node_ref(uid))
        before_generation = runtime.graph.generation
        changed = runtime.apply_environment_evidence_confidence({77: 0.25})
        assert changed == 1
        assert runtime.graph.payloads[uid]["evidence_confidence"] == 0.25
        assert runtime.graph.versions.get(node_ref(uid)) == before_version + 1
        assert runtime.graph.generation == before_generation + 1
    finally:
        runtime.close()


def test_low_level_delete_callback_releases_grounding_payload_reference(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "delete-index")
    try:
        uid = MemoryUid(22, 33)
        runtime._grounding_action_payload_by_low[int(uid.lo)] = (uid, {"action_id": 1})
        runtime.on_low_level_deleted((uid,))
        assert int(uid.lo) not in runtime._grounding_action_payload_by_low
    finally:
        runtime.close()


def test_grounding_validation_ids_are_bounded() -> None:
    registry = GroundingRegistry()
    for index in range(80):
        registry.observe(
            GroundingEvidence(
                1,
                2,
                3,
                4,
                5,
                index,
                recurrent_symbol=True,
                validation_trial_id=f"trial-{index}",
            )
        )
    row = next(iter(registry.states.values()))
    assert len(row.validation_trial_ids) == 64
    assert row.validation_trial_ids[0] == "trial-16"
    assert row.validation_trial_ids[-1] == "trial-79"


def test_snapshot_streams_graph_without_calling_full_graph_state_dict(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "snapshot")
    try:
        runtime.graph.__dict__["state_dict"] = lambda: (_ for _ in ()).throw(AssertionError("full graph materialized"))
        result = runtime.snapshot()
        assert (result.path / "manifest.json").exists()
        assert (result.path / "COMPLETE").exists()
        runtime.graph.__dict__.pop("state_dict", None)
    finally:
        runtime.close()

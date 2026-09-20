from __future__ import annotations

from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.config import RuntimeConfig, ScientificConfig
from v9.runtime.epoch_inference_view import EpochInferenceView
from v9.runtime.policy_projection import build_policy_projection
from v9.runtime.scientific_modes import ScientificVisibilityMode
from v9.research.experiment_manifest import ExperimentManifest, InteractionOpportunityManifest
from v9.curriculum import EnvironmentSpec
from v9.runtime.parallel_memory_coordinator import run_parallel_memory_jobs


def _projection():
    return build_policy_projection(
        ActorPolicySnapshot.build(
            generation=0,
            normalized_action_supports={1: 2.0},
            hgt_action_scores={},
            model_version="untrained",
        )
    )


def test_epoch_view_pins_complete_identity() -> None:
    store = CanonicalStore()
    with EpochInferenceView(
        store=store,
        experiment_manifest_id="a" * 64,
        sampling_epoch_id=4,
        policy_projection=_projection(),
        model_version="untrained",
        stage_state={"stage": 1},
        normalization_state={"generation": 2},
        graph_schema_version=1,
        feature_schema_version=1,
        scientific_config_id="b" * 64,
    ) as view:
        assert view.identity.canonical_handle_checksum == store.current_handle.checksum
        assert len(view.identity.checksum) == 64


def test_runtime_builds_matched_epoch_view_from_one_complete_cut(tmp_path) -> None:
    from v9 import ContinuousMemoryRuntime

    scientific = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    manifest = ExperimentManifest(
        scientific.config_id,
        InteractionOpportunityManifest(()),
        visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
    )
    manifest_path = manifest.write(tmp_path / "experiment.json")
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path / "run",
            restore=False,
            enable_snapshots=False,
            enable_peers=False,
            enable_canonical_durability=True,
            experiment_manifest=manifest_path,
            scientific=scientific,
        )
    )
    try:
        with runtime.create_epoch_inference_view(sampling_epoch_id=3) as view:
            assert view.identity.experiment_manifest_id == manifest.manifest_id.value
            assert view.identity.sampling_epoch_id == 3
            assert view.canonical_handle is runtime.canonical_state_handle
            assert view.policy_projection.snapshot.model_version == runtime.unified_telemetry.model_version
    finally:
        runtime.close(normal=False)


def test_matched_actors_report_the_exact_bound_epoch_view(tmp_path) -> None:
    from v9 import ContinuousMemoryRuntime

    scientific = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    manifest_path = ExperimentManifest(
        scientific.config_id,
        InteractionOpportunityManifest(()),
        visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
    ).write(tmp_path / "experiment.json")
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path / "run",
            restore=False,
            enable_snapshots=False,
            enable_peers=False,
            enable_canonical_durability=True,
            experiment_manifest=manifest_path,
            scientific=scientific,
        )
    )
    runtime.start()
    try:
        with runtime.create_epoch_inference_view(sampling_epoch_id=1) as view:
            rows = run_parallel_memory_jobs(
                runtime,
                [(1, EnvironmentSpec("synthetic_symbolic", "synthetic-symbolic"), 4, 7)],
                actor_limit=2,
                stage_workers=1,
                shards=1,
                queue_capacity=64,
                epsilon=0.0,
                env_root=None,
                alfred_backend_factory=None,
                start_method=None,
                progress_interval_seconds=60.0,
                ingest_workers=1,
                derivation_workers=1,
                ingest_queue_capacity=64,
                derivation_queue_capacity=64,
                publication_queue_capacity=64,
                allow_policy_refresh=False,
                bound_epoch_view=view,
            )
            assert {row.epoch_inference_view_id for row in rows} == {view.identity.checksum}
            assert {row.policy_refreshes for row in rows} == {0}
    finally:
        runtime.close(normal=False)

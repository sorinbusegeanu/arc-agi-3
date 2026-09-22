from __future__ import annotations

from pathlib import Path

from v9 import ContinuousMemoryRuntime
from v9.memory import MemoryLevel, MemoryType
from v9.runtime.canonical_commit import apply_canonical_commit_batch
from v9.runtime.config import RuntimeConfig, ScientificConfig
from v9.runtime.memory_admission import should_retain_concrete
from v9.runtime.memory_pipeline import IngestionTask, build_commit_plan, prepare_ingestion
from v9.runtime.multiprocess import EncodedTransition


def test_admission_keeps_bootstrap_boundaries_surprise_and_sparse_milestones() -> None:
    scientific = ScientificConfig(
        concrete_admission_representatives_per_signature=4,
        concrete_admission_prediction_error_threshold=0.5,
        concrete_admission_future_option_threshold=1.0,
    )

    first = should_retain_concrete(
        scientific,
        prior_support=0,
        retained_representatives=0,
        novel_context=True,
    )
    assert not first.retain
    assert first.reason == "unconfirmed_novelty"
    assert should_retain_concrete(
        scientific,
        prior_support=1,
        retained_representatives=0,
        novel_context=True,
    ).reason == "recurrent_novel_context"
    assert should_retain_concrete(
        scientific,
        prior_support=3,
        retained_representatives=2,
    ).retain
    assert not should_retain_concrete(
        scientific,
        prior_support=4,
        retained_representatives=4,
    ).retain
    assert should_retain_concrete(
        scientific,
        prior_support=7,
        retained_representatives=4,
    ).reason == "support_milestone"

    class Boundary:
        task_success = True
        task_failure = False
        task_truncated = False
        primary_valence = 0

    assert should_retain_concrete(
        scientific, prior_support=20, context=Boundary()
    ).reason == "boundary"
    assert should_retain_concrete(
        scientific,
        prior_support=20,
        isf_static=(0.0, 0.0, 0.75, 0.0, 0.0),
    ).reason == "prediction_error"


def test_redundant_interactions_advance_support_without_materializing_every_pair(
    tmp_path: Path,
) -> None:
    scientific = ScientificConfig(
        concrete_admission_representatives_per_signature=4,
        concrete_admission_prediction_error_threshold=1_000_000.0,
        concrete_admission_future_option_threshold=1_000_000.0,
        resident_m0_limit=50_000,
        resident_m1_grounded_limit=50_000,
    )
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig(
            tmp_path / "run",
            restore=False,
            enable_snapshots=False,
            scientific=scientific,
        )
    )
    try:
        plans = []
        signature = None
        last_grounding_uid = None
        for index in range(15):
            transition = EncodedTransition(
                actor_id=1,
                producer_sequence=index + 1,
                global_step=index,
                environment_identity=("synthetic", "admission", "default", "seed=0"),
                episode_id=1,
                observation_schema_id=1,
                before_signature=10,
                action_id=2,
                after_signature=11,
                available_actions_after=4,
                primary_valence=0,
                symbols=(),
                curriculum_step="broad",
                game_scenario="admission",
            )
            prepared = prepare_ingestion(IngestionTask(index + 1, index + 1, transition))
            plan = build_commit_plan(prepared)
            signature = int(plan.relation.structural_signature)
            last_grounding_uid = plan.interaction_grounding.uid
            plans.append(plan)

        apply_canonical_commit_batch(runtime, tuple(plans))
        runtime.flush_deferred_memory_updates()

        assert signature is not None
        assert runtime.signature_support(signature) == 15
        assert float(runtime._v978_m1n_evidence[signature]["support"]) == 15.0
        assert float(runtime.graph.payloads[plans[-1].relation.uid]["support"]) == 15.0
        m0 = sum(
            1
            for node in runtime.graph.nodes.values()
            if node.level is MemoryLevel.M0 and node.memory_type is MemoryType.EPISODE
        )
        m1g = sum(
            1
            for node in runtime.graph.nodes.values()
            if node.level is MemoryLevel.M1
            and node.memory_type is MemoryType.GROUNDED_CONTINGENCY
        )
        assert m0 == 5
        assert m1g == 1
        assert runtime.telemetry["concrete_admission_retained_events"] == 5
        assert runtime.telemetry["concrete_admission_skipped_events"] == 10
        # First observation avoids both its unique M0 and not-yet-materialized
        # canonical M1G; later skipped observations avoid only their unique M0.
        assert runtime.telemetry["concrete_nodes_avoided"] == 11
        latest = runtime._latest_interaction_grounding[
            (plans[-1].interaction_grounding.environment_instance_id, 1)
        ]
        assert latest.uid == last_grounding_uid
    finally:
        runtime.close(normal=False)



def test_prepare_ingestion_uses_true_future_option_delta() -> None:
    transition = EncodedTransition(
        actor_id=1,
        producer_sequence=1,
        global_step=0,
        environment_identity=("synthetic", "options", "default", "seed=0"),
        episode_id=1,
        observation_schema_id=1,
        before_signature=10,
        action_id=2,
        after_signature=11,
        available_actions_after=8,
        primary_valence=0,
        symbols=(),
        curriculum_step="broad",
        game_scenario="options",
        future_option_delta=-2.0,
    )
    prepared = prepare_ingestion(IngestionTask(1, 1, transition))
    assert prepared.event is not None
    assert prepared.event.experience.future_option_delta == -2.0

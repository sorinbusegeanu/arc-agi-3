from __future__ import annotations

from types import SimpleNamespace

from v7.derivation.scientific import ScientificDerivationKernels, TYPE_CONCEPT
from v7.environment.ablation import (
    CognitionAblation,
    ablation_names,
    parse_ablation_spec,
)
from v7.environment.cognition import LocalCognitionOverlay
from v7.evaluation.cognition_metrics import CognitionMetricsAccumulator
from v7.memory.concept_validation import (
    ConceptValidationStatus,
    EmpiricalConceptValidator,
)
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import NodeMutation
from v7.memory.writer import CanonicalMemoryWriter


def test_m2_identity_is_action_independent() -> None:
    members = (MemoryId(1), MemoryId(2))
    left = ScientificDerivationKernels.m2_family(
        action_id=1,
        member_ids=members,
        outcome_class=99,
    )
    right = ScientificDerivationKernels.m2_family(
        action_id=7,
        member_ids=members,
        outcome_class=99,
    )
    assert left.key == right.key
    assert left.key.parts == (99,)


def test_context_lattice_contains_behavioral_structural_combined_and_exact() -> None:
    overlay = LocalCognitionOverlay()
    first = overlay.build_context(structural_signature=11, exact_signature=22)
    assert len(first.signatures) == 5
    overlay.record_step(
        contexts=first.signatures,
        next_contexts=(),
        action_id=3,
        outcome_signature=44,
        terminal_polarity=0,
        prediction_error=0.0,
        future_option_delta=0.0,
        changed=True,
    )
    second = overlay.build_context(structural_signature=11, exact_signature=23)
    assert second.signatures[0] == first.signatures[0]
    assert second.signatures[1] != first.signatures[1]
    assert second.signatures[2] == first.signatures[2]
    assert second.signatures[3] != first.signatures[3]
    assert second.signatures[4] != first.signatures[4]


def test_concept_validation_reaches_trusted_only_with_transfer_evidence() -> None:
    writer = CanonicalMemoryWriter()
    concept = MemoryId(1)
    writer.apply_mutation_batch(
        (NodeMutation(concept, MemoryLevel.M4, TYPE_CONCEPT, support_delta=4),)
    )
    writer.commit_generation()
    validator = EmpiricalConceptValidator()
    untested = validator.evaluate(
        writer.published_view,
        transfer_summary={},
        memory_ids=(concept,),
    )[0]
    assert untested.candidate
    assert untested.structural_supported
    assert untested.transfer_candidate
    assert not untested.validated
    assert not untested.trusted

    trusted = validator.evaluate(
        writer.published_view,
        transfer_summary={concept: (4, 3, 0.75)},
        memory_ids=(concept,),
    )[0]
    assert trusted.validated
    assert trusted.trusted
    assert trusted.next_flags & int(ConceptValidationStatus.TRANSFER_VALIDATED)
    assert trusted.next_flags & int(ConceptValidationStatus.TRUSTED)


def test_phase6_ablation_spec_is_deterministic() -> None:
    mask = parse_ablation_spec("planning,strategy,development")
    assert mask == int(
        CognitionAblation.PERSISTENT_PLANNING
        | CognitionAblation.STRATEGY_EXECUTION
        | CognitionAblation.DEVELOPMENTAL_POLICY
    )
    assert ablation_names(mask) == (
        "persistent_planning",
        "strategy_execution",
        "developmental_policy",
    )


def _batch(
    *,
    game: str,
    wins: int,
    level_steps: tuple[int, ...],
    failure_keys: tuple[tuple[int, int], ...] = (),
):
    trajectories = tuple(
        SimpleNamespace(success=True, steps_to_success=steps, level_key=f"level_{index:04d}")
        for index, steps in enumerate(level_steps)
    )
    evidence = [
        SimpleNamespace(
            terminal_polarity=-1,
            context_signature=context,
            action_id=action,
            selection_mode="memory",
            development_stage="PLANNING",
        )
        for context, action in failure_keys
    ]
    return SimpleNamespace(
        game_id=game,
        wins=wins,
        trajectories=trajectories,
        evidence=tuple(evidence),
    )


def test_cognition_metrics_measure_retention_rediscovery_and_failure_repetition() -> None:
    metrics = CognitionMetricsAccumulator()
    metrics.observe_epoch(
        0,
        (
            _batch(
                game="g1",
                wins=1,
                level_steps=(10,),
                failure_keys=((5, 2),),
            ),
            _batch(game="g2", wins=0, level_steps=()),
        ),
    )
    metrics.observe_epoch(
        1,
        (
            _batch(
                game="g1",
                wins=1,
                level_steps=(7,),
                failure_keys=((5, 2),),
            ),
            _batch(game="g2", wins=1, level_steps=(20,)),
        ),
    )
    snapshot = metrics.snapshot(transfer_trials=4, transfer_successes=3)
    assert snapshot.solved_game_count_by_epoch == (1, 2)
    assert snapshot.ever_solved_game_count == 2
    assert snapshot.repeat_solution_rate == 1.0
    assert snapshot.solution_retention_rate == 1.0
    assert snapshot.mean_successful_trajectory_length == 37 / 3
    assert snapshot.mean_steps_to_rediscover_solved_level == 7.0
    assert snapshot.cross_game_transfer_success_rate == 0.75
    assert snapshot.failure_repetition_rate == 0.5
    assert snapshot.selection_modes == {"memory": 2}
    assert snapshot.development_stages == {"PLANNING": 2}

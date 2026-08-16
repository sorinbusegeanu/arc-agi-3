from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v8.arena import NodeRecord
from v8.development import derive_proposal
from v8.dirty import DirtyAccumulator
from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceLedger, EvidenceRecord
from v8.isf import score_memory
from v8.lifecycle import LifecycleController
from v8.model import (
    CognitiveState,
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    ValidationState,
)
from v8.normalized_memory_v086 import is_grounded_contingency, is_normalized_contingency
from v8.outcomes import OutcomeEquivalenceEstimator
from v8.replay import ReplayScheduler
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig
from v8.strategies import StrategyEstimator
from v8.transfer import TransferValidator
from v8.world_model import WorldModelEstimator


def node(
    level: MemoryLevel,
    memory_type: MemoryType,
    key: tuple[int, ...],
    *,
    support: int = 2,
    prediction_error: float = 0.0,
    cognitive_state: int = int(CognitiveState.ACTIVE),
    validation_state: int = int(ValidationState.STRUCTURAL),
    success_sum: float = 0.0,
    cost_sum: float = 0.0,
    attempt_weight: float = 0.0,
) -> NodeRecord:
    return NodeRecord(
        uid=MemoryUid.from_key(level, memory_type, key),
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=int(support),
        significance_sum=1.0,
        prediction_error_sum=float(prediction_error) * max(1, support),
        learning_value_sum=1.0,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=1.0,
        score_weight=float(max(1, support)),
        updated_watermark=10,
        game_mask=0,
        cognitive_state=int(cognitive_state),
        validation_state=int(validation_state),
        success_sum=float(success_sum),
        cost_sum=float(cost_sum),
        attempt_weight=float(attempt_weight),
    )


class DependencyAndCausalityTests(unittest.TestCase):
    def test_dirty_accumulator_preserves_multiplicity(self) -> None:
        dirty: DirtyAccumulator[str] = DirtyAccumulator()
        dirty.add("x", "old", version=1, multiplicity=2)
        dirty.add("x", "new", version=2, multiplicity=3)
        rows = dirty.drain()
        self.assertEqual(len(rows), 1)
        _key, item = rows[0]
        self.assertEqual(item.payload, "new")
        self.assertEqual(item.multiplicity, 5)
        self.assertEqual(item.version, 2)

    def test_coalesced_proposal_scales_support_and_causal_error(self) -> None:
        event = ExperienceEvent(
            EventId.from_producer(1, 1),
            10,
            1,
            1,
            99,
            10,
            7,
            2,
            123,
            456,
            789,
            1.0,
            4,
            0,
            555,
            8,
            0.75,
        )
        proposal = derive_proposal(MemoryLevel.M1, PipelineEvent(event, multiplicity=4))
        self.assertEqual(proposal.support_delta, 4)
        self.assertAlmostEqual(proposal.prediction_error_sum, 3.0)
        self.assertAlmostEqual(proposal.score_weight, 4.0)


class DevelopmentalControlTests(unittest.TestCase):
    def test_isf_drives_bounded_replay_attention(self) -> None:
        high = node(MemoryLevel.M4, MemoryType.CONCEPT, (1, 1), support=8, prediction_error=0.9)
        low = node(MemoryLevel.M4, MemoryType.CONCEPT, (2, 1), support=8)
        self.assertGreater(score_memory(high).total, score_memory(low).total)
        rows = ReplayScheduler(min_priority=0.0).candidates((low, high), budget=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].uid, high.uid)

    def test_world_model_requires_multiple_concept_consequences(self) -> None:
        a = node(MemoryLevel.M5, MemoryType.CONSEQUENCE, (10, 11, 99, 1), support=3)
        b = node(MemoryLevel.M5, MemoryType.CONSEQUENCE, (20, 21, 99, 1), support=4)
        components = WorldModelEstimator().propose((a, b))
        self.assertEqual(len(components), 1)
        self.assertEqual(set(components[0].consequences), {a.uid, b.uid})

    def test_outcome_estimator_proposes_real_coarse_merge(self) -> None:
        a = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3), support=3)
        b = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 4), support=4)
        estimator = OutcomeEquivalenceEstimator()
        classes = estimator.rebuild((a, b))
        self.assertEqual(len(classes), 1)
        revision = estimator.merge_revision(classes[0])
        self.assertIsNotNone(revision)
        self.assertEqual(revision.kind, "MERGE")
        self.assertEqual(set(revision.sources), {a.uid, b.uid})
        self.assertEqual(revision.target, MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2)))

    def test_strategy_estimator_uses_observed_reliability_and_cost(self) -> None:
        row = node(
            MemoryLevel.M7,
            MemoryType.STRATEGY,
            (1, 11, 22, 33),
            support=10,
            success_sum=3,
            cost_sum=20,
            attempt_weight=4,
        )
        evidence = StrategyEstimator().evaluate((row,))[0]
        self.assertAlmostEqual(evidence.reliability, 0.75)
        self.assertAlmostEqual(evidence.mean_cost, 5.0)

    def test_lifecycle_finalizes_unprotected_retirement(self) -> None:
        row = node(
            MemoryLevel.M4,
            MemoryType.CONCEPT,
            (7, 1),
            support=1,
            cognitive_state=int(CognitiveState.RETIRE_PENDING),
            validation_state=int(ValidationState.STRUCTURAL),
        )
        controller = LifecycleController()
        decision = controller.finalize_retirement(row, protected_by_dependencies=False)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.cognitive_state, int(CognitiveState.RETIRED))


class ScientificIntegrityTests(unittest.TestCase):
    def test_evidence_ledger_restart_state_round_trip(self) -> None:
        ledger = EvidenceLedger()
        row = EvidenceRecord.for_uid(
            "heldout",
            MemoryUid(1, 2),
            evidence_kind="transfer_trial_pass",
            watermark=10,
            raw_value=0.5,
            normalized_value=0.5,
            developmental_stage=4,
            validation_state=int(ValidationState.VALIDATED),
            target_game_hash=200,
            provenance_games=(100,),
            causal_intervention="matched_arc_memory_ablation",
            effect_direction=1,
            graph_generation=7,
        )
        ledger.append(row)
        restored = EvidenceLedger()
        restored.load_state(ledger.state_dict())
        self.assertEqual(restored.cut(10), (row,))

    def test_transfer_rejects_non_held_out_target(self) -> None:
        validator = TransferValidator(effect_threshold=0.0)
        uid = MemoryUid(1, 2)
        trial = validator.record_trial(
            uid,
            target_game_hash=100,
            metric_on=1.0,
            metric_off=0.0,
            formation_games=(100, 200),
        )
        self.assertFalse(trial.passed)

    def test_h06_requires_causal_held_out_positive_trial(self) -> None:
        evaluator = ScientificHypothesisEvaluator()
        bad = EvidenceRecord.for_uid(
            "bad",
            MemoryUid(1, 2),
            evidence_kind="transfer_trial_pass",
            watermark=10,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=4,
            validation_state=int(ValidationState.VALIDATED),
            target_game_hash=100,
            provenance_games=(100,),
        )
        status = evaluator.status_map(evaluator.evaluate((bad,)))
        self.assertNotEqual(status["H06"], "VALID")
        good = EvidenceRecord.for_uid(
            "good",
            MemoryUid(1, 2),
            evidence_kind="transfer_trial_pass",
            watermark=11,
            raw_value=0.5,
            normalized_value=0.5,
            developmental_stage=4,
            validation_state=int(ValidationState.VALIDATED),
            target_game_hash=200,
            provenance_games=(100,),
            causal_intervention="matched_arc_memory_ablation",
            effect_direction=1,
        )
        status = evaluator.status_map(evaluator.evaluate((good,)))
        self.assertEqual(status["H06"], "VALID")


class RuntimeDurabilityTests(unittest.TestCase):
    def test_exact_game_provenance_survives_bitmask_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ContinuousMemoryRuntime(
                V8RuntimeConfig.from_path(
                    tmp,
                    shards=1,
                    stage_workers=1,
                    enable_snapshots=False,
                    enable_peers=False,
                    node_capacity_per_shard=1000,
                    edge_capacity_per_shard=3000,
                    action_capacity_per_shard=128,
                )
            )
            runtime.start()
            for producer, game_hash in ((1, 1), (2, 65)):
                runtime.submit(
                    runtime.make_experience(
                        producer_id=producer,
                        producer_sequence=1,
                        source_game_hash=game_hash,
                        global_step=producer,
                        context_signature=10,
                        action_id=1,
                        outcome_signature=100,
                        family_signature=200,
                        carrier_signature=300,
                        next_context_signature=11,
                    )
                )
            runtime.wait_quiescent(timeout=10)
            m1 = runtime.read_view.node_records(level=MemoryLevel.M1)
            grounded = [row for row in m1 if is_grounded_contingency(row)]
            normalized = [row for row in m1 if is_normalized_contingency(row)]
            self.assertEqual(len(grounded), 2)
            self.assertEqual(
                {runtime.read_view.source_games(row.uid) for row in grounded},
                {frozenset({1}), frozenset({65})},
            )
            self.assertTrue(normalized)
            self.assertTrue(
                any(runtime.read_view.source_games(row.uid) == frozenset({1, 65}) for row in normalized)
            )
            runtime.close(normal=True, timeout=10)

    def test_peer_evidence_restores_with_final_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V8RuntimeConfig.from_path(
                tmp,
                shards=1,
                stage_workers=1,
                enable_snapshots=True,
                enable_peers=True,
                snapshot_interval_seconds=3600,
                node_capacity_per_shard=1000,
                edge_capacity_per_shard=3000,
                action_capacity_per_shard=128,
            )
            runtime = ContinuousMemoryRuntime(config)
            runtime.start()
            row = EvidenceRecord.for_uid(
                "persistent",
                MemoryUid(1, 2),
                evidence_kind="preference_probe",
                watermark=0,
                raw_value=1.0,
                normalized_value=1.0,
                developmental_stage=6,
                validation_state=int(ValidationState.STRUCTURAL),
                causal_intervention="clean_choice_probe",
            )
            runtime.peers.ledger.append(row)
            runtime.close(normal=True, timeout=15)

            restored = ContinuousMemoryRuntime(config)
            self.assertTrue(restored.peers.ledger.contains("persistent"))
            restored.close(normal=False)


if __name__ == "__main__":
    unittest.main()

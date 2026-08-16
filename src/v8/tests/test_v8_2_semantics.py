from __future__ import annotations

import unittest

import v8.development as development
import v8.runtime as runtime_module
from v8.arena import EdgeRecord, NodeRecord
from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
)
from v8.observation_contract import ARC_GRID_CONTRACT, changed_cell_distance, grid_transformation
from v8.outcomes import OutcomeEquivalenceEstimator
from v8.promotion import EvidenceGatedPromotionEngine
from v8.pruning import PruningPlanner
from v8.structural_correspondence import StructuralCorrespondenceEstimator
from v8.transfer import TransferValidator


def node(
    level: MemoryLevel,
    memory_type: MemoryType,
    key: tuple[int, ...],
    *,
    support: int = 2,
    significance: float = 0.5,
    learning: float = 0.5,
    transfer: float = 0.0,
    explanatory: float = 0.0,
    future: float = 0.0,
    game_mask: int = 0,
    cognitive_state: int = int(CognitiveState.ACTIVE),
    validation_state: int = int(ValidationState.STRUCTURAL),
) -> NodeRecord:
    uid = MemoryUid.from_key(level, memory_type, key)
    weight = 1.0
    return NodeRecord(
        uid=uid,
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=support,
        significance_sum=significance * weight,
        prediction_error_sum=0.0,
        learning_value_sum=learning * weight,
        transfer_prior_sum=transfer * weight,
        explanatory_sum=explanatory * weight,
        future_option_sum=future * weight,
        score_weight=weight,
        updated_watermark=10,
        game_mask=game_mask,
        cognitive_state=cognitive_state,
        validation_state=validation_state,
    )


def edge(
    source: MemoryUid,
    relation: RelationType,
    target: MemoryUid,
    *,
    score: float = 0.0,
) -> EdgeRecord:
    return EdgeRecord(
        source,
        int(relation),
        target,
        1,
        10,
        score_sum=score,
        score_weight=1.0 if score else 0.0,
    )


class ObservationContractTests(unittest.TestCase):
    def test_arc_contract_declares_structure_and_forbids_semantic_reward(self) -> None:
        self.assertIn("four_neighbor_adjacency", ARC_GRID_CONTRACT.primitive_relations)
        self.assertIn("reward", ARC_GRID_CONTRACT.forbidden_semantic_fields)
        self.assertIn("terminal_value", ARC_GRID_CONTRACT.forbidden_semantic_fields)
        self.assertTrue(ARC_GRID_CONTRACT.digest)

    def test_reference_grid_transformation_is_structural(self) -> None:
        before = ((0, 1), (2, 3))
        after = ((0, 4), (2, 3))
        delta = grid_transformation(before, after)
        self.assertEqual(delta, ((0, 1, 1, 4),))
        self.assertEqual(changed_cell_distance(delta, delta), 0.0)


class DevelopmentalFormationTests(unittest.TestCase):
    def test_live_runtime_raw_pipeline_stops_at_m1(self) -> None:
        self.assertEqual(tuple(stage.level for stage in runtime_module.STAGES), (MemoryLevel.M0, MemoryLevel.M1))
        self.assertEqual(len(development.STAGES), 8)

    def test_one_contingency_cannot_directly_create_family(self) -> None:
        rows = (node(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 2, 77, 11), support=5),)
        candidates = EvidenceGatedPromotionEngine().propose(rows, ())
        self.assertFalse(any(item.level == MemoryLevel.M2 for item in candidates))

    def test_multiple_established_contingencies_can_create_family(self) -> None:
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 2, 77, 11), support=5),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (20, 2, 77, 21), support=4),
        )
        candidates = EvidenceGatedPromotionEngine().propose(rows, ())
        families = [item for item in candidates if item.level == MemoryLevel.M2]
        self.assertEqual(len(families), 1)
        self.assertEqual(len(families[0].parents), 2)
        self.assertGreater(families[0].evidence_value, 0.0)

    def test_concept_candidate_requires_compression_explanation_and_transfer_prior(self) -> None:
        weak = node(
            MemoryLevel.M3,
            MemoryType.ROLE,
            (7, 1),
            support=4,
            explanatory=3.0,
            transfer=0.0,
        )
        strong = node(
            MemoryLevel.M3,
            MemoryType.ROLE,
            (8, 1),
            support=4,
            explanatory=3.0,
            transfer=0.75,
        )
        candidates = EvidenceGatedPromotionEngine().propose((weak, strong), ())
        concepts = [item for item in candidates if item.level == MemoryLevel.M4]
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].key_parts, strong.key_parts)


class StructuralTransferTests(unittest.TestCase):
    def test_correspondence_descriptors_use_one_edge_pass(self) -> None:
        class CountingEstimator(StructuralCorrespondenceEstimator):
            descriptor_passes = 0

            @classmethod
            def _descriptors(cls, uids, edges, by_uid):
                cls.descriptor_passes += 1
                return super()._descriptors(uids, edges, by_uid)

        left = node(MemoryLevel.M3, MemoryType.ROLE, (1, 1))
        middle = node(MemoryLevel.M3, MemoryType.ROLE, (2, 1))
        right = node(MemoryLevel.M3, MemoryType.ROLE, (3, 1))
        graph = (
            edge(left.uid, RelationType.SIMILAR_TO, middle.uid, score=0.9),
            edge(left.uid, RelationType.SIMILAR_TO, right.uid, score=0.8),
        )

        CountingEstimator().evaluate((left, middle, right), graph)

        self.assertEqual(CountingEstimator.descriptor_passes, 1)

    def test_similarity_requires_formal_correspondence_before_transfer_candidate(self) -> None:
        left = node(MemoryLevel.M3, MemoryType.ROLE, (1, 1), support=4, game_mask=1)
        right = node(MemoryLevel.M3, MemoryType.ROLE, (2, 1), support=4, game_mask=2)
        lower_left = node(MemoryLevel.M2, MemoryType.FAMILY, (11,), support=4)
        lower_right = node(MemoryLevel.M2, MemoryType.FAMILY, (12,), support=4)
        similarity = edge(left.uid, RelationType.SIMILAR_TO, right.uid, score=0.9)
        graph = (
            edge(left.uid, RelationType.EXPLAINS, lower_left.uid),
            edge(right.uid, RelationType.EXPLAINS, lower_right.uid),
            similarity,
        )
        rows = (left, right, lower_left, lower_right)
        correspondence = StructuralCorrespondenceEstimator(theta_struct=0.5).evaluate(rows, graph)
        self.assertEqual(len(correspondence), 1)
        self.assertTrue(correspondence[0].admissible)
        self.assertEqual(correspondence[0].epsilon_struct, 0.0)

        validator = TransferValidator()
        self.assertEqual(validator.candidates(rows, graph), ())
        correspondence_edge = edge(
            left.uid,
            RelationType.TRANSFER_CORRESPONDENCE,
            right.uid,
            score=1.0,
        )
        candidates = validator.candidates(rows, graph + (correspondence_edge,))
        self.assertEqual(len(candidates), 2)


class OutcomeAndForgettingTests(unittest.TestCase):
    def test_outcome_merge_requires_declared_persistent_class_criteria(self) -> None:
        a = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 9, 1), support=2)
        b = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 9, 2), support=2)
        estimator = OutcomeEquivalenceEstimator()
        classes = estimator.rebuild((a, b))
        self.assertEqual(len(classes), 1)
        self.assertTrue(classes[0].persistent)
        self.assertLessEqual(classes[0].within_class_diameter, estimator.max_diameter)
        self.assertIsNotNone(estimator.merge_revision(classes[0]))

    def test_low_level_memory_needs_semantic_replacement_before_retirement(self) -> None:
        lower = node(
            MemoryLevel.M2,
            MemoryType.FAMILY,
            (1,),
            cognitive_state=int(CognitiveState.RETIRE_PENDING),
        )
        higher = node(MemoryLevel.M3, MemoryType.ROLE, (2, 1))
        planner = PruningPlanner()
        no_replacement = planner.candidates((lower, higher), ())
        self.assertFalse(no_replacement[0].safe_to_retire)
        with_replacement = planner.candidates(
            (lower, higher),
            (edge(higher.uid, RelationType.SUPERSEDES, lower.uid),),
        )
        self.assertTrue(with_replacement[0].safe_to_retire)


class TraceabilityTests(unittest.TestCase):
    def test_developmental_order_can_invalidate_late_claim_without_precursor(self) -> None:
        evidence = EvidenceRecord.for_uid(
            "carrier-only",
            MemoryUid(1, 2),
            evidence_kind="carrier_emergence",
            watermark=10,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=2,
            validation_state=int(ValidationState.STRUCTURAL),
        )
        decisions = ScientificHypothesisEvaluator().evaluate((evidence,))
        h04 = next(item for item in decisions if item.hypothesis_id == "H04")
        self.assertEqual(h04.ordering_gate, "FAIL")
        self.assertEqual(h04.final_decision, "INVALID")
        self.assertTrue(h04.paper_claim)


if __name__ == "__main__":
    unittest.main()

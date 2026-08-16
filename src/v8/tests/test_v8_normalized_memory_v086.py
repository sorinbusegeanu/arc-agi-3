from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

import v8
from v8 import development
from v8.arena import NodeRecord
from v8.behavior_recovery import CausalEvidenceGatedPromotionEngine
from v8.learning_blockers_v055 import control_context_signature
from v8.model import (
    CognitiveState,
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    ValidationState,
    decode_pipeline,
    encode_pipeline,
)
from v8.normalized_memory_v086 import (
    PIPELINE_PACKET_SIZE_V086,
    V86PipelineEvent,
    derive_normalized_proposals,
    is_grounded_contingency,
    is_normalized_contingency,
    normalization_metrics,
)
from v8.structural_events import (
    MAX_NORMALIZED_FACTS_PER_EVENT,
    NormalizedPrimitive,
    StructuralFact,
    extract_normalized_fact_tokens,
    grounded_context_signature,
    normalized_fact_kind,
)


def event(
    *,
    sequence: int,
    context: int,
    action: int = 1,
    outcome: int = 100,
    changed: int = 1,
    source_game_hash: int = 1,
    prediction_error: float = 0.0,
) -> ExperienceEvent:
    return ExperienceEvent(
        event_id=EventId.from_producer(77, sequence),
        watermark=sequence,
        producer_id=77,
        producer_sequence=sequence,
        source_game_hash=source_game_hash,
        global_step=sequence,
        context_signature=context,
        action_id=action,
        outcome_signature=outcome,
        family_signature=200 + outcome,
        carrier_signature=300,
        future_option_delta=0.0,
        changed_cells=changed,
        terminal_polarity=0,
        trajectory_signature=400 + sequence,
        next_context_signature=context + 1,
        prediction_error=prediction_error,
    )


def node(key, *, support=5):
    uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
    return NodeRecord(
        uid=uid,
        fingerprint=1,
        level=int(MemoryLevel.M1),
        memory_type=int(MemoryType.CONTINGENCY),
        key_parts=tuple(key),
        support_count=int(support),
        significance_sum=float(support),
        prediction_error_sum=0.0,
        learning_value_sum=float(support),
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=0.0,
        score_weight=float(support),
        updated_watermark=10,
        game_mask=1,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.VALIDATED),
    )


class NormalizedMemoryV086Tests(unittest.TestCase):
    def test_pipeline_round_trip_preserves_normalization_sidecar(self):
        token = StructuralFact(NormalizedPrimitive.COMPONENT_RELOCATED, 123, 456).token
        row = V86PipelineEvent(
            event(sequence=1, context=10),
            MemoryUid(9, 11),
            0,
            1,
            777,
            888,
            3,
            (token,),
        )
        restored = decode_pipeline(encode_pipeline(row))
        self.assertEqual(restored, row)
        self.assertEqual(len(encode_pipeline(row)), PIPELINE_PACKET_SIZE_V086)

    def test_game_scope_and_history_both_change_grounded_context(self):
        prior = os.environ.get("ARC_AGI3_V8_CONTROL_SCOPE")
        grid = np.asarray([[1, 0], [0, 2]], dtype=np.int64)
        try:
            os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = "game-a"
            game_a = control_context_signature(grid)
            os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = "game-b"
            game_b = control_context_signature(grid)
            self.assertNotEqual(game_a, game_b)
            self.assertNotEqual(
                grounded_context_signature(game_a, 10),
                grounded_context_signature(game_a, 11),
            )
        finally:
            if prior is None:
                os.environ.pop("ARC_AGI3_V8_CONTROL_SCOPE", None)
            else:
                os.environ["ARC_AGI3_V8_CONTROL_SCOPE"] = prior

    def test_color_renamed_equivalent_effects_share_normalized_fact(self):
        before_a = np.asarray([[1, 0], [0, 0]], dtype=np.int64)
        after_a = np.asarray([[0, 0], [1, 0]], dtype=np.int64)
        before_b = np.asarray([[7, 0], [0, 0]], dtype=np.int64)
        after_b = np.asarray([[0, 0], [7, 0]], dtype=np.int64)
        a = extract_normalized_fact_tokens(before_a, after_a)
        b = extract_normalized_fact_tokens(before_b, after_b)
        self.assertEqual(a, b)
        self.assertEqual(normalized_fact_kind(a[0]), NormalizedPrimitive.COMPONENT_RELOCATED)

    def test_normalization_is_bounded(self):
        before = np.zeros((10, 10), dtype=np.int64)
        after = np.indices((10, 10)).sum(axis=0) % 3
        tokens = extract_normalized_fact_tokens(
            before,
            after,
            before_actions=(1, 2),
            after_actions=(1, 2, 6),
            elapsed_since_change=7,
        )
        self.assertLessEqual(len(tokens), MAX_NORMALIZED_FACTS_PER_EVENT)
        self.assertTrue(tokens)

    def test_m1n_is_one_part_and_links_back_to_grounded_m1g(self):
        token = StructuralFact(NormalizedPrimitive.COMPONENT_ATTRIBUTE_CHANGED, 100, 200).token
        pipeline = V86PipelineEvent(
            event(sequence=2, context=20),
            normalized_facts=(token,),
        )
        grounded = development.derive_proposal(MemoryLevel.M1, pipeline)
        normalized = derive_normalized_proposals(pipeline, grounded)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(len(normalized[0].key_parts), 1)
        self.assertEqual(normalized[0].parent_uid, grounded.uid)
        self.assertTrue(is_normalized_contingency(SimpleNamespace(
            level=MemoryLevel.M1,
            memory_type=MemoryType.CONTINGENCY,
            key_parts=normalized[0].key_parts,
        )))

    def test_old_four_part_m1_remains_grounded(self):
        self.assertTrue(is_grounded_contingency(node((10, 1, 100, 11))))

    def test_m2_formation_uses_m1n_when_normalized_evidence_exists(self):
        token_a = StructuralFact(NormalizedPrimitive.COMPONENT_RELOCATED, 100).token
        token_b = StructuralFact(NormalizedPrimitive.COMPONENT_RELOCATED, 200).token
        normalized_a = node((token_a,))
        normalized_b = node((token_b,))
        grounded_a = node((10, 1, 100, 11))
        grounded_b = node((20, 1, 100, 21))
        engine = CausalEvidenceGatedPromotionEngine(
            min_contingency_support=1,
            min_family_members=2,
            min_family_compression=0.0,
        )
        candidates = engine.propose(
            (normalized_a, normalized_b, grounded_a, grounded_b),
            (),
            budget=32,
        )
        families = [row for row in candidates if int(row.level) == int(MemoryLevel.M2)]
        self.assertTrue(families)
        normalized_uids = {normalized_a.uid, normalized_b.uid}
        for family in families:
            self.assertTrue(set(family.parents).issubset(normalized_uids))

    def test_changed_cell_magnitude_no_longer_determines_grounded_learning_value(self):
        token = StructuralFact(NormalizedPrimitive.COMPONENT_GEOMETRY_CHANGED, 999).token
        small = development.derive_proposal(
            MemoryLevel.M1,
            V86PipelineEvent(event(sequence=3, context=30, changed=1), normalized_facts=(token,)),
        )
        large = development.derive_proposal(
            MemoryLevel.M1,
            V86PipelineEvent(event(sequence=4, context=40, changed=100), normalized_facts=(token,)),
        )
        self.assertAlmostEqual(small.learning_value_sum, large.learning_value_sum)
        self.assertAlmostEqual(small.significance_sum, large.significance_sum)

    def test_metrics_distinguish_m1g_and_m1n(self):
        token = StructuralFact(NormalizedPrimitive.COMPONENT_CREATED, 123).token
        rows = (
            node((10, 1, 100, 11)),
            node((token,)),
        )

        class View:
            def node_records(self, level=None):
                if level == MemoryLevel.M1:
                    return rows
                return ()

        metrics = normalization_metrics(View())
        self.assertEqual(metrics["m1g_nodes"], 1)
        self.assertEqual(metrics["m1n_nodes"], 1)

    def test_runtime_exposes_v86_memory_semantics(self):
        self.assertEqual(
            v8.ContinuousMemoryRuntime.memory_semantics_version,
            "v8.6-grounded-normalized",
        )


if __name__ == "__main__":
    unittest.main()

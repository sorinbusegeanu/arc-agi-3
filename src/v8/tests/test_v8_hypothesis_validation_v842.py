from __future__ import annotations

import unittest

import v8
from v8.arena import NodeRecord
from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.hypothesis_validation_v842 import (
    _OLD_PROXY_TRANSFER_INTERVENTION,
    _hypothesis_status_line_v842,
)
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.similarity import BoundedNeighborhoodSimilarity


def node(uid: MemoryUid, *, game_mask: int) -> NodeRecord:
    return NodeRecord(
        uid,
        1,
        int(MemoryLevel.M3),
        int(MemoryType.ROLE),
        (uid.lo, 0),
        4,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1,
        game_mask,
        int(CognitiveState.ACTIVE),
        int(ValidationState.STRUCTURAL),
        0.0,
        0.0,
        0.0,
    )


def evidence(
    evidence_id: str,
    kind: str,
    *,
    watermark: int = 10,
    target: int = 0,
    provenance: tuple[int, ...] = (),
    causal: str = "",
    effect: int = 1,
) -> EvidenceRecord:
    return EvidenceRecord.for_uid(
        evidence_id,
        MemoryUid(10, len(evidence_id)),
        evidence_kind=kind,
        watermark=watermark,
        raw_value=1.0,
        normalized_value=1.0,
        developmental_stage=1,
        validation_state=int(ValidationState.VALIDATED),
        target_game_hash=target,
        provenance_games=provenance,
        causal_intervention=causal,
        effect_direction=effect,
    )


class LegacyEvidenceTests(unittest.TestCase):
    def test_structural_proxy_transfer_cannot_validate_h06(self) -> None:
        row = evidence(
            "legacy-transfer",
            "transfer_trial_pass",
            target=2,
            provenance=(1,),
            causal=_OLD_PROXY_TRANSFER_INTERVENTION,
            effect=1,
        )
        status = {
            decision.hypothesis_id: decision.final_decision
            for decision in ScientificHypothesisEvaluator().evaluate((row,))
        }
        self.assertNotEqual(status["H06"], "VALID")

    def test_real_causal_heldout_transfer_still_validates_h06(self) -> None:
        row = evidence(
            "behavioral-transfer",
            "transfer_trial_pass",
            target=2,
            provenance=(1,),
            causal="behavioral_candidate_vs_baseline",
            effect=1,
        )
        status = {
            decision.hypothesis_id: decision.final_decision
            for decision in ScientificHypothesisEvaluator().evaluate((row,))
        }
        self.assertEqual(status["H06"], "VALID")


class SimilarityBreadthTests(unittest.TestCase):
    def test_similarity_reserves_cross_game_candidate_under_tight_budget(self) -> None:
        rows = (
            node(MemoryUid(50, 1), game_mask=1),
            node(MemoryUid(50, 2), game_mask=1),
            node(MemoryUid(50, 3), game_mask=2),
        )
        estimator = BoundedNeighborhoodSimilarity(
            max_candidates=1,
            top_results=1,
            threshold=0.0,
        )
        result = estimator.evaluate(rows, ())
        masks = {row.uid: int(row.game_mask) for row in rows}
        self.assertTrue(result)
        self.assertTrue(
            any(
                masks[item.source_uid] != masks[item.target_uid]
                for item in result
            )
        )


class RuntimeHypothesisLineTests(unittest.TestCase):
    def test_stdout_cut_excludes_future_evidence(self) -> None:
        future = evidence(
            "future-h01",
            "contingency_recurrence",
            watermark=100,
        )
        line = _hypothesis_status_line_v842((future,), 50)
        self.assertIn("H01=INSUFFICIENT_EVIDENCE", line)


if __name__ == "__main__":
    unittest.main()

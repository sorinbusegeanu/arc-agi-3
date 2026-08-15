from __future__ import annotations

import unittest

from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
)
from v8.peers import DevelopmentalPeerSupervisor
from v8.similarity import BoundedNeighborhoodSimilarity
from v8.transfer import TransferValidator


def role_node(
    key: tuple[int, ...],
    *,
    watermark: int,
    transfer_prior: float = 0.0,
) -> NodeRecord:
    return NodeRecord(
        uid=MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, key),
        fingerprint=watermark + 100,
        level=int(MemoryLevel.M3),
        memory_type=int(MemoryType.ROLE),
        key_parts=key,
        support_count=3,
        significance_sum=1.0,
        prediction_error_sum=0.0,
        learning_value_sum=1.0,
        transfer_prior_sum=float(transfer_prior),
        explanatory_sum=1.0,
        future_option_sum=0.0,
        score_weight=1.0,
        updated_watermark=watermark,
        game_mask=0,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.STRUCTURAL),
    )


class BoundedSimilarityTests(unittest.TestCase):
    def test_identical_radius_one_structure_scores_one(self) -> None:
        a = role_node((1, 0), watermark=10)
        b = role_node((2, 0), watermark=11)
        child_a = role_node((10, 0), watermark=8)
        child_b = role_node((20, 0), watermark=9)
        edges = (
            EdgeRecord(a.uid, int(RelationType.EXPLAINS), child_a.uid, 1, 10),
            EdgeRecord(b.uid, int(RelationType.EXPLAINS), child_b.uid, 1, 11),
        )
        estimator = BoundedNeighborhoodSimilarity(threshold=0.0)
        descriptors = estimator.descriptors((a, b, child_a, child_b), edges)
        score = estimator.score(descriptors[a.uid], descriptors[b.uid])
        self.assertAlmostEqual(score.score, 1.0)

    def test_candidate_search_is_bounded_and_incremental(self) -> None:
        nodes = tuple(role_node((index, 0), watermark=index + 1) for index in range(40))
        estimator = BoundedNeighborhoodSimilarity(
            max_candidates=5,
            top_results=2,
            threshold=0.0,
        )
        estimator.evaluate(nodes, ())
        first_comparisons = estimator.candidate_comparisons
        self.assertLessEqual(first_comparisons, len(nodes) * 5)
        second = estimator.evaluate(nodes, ())
        self.assertEqual(second, ())
        self.assertEqual(estimator.candidate_comparisons, first_comparisons)

    def test_state_round_trip_preserves_dirty_versions(self) -> None:
        nodes = (role_node((1, 0), watermark=4), role_node((2, 0), watermark=5))
        estimator = BoundedNeighborhoodSimilarity(threshold=0.0)
        estimator.evaluate(nodes, ())
        restored = BoundedNeighborhoodSimilarity(threshold=0.0)
        restored.load_state(estimator.state_dict())
        before = restored.candidate_comparisons
        self.assertEqual(restored.evaluate(nodes, ()), ())
        self.assertEqual(restored.candidate_comparisons, before)

    def test_similarity_prior_nominates_single_game_transfer_candidate(self) -> None:
        row = role_node((1, 0), watermark=5, transfer_prior=0.8)
        candidates = TransferValidator().candidates(
            (row,),
            provenance=lambda _uid: frozenset({101}),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].formation_games, (101,))
        self.assertGreaterEqual(candidates[0].structural_score, 0.8)


class _ReadView:
    def __init__(self, nodes: tuple[NodeRecord, ...]) -> None:
        self._nodes = nodes
        self._games = {
            nodes[0].uid: frozenset({101}),
            nodes[1].uid: frozenset({202}),
        }

    def node_records(self, *, level=None):
        if level is None:
            return self._nodes
        return tuple(row for row in self._nodes if int(row.level) == int(level))

    def edge_records(self):
        return ()

    def source_games(self, uid: MemoryUid):
        return self._games.get(uid, frozenset())


class SimilarityPeerIntegrationTests(unittest.TestCase):
    def test_peer_emits_similarity_proposal_without_merging_identity(self) -> None:
        nodes = (
            role_node((11, 0), watermark=10),
            role_node((22, 0), watermark=11),
        )
        proposals = []
        peer = DevelopmentalPeerSupervisor(
            read_view=_ReadView(nodes),
            submit_proposal=proposals.append,
            watermark=lambda: 11,
            generation=lambda: 3,
            interval_seconds=1.0,
        )
        peer.run_once()
        similar = [
            proposal
            for proposal in proposals
            if int(proposal.relation_type) == int(RelationType.SIMILAR_TO)
        ]
        self.assertTrue(similar)
        self.assertNotEqual(similar[0].uid, similar[0].parent_uid)
        self.assertIn(similar[0].uid, {nodes[0].uid, nodes[1].uid})
        self.assertIn(similar[0].parent_uid, {nodes[0].uid, nodes[1].uid})
        self.assertEqual(len({nodes[0].uid, nodes[1].uid}), 2)
        self.assertGreater(peer.metrics().similarity_comparisons, 0)


if __name__ == "__main__":
    unittest.main()

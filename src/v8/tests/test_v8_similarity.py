from __future__ import annotations

import threading
import unittest
from struct import Struct
from unittest import mock

from v8.arena import EdgeRecord, NodeRecord, SharedEdgeArena
from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RELATION_PROPOSAL_PACKET_SIZE,
    RelationType,
    ValidationState,
    decode_relation_proposal,
    encode_proposal,
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
        self.assertIsNone(score.context_score)
        self.assertEqual(descriptors[a.uid].dependency_signature, 0)
        self.assertEqual(descriptors[b.uid].dependency_signature, 0)

    def test_real_dependency_edges_activate_dependency_similarity(self) -> None:
        a = role_node((1, 0), watermark=10)
        b = role_node((2, 0), watermark=11)
        child_a = role_node((10, 0), watermark=8)
        child_b = role_node((20, 0), watermark=9)
        edges = (
            EdgeRecord(a.uid, int(RelationType.DEPENDS_ON), child_a.uid, 1, 10),
            EdgeRecord(b.uid, int(RelationType.DEPENDS_ON), child_b.uid, 1, 11),
        )
        estimator = BoundedNeighborhoodSimilarity(threshold=0.0)
        descriptors = estimator.descriptors((a, b, child_a, child_b), edges)
        score = estimator.score(descriptors[a.uid], descriptors[b.uid])
        self.assertNotEqual(descriptors[a.uid].dependency_signature, 0)
        self.assertEqual(score.dependency_score, 1.0)

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

    def test_scored_similarity_requires_structural_correspondence_before_transfer_candidates(self) -> None:
        a = role_node((1, 0), watermark=5)
        b = role_node((2, 0), watermark=6)
        source, target = sorted((a.uid, b.uid))
        similarity = EdgeRecord(
            source,
            int(RelationType.SIMILAR_TO),
            target,
            2,
            6,
            1.6,
            2.0,
            5,
            6,
        )
        games = {a.uid: frozenset({101}), b.uid: frozenset({202})}
        validator = TransferValidator()
        self.assertEqual(
            validator.candidates(
                (a, b),
                (similarity,),
                provenance=lambda uid: games[uid],
            ),
            (),
        )
        correspondence = EdgeRecord(
            source,
            int(RelationType.TRANSFER_CORRESPONDENCE),
            target,
            2,
            6,
            1.6,
            2.0,
            5,
            6,
        )
        candidates = validator.candidates(
            (a, b),
            (similarity, correspondence),
            provenance=lambda uid: games[uid],
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(abs(item.structural_score - 0.8) < 1e-9 for item in candidates))
        self.assertEqual({item.correspondence_uid for item in candidates}, {a.uid, b.uid})

    def test_same_game_correspondence_does_not_nominate_transfer(self) -> None:
        a = role_node((1, 0), watermark=5)
        b = role_node((2, 0), watermark=6)
        source, target = sorted((a.uid, b.uid))
        edge = EdgeRecord(
            source,
            int(RelationType.TRANSFER_CORRESPONDENCE),
            target,
            1,
            6,
            0.9,
            1.0,
        )
        candidates = TransferValidator().candidates(
            (a, b),
            (edge,),
            provenance=lambda _uid: frozenset({101}),
        )
        self.assertEqual(candidates, ())

    def test_transfer_candidates_reuse_coherent_read_view_indexes(self) -> None:
        a = role_node((1, 0), watermark=5)
        b = role_node((2, 0), watermark=6)
        unrelated = role_node((3, 0), watermark=7)
        source, target = sorted((a.uid, b.uid))
        correspondence = EdgeRecord(
            source,
            int(RelationType.TRANSFER_CORRESPONDENCE),
            target,
            1,
            6,
            0.9,
            1.0,
        )

        class CachedView:
            _v839_transfer_version = (8,)
            _v839_provenance_version = (8,)
            _v839_transfer_edges = (correspondence,)
            _v839_direct_games = {
                a.uid: {101},
                b.uid: {202},
                unrelated.uid: {303},
            }
            _v839_provenance_parents = {}

            def edge_records(self):
                raise AssertionError("coherent transfer indexes must avoid edge scan")

            def source_games(self, _uid):
                raise AssertionError("coherent direct/parent indexes must be reused")

        original = TransferValidator._provenance_from_indexes
        with mock.patch.object(
            TransferValidator,
            "_provenance_from_indexes",
            wraps=original,
        ) as provenance:
            candidates = TransferValidator().candidates(
                (a, b, unrelated), provenance=CachedView().source_games
            )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {item.formation_games for item in candidates},
            {(101,), (202,)},
        )
        self.assertEqual(set(provenance.call_args.args[0]), {a.uid, b.uid})

    def test_transfer_candidate_scan_stops_before_cold_edge_read_after_cancel(self) -> None:
        a = role_node((1, 0), watermark=5)
        cancel = threading.Event()
        cancel.set()

        class ColdView:
            def edge_records(self):
                raise AssertionError("cancelled peer transfer must not read graph")

            def source_games(self, _uid):
                return frozenset()

        self.assertEqual(
            TransferValidator().candidates(
                (a,),
                provenance=ColdView().source_games,
                cancel_event=cancel,
            ),
            (),
        )

    def test_transfer_provenance_scan_honors_cancel_after_start(self) -> None:
        a = role_node((1, 0), watermark=5)
        cancel = threading.Event()
        edge = EdgeRecord(
            a.uid,
            int(RelationType.EXPLAINS),
            MemoryUid(9, 9),
            1,
            1,
        )

        class CancellingEdges:
            def __iter__(self):
                for index in range(5000):
                    if index == 1:
                        cancel.set()
                    yield edge

        self.assertIsNone(
            TransferValidator._provenance_from_edges(
                (a.uid,), CancellingEdges(), cancel_event=cancel
            )
        )

    def test_transfer_provenance_expansion_honors_cancel_after_edge_scan(self) -> None:
        cancel = threading.Event()
        uid = MemoryUid(7, 7)

        class CancellingDirect(dict):
            def get(self, key, default=None):
                cancel.set()
                return super().get(key, default)

        self.assertIsNone(
            TransferValidator._provenance_from_indexes(
                (uid,) * 5000,
                CancellingDirect(),
                {},
                cancel_event=cancel,
            )
        )

    def test_similarity_memory_proposal_serializes_as_relation_packet(self) -> None:
        a = role_node((1, 0), watermark=10)
        b = role_node((2, 0), watermark=11)
        source, target = sorted((a.uid, b.uid))
        proposal = MemoryProposal(
            uid=source,
            fingerprint=a.fingerprint,
            event_id=EventId.from_producer(99, 1),
            watermark=11,
            level=MemoryLevel.M3,
            memory_type=MemoryType.ROLE,
            key_parts=a.key_parts,
            support_delta=0,
            transfer_prior_sum=0.8,
            score_weight=0.0,
            parent_uid=target,
            relation_type=RelationType.SIMILAR_TO,
        )
        payload = encode_proposal(proposal)
        self.assertEqual(len(payload), RELATION_PROPOSAL_PACKET_SIZE)
        relation = decode_relation_proposal(payload)
        self.assertEqual(relation.source_uid, source)
        self.assertEqual(relation.target_uid, target)
        self.assertEqual(relation.relation_type, RelationType.SIMILAR_TO)
        self.assertEqual(relation.support_delta, 1)
        self.assertAlmostEqual(relation.score_sum, 0.8)
        self.assertAlmostEqual(relation.score_weight, 1.0)

    def test_edge_arena_round_trip_preserves_similarity_score_and_versions(self) -> None:
        a = role_node((1, 0), watermark=10)
        b = role_node((2, 0), watermark=11)
        arena = SharedEdgeArena(capacity=4)
        try:
            edge = EdgeRecord(a.uid, int(RelationType.SIMILAR_TO), b.uid, 3, 11, 2.4, 3.0, 10, 11)
            arena.begin_write()
            arena.write(0, edge)
            arena.end_write(count=1)
            restored = arena.read(0)
            self.assertAlmostEqual(restored.score, 0.8)
            self.assertEqual(restored.source_version, 10)
            self.assertEqual(restored.target_version, 11)
        finally:
            arena.dispose()

    def test_edge_arena_migrates_legacy_snapshot(self) -> None:
        a = role_node((1, 0), watermark=10)
        b = role_node((2, 0), watermark=11)
        header = Struct("<QQ")
        legacy = Struct("<QQHQQqQ")
        payload = header.pack(1, 0) + legacy.pack(
            a.uid.hi,
            a.uid.lo,
            int(RelationType.EXPLAINS),
            b.uid.hi,
            b.uid.lo,
            2,
            11,
        )
        arena = SharedEdgeArena(capacity=4)
        try:
            arena.load_snapshot(payload)
            restored = arena.read(0)
            self.assertEqual(restored.support_count, 2)
            self.assertEqual(restored.score_weight, 0.0)
            self.assertEqual(restored.updated_watermark, 11)
        finally:
            arena.dispose()


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

        relation = decode_relation_proposal(encode_proposal(similar[0]))
        self.assertEqual(relation.relation_type, RelationType.SIMILAR_TO)
        self.assertNotEqual(relation.source_uid, relation.target_uid)


if __name__ == "__main__":
    unittest.main()

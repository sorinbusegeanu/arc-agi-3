from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import v8
from v8 import click_state_learning_v857 as click_v857
from v8 import transfer_correspondence_v857 as transfer_v857
from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    decode_relation_proposal,
    encode_proposal,
)
from v8.peers import DevelopmentalPeerSupervisor
from v8.sampling_portfolio_v831 import PortfolioSampler
from v8.similarity import BoundedNeighborhoodSimilarity
from v8.transfer import TransferValidator


class ClickStateLearningV857Tests(unittest.TestCase):
    @staticmethod
    def _frame_and_env():
        frame = np.zeros((16, 16), dtype=np.int64)
        frame[0:8, 0:8] = 2
        frame[8:16, 0:8] = 2
        frame[0:8, 8:16] = 5
        frame[8:16, 8:16] = 5
        game = SimpleNamespace(
            camera=SimpleNamespace(width=2, height=2, x=0, y=0),
            current_level=SimpleNamespace(grid_size=(2, 2)),
        )
        env = SimpleNamespace(
            env=SimpleNamespace(_game=game),
            _last_grid=frame,
            _v848_replayable_clicks=set(),
            reset_count=0,
            last_levels_completed=0,
        )
        return frame, env

    def test_all_observable_cell_states_are_exposed(self) -> None:
        frame, env = self._frame_and_env()
        tokens = click_v857._all_cell_click_tokens(env, frame)
        self.assertEqual(len(tokens), 4)

        from v8.learning_blockers_v055 import unpack_action_choice

        colors = {
            int(frame[payload["y"], payload["x"]])
            for token in tokens
            for action, payload in (unpack_action_choice(token),)
            if action == 6 and payload is not None
        }
        self.assertEqual(colors, {2, 5})

        pages = click_v857._exact_click_pages_v857(env, frame)
        self.assertEqual({token for page in pages for token in page}, set(tokens))

    def test_productive_coordinate_gets_exactly_one_bounded_second_click(self) -> None:
        frame, env = self._frame_and_env()
        tokens = click_v857._all_cell_click_tokens(env, frame)
        target = tokens[0]
        env._v848_replayable_clicks = {target}

        sampler = PortfolioSampler("gp03-repeat-fixture", seed=0)
        sampler.begin_lease(0)
        sampler._v848_scan_reset_count = 0
        sampler._v848_scan_level = 0
        sampler._v848_scan_tried = set(tokens)
        sampler._v848_scan_available = ()
        sampler._v857_repeat_stamp = (0, 0)
        sampler._v857_repeat_tried = set()
        sampler._v857_repeat_available = ()

        with patch.object(click_v857, "_BASE_PREPARE_STEP", return_value=True):
            self.assertFalse(click_v857._sampler_prepare_step_v857(sampler, env))
        self.assertEqual(sampler._v857_repeat_available, (target,))

        with patch.object(click_v857, "_BASE_FORCED_ACTION", return_value=None):
            selected = click_v857._sampler_forced_action_v857(
                sampler,
                level=0,
                context=1,
                actions=tuple(tokens),
                history=(),
            )
        self.assertEqual(selected, target)
        self.assertEqual(sampler._v857_repeat_tried, {target})

        with patch.object(click_v857, "_BASE_PREPARE_STEP", return_value=True):
            self.assertTrue(click_v857._sampler_prepare_step_v857(sampler, env))
        self.assertEqual(sampler._v857_repeat_available, ())

    def test_unproductive_coordinate_is_never_repeated(self) -> None:
        frame, env = self._frame_and_env()
        tokens = click_v857._all_cell_click_tokens(env, frame)
        sampler = PortfolioSampler("gp03-noop-fixture", seed=0)
        sampler.begin_lease(0)
        sampler._v848_scan_reset_count = 0
        sampler._v848_scan_level = 0
        sampler._v848_scan_tried = set(tokens)
        sampler._v857_repeat_stamp = (0, 0)
        sampler._v857_repeat_tried = set()

        with patch.object(click_v857, "_BASE_PREPARE_STEP", return_value=True):
            self.assertTrue(click_v857._sampler_prepare_step_v857(sampler, env))

    def test_observed_nonclick_noops_reclassify_native_mixed_as_click(self) -> None:
        row = {
            "native_types": {1, 2, 3, 4, 6},
            "movement_actions_executed": 16,
            "movement_productive": 0,
            "movement_level_advances": 0,
        }
        self.assertEqual(click_v857._space_type_v857(row), "click")
        row["movement_productive"] = 1
        self.assertEqual(click_v857._space_type_v857(row), "mixed")



def _role_node(key: tuple[int, ...], watermark: int) -> NodeRecord:
    return NodeRecord(
        uid=MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, key),
        fingerprint=watermark + 100,
        level=int(MemoryLevel.M3),
        memory_type=int(MemoryType.ROLE),
        key_parts=key,
        support_count=4,
        significance_sum=1.0,
        prediction_error_sum=0.0,
        learning_value_sum=1.0,
        transfer_prior_sum=0.0,
        explanatory_sum=1.0,
        future_option_sum=0.0,
        score_weight=1.0,
        updated_watermark=watermark,
        game_mask=0,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.STRUCTURAL),
    )


class _TransferReadView:
    def __init__(self, nodes, games):
        self._nodes = tuple(nodes)
        self._games = dict(games)

    def node_records(self, *, level=None):
        if level is None:
            return self._nodes
        return tuple(row for row in self._nodes if int(row.level) == int(level))

    def edge_records(self):
        return ()

    def source_games(self, uid):
        return frozenset(self._games.get(uid, ()))


class TransferCorrespondenceV857Tests(unittest.TestCase):
    def test_high_confidence_cross_game_similarity_emits_formal_correspondence(self) -> None:
        a = _role_node((11, 0), 10)
        b = _role_node((22, 0), 11)
        view = _TransferReadView((a, b), {a.uid: {101}, b.uid: {202}})
        proposals = []
        peer = DevelopmentalPeerSupervisor(
            read_view=view,
            submit_proposal=proposals.append,
            watermark=lambda: 11,
            generation=lambda: 3,
            interval_seconds=1.0,
        )

        peer.run_once()

        similar = [row for row in proposals if row.relation_type == RelationType.SIMILAR_TO]
        correspondence = [
            row
            for row in proposals
            if row.relation_type == RelationType.TRANSFER_CORRESPONDENCE
        ]
        self.assertTrue(similar)
        self.assertTrue(correspondence)
        self.assertGreaterEqual(
            float(correspondence[0].transfer_prior_sum),
            transfer_v857._CORRESPONDENCE_THRESHOLD,
        )

        relation = decode_relation_proposal(encode_proposal(correspondence[0]))
        self.assertEqual(relation.relation_type, RelationType.TRANSFER_CORRESPONDENCE)
        edge = EdgeRecord(
            relation.source_uid,
            int(relation.relation_type),
            relation.target_uid,
            relation.support_delta,
            relation.watermark,
            relation.score_sum,
            relation.score_weight,
            relation.source_version,
            relation.target_version,
        )
        candidates = TransferValidator().candidates(
            (a, b),
            (edge,),
            provenance=view.source_games,
        )
        self.assertEqual(len(candidates), 2)

    def test_same_formation_scope_does_not_emit_correspondence(self) -> None:
        a = _role_node((31, 0), 20)
        b = _role_node((32, 0), 21)
        view = _TransferReadView((a, b), {a.uid: {101}, b.uid: {101}})
        proposals = []
        peer = DevelopmentalPeerSupervisor(
            read_view=view,
            submit_proposal=proposals.append,
            watermark=lambda: 21,
            generation=lambda: 4,
            interval_seconds=1.0,
        )
        peer.run_once()
        self.assertFalse(
            any(
                row.relation_type == RelationType.TRANSFER_CORRESPONDENCE
                for row in proposals
            )
        )

    def test_v1_similarity_snapshot_is_replayed_once_for_correspondence_migration(self) -> None:
        a = _role_node((41, 0), 30)
        b = _role_node((42, 0), 31)
        first = BoundedNeighborhoodSimilarity(threshold=0.0)
        self.assertTrue(first.evaluate((a, b), ()))
        state = first.state_dict()
        self.assertEqual(int(state["version"]), 2)
        state["version"] = 1

        restored = BoundedNeighborhoodSimilarity(threshold=0.0)
        restored.load_state(state)
        self.assertTrue(restored.evaluate((a, b), ()))

        current = restored.state_dict()
        again = BoundedNeighborhoodSimilarity(threshold=0.0)
        again.load_state(current)
        self.assertEqual(again.evaluate((a, b), ()), ())


if __name__ == "__main__":
    unittest.main()

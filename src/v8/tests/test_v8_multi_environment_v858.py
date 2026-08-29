from __future__ import annotations

import unittest

import numpy as np

import v8
from v8.environments import (
    CHESS_GYM_ID,
    ChessAdapter,
    DiscreteActionCodec,
    DiscreteObservationCodec,
    GymDiscreteAdapter,
    decode_chess_move,
    encode_chess_move,
)
from v8.structural_events import is_normalized_fact_token


class MultiEnvironmentV858Tests(unittest.TestCase):
    def test_discrete_codecs_are_reversible_and_schema_scoped(self) -> None:
        actions4 = DiscreteActionCodec(4)
        actions6 = DiscreteActionCodec(6)
        observations = DiscreteObservationCodec(16)
        self.assertEqual(actions4.decode(actions4.encode(3)), 3)
        self.assertEqual(observations.encode(15), 15)
        self.assertNotEqual(actions4.schema.schema_id, actions6.schema.schema_id)
        with self.assertRaises(ValueError):
            actions4.encode(4)

    def test_frozenlake_deterministic_success_and_sparse_valence(self) -> None:
        adapter = GymDiscreteAdapter(
            "FrozenLake-v1",
            seed=7,
            make_kwargs={"is_slippery": False},
        )
        try:
            self.assertEqual(adapter.observe(), 0)
            self.assertEqual(adapter.available_actions(), (0, 1, 2, 3))
            before = adapter.observe()
            before_actions = adapter.available_actions()
            for action in (1, 1, 2, 1, 2, 2):
                before = adapter.observe()
                before_actions = adapter.available_actions()
                after = adapter.step(action)
                facts = adapter.normalized_fact_tokens(
                    before,
                    after,
                    before_actions=before_actions,
                    after_actions=adapter.available_actions(),
                )
                self.assertTrue(facts)
                self.assertTrue(all(is_normalized_fact_token(token) for token in facts))
            boundary = adapter.cognitive_boundary_event()
            self.assertFalse(boundary.continuation)
            self.assertEqual(boundary.primary_valence, 1)
            self.assertEqual(adapter.telemetry.reward, 1.0)
        finally:
            adapter.close()

    def test_frozenlake_hole_is_negative_terminal(self) -> None:
        adapter = GymDiscreteAdapter(
            "FrozenLake-v1",
            seed=11,
            make_kwargs={"is_slippery": False},
        )
        try:
            adapter.step(2)
            adapter.step(1)
            boundary = adapter.cognitive_boundary_event()
            self.assertFalse(boundary.continuation)
            self.assertEqual(boundary.primary_valence, -1)
            self.assertEqual(adapter.telemetry.reward, 0.0)
        finally:
            adapter.close()

    def test_chess_move_codec_round_trip(self) -> None:
        import chess

        move = chess.Move.from_uci("e2e4")
        token = encode_chess_move(move)
        self.assertEqual(decode_chess_move(token), move)
        promotion = chess.Move.from_uci("a7a8q")
        self.assertEqual(decode_chess_move(encode_chess_move(promotion)), promotion)

    def test_chess_is_registered_as_gymnasium_environment(self) -> None:
        import gymnasium as gym

        env = gym.make(CHESS_GYM_ID, opponent="first")
        try:
            observation, info = env.reset(seed=3)
            self.assertTrue(env.observation_space.contains(observation))
            legal = tuple(info["legal_actions"])
            self.assertEqual(len(legal), 20)
            self.assertTrue(all(env.action_space.contains(token) for token in legal))
            observation, reward, terminated, truncated, info = env.step(legal[0])
            self.assertTrue(env.observation_space.contains(observation))
            self.assertIn(reward, (-1.0, 0.0, 1.0))
            self.assertFalse(truncated)
            self.assertIn("intermediate_observation", info)
            self.assertIsInstance(bool(terminated), bool)
        finally:
            env.close()

    def test_chess_adapter_exposes_variable_legal_action_subset_and_micro_time(self) -> None:
        adapter = ChessAdapter(seed=5, opponent="first")
        try:
            before = adapter.observe()
            before_actions = adapter.available_actions()
            self.assertEqual(len(before_actions), 20)
            after = adapter.step(before_actions[0])
            after_actions = adapter.available_actions()
            self.assertFalse(np.array_equal(before, after))
            self.assertTrue(after_actions)
            self.assertNotEqual(set(before_actions), set(after_actions))
            trace = adapter.cognitive_within_action_trace()
            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertGreaterEqual(trace.frame_count, 1)
            self.assertLessEqual(trace.frame_count, 2)
            facts = adapter.normalized_fact_tokens(
                before,
                after,
                before_actions=before_actions,
                after_actions=after_actions,
            )
            self.assertTrue(all(is_normalized_fact_token(token) for token in facts))
            transition = adapter.cognitive_transition(
                before_observation=before,
                after_observation=after,
                action_token=before_actions[0],
                available_actions_before=before_actions,
                available_actions_after=after_actions,
            )
            self.assertTrue(transition.structural_changed)
            self.assertEqual(transition.action_token, before_actions[0])
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import v8
from v7.environment.arc_adapter import ArcGridEnvironment
from v8.action_targeting_v810 import (
    is_structural_click_token,
    native_action_id,
    prefer_persisted_scores,
    structural_click_targets,
)
from v8.learning_blockers_v055 import pack_action_choice
from v8.learning_fixes_v088 import ActorProgress
from v8.model import stable_u64
from v8.normalized_memory_v086_fixups import _grounded_context
from v8.publication import ActionScore
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


class _FakeRaw:
    def __init__(self, grid, *, actions=(1, 6), levels=0, state="NOT_FINISHED"):
        self.frame = np.asarray(grid, dtype=np.int64)
        self.available_actions = list(actions)
        self.levels_completed = int(levels)
        self.state = state


class _FakeEnv:
    def __init__(self):
        self.calls = []
        self.raw = _FakeRaw([[0, 2, 0], [0, 2, 0], [0, 2, 0]])

    def reset(self):
        self.raw = _FakeRaw([[0, 2, 0], [0, 2, 0], [0, 2, 0]])
        return self.raw

    @staticmethod
    def _action_id(action):
        try:
            return int(action)
        except (TypeError, ValueError):
            value = getattr(action, "value", None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
            text = str(action)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return int(digits)
            raise

    def step(self, action, data=None):
        self.calls.append((self._action_id(action), data))
        return self.raw


def _factory(**_kwargs):
    return _FakeEnv()


class StructuralClickTests(unittest.TestCase):
    def test_structural_target_survives_color_change_and_translation(self):
        first = np.zeros((7, 7), dtype=np.int64)
        first[1, 1:3] = 2
        first[2, 1] = 2
        second = np.zeros((9, 9), dtype=np.int64)
        second[4, 5:7] = 7
        second[5, 5] = 7

        first_center = next(
            row for row in structural_click_targets(first) if row.kind == "component_center"
        )
        second_center = next(
            row for row in structural_click_targets(second) if row.kind == "component_center"
        )

        self.assertEqual(first_center.token, second_center.token)
        self.assertNotEqual((first_center.x, first_center.y), (second_center.x, second_center.y))
        self.assertTrue(is_structural_click_token(first_center.token))
        self.assertEqual(native_action_id(first_center.token), 6)

    def test_changed_component_has_explicit_target_role(self):
        grid = np.zeros((5, 5), dtype=np.int64)
        grid[1:4, 2] = 3
        rows = structural_click_targets(grid, last_changed=((2, 3),))
        changed = [row for row in rows if row.kind == "changed_component"]
        self.assertTrue(changed)
        self.assertEqual((changed[0].x, changed[0].y), (2, 3))

    def test_environment_exposes_structural_targets_not_coordinate_pages(self):
        env = ArcGridEnvironment(game_id="fixture", env_factory=_factory)
        actions = env.available_actions()
        self.assertIn(1, actions)
        self.assertNotIn(6, actions)
        targets = [value for value in actions if is_structural_click_token(value)]
        self.assertTrue(targets)
        self.assertLessEqual(len(targets), 96)

        target = next(
            row
            for row in structural_click_targets(env.observe())
            if row.kind == "component_center"
        )
        self.assertIn(target.token, actions)
        env.step(target.token)
        self.assertEqual(env.env.calls[-1], (6, {"x": target.x, "y": target.y}))
        self.assertEqual(set(actions), set(env.available_actions()))

    def test_legacy_absolute_click_remains_executable_action_type(self):
        legacy = pack_action_choice(6, 12, 34)
        self.assertEqual(native_action_id(legacy), 6)
        self.assertFalse(is_structural_click_token(legacy))


class RestartMemoryPolicyTests(unittest.TestCase):
    def test_positive_persisted_memory_prevents_mandatory_unseen_override(self):
        rows = (
            ActionScore(1, 8, 0.75, 1),
            ActionScore(2, 0, 0.0, 0),
        )
        stabilized = prefer_persisted_scores(rows)
        self.assertEqual(stabilized[0].support_count, 8)
        self.assertEqual(stabilized[1].support_count, 1)
        self.assertEqual(
            min(stabilized, key=lambda row: (-row.score, -row.support_count, row.action_id)).action_id,
            1,
        )
        exploratory = prefer_persisted_scores(rows, force_random=True)
        self.assertEqual(exploratory[1].support_count, 0)

    def test_restored_action_memory_is_immediately_preferred_over_unseen(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = V8RuntimeConfig.from_path(
                tmp,
                shards=1,
                stage_workers=1,
                enable_snapshots=True,
                restore=True,
                enable_peers=False,
                snapshot_interval_seconds=3600,
                node_capacity_per_shard=1024,
                edge_capacity_per_shard=2048,
                action_capacity_per_shard=128,
            )
            game_hash = stable_u64("restart-policy", person=b"v8-game")
            raw_context = 100
            grounded_context = _grounded_context(game_hash, raw_context)

            runtime = ContinuousMemoryRuntime(config)
            runtime.start()
            for sequence in range(1, 5):
                runtime.submit(
                    runtime.make_experience(
                        producer_id=1,
                        producer_sequence=sequence,
                        source_game_hash=game_hash,
                        global_step=sequence,
                        context_signature=raw_context,
                        action_id=1,
                        outcome_signature=1001,
                        family_signature=2001,
                        carrier_signature=3001,
                        changed_cells=1,
                        terminal_polarity=1,
                        next_context_signature=101,
                    )
                )
            runtime.wait_quiescent(timeout=20)
            runtime.close(normal=True, timeout=30)

            restored = ContinuousMemoryRuntime(config)
            try:
                restored.start()
                scores = restored.read_view.score_actions(grounded_context, (1, 2))
                by_action = {row.action_id: row for row in scores}
                self.assertGreater(by_action[1].score, 0.0)
                self.assertGreater(by_action[2].support_count, 0)
                unseen = [row.action_id for row in scores if row.support_count == 0]
                self.assertEqual(unseen, [])
                self.assertEqual(
                    min(scores, key=lambda row: (-row.score, -row.support_count, row.action_id)).action_id,
                    1,
                )
            finally:
                restored.close(normal=False)


class CompactProgressTests(unittest.TestCase):
    def test_solved_game_output_uses_best_and_last_only(self):
        from v8 import diagnostics

        rows = (
            ActorProgress(
                1,
                "tp01",
                1000,
                3,
                0,
                5,
                first_win_step=869,
                best_win_steps=766,
                last_win_steps=812,
            ),
        )
        line = diagnostics.format_game_rate_line(rows)
        self.assertIn("tp01:B=766,L=812", line)
        self.assertNotIn("first=", line)
        self.assertNotIn("best_win_actions=", line)
        self.assertNotIn("last_win_actions=", line)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock, patch

import v8
from v8.model import stable_u64
from v8.normalized_memory_v086_fixups import _grounded_context
from v8.publication import LiveReadView
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig
from v8 import restart_memory_v815 as restart
from v8 import restart_memory_v815_fixups as restart_fixups
import v8.restart_control_safety_v817 as safety


class RestartControlSafetyV817Tests(unittest.TestCase):
    def test_actor_score_bypasses_context_free_v815_wrapper(self) -> None:
        view = type("FakeView", (), {"_behavior_actor_mode": True})()
        sentinel = (object(),)
        with patch.object(restart_fixups, "_refresh_if_published", Mock()), patch.object(
            restart, "_BASE_SCORE_ACTIONS", Mock(return_value=sentinel)
        ) as base:
            rows = LiveReadView.score_actions(view, 123, (1, 2))
        self.assertIs(rows, sentinel)
        base.assert_called_once_with(view, 123, (1, 2))

    def test_actor_plan_bypasses_same_game_and_phase_v815_fallbacks(self) -> None:
        view = type("FakeView", (), {"_behavior_actor_mode": True})()
        sentinel = (object(),)
        with patch.object(restart_fixups, "_refresh_if_published", Mock()), patch.object(
            restart, "_BASE_PLAN_CANDIDATES", Mock(return_value=sentinel)
        ) as base:
            rows = LiveReadView.plan_candidates(view, 456, (1, 2), outcome_uid=None)
        self.assertIs(rows, sentinel)
        base.assert_called_once_with(view, 456, (1, 2), outcome_uid=None)

    def test_actor_refresh_does_not_build_game_wide_restart_indexes(self) -> None:
        view = type("FakeView", (), {"_behavior_actor_mode": True})()
        with patch.object(restart, "_BASE_VIEW_REFRESH", Mock(return_value=None)) as base, patch.object(
            safety, "_BASE_REFRESH", Mock(side_effect=AssertionError("v8.15 refresh used"))
        ):
            LiveReadView._refresh_strategy_cache(view)
        base.assert_called_once_with(view)

    def test_restored_memory_controls_exact_context_but_not_unrelated_context(self) -> None:
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
            game_hash = stable_u64("v817-restart-policy", person=b"v8-game")
            raw_context = 100
            exact_context = _grounded_context(game_hash, raw_context)

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
            restored.start()
            try:
                view = restored.read_view
                view._behavior_actor_mode = True

                exact = {row.action_id: row for row in view.score_actions(exact_context, (1, 2))}
                self.assertGreater(exact[1].score, 0.0)

                unrelated_context = exact_context ^ 0x5A5A5A5A
                unrelated = {
                    row.action_id: row
                    for row in view.score_actions(unrelated_context, (1, 2))
                }
                self.assertEqual(unrelated[1].support_count, 0)
                self.assertEqual(unrelated[1].score, 0.0)
                self.assertEqual(unrelated[2].support_count, 0)
            finally:
                restored.close(normal=False)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest

from v8.development import RAW_STAGES
from v8.model import stable_u64
from v8.normalized_memory_v086_fixups import _grounded_context
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig
import v8.runtime as runtime_module


class RestartPersistenceTests(unittest.TestCase):
    def _config(self, root: str) -> V8RuntimeConfig:
        return V8RuntimeConfig.from_path(
            root,
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

    def test_raw_runtime_topology_is_explicitly_m0_m1_only(self) -> None:
        self.assertEqual(tuple(runtime_module.STAGES), tuple(RAW_STAGES))
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ContinuousMemoryRuntime(self._config(tmp))
            try:
                self.assertEqual(len(runtime._stage_rings), 2)
                self.assertEqual(len(runtime._stage_processes), 2)
            finally:
                runtime.close(normal=False)

    def test_primary_valence_action_policy_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp)
            game_hash = stable_u64("persist01", person=b"v8-game")
            raw_context = 100
            raw_next_context = 101
            good_action = 1
            bad_action = 2
            grounded_context = _grounded_context(game_hash, raw_context)

            runtime = ContinuousMemoryRuntime(config)
            runtime.start()
            sequence = 0
            for action, polarity in ((good_action, 1),) * 4 + ((bad_action, -1),) * 4:
                sequence += 1
                runtime.submit(
                    runtime.make_experience(
                        producer_id=1,
                        producer_sequence=sequence,
                        source_game_hash=game_hash,
                        global_step=sequence,
                        context_signature=raw_context,
                        action_id=action,
                        outcome_signature=1000 + action,
                        family_signature=2000 + action,
                        carrier_signature=3000 + action,
                        future_option_delta=0.0,
                        changed_cells=1,
                        terminal_polarity=polarity,
                        next_context_signature=raw_next_context,
                    )
                )
            runtime.wait_quiescent(timeout=20)

            before_rows = {
                row.action_id: row
                for row in runtime.read_view.score_actions(
                    grounded_context, (good_action, bad_action)
                )
            }
            self.assertGreater(before_rows[good_action].support_count, 0)
            self.assertGreater(before_rows[bad_action].support_count, 0)
            self.assertGreater(before_rows[good_action].score, before_rows[bad_action].score)
            self.assertEqual(
                runtime.read_view.best_action(
                    grounded_context, (good_action, bad_action)
                ),
                good_action,
            )
            runtime.close(normal=True, timeout=30)

            restored = ContinuousMemoryRuntime(config)
            try:
                after_rows = {
                    row.action_id: row
                    for row in restored.read_view.score_actions(
                        grounded_context, (good_action, bad_action)
                    )
                }
                self.assertEqual(
                    after_rows[good_action].support_count,
                    before_rows[good_action].support_count,
                )
                self.assertEqual(
                    after_rows[bad_action].support_count,
                    before_rows[bad_action].support_count,
                )
                self.assertAlmostEqual(
                    after_rows[good_action].score,
                    before_rows[good_action].score,
                    places=12,
                )
                self.assertAlmostEqual(
                    after_rows[bad_action].score,
                    before_rows[bad_action].score,
                    places=12,
                )
                self.assertEqual(
                    restored.read_view.best_action(
                        grounded_context, (good_action, bad_action)
                    ),
                    good_action,
                )
            finally:
                restored.close(normal=False)


if __name__ == "__main__":
    unittest.main()

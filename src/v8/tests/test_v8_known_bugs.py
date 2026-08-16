from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v8.model import MemoryLevel
from v8.normalized_memory_v086 import is_grounded_contingency, is_normalized_contingency
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


class KnownBugTests(unittest.TestCase):
    def config(self, root: Path, *, snapshots: bool = False, restore: bool = False) -> V8RuntimeConfig:
        return V8RuntimeConfig(
            root=root,
            shards=2,
            stage_workers=1,
            stage_ring_capacity=128,
            shard_ring_capacity=128,
            node_capacity_per_shard=2048,
            edge_capacity_per_shard=4096,
            action_capacity_per_shard=128,
            snapshot_interval_seconds=9999.0,
            enable_snapshots=snapshots,
            restore=restore,
        )

    def experience(self, runtime: ContinuousMemoryRuntime):
        return runtime.make_experience(
            producer_id=7,
            producer_sequence=1,
            source_game_hash=1,
            global_step=1,
            context_signature=10,
            action_id=2,
            outcome_signature=100,
            family_signature=200,
            carrier_signature=300,
            future_option_delta=1.0,
            changed_cells=4,
            trajectory_signature=400,
        )

    def test_constructing_unsubmitted_experience_does_not_advance_watermark(self) -> None:
        root = Path(tempfile.mkdtemp())
        runtime = ContinuousMemoryRuntime(self.config(root))
        try:
            self.assertEqual(runtime.watermark, 0)
            self.experience(runtime)
            self.assertEqual(runtime.watermark, 0)
        finally:
            runtime.close(normal=False)

    def test_duplicate_event_after_restart_does_not_double_count_support(self) -> None:
        root = Path(tempfile.mkdtemp())
        config = self.config(root, snapshots=True, restore=True)
        runtime = ContinuousMemoryRuntime(config)
        event = self.experience(runtime)
        runtime.start()
        runtime.submit(event)
        runtime.wait_quiescent(timeout=20)
        runtime.close(normal=True, timeout=30)

        restored = ContinuousMemoryRuntime(config)
        try:
            restored.start()
            restored.submit(event)
            restored.wait_quiescent(timeout=20)
            rows = restored.read_view.node_records(level=MemoryLevel.M1)
            grounded = [row for row in rows if is_grounded_contingency(row)]
            normalized = [row for row in rows if is_normalized_contingency(row)]
            self.assertEqual(len(grounded), 1)
            self.assertEqual(grounded[0].support_count, 1)
            self.assertTrue(normalized)
            self.assertTrue(all(row.support_count == 1 for row in normalized))
        finally:
            restored.close(normal=False)


if __name__ == "__main__":
    unittest.main()

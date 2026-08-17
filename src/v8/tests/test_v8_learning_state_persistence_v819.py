from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import v8
from v8.adaptive_learning_allocation_v819 import (
    FrontierCandidate,
    FrontierScope,
    FrontierSource,
    GameLearningState,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


class LearningStateSnapshotPersistenceV819Tests(unittest.TestCase):
    def config(self, root: Path, *, restore: bool) -> V8RuntimeConfig:
        return V8RuntimeConfig(
            root=root,
            shards=1,
            stage_workers=1,
            stage_ring_capacity=128,
            shard_ring_capacity=128,
            node_capacity_per_shard=2048,
            edge_capacity_per_shard=4096,
            action_capacity_per_shard=512,
            snapshot_interval_seconds=9999.0,
            enable_snapshots=True,
            restore=restore,
            enable_peers=False,
        )

    def test_final_snapshot_restores_learning_state_and_frontier(self) -> None:
        root = Path(tempfile.mkdtemp())
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
        strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (4, 5, 6, 7))
        scope = FrontierScope("ez01", 5, 123, outcome.hi, outcome.lo)
        candidate = FrontierCandidate(
            strategy,
            "persisted-trajectory",
            987,
            42,
            3,
            3,
            int(ValidationState.TESTED),
            FrontierSource.SAMPLER,
            10,
        )

        first = ContinuousMemoryRuntime(self.config(root, restore=False))
        try:
            first.start()
            first._v819_adaptive_learning.observe_frontier_candidate(
                scope,
                candidate,
                terminal_state="WIN",
                generation=10,
            )
            first.close(normal=True, timeout=30.0)
        except BaseException:
            first.close(normal=False)
            raise

        restored = ContinuousMemoryRuntime(self.config(root, restore=True))
        try:
            coordinator = restored._v819_adaptive_learning
            self.assertEqual(
                coordinator.game_state("ez01"),
                GameLearningState.SOLVED_OPTIMIZING,
            )
            best = coordinator.frontier.best_for_game("ez01")
            self.assertIsNotNone(best)
            self.assertEqual(best[1].trajectory_id, "persisted-trajectory")
            self.assertEqual(best[1].cost, 42)
            telemetry = {row.game_id: row for row in coordinator.telemetry()}
            self.assertEqual(telemetry["ez01"].sample_steps, 0)
        finally:
            restored.close(normal=False)


if __name__ == "__main__":
    unittest.main()

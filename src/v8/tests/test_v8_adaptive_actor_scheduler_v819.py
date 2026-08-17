from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import v8
from v8 import actor as actor_module
from v8.adaptive_learning_allocation_v819 import AdaptiveLearningConfig
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


class AdaptiveActorSchedulerV819Tests(unittest.TestCase):
    def config(self, root: Path) -> V8RuntimeConfig:
        return V8RuntimeConfig(
            root=root,
            shards=1,
            stage_workers=1,
            stage_ring_capacity=512,
            shard_ring_capacity=512,
            node_capacity_per_shard=12000,
            edge_capacity_per_shard=24000,
            action_capacity_per_shard=2048,
            snapshot_interval_seconds=9999.0,
            enable_snapshots=False,
            restore=False,
            enable_peers=False,
        )

    def test_global_credit_budget_is_preserved_across_multiple_leases(self) -> None:
        root = Path(tempfile.mkdtemp())
        runtime = ContinuousMemoryRuntime(self.config(root))
        runtime._v819_adaptive_learning.config = AdaptiveLearningConfig(lease_steps=64)
        job = actor_module.ActorJob(
            actor_id=1,
            game_id="ez01",
            steps=130,
            seed=7,
            env_root=None,
            epsilon=0.10,
            graph_check_steps=1000,
        )
        try:
            results = actor_module.run_actor_jobs(
                runtime,
                (job,),
                timeout=60.0,
                progress_interval_seconds=5.0,
            )
            self.assertEqual(sum(int(row.steps) for row in results), 130)
            run = runtime._v819_adaptive_learning._run["ez01"]
            self.assertEqual(run.sample_steps, 130)
            self.assertEqual(run.leases, 3)
            self.assertTrue((root / "sampling_allocation.log").exists())
        finally:
            runtime.close(normal=False)


if __name__ == "__main__":
    unittest.main()

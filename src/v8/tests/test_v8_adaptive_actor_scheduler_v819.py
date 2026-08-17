from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import v8
from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819 as v819
from v8.adaptive_learning_allocation_v819 import AdaptiveLearningConfig
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


def _fake_persistent_lease_worker(
    *,
    worker_id,
    assignment_queue,
    event_queue,
    ready_event,
    experience_ring_args,
    read_descriptors,
    watermark,
    stop_event,
    actor_throttle,
    snapshot_freeze,
):
    del experience_ring_args, read_descriptors, watermark, actor_throttle, snapshot_freeze
    ready_event.set()
    while not stop_event.is_set():
        try:
            lease = assignment_queue.get(timeout=0.10)
        except queue.Empty:
            continue
        if lease is None:
            return
        result = actor_module.ActorResult(
            int(worker_id),
            str(lease.game_id),
            int(lease.steps),
            0,
            0,
            0,
            0,
            0,
            0,
            (),
            (),
            (),
            None,
        )
        event_queue.put(v819._LeaseResult(int(worker_id), lease, result))


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
            game_id="synthetic-game",
            steps=130,
            seed=7,
            env_root=None,
            epsilon=0.10,
            graph_check_steps=1000,
        )
        try:
            with patch.object(
                v819,
                "_persistent_lease_worker",
                _fake_persistent_lease_worker,
            ):
                results = actor_module.run_actor_jobs(
                    runtime,
                    (job,),
                    timeout=60.0,
                    progress_interval_seconds=5.0,
                )
            self.assertEqual(sum(int(row.steps) for row in results), 130)
            run = runtime._v819_adaptive_learning._run["synthetic-game"]
            self.assertEqual(run.sample_steps, 130)
            self.assertEqual(run.leases, 3)
            self.assertTrue((root / "sampling_allocation.log").exists())
        finally:
            runtime.close(normal=False)


if __name__ == "__main__":
    unittest.main()

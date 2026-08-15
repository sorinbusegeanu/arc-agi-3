from __future__ import annotations

import tempfile
import unittest

from v8.actor import _trajectory_step_cost
from v8.capacity import snapshot_usage
from v8.model import MemoryLevel
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


class PredictionBootstrapTests(unittest.TestCase):
    def test_actor_expectation_is_inactive_until_supported_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ContinuousMemoryRuntime(
                V8RuntimeConfig.from_path(
                    tmp,
                    shards=1,
                    stage_workers=1,
                    enable_snapshots=False,
                    enable_peers=False,
                    node_capacity_per_shard=1000,
                    edge_capacity_per_shard=3000,
                    action_capacity_per_shard=128,
                )
            )
            runtime.start()
            for sequence in (1, 2):
                runtime.submit(
                    runtime.make_experience(
                        producer_id=1,
                        producer_sequence=sequence,
                        source_game_hash=1,
                        global_step=sequence,
                        context_signature=10,
                        action_id=2,
                        outcome_signature=100,
                        family_signature=200,
                        carrier_signature=300,
                        next_context_signature=11,
                    )
                )
            runtime.wait_quiescent(timeout=15)
            self.assertEqual(runtime.read_view.outcome_distribution(10, 2), {})

            runtime.submit(
                runtime.make_experience(
                    producer_id=1,
                    producer_sequence=3,
                    source_game_hash=1,
                    global_step=3,
                    context_signature=10,
                    action_id=2,
                    outcome_signature=100,
                    family_signature=200,
                    carrier_signature=300,
                    next_context_signature=11,
                )
            )
            runtime.wait_quiescent(timeout=15)
            self.assertEqual(runtime.read_view.outcome_distribution(10, 2), {100: 1.0})
            runtime.close(normal=True, timeout=15)


class TrajectoryCostTests(unittest.TestCase):
    def test_blocked_repeat_loop_and_failure_raise_cost(self) -> None:
        clean = _trajectory_step_cost(
            context=1,
            after_context=2,
            changed_cells=4,
            negative_outcome=False,
            recent_contexts=(3, 4),
        )
        costly = _trajectory_step_cost(
            context=1,
            after_context=1,
            changed_cells=0,
            negative_outcome=True,
            recent_contexts=(1, 3),
        )
        self.assertEqual(clean, 1.0)
        self.assertGreater(costly, clean)


class MultiprocessingSafetyTests(unittest.TestCase):
    def test_runtime_never_uses_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ContinuousMemoryRuntime(
                V8RuntimeConfig.from_path(
                    tmp,
                    shards=1,
                    stage_workers=1,
                    enable_snapshots=False,
                    enable_peers=False,
                    node_capacity_per_shard=128,
                    edge_capacity_per_shard=256,
                    action_capacity_per_shard=64,
                )
            )
            self.assertNotEqual(runtime._mp_ctx.get_start_method(), "fork")
            runtime.close(normal=False)

    def test_snapshot_service_never_uses_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ContinuousMemoryRuntime(
                V8RuntimeConfig.from_path(
                    tmp,
                    shards=1,
                    stage_workers=1,
                    enable_snapshots=True,
                    enable_peers=False,
                    node_capacity_per_shard=128,
                    edge_capacity_per_shard=256,
                    action_capacity_per_shard=64,
                )
            )
            self.assertIsNotNone(runtime.snapshot_service)
            self.assertNotEqual(
                runtime.snapshot_service.multiprocessing_start_method,
                "fork",
            )
            runtime.close(normal=False)


class ChunkCapacityContinuationTests(unittest.TestCase):
    def test_capacity_planner_reads_content_addressed_final_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V8RuntimeConfig.from_path(
                tmp,
                shards=1,
                stage_workers=1,
                enable_snapshots=True,
                enable_peers=False,
                snapshot_interval_seconds=3600,
                node_capacity_per_shard=1000,
                edge_capacity_per_shard=3000,
                action_capacity_per_shard=128,
            )
            runtime = ContinuousMemoryRuntime(config)
            runtime.start()
            runtime.submit(
                runtime.make_experience(
                    producer_id=1,
                    producer_sequence=1,
                    source_game_hash=42,
                    global_step=1,
                    context_signature=10,
                    action_id=1,
                    outcome_signature=100,
                    family_signature=200,
                    carrier_signature=300,
                    next_context_signature=11,
                )
            )
            runtime.wait_quiescent(timeout=15)
            runtime.close(normal=True, timeout=20)

            usage = snapshot_usage(tmp)
            self.assertGreater(usage.node_count, 0)
            self.assertGreater(usage.edge_count, 0)
            self.assertGreaterEqual(usage.node_capacity, 1000)
            self.assertGreaterEqual(usage.edge_capacity, 3000)


if __name__ == "__main__":
    unittest.main()

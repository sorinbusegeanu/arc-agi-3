from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - install current runtime stack
from v8 import actor_throughput_v853 as v853
from v8 import capacity
from v8 import memory_efficiency_v851 as memory
from v8 import memory_efficiency_v851_integrity as integrity
from v8 import memory_efficiency_v852_review_fix as v852
from v8 import runtime_stack_v88
from v8.scheduler import ResourceController


class ActorThroughputV853Tests(unittest.TestCase):
    def _decide(self, *, stage_depth=0, shard_depth=0, memory_count=0, memory_capacity=1000):
        return ResourceController().decide(
            stage_depths=(int(stage_depth),),
            shard_depths=(int(shard_depth),),
            stage_capacity=1000,
            shard_capacity=1000,
            memory_count=int(memory_count),
            memory_capacity=int(memory_capacity),
        )

    def test_v853_remains_final_post_layer(self):
        self.assertEqual(runtime_stack_v88._POST_LAYERS[-1], "actor_throughput_v853")
        self.assertEqual(runtime_stack_v88._FINAL_LAYERS[-1], "performance_memory_v854")
        self.assertTrue(v853._INSTALLED)

    def test_dense_arena_alone_never_throttles_actor(self):
        with patch.object(
            memory,
            "_system_memory",
            return_value={"used_pct": 20.0},
        ):
            decision = self._decide(memory_count=999, memory_capacity=1000)
        self.assertEqual(decision.actor_throttle_seconds, 0.0)
        self.assertEqual(decision.reason, "normal")

    def test_real_ring_backlog_still_throttles_actor(self):
        with patch.object(
            memory,
            "_system_memory",
            return_value={"used_pct": 20.0},
        ):
            decision = self._decide(stage_depth=950, memory_count=999, memory_capacity=1000)
        self.assertEqual(decision.actor_throttle_seconds, 0.010)
        self.assertEqual(decision.reason, "critical memory/backlog pressure")

    def test_real_ram_pressure_still_throttles_actor(self):
        with patch.object(
            memory,
            "_system_memory",
            return_value={"used_pct": 93.0},
        ):
            decision = self._decide(memory_count=0, memory_capacity=1000)
        self.assertEqual(decision.actor_throttle_seconds, 0.010)
        self.assertEqual(decision.reason, "critical real-RAM pressure")

    def test_restored_action_table_shrinks_from_historical_capacity(self):
        historical = capacity.SnapshotUsage(
            node_count=100,
            edge_count=200,
            node_capacity=100_000,
            edge_capacity=200_000,
            action_capacity=1_000_000,
        )
        with (
            patch.object(capacity, "snapshot_usage", return_value=historical),
            patch.object(v852, "_occupied_action_count", return_value=100),
        ):
            plan = capacity.plan_capacities(
                total_steps=100,
                shards=1,
                root="unused",
                restore=True,
            )
        self.assertEqual(plan.action_capacity_per_shard, integrity._MIN_ACTION_CAPACITY)
        self.assertLess(plan.action_capacity_per_shard, historical.action_capacity)

    def test_restored_action_table_keeps_required_growth_headroom(self):
        occupied = 30_000
        with patch.object(v852, "_occupied_action_count", return_value=occupied):
            plan = capacity.plan_capacities(
                total_steps=10_000,
                shards=1,
                root="unused",
                restore=True,
            )
        required = (
            occupied
            + 15_000
            + capacity.FIXED_HEADROOM
        ) / 0.70
        self.assertGreaterEqual(plan.action_capacity_per_shard, int(required))

    def test_resource_decision_is_exposed_in_metrics_payload(self):
        controller = ResourceController()
        with patch.object(
            memory,
            "_system_memory",
            return_value={"used_pct": 20.0},
        ):
            decision = controller.decide(
                stage_depths=(0,),
                shard_depths=(0,),
                stage_capacity=1000,
                shard_capacity=1000,
                memory_count=999,
                memory_capacity=1000,
            )
        payload = v853._decision_payload(decision)
        self.assertEqual(payload["actor_throttle_seconds"], 0.0)
        self.assertFalse(payload["arena_occupancy_throttles_actors"])
        self.assertEqual(payload["reason"], "normal")


if __name__ == "__main__":
    unittest.main()

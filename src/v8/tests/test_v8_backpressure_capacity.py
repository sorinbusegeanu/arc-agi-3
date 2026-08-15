from __future__ import annotations

import multiprocessing as mp
import tempfile
import threading
import time
import unittest
from pathlib import Path

from v8.capacity import (
    DEFAULT_ACTION_CAPACITY,
    DEFAULT_EDGE_CAPACITY,
    DEFAULT_NODE_CAPACITY,
    plan_capacities,
)
from v8.development import _put_with_backpressure, derive_proposal
from v8.model import EventId, ExperienceEvent, MemoryLevel, MemoryUid, PipelineEvent
from v8.ring import SharedRingBuffer


class BackpressureTests(unittest.TestCase):
    def test_temporary_full_ring_waits_instead_of_failing(self) -> None:
        ring = SharedRingBuffer(capacity=1, slot_size=8)
        stop = mp.Event()
        try:
            self.assertTrue(ring.put(b"first", timeout=0.1))

            def release_slot() -> None:
                time.sleep(0.10)
                self.assertEqual(ring.get(timeout=1.0), b"first")

            thread = threading.Thread(target=release_slot)
            thread.start()
            started = time.monotonic()
            self.assertTrue(
                _put_with_backpressure(
                    ring,
                    b"next",
                    stop,
                    retry_seconds=0.01,
                )
            )
            elapsed = time.monotonic() - started
            thread.join(timeout=1.0)
            self.assertGreaterEqual(elapsed, 0.05)
            self.assertEqual(ring.get(timeout=0.1), b"next")
        finally:
            ring.dispose()

    def test_backpressure_stops_cleanly_when_runtime_stops(self) -> None:
        ring = SharedRingBuffer(capacity=1, slot_size=8)
        stop = mp.Event()
        try:
            self.assertTrue(ring.put(b"first", timeout=0.1))
            stop.set()
            self.assertFalse(
                _put_with_backpressure(
                    ring,
                    b"next",
                    stop,
                    retry_seconds=0.01,
                )
            )
        finally:
            ring.dispose()


class CapacityTests(unittest.TestCase):
    def test_long_run_auto_capacity_scales_with_step_budget(self) -> None:
        root = Path(tempfile.mkdtemp())
        plan = plan_capacities(
            total_steps=1_000_000,
            shards=4,
            root=root,
            restore=False,
        )
        self.assertGreater(plan.node_capacity_per_shard, DEFAULT_NODE_CAPACITY)
        self.assertGreater(plan.edge_capacity_per_shard, DEFAULT_EDGE_CAPACITY)
        self.assertGreater(plan.action_capacity_per_shard, DEFAULT_ACTION_CAPACITY)
        self.assertGreaterEqual(plan.node_capacity_per_shard, 1_000_000)
        self.assertGreaterEqual(plan.edge_capacity_per_shard, 1_000_000)

    def test_manual_capacity_override_is_preserved_without_restore(self) -> None:
        plan = plan_capacities(
            total_steps=10_000_000,
            shards=4,
            restore=False,
            node_override=123_456,
            edge_override=234_567,
            action_override=34_567,
        )
        self.assertEqual(plan.node_capacity_per_shard, 123_456)
        self.assertEqual(plan.edge_capacity_per_shard, 234_567)
        self.assertEqual(plan.action_capacity_per_shard, 34_567)


class StrategyIdentityTests(unittest.TestCase):
    def _event(self, trajectory_signature: int) -> ExperienceEvent:
        return ExperienceEvent(
            event_id=EventId.from_producer(1, trajectory_signature),
            watermark=1,
            producer_id=1,
            producer_sequence=trajectory_signature,
            source_game_hash=1,
            global_step=trajectory_signature,
            context_signature=10,
            action_id=2,
            outcome_signature=100,
            family_signature=200,
            carrier_signature=300,
            future_option_delta=1.0,
            changed_cells=4,
            terminal_polarity=0,
            trajectory_signature=trajectory_signature,
        )

    def test_m7_identity_does_not_explode_with_rolling_trajectory_signature(self) -> None:
        outcome_uid = MemoryUid(11, 22)
        first = derive_proposal(
            MemoryLevel.M7,
            PipelineEvent(self._event(400), parent_uid=outcome_uid, current_level=6),
        )
        second = derive_proposal(
            MemoryLevel.M7,
            PipelineEvent(self._event(999), parent_uid=outcome_uid, current_level=6),
        )
        self.assertEqual(first.uid, second.uid)
        self.assertEqual(first.key_parts, second.key_parts)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v8.model import (
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    decode_pipeline,
    encode_pipeline,
)
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


class IdentityTests(unittest.TestCase):
    def test_memory_uid_is_deterministic_and_level_scoped(self) -> None:
        a = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 2, 30))
        b = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 2, 30))
        c = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, (10, 2, 30))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_pipeline_binary_codec_round_trip(self) -> None:
        event = ExperienceEvent(
            EventId.from_producer(3, 9),
            15,
            3,
            9,
            22,
            15,
            101,
            2,
            303,
            404,
            505,
            1.5,
            7,
            0,
            606,
        )
        original = PipelineEvent(event, MemoryUid(9, 11), 4)
        self.assertEqual(decode_pipeline(encode_pipeline(original)), original)


class RuntimeTests(unittest.TestCase):
    def config(self, root: Path, *, snapshots: bool = True, restore: bool = True) -> V8RuntimeConfig:
        return V8RuntimeConfig(
            root=root,
            shards=2,
            stage_workers=2,
            stage_ring_capacity=256,
            shard_ring_capacity=256,
            node_capacity_per_shard=5000,
            edge_capacity_per_shard=10000,
            action_capacity_per_shard=512,
            snapshot_interval_seconds=9999.0,
            enable_snapshots=snapshots,
            restore=restore,
        )

    def populate(self, runtime: ContinuousMemoryRuntime, count: int = 100) -> None:
        # Recurrent lower-level contingencies are intentional: v8.2 may form M2+
        # only after accumulated lower-level evidence, not by direct raw-event projection.
        for index in range(count):
            context = index % 3
            action = index % 2
            runtime.submit(
                runtime.make_experience(
                    producer_id=1 + index % 4,
                    producer_sequence=1 + index // 4,
                    source_game_hash=1 + context,
                    global_step=index,
                    context_signature=10 + context,
                    action_id=action,
                    outcome_signature=100 + action,
                    family_signature=200 + action,
                    carrier_signature=300 + context,
                    future_option_delta=1.0,
                    changed_cells=1 + action,
                    trajectory_signature=400 + index % 11,
                )
            )

    def test_continuous_pipeline_populates_all_memory_levels(self) -> None:
        root = Path(tempfile.mkdtemp())
        runtime = ContinuousMemoryRuntime(self.config(root, snapshots=False, restore=False))
        try:
            runtime.start()
            self.populate(runtime, 120)
            runtime.wait_quiescent(timeout=20)
            counts = runtime.read_view.level_counts()
            for level in MemoryLevel:
                self.assertGreater(counts[int(level)], 0, f"M{int(level)} remained empty")
            self.assertGreater(runtime.read_view.edge_count, 0)
            self.assertEqual(runtime.metrics()["unsaved_tail"], runtime.watermark)
        finally:
            runtime.close(normal=False)

    def test_repeated_experience_collapses_to_canonical_memories(self) -> None:
        root = Path(tempfile.mkdtemp())
        runtime = ContinuousMemoryRuntime(self.config(root, snapshots=False, restore=False))
        try:
            runtime.start()
            for sequence in range(1, 11):
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
                        future_option_delta=1.0,
                        changed_cells=4,
                        trajectory_signature=400,
                    )
                )
            runtime.wait_quiescent(timeout=20)
            m1 = runtime.read_view.node_records(level=MemoryLevel.M1)
            self.assertEqual(len(m1), 1)
            self.assertEqual(m1[0].support_count, 10)
        finally:
            runtime.close(normal=False)

    def test_final_snapshot_restores_exact_ram_state(self) -> None:
        root = Path(tempfile.mkdtemp())
        config = self.config(root, snapshots=True, restore=True)
        runtime = ContinuousMemoryRuntime(config)
        runtime.start()
        self.populate(runtime, 80)
        runtime.wait_quiescent(timeout=20)
        before = runtime.read_view.state_digest()
        watermark = runtime.watermark
        final = runtime.close(normal=True, timeout=30)
        self.assertIsNotNone(final)
        self.assertEqual(final.watermark, watermark)
        self.assertTrue((root / "RUN_COMPLETE.json").is_file())

        restored = ContinuousMemoryRuntime(config)
        try:
            restored.start()
            after = restored.read_view.state_digest()
            self.assertEqual(before, after)
            self.assertEqual(restored.watermark, watermark)
            self.assertEqual(restored.metrics()["saved_watermark"], watermark)
        finally:
            restored.close(normal=False)


if __name__ == "__main__":
    unittest.main()

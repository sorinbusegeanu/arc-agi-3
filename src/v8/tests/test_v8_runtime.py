from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from v8.behavior_recovery import strategy_can_control
from v8.model import (
    CognitiveState,
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    ValidationState,
    decode_pipeline,
    encode_pipeline,
)
from v8.normalized_memory_v086 import is_grounded_contingency, is_normalized_contingency
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

    def test_continuous_pipeline_allows_probe_scaffold_without_prevalidation_control(self) -> None:
        root = Path(tempfile.mkdtemp())
        runtime = ContinuousMemoryRuntime(self.config(root, snapshots=False, restore=False))
        try:
            runtime.start()
            self.populate(runtime, 120)
            runtime.wait_quiescent(timeout=20)
            counts = runtime.read_view.level_counts()
            # Raw and structural evidence develops through M4. v8.8 may create
            # probationary M5-M7 descendants so M4 transfer can be tested without
            # circular dependence on an already validated M7 strategy.
            for level in (MemoryLevel.M0, MemoryLevel.M1, MemoryLevel.M2, MemoryLevel.M3, MemoryLevel.M4):
                self.assertGreater(counts[int(level)], 0, f"M{int(level)} remained empty")
            self.assertGreater(counts[int(MemoryLevel.M5)], 0, "M5 probe scaffold was not formed")

            higher = tuple(
                row
                for row in runtime.read_view.node_records()
                if int(row.level) >= int(MemoryLevel.M5)
            )
            self.assertTrue(higher)
            for row in higher:
                self.assertNotEqual(
                    int(row.validation_state),
                    int(ValidationState.VALIDATED),
                    f"M{int(row.level)} validated before empirical M4 transfer",
                )
                self.assertNotIn(
                    int(row.cognitive_state),
                    {int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)},
                    f"M{int(row.level)} became validated control before empirical M4 transfer",
                )
                if int(row.level) == int(MemoryLevel.M7) and len(row.key_parts) >= 3:
                    outcome_uid = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
                    self.assertFalse(
                        strategy_can_control(runtime.read_view, row.uid, outcome_uid),
                        "provisional M7 scaffold became normal control before M4 validation",
                    )

            self.assertGreater(runtime.read_view.edge_count, 0)
            self.assertEqual(runtime.metrics()["unsaved_tail"], runtime.watermark)
            normalization = runtime.metrics()["memory_normalization"]
            self.assertGreater(normalization["m1g_nodes"], 0)
            self.assertGreater(normalization["m1n_nodes"], 0)
            self.assertGreater(normalization["m2_from_m1n"], 0)
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
            grounded = [row for row in m1 if is_grounded_contingency(row)]
            normalized = [row for row in m1 if is_normalized_contingency(row)]
            self.assertEqual(len(grounded), 1)
            self.assertEqual(grounded[0].support_count, 10)
            self.assertGreaterEqual(len(normalized), 1)
            self.assertTrue(all(row.support_count == 10 for row in normalized))
        finally:
            runtime.close(normal=False)

    def test_final_snapshot_restores_exact_ram_state(self) -> None:
        root = Path(tempfile.mkdtemp())
        config = self.config(root, snapshots=True, restore=True)
        runtime = ContinuousMemoryRuntime(config)
        runtime.start()
        self.populate(runtime, 80)
        runtime.wait_quiescent(timeout=20)
        watermark = runtime.watermark
        final = runtime.close(normal=True, timeout=30)
        self.assertIsNotNone(final)
        self.assertEqual(final.watermark, watermark)
        self.assertTrue((root / "RUN_COMPLETE.json").is_file())

        manifest = json.loads((Path(final.path) / "manifest.json").read_text(encoding="utf-8"))
        restored = ContinuousMemoryRuntime(config)
        try:
            restored.start()
            for shard_manifest, nodes, edges, actions in zip(
                manifest["shards"],
                restored.read_view._nodes,
                restored.read_view._edges,
                restored.read_view._actions,
                strict=True,
            ):
                for label, arena in (("nodes", nodes), ("edges", edges), ("actions", actions)):
                    digest = hashlib.sha256(arena.snapshot_bytes()).hexdigest()
                    self.assertEqual(digest, shard_manifest[label]["sha256"])
            self.assertEqual(restored.watermark, watermark)
            self.assertEqual(restored.metrics()["saved_watermark"], watermark)
        finally:
            restored.close(normal=False)

    def test_periodic_snapshot_drains_writes_without_advancing_peer_fixed_point(self) -> None:
        runtime = object.__new__(ContinuousMemoryRuntime)
        runtime.snapshot_service = Mock()
        runtime._maintenance_lock = threading.Lock()
        runtime._snapshot_freeze = threading.Event()
        runtime._snapshot_id = 4
        runtime._watermark = SimpleNamespace(value=11, get_lock=nullcontext)
        runtime._generation = SimpleNamespace(value=17, get_lock=nullcontext)
        runtime._auxiliary_state_json = Mock(return_value="{}")
        runtime.wait_quiescent = Mock()
        runtime.peers = Mock()
        runtime.peers.wait_idle.return_value = True

        runtime.request_consistent_snapshot(timeout=2.0)

        runtime.peers.pause.assert_called_once_with()
        runtime.peers.wait_idle.assert_called_once()
        runtime.wait_quiescent.assert_called_once()
        self.assertFalse(runtime.wait_quiescent.call_args.kwargs["settle_peers"])
        self.assertFalse(runtime.wait_quiescent.call_args.kwargs["resume_peers"])
        runtime.snapshot_service.request_consistent_capture.assert_called_once()
        runtime.peers.resume.assert_called_once_with()
        self.assertFalse(runtime._snapshot_freeze.is_set())


if __name__ == "__main__":
    unittest.main()

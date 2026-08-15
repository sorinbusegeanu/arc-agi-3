from __future__ import annotations

import tempfile
import unittest

from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    ValidationState,
    proposal_fingerprint,
)
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


class RetiredCompactionTests(unittest.TestCase):
    def test_retired_node_reclaims_ram_and_shard_continues(self) -> None:
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
            runtime.start()
            key = (999, 1)
            uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, key)
            runtime.submit_proposal(
                MemoryProposal(
                    uid=uid,
                    fingerprint=proposal_fingerprint(MemoryLevel.M4, MemoryType.CONCEPT, key),
                    event_id=EventId.from_producer(99, 1),
                    watermark=1,
                    level=MemoryLevel.M4,
                    memory_type=MemoryType.CONCEPT,
                    key_parts=key,
                    cognitive_state=int(CognitiveState.RETIRED),
                    validation_state=int(ValidationState.STRUCTURAL),
                )
            )
            runtime.wait_quiescent(timeout=10)
            self.assertTrue(runtime.read_view.has_uid(uid))
            before = runtime.read_view.memory_count

            result = runtime.compact_retired_memory(timeout=10)
            self.assertEqual(result.retired_nodes, 1)
            self.assertEqual(runtime.read_view.memory_count, before - 1)
            self.assertFalse(runtime.read_view.has_uid(uid))
            self.assertTrue((runtime.root / "archive" / "retired_memory.jsonl").is_file())

            key2 = (1000, 1)
            uid2 = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, key2)
            runtime.submit_proposal(
                MemoryProposal(
                    uid=uid2,
                    fingerprint=proposal_fingerprint(MemoryLevel.M4, MemoryType.CONCEPT, key2),
                    event_id=EventId.from_producer(99, 2),
                    watermark=2,
                    level=MemoryLevel.M4,
                    memory_type=MemoryType.CONCEPT,
                    key_parts=key2,
                )
            )
            runtime.wait_quiescent(timeout=10)
            self.assertTrue(runtime.read_view.has_uid(uid2))
            runtime.close(normal=True, timeout=10)


if __name__ == "__main__":
    unittest.main()

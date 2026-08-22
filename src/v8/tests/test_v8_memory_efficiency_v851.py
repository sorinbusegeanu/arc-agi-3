from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from v8 import capacity
from v8 import memory_efficiency_v851 as memory
from v8 import memory_efficiency_v851_fixups as fixups
from v8 import memory_efficiency_v851_integrity as integrity
from v8 import memory_storage_v851 as storage
from v8 import runtime_stack_v88
from v8.actor_read_view_v851 import ActorReadView
from v8.arena import ActionRecord, SharedActionArena, SharedEdgeArena, SharedNodeArena
from v8.evidence import EvidenceRecord
from v8.evidence_memory_v851 import DiskBackedEvidenceLedger, _ROOT_ENV
from v8.model import MemoryLevel, MemoryUid, ValidationState
from v8.publication import ShardReadDescriptor


class MemoryEfficiencyV851Tests(unittest.TestCase):
    def test_runtime_stack_installs_v851_layers_last(self):
        self.assertEqual(
            runtime_stack_v88._POST_LAYERS[-3:],
            (
                "memory_efficiency_v851",
                "memory_efficiency_v851_fixups",
                "memory_efficiency_v851_integrity",
            ),
        )
        self.assertTrue(memory._INSTALLED)
        self.assertTrue(fixups._INSTALLED)
        self.assertTrue(integrity._INSTALLED)
        self.assertTrue(storage._INSTALLED)

    def test_actor_read_view_does_not_retain_full_record_cut_cache(self):
        nodes = SharedNodeArena(capacity=16)
        edges = SharedEdgeArena(capacity=16)
        actions = SharedActionArena(capacity=16)
        descriptor = ShardReadDescriptor(nodes.descriptor, edges.descriptor, actions.descriptor)
        view = None
        try:
            view = ActorReadView((descriptor,), refresh_interval_seconds=None)
            self.assertEqual(view._record_cache, {})
            self.assertEqual(view.node_records(), ())
            self.assertEqual(view.edge_records(), ())
            self.assertEqual(view._record_cache, {})
        finally:
            if view is not None:
                view.close()
            nodes.dispose()
            edges.dispose()
            actions.dispose()

    def test_action_arena_snapshot_rehashes_when_capacity_changes(self):
        source = SharedActionArena(capacity=16)
        target = SharedActionArena(capacity=32)
        record = ActionRecord(1234567, 3, 7, 4.5, 7.0, 99)
        try:
            storage._action_insert(source, record)
            payload = source.snapshot_bytes()
            target.load_snapshot(payload)
            restored = target.lookup(record.context_signature, record.action_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.context_signature, record.context_signature)
            self.assertEqual(restored.action_id, record.action_id)
            self.assertEqual(restored.support_count, record.support_count)
        finally:
            source.dispose()
            target.dispose()

    def test_disk_evidence_is_bounded_and_rolls_back_to_restore_watermark(self):
        prior = os.environ.get(_ROOT_ENV)
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[_ROOT_ENV] = tmp
            ledger = DiskBackedEvidenceLedger()
            try:
                uid = MemoryUid(1, 2)
                first = EvidenceRecord.for_uid(
                    "e1",
                    uid,
                    evidence_kind="test",
                    watermark=5,
                    raw_value=1.0,
                    normalized_value=1.0,
                    developmental_stage=int(MemoryLevel.M4),
                    validation_state=int(ValidationState.VALIDATED),
                    causal_intervention="matched_test",
                    effect_direction=1,
                )
                second = EvidenceRecord.for_uid(
                    "e2",
                    uid,
                    evidence_kind="test",
                    watermark=10,
                    raw_value=1.0,
                    normalized_value=1.0,
                    developmental_stage=int(MemoryLevel.M4),
                    validation_state=int(ValidationState.VALIDATED),
                )
                self.assertTrue(ledger.append(first))
                self.assertTrue(ledger.append(second))
                self.assertEqual(ledger.count(), 2)
                self.assertEqual(ledger.state_dict()["records"], [])
                self.assertIn(uid, ledger.positive_effect_uids())
                self.assertIn(uid, ledger.protected_uids())
                self.assertEqual(ledger.truncate_after(5), 1)
                self.assertEqual(ledger.count(), 1)
                self.assertEqual(tuple(row.evidence_id for row in ledger.cut(5)), ("e1",))
            finally:
                ledger.close()
        if prior is None:
            os.environ.pop(_ROOT_ENV, None)
        else:
            os.environ[_ROOT_ENV] = prior

    def test_noncausal_observation_does_not_pin_memory_forever(self):
        prior = os.environ.get(_ROOT_ENV)
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[_ROOT_ENV] = tmp
            ledger = DiskBackedEvidenceLedger()
            try:
                uid = MemoryUid(3, 4)
                row = EvidenceRecord.for_uid(
                    "weak",
                    uid,
                    evidence_kind="recurrence",
                    watermark=1,
                    raw_value=0.1,
                    normalized_value=0.1,
                    developmental_stage=int(MemoryLevel.M2),
                    validation_state=int(ValidationState.STRUCTURAL),
                )
                ledger.append(row)
                self.assertNotIn(uid, ledger.protected_uids())
            finally:
                ledger.close()
        if prior is None:
            os.environ.pop(_ROOT_ENV, None)
        else:
            os.environ[_ROOT_ENV] = prior

    def test_capacity_plan_does_not_keep_historical_capacity_or_large_legacy_floor(self):
        historical = capacity.SnapshotUsage(
            node_count=100,
            edge_count=200,
            node_capacity=9_000_000,
            edge_capacity=18_000_000,
            action_capacity=1_000_000,
        )
        with patch.object(capacity, "snapshot_usage", return_value=historical):
            plan = capacity.plan_capacities(
                total_steps=0,
                shards=1,
                root="unused",
                restore=True,
            )
        self.assertEqual(plan.node_capacity_per_shard, integrity._MIN_NODE_CAPACITY)
        self.assertEqual(plan.edge_capacity_per_shard, integrity._MIN_EDGE_CAPACITY)
        self.assertEqual(plan.action_capacity_per_shard, integrity._MIN_ACTION_CAPACITY)
        self.assertLess(plan.node_capacity_per_shard, historical.node_capacity)
        self.assertLess(plan.edge_capacity_per_shard, historical.edge_capacity)
        self.assertLess(plan.action_capacity_per_shard, historical.action_capacity)

    def test_real_process_memory_probe_is_available(self):
        row = memory._smaps_rollup(os.getpid())
        self.assertGreaterEqual(row["rss_bytes"], 0)
        self.assertGreaterEqual(row["pss_bytes"], 0)
        self.assertGreaterEqual(row["uss_bytes"], 0)
        system = memory._system_memory()
        self.assertIn("used_pct", system)


if __name__ == "__main__":
    unittest.main()

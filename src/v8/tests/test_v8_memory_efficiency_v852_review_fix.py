from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from v8 import actor_read_view_v851 as actor_read
from v8 import capacity
from v8 import memory_efficiency_v851 as memory
from v8 import memory_efficiency_v851_integrity as integrity
from v8 import memory_efficiency_v852_review_fix as v852
from v8 import publication
from v8 import runtime_stack_v88
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState


class MemoryEfficiencyV852ReviewFixTests(unittest.TestCase):
    def test_v852_precedes_v853_final_layer(self):
        self.assertEqual(
            runtime_stack_v88._POST_LAYERS[-2:],
            (
                "memory_efficiency_v852_review_fix",
                "actor_throughput_v853",
            ),
        )
        self.assertTrue(v852._INSTALLED)

    def test_actor_compact_view_refreshes_only_when_invalidated(self):
        original = v852._BASE_ACTOR_REFRESH
        calls = []

        def fake_base(instance):
            calls.append(instance)

        dummy = SimpleNamespace(
            _strategy_cache_stale=False,
            _refresh_interval_seconds=None,
            _strategy_version=(2, 2),
            _next_strategy_refresh=0.0,
        )
        try:
            v852._BASE_ACTOR_REFRESH = fake_base
            v852._actor_refresh_strategy_cache_v852(dummy)
            self.assertEqual(calls, [])
            dummy._strategy_cache_stale = True
            v852._actor_refresh_strategy_cache_v852(dummy)
            self.assertEqual(calls, [dummy])
        finally:
            v852._BASE_ACTOR_REFRESH = original

    def test_actor_compact_session_cut_defers_live_graph_invalidation(self):
        original = v852._BASE_ACTOR_REFRESH
        calls = []
        dummy = SimpleNamespace(
            _v851_ready=True,
            _strategy_cache_stale=True,
            _refresh_interval_seconds=None,
            _strategy_version=(20, 40),
        )
        try:
            v852._BASE_ACTOR_REFRESH = lambda instance: calls.append(instance)
            v852._actor_refresh_strategy_cache_v852(dummy)
        finally:
            v852._BASE_ACTOR_REFRESH = original
        self.assertEqual(calls, [])
        self.assertFalse(dummy._strategy_cache_stale)

    def test_depends_on_is_strategy_lineage(self):
        relation = int(RelationType.DEPENDS_ON)
        self.assertIn(relation, publication._LINEAGE_RELATIONS)
        self.assertIn(relation, actor_read._LINEAGE_RELATIONS)

    def test_failed_or_quarantined_ancestor_never_enables_transfer(self):
        child = MemoryUid(1, 1)
        parent = MemoryUid(2, 2)
        dummy = SimpleNamespace(
            _parents={child: {parent}},
            _node_by_uid={},
        )

        def row(*, cognitive, validation, games=5):
            return SimpleNamespace(
                level=int(MemoryLevel.M4),
                cognitive_state=int(cognitive),
                validation_state=int(validation),
                game_evidence_count=int(games),
            )

        dummy._node_by_uid[parent] = row(
            cognitive=CognitiveState.ACTIVE,
            validation=ValidationState.FAILED,
        )
        self.assertFalse(v852._has_transferable_ancestor_v852(dummy, child))

        dummy._node_by_uid[parent] = row(
            cognitive=CognitiveState.QUARANTINED,
            validation=ValidationState.VALIDATED,
        )
        self.assertFalse(v852._has_transferable_ancestor_v852(dummy, child))

        dummy._node_by_uid[parent] = row(
            cognitive=CognitiveState.ACTIVE,
            validation=ValidationState.VALIDATED,
            games=1,
        )
        self.assertTrue(v852._has_transferable_ancestor_v852(dummy, child))

    def test_transferable_ancestor_cache_is_reused_and_graph_scoped(self):
        child = MemoryUid(1, 1)
        parent = MemoryUid(2, 2)

        class GuardedParents(dict):
            fail = False

            def get(self, *args, **kwargs):
                if self.fail:
                    raise AssertionError("cached query rescanned lineage")
                return super().get(*args, **kwargs)

        parents = GuardedParents({child: {parent}})
        dummy = SimpleNamespace(
            _parents=parents,
            _node_by_uid={
                parent: SimpleNamespace(
                    level=int(MemoryLevel.M4),
                    cognitive_state=int(CognitiveState.ACTIVE),
                    validation_state=int(ValidationState.VALIDATED),
                    game_evidence_count=2,
                )
            },
            _strategy_version=(),
        )

        self.assertTrue(v852._has_transferable_ancestor_v852(dummy, child))

        # Keep the graph identity stable to prove the result is reused.
        parents.fail = True
        self.assertTrue(v852._has_transferable_ancestor_v852(dummy, child))

        # Publication replaces the index objects on each coherent graph cut.
        dummy._parents = {}
        self.assertFalse(v852._has_transferable_ancestor_v852(dummy, child))

    def test_restored_action_capacity_does_not_grow_from_table_size_alone(self):
        historical = capacity.SnapshotUsage(
            node_count=100,
            edge_count=200,
            node_capacity=100_000,
            edge_capacity=200_000,
            action_capacity=65_536,
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

    def test_snapshot_retention_keeps_latest_and_run_complete_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            chunks = root / "snapshot_chunks"
            snapshots.mkdir()
            chunks.mkdir()
            for index in range(1, 6):
                name = f"snapshot-{index:020d}"
                snapshot = snapshots / name
                snapshot.mkdir()
                digest = f"chunk{index}"
                (chunks / f"{digest}.bin").write_bytes(bytes([index]))
                manifest = {
                    "shards": [
                        {
                            "nodes": {"chunks": [{"sha256": digest, "bytes": 1}]},
                            "edges": {"chunks": []},
                            "actions": {"chunks": []},
                        }
                    ]
                }
                (snapshot / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                (snapshot / "COMPLETE").write_text("ok\n", encoding="ascii")
            (root / "RUN_COMPLETE.json").write_text(
                json.dumps({"snapshot_id": 1}), encoding="utf-8"
            )

            v852._prune_snapshot_storage(root, retain=2)

            remaining = {path.name for path in snapshots.glob("snapshot-*")}
            self.assertEqual(
                remaining,
                {
                    "snapshot-00000000000000000001",
                    "snapshot-00000000000000000004",
                    "snapshot-00000000000000000005",
                },
            )
            remaining_chunks = {path.stem for path in chunks.glob("*.bin")}
            self.assertEqual(remaining_chunks, {"chunk1", "chunk4", "chunk5"})

    def test_compaction_preflight_fails_before_mutation(self):
        retired_uid = MemoryUid(10, 10)
        active_uid = MemoryUid(20, 20)
        game_uid = MemoryUid(0, 999)
        retired = SimpleNamespace(
            uid=retired_uid,
            cognitive_state=int(CognitiveState.RETIRED),
            updated_watermark=1,
            level=int(MemoryLevel.M0),
            memory_type=int(MemoryType.EPISODE),
            key_parts=(1,),
        )
        active = SimpleNamespace(
            uid=active_uid,
            cognitive_state=int(CognitiveState.ACTIVE),
            updated_watermark=2,
            level=int(MemoryLevel.M4),
            memory_type=int(MemoryType.CONCEPT),
            key_parts=(2,),
        )
        provenance = SimpleNamespace(
            source_uid=retired_uid,
            relation_type=int(RelationType.GAME_PROVENANCE),
            target_uid=game_uid,
            support_count=1,
            updated_watermark=1,
        )
        supersedes = SimpleNamespace(
            source_uid=active_uid,
            relation_type=int(RelationType.SUPERSEDES),
            target_uid=retired_uid,
            support_count=1,
            updated_watermark=2,
        )

        class FakeNodeArena:
            count = 2

            def __init__(self):
                self.begin_calls = 0

            def records(self):
                return (retired, active)

            def begin_write(self):
                self.begin_calls += 1

            def close(self):
                return None

        class FakeEdgeArena:
            count = 2
            capacity = 0

            def __init__(self):
                self.begin_calls = 0

            def records(self):
                return (provenance, supersedes)

            def begin_write(self):
                self.begin_calls += 1

            def close(self):
                return None

        nodes = FakeNodeArena()
        edges = FakeEdgeArena()
        descriptor = SimpleNamespace(nodes="nodes", edges="edges")
        with tempfile.TemporaryDirectory() as tmp:
            from v8.arena import SharedEdgeArena, SharedNodeArena

            with (
                patch.object(SharedNodeArena, "attach", return_value=nodes),
                patch.object(SharedEdgeArena, "attach", return_value=edges),
            ):
                with self.assertRaises(MemoryError):
                    v852._compact_retired_memory_v852(
                        (descriptor,),
                        archive_path=Path(tmp) / "archive.jsonl",
                    )
        self.assertEqual(nodes.begin_calls, 0)
        self.assertEqual(edges.begin_calls, 0)

    def test_runtime_compaction_failure_restarts_shard_workers(self):
        original = v852._BASE_RUNTIME_COMPACT

        class Stop:
            def is_set(self):
                return False

        class Process:
            def __init__(self):
                self.started = False
                self.alive = False

            def start(self):
                self.started = True
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                return None

            def terminate(self):
                self.alive = False

        built = []
        runtime = SimpleNamespace(
            _started=True,
            _closed=False,
            _stop=Stop(),
            config=SimpleNamespace(shards=1),
            _shard_processes=[],
        )

        def build(_shard_id):
            process = Process()
            built.append(process)
            return process

        runtime._build_shard_process = build

        def fail(_runtime, *args, **kwargs):
            raise RuntimeError("boom")

        try:
            v852._BASE_RUNTIME_COMPACT = fail
            with self.assertRaisesRegex(RuntimeError, "boom"):
                v852._runtime_compact_v852(runtime)
        finally:
            v852._BASE_RUNTIME_COMPACT = original
        self.assertEqual(len(built), 1)
        self.assertTrue(built[0].started)

    def test_mp_resource_helpers_close_queues_and_process_handles(self):
        queue = SimpleNamespace(close=Mock(), join_thread=Mock())
        process = SimpleNamespace(is_alive=Mock(return_value=False), close=Mock())
        v852._close_mp_queue(queue)
        v852._close_process_handle(process)
        queue.close.assert_called_once_with()
        queue.join_thread.assert_called_once_with()
        process.close.assert_called_once_with()

    def test_adaptive_actor_memory_monitor_preserves_public_actor_authority(self):
        original = v852._BASE_ADAPTIVE_WORKER
        writes = []

        def base_worker(**kwargs):
            return "done"

        def write_actor_memory(root, job, *, peak_pss, peak_uss, finished):
            writes.append((Path(root), int(job.actor_id), bool(finished)))
            return int(peak_pss), int(peak_uss)

        with tempfile.TemporaryDirectory() as tmp:
            try:
                v852._BASE_ADAPTIVE_WORKER = base_worker
                with patch.object(memory, "_write_actor_memory", side_effect=write_actor_memory):
                    result = v852._adaptive_worker_v852(
                        worker_id=7,
                        trajectory_root=str(Path(tmp) / "trajectory_optimizer"),
                    )
            finally:
                v852._BASE_ADAPTIVE_WORKER = original
        self.assertEqual(result, "done")
        self.assertGreaterEqual(len(writes), 2)
        self.assertEqual(writes[0][1], 7)
        self.assertTrue(writes[-1][2])


if __name__ == "__main__":
    unittest.main()

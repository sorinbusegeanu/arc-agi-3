from pathlib import Path

import pytest

from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.mutation.proposals import MutationKind, MutationProposal, MutationWrite
from v9.mutation.read_sets import ReadSet
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.publication import CanonicalGraph
from v9.runtime.canonical_store import canonical_value
from v9.runtime.canonical_commit import apply_canonical_commit_batch
from v9.benchmarks.ingestion_drain import _plans


def _proposal(partitions: int = 1, *, ordinal: int = 0) -> MutationProposal:
    uid = MemoryUid(7, 9 + int(ordinal))
    return MutationProposal.build(
        MutationKind.UPSERT_NODE,
        target_partitions=(uid.shard(partitions),),
        read_set=ReadSet.build((), maximum_size=8),
        evidence_refs=(),
        causal_watermark=1,
        writes=(
            MutationWrite(
                node=CanonicalNode(uid, MemoryLevel.M2, MemoryType.FAMILY, (11,), 1),
                payload={"support": 1, "ordinal": int(ordinal)},
            ),
        ),
    )


def test_durability_failure_prevents_live_graph_visibility() -> None:
    graph = CanonicalGraph(1)

    def fail(**_kwargs: object) -> None:
        raise OSError("injected fsync failure")

    graph.configure_durable_commit(fail)
    with pytest.raises(OSError, match="injected fsync failure"):
        graph.publish(_proposal())
    assert graph.generation == 0
    assert graph.nodes == {}


def test_runtime_graph_publication_advances_wal_store_then_visibility(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path,
            restore=False,
            enable_snapshots=False,
            enable_canonical_durability=True,
        )
    )
    result = runtime.graph.publish(_proposal(runtime.graph.partition_count))
    assert result.outcome.value == "ACCEPTED"
    assert runtime.persistence_frontiers.wal_durable_lsn == 1
    assert runtime.persistence_frontiers.canonical_applied_lsn == 1
    assert runtime.canonical_state_handle.generation == 1
    assert MemoryUid(7, 9) in runtime.graph.nodes


def test_runtime_restores_exact_canonical_snapshot_and_wal_frontier(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, restore=False, enable_canonical_durability=True)
    )
    runtime.graph.publish(_proposal(runtime.graph.partition_count))
    expected = runtime.canonical_state_handle
    runtime.snapshot()

    restored = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, enable_canonical_durability=True)
    )
    assert restored.canonical_state_handle == expected
    assert restored.persistence_frontiers.wal_durable_lsn == 1
    assert restored.persistence_frontiers.snapshot_applied_lsn == 1
    assert MemoryUid(7, 9) in restored.graph.nodes


def test_runtime_reclaims_only_snapshot_and_hgt_acknowledged_wal(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, restore=False, enable_canonical_durability=True)
    )
    for ordinal in range(3):
        runtime.graph.publish(_proposal(runtime.graph.partition_count, ordinal=ordinal))
    runtime.snapshot()
    assert runtime.reclaim_canonical_wal() == 0

    runtime.advance_hgt_checkpoint(2)
    assert runtime.reclaim_canonical_wal() == 2
    assert tuple(frame.wal_lsn for frame in runtime.canonical_wal.recover().frames) == (3,)
    restored = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, enable_canonical_durability=True)
    )
    assert restored.canonical_wal.wal_reclaim_lsn == 2
    assert restored.persistence_frontiers.hgt_checkpoint_lsn == 2


def test_runtime_restores_preserved_snapshot_after_interrupted_directory_swap(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, restore=False, enable_canonical_durability=True)
    )
    runtime.graph.publish(_proposal(runtime.graph.partition_count))
    runtime.snapshot()
    expected = runtime.canonical_state_handle
    snapshot = max((tmp_path / "canonical" / "snapshots").glob("snapshot-*"))
    preserved = snapshot.with_name(f".{snapshot.name}.previous")
    snapshot.rename(preserved)

    restored = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path, enable_canonical_durability=True)
    )
    assert restored.canonical_state_handle == expected
    assert restored.persistence_frontiers.snapshot_applied_lsn == 1


def test_ingestion_tail_keeps_compatibility_graph_equal_to_immutable_root(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path,
            restore=False,
            enable_snapshots=False,
            enable_canonical_durability=True,
        )
    )
    apply_canonical_commit_batch(runtime, _plans(20))
    runtime.flush_deferred_memory_updates()
    payloads = runtime.canonical_store.collection(runtime.canonical_state_handle, "payload")
    graph_rows = runtime.canonical_store.collection(runtime.canonical_state_handle, "graph")
    assert len(graph_rows) == len(runtime.graph.nodes) + len(runtime.graph.edges)
    for uid, payload in runtime.graph.payloads.items():
        key = f"node:{uid.hi:016x}{uid.lo:016x}"
        assert canonical_value(payloads[key]) == canonical_value(payload)


def test_resident_deletion_is_durable_before_visibility(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path,
            restore=False,
            enable_snapshots=False,
            enable_canonical_durability=True,
        )
    )
    target = MemoryUid(11, 12)
    replacement = MemoryUid(21, 22)
    runtime._publish(
        CanonicalNode(target, MemoryLevel.M0, MemoryType.EPISODE, (1,), 1),
        {"action_id": 1},
        (target,),
    )
    runtime._publish(
        CanonicalNode(replacement, MemoryLevel.M2, MemoryType.FAMILY, (2,), 2),
        {"parents": [[target.hi, target.lo]]},
        (target,),
    )
    assert runtime.graph.delete_low_level_nodes_batch(
        ((target, replacement, "durable-test"),)
    ) == (target,)
    handle = runtime.canonical_state_handle
    key = f"node:{target.hi:016x}{target.lo:016x}"
    assert target not in runtime.graph.nodes
    assert runtime.canonical_store.read_from_handle(handle, "graph", key) is None
    assert runtime.persistence_frontiers.wal_durable_lsn == handle.canonical_applied_lsn


def test_resident_deletion_fsync_failure_keeps_complete_old_handle(tmp_path: Path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path,
            restore=False,
            enable_snapshots=False,
            enable_canonical_durability=True,
        )
    )
    target = MemoryUid(31, 32)
    replacement = MemoryUid(41, 42)
    runtime._publish(
        CanonicalNode(target, MemoryLevel.M0, MemoryType.EPISODE, (1,), 1),
        {"action_id": 1},
        (target,),
    )
    runtime._publish(
        CanonicalNode(replacement, MemoryLevel.M2, MemoryType.FAMILY, (2,), 2),
        {"parents": [[target.hi, target.lo]]},
        (target,),
    )
    old_handle = runtime.canonical_state_handle

    def fail(_descriptor: int) -> None:
        raise OSError("injected delete fsync failure")

    runtime.canonical_wal._fsync = fail
    with pytest.raises(OSError, match="injected delete fsync failure"):
        runtime.graph.delete_low_level_nodes_batch(((target, replacement, "crash"),))
    assert target in runtime.graph.nodes
    assert runtime.canonical_state_handle == old_handle

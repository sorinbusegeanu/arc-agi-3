from __future__ import annotations

import pytest

from v9.runtime.storage_governor import DurableStorageClass, StorageGovernor, StorageObject


def _ceilings(value: int = 1000):
    return {kind: value for kind in DurableStorageClass}


def test_per_class_and_aggregate_storage_ceilings(tmp_path) -> None:
    governor = StorageGovernor(tmp_path, class_ceilings=_ceilings(100), aggregate_ceiling=150, minimum_free_bytes=0, minimum_free_fraction=0)
    path = tmp_path / "wal"
    path.write_bytes(b"x" * 80)
    governor.register(StorageObject("wal-1", DurableStorageClass.WAL, path, 80))
    with pytest.raises(OverflowError, match="wal"):
        governor.register(StorageObject("wal-2", DurableStorageClass.WAL, tmp_path / "wal2", 30))


def test_zero_byte_storage_objects_cannot_grow_without_bound(tmp_path) -> None:
    governor = StorageGovernor(
        tmp_path,
        class_ceilings=_ceilings(),
        aggregate_ceiling=5000,
        minimum_free_bytes=0,
        minimum_free_fraction=0,
        max_objects=1,
    )
    governor.register(StorageObject("one", DurableStorageClass.WAL, tmp_path / "one", 0))
    with pytest.raises(OverflowError, match="object-count"):
        governor.register(StorageObject("two", DurableStorageClass.WAL, tmp_path / "two", 0))


def test_restart_reconciles_every_durable_storage_class_from_filesystem(tmp_path) -> None:
    files = {
        "canonical/commit.wal": DurableStorageClass.WAL,
        "canonical/chunks/chunk.bin": DurableStorageClass.CANONICAL_CHUNK,
        "snapshots/snapshot-1/manifest.json": DurableStorageClass.SNAPSHOT,
        "hgt/training_evidence/segments/segment-1.jsonl": DurableStorageClass.TRAINING_EVIDENCE,
        "models/hgt-1.pt": DurableStorageClass.MODEL_CHECKPOINT,
        "models/optimizer-1.pt": DurableStorageClass.OPTIMIZER_CHECKPOINT,
        "indexes/signatures.sqlite": DurableStorageClass.SIGNATURE_INDEX,
        "hgt/training_cuts/cut-1.json": DurableStorageClass.POLICY_VIEW_MANIFEST,
        "reports/result.json": DurableStorageClass.EVALUATION_ARTIFACT,
    }
    for relative in files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    governor = StorageGovernor(
        tmp_path,
        class_ceilings=_ceilings(100),
        aggregate_ceiling=1000,
        minimum_free_bytes=0,
        minimum_free_fraction=0,
    )

    governor.reconcile_filesystem()

    assert dict(governor.status().bytes_by_class) == {
        kind.value: 1 for kind in DurableStorageClass
    }


def test_reconcile_preserves_live_references_and_reclaimability(tmp_path) -> None:
    path = tmp_path / "canonical" / "chunks" / "chunk.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"first")
    governor = StorageGovernor(
        tmp_path,
        class_ceilings=_ceilings(100),
        aggregate_ceiling=1000,
        minimum_free_bytes=0,
        minimum_free_fraction=0,
    )
    governor.reconcile_filesystem()
    object_id = "canonical/chunks/chunk.bin"
    governor.add_reference(object_id, "snapshot-1")
    governor.mark_reclaimable(object_id)
    path.write_bytes(b"replacement")

    governor.reconcile_filesystem()

    assert governor.collect() == ()
    assert path.exists()
    assert dict(governor.status().bytes_by_class)[DurableStorageClass.CANONICAL_CHUNK.value] == len(b"replacement")
    governor.release_reference(object_id, "snapshot-1")
    assert governor.collect() == (object_id,)

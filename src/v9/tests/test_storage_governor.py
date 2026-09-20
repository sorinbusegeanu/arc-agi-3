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

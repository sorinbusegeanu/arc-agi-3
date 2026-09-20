from __future__ import annotations

from v9.runtime.storage_governor import DurableStorageClass, StorageGovernor, StorageObject


def test_storage_gc_never_deletes_referenced_object(tmp_path) -> None:
    governor = StorageGovernor(tmp_path, class_ceilings={kind: 1000 for kind in DurableStorageClass}, aggregate_ceiling=5000, minimum_free_bytes=0, minimum_free_fraction=0)
    path = tmp_path / "chunk"
    path.write_bytes(b"chunk")
    governor.register(StorageObject("chunk", DurableStorageClass.CANONICAL_CHUNK, path, 5, frozenset(("epoch-view",)), True))
    assert governor.collect() == ()
    governor.release_reference("chunk", "epoch-view")
    assert governor.collect() == ("chunk",)
    assert not path.exists()

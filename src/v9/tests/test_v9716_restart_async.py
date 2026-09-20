from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_wal import CanonicalCommitWAL, recover_canonical_store


def test_async_snapshot_plus_wal_recovers_complete_handles(tmp_path) -> None:
    identity = {"experiment_id": "e", "mode": "ASYNC_DEVELOPMENT"}
    store = CanonicalStore()
    first = store.begin_overlay(1)
    first.put("graph", "a", 1)
    store.finalize_overlay(first)
    snapshot = store.write_snapshot(tmp_path / "snapshot", scientific_identity=identity)
    wal = CanonicalCommitWAL(tmp_path / "canonical.wal")
    wal.append_group(({"transaction_id": "already-snapshotted"},))
    wal.append_group(({"transaction_id": "next", "mutations": ({"collection": "graph", "key": "b", "value": 2},)},))
    recovered = recover_canonical_store(snapshot_path=snapshot, wal=wal, expected_scientific_identity=identity)
    assert recovered.replayed_lsns == (2,)
    assert recovered.store.read_from_handle(recovered.store.current_handle, "graph", "a") == 1
    assert recovered.store.read_from_handle(recovered.store.current_handle, "graph", "b") == 2

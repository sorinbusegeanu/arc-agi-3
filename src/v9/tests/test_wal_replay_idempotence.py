from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_wal import CanonicalCommitWAL


def test_wal_is_durable_before_overlay_becomes_visible(tmp_path) -> None:
    store = CanonicalStore()
    wal = CanonicalCommitWAL(tmp_path / "canonical.wal")
    overlay = store.begin_overlay(1)
    overlay.put("graph", "node", "value")
    observed = []

    def after_durable(frame) -> None:
        observed.append((wal.wal_durable_lsn, store.current_handle.canonical_applied_lsn, frame.wal_lsn))

    handle = wal.commit_overlay(store, overlay, transaction_id="tx-1", after_durable=after_durable)
    assert observed == [(1, 0, 1)]
    assert handle.canonical_applied_lsn == 1
    assert store.read_from_handle(handle, "graph", "node") == "value"


def test_post_wal_pre_publish_recovery_replays_once(tmp_path) -> None:
    wal = CanonicalCommitWAL(tmp_path / "canonical.wal")
    wal.append_group(({"transaction_id": "tx-1", "mutations": ({"collection": "graph", "key": "node", "value": 1},)},))
    store = CanonicalStore()
    for frame in wal.recover().frames:
        if frame.wal_lsn <= store.current_handle.canonical_applied_lsn:
            continue
        overlay = store.begin_overlay(frame.wal_lsn)
        for mutation in frame.mutations:
            overlay.put(mutation["collection"], mutation["key"], mutation["value"])
        store.finalize_overlay(overlay)
    assert store.current_handle.canonical_applied_lsn == 1
    assert store.read_from_handle(store.current_handle, "graph", "node") == 1

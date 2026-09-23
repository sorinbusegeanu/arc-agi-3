from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.storage_governor import CanonicalSnapshotPin


def test_snapshot_manifest_records_exact_pinned_applied_lsn() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(4)
    overlay.put("graph", "a", 1)
    handle = store.finalize_overlay(overlay)
    with CanonicalSnapshotPin(store) as snapshot:
        manifest = snapshot.manifest()
        assert manifest["snapshot_applied_lsn"] == 4
        assert manifest["canonical_handle_checksum"] == handle.checksum

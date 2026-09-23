from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore


def test_snapshot_restores_exact_handle_root_frontier_checksum(tmp_path) -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(3)
    overlay.put("payload", "node", {"value": 1})
    expected = store.finalize_overlay(overlay)
    path = store.write_snapshot(tmp_path / "snapshot")
    restored = CanonicalStore.from_snapshot(path)
    assert restored.current_handle == expected
    assert restored.read_from_handle(expected, "payload", "node") == {"value": 1}

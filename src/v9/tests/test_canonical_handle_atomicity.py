from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore


def test_reader_pins_one_root_frontier_generation() -> None:
    store = CanonicalStore()
    first = store.begin_overlay(1)
    first.put("graph", "node", 1)
    old = store.finalize_overlay(first)
    pin = store.pin(old)
    second = store.begin_overlay(2)
    second.put("graph", "node", 2)
    new = store.finalize_overlay(second)
    assert pin.handle.canonical_applied_lsn == 1
    assert store.read_from_handle(pin.handle, "graph", "node") == 1
    assert store.read_from_handle(new, "graph", "node") == 2
    pin.release()


def test_handle_checksum_couples_roots_and_applied_lsn() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(7)
    overlay.put("payload", "node", {"x": 1})
    handle = store.finalize_overlay(overlay)
    assert len(handle.checksum) == 64
    assert handle.canonical_applied_lsn == 7

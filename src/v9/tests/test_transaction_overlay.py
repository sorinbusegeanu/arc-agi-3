from __future__ import annotations

import pytest

from v9.runtime.canonical_store import CanonicalStore


def test_overlay_reads_delta_before_base_and_abort_is_invisible() -> None:
    store = CanonicalStore()
    initial = store.begin_overlay(1)
    initial.put("payload", "a", 1)
    store.finalize_overlay(initial)
    overlay = store.begin_overlay(2)
    overlay.put("payload", "a", 2)
    assert overlay.get(store, "payload", "a") == 2
    store.abort_overlay(overlay)
    assert store.read_from_handle(store.current_handle, "payload", "a") == 1


def test_overlay_enforces_entry_and_byte_bounds() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(1, max_entries=1, max_bytes=64)
    overlay.put("graph", "a", 1)
    with pytest.raises(OverflowError, match="entry ceiling"):
        overlay.put("graph", "b", 2)

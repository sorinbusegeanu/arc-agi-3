from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore


def test_gc_never_deletes_chunks_reachable_from_pinned_handle() -> None:
    store = CanonicalStore(max_chunk_entries=1)
    first = store.begin_overlay(1)
    first.put("graph", "a", 1)
    old = store.finalize_overlay(first)
    pinned = store.pin(old)
    second = store.begin_overlay(2)
    second.put("graph", "a", 2)
    store.finalize_overlay(second)
    assert store.collect_garbage(max_chunks=10) == 0
    assert store.read_from_handle(old, "graph", "a") == 1
    pinned.release()
    assert store.collect_garbage(max_chunks=10) == 1


def test_gc_work_is_bounded_by_cursor_budget() -> None:
    store = CanonicalStore(max_chunk_entries=1)
    for lsn in range(1, 5):
        overlay = store.begin_overlay(lsn)
        overlay.put("graph", "a", lsn)
        store.finalize_overlay(overlay)
    assert store.collect_garbage(max_chunks=1) == 1
    assert store.orphan_count >= 1

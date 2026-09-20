from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_transaction import CanonicalCollection


def test_store_publishes_immutable_bounded_chunks() -> None:
    store = CanonicalStore(max_chunk_entries=2, max_chunk_bytes=4096)
    overlay = store.begin_overlay(1)
    overlay.put(CanonicalCollection.GRAPH, "a", {"value": 1})
    overlay.put(CanonicalCollection.GRAPH, "b", {"value": 2})
    overlay.put(CanonicalCollection.GRAPH, "c", {"value": 3})
    handle = store.finalize_overlay(overlay)
    assert handle.canonical_applied_lsn == 1
    assert len(handle.graph_root) == 2
    assert all(len(store.chunk(chunk_id).entries) <= 2 for chunk_id in handle.graph_root)
    assert store.read_from_handle(handle, CanonicalCollection.GRAPH, "c") == {"value": 3}


def test_unchanged_chunks_are_structurally_shared() -> None:
    store = CanonicalStore(max_chunk_entries=2, max_chunk_bytes=4096)
    first = store.begin_overlay(1)
    for key in ("a", "b", "c", "d"):
        first.put("graph", key, key.upper())
    old = store.finalize_overlay(first)
    second = store.begin_overlay(2)
    second.put("graph", "a", "changed")
    new = store.finalize_overlay(second)
    assert set(old.graph_root).intersection(new.graph_root)
    assert store.read_from_handle(old, "graph", "a") == "A"
    assert store.read_from_handle(new, "graph", "a") == "changed"

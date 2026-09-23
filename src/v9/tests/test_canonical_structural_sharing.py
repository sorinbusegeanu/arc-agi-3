from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore


def test_one_transaction_does_not_clone_unrelated_collections() -> None:
    store = CanonicalStore(max_chunk_entries=1)
    first = store.begin_overlay(1)
    first.put("graph", "node", "graph-v1")
    first.put("payload", "node", "payload-v1")
    old = store.finalize_overlay(first)
    second = store.begin_overlay(2)
    second.put("payload", "node", "payload-v2")
    new = store.finalize_overlay(second)
    assert old.graph_root == new.graph_root
    assert old.payload_root != new.payload_root

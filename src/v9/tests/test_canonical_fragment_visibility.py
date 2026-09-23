from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_transaction import execute_private_continuations


def test_private_fragments_never_change_current_handle() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(1)
    overlay.put("graph", "node", 1)
    before = store.current_handle
    observed = []
    execute_private_continuations(
        overlay,
        max_fragment_bytes=64,
        primitive=lambda _fragment: observed.append(store.current_handle.checksum),
    )
    assert observed == [before.checksum]
    assert store.current_handle is before
    published = store.finalize_overlay(overlay)
    assert published is store.current_handle
    assert published.checksum != before.checksum

from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_transaction import compile_continuation_fragments, execute_private_continuations


def test_continuation_fragments_obey_byte_bound() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(1)
    for index in range(8):
        overlay.put("payload", index, "x" * 20)
    fragments = compile_continuation_fragments(overlay, max_fragment_bytes=80)
    assert len(fragments) > 1
    assert all(fragment.encoded_bytes <= 80 for fragment in fragments)
    timings = execute_private_continuations(overlay, max_fragment_bytes=80)
    assert tuple(row.fragment_index for row in timings) == tuple(range(len(fragments)))

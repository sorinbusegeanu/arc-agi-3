from __future__ import annotations

from v9.runtime.signature_index import SignatureIndexStore


def test_signature_index_resident_state_is_bounded(tmp_path) -> None:
    store = SignatureIndexStore(tmp_path / "signatures.sqlite", delta_limit=4, page_cache_limit=2, dirty_window_limit=3)
    try:
        for signature in range(10):
            store.observe(signature, support_delta=signature + 1)
        assert store.resident_delta_entries < 4
        assert store.resident_cache_entries <= 2
        dirty = store.dirty_window()
        assert len(dirty) == 3
        assert tuple(row.priority for row in dirty) == tuple(sorted((row.priority for row in dirty), reverse=True))
    finally:
        store.close()


def test_signature_index_supports_full_uint64_identity(tmp_path) -> None:
    signature = (1 << 64) - 1
    with SignatureIndexStore(tmp_path / "signatures.sqlite") as store:
        store.observe(signature)
        store.flush()
        assert store.get(signature).support == 1

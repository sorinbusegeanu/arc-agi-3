from __future__ import annotations

from v9.runtime.signature_index import SignatureIndexStore


def test_dirty_state_and_scan_cursor_survive_restart(tmp_path) -> None:
    path = tmp_path / "signatures.sqlite"
    with SignatureIndexStore(path) as store:
        for signature in (10, 20, 30):
            store.observe(signature)
        assert tuple(row.signature for row in store.scan_page(limit=2)) == (10, 20)
    with SignatureIndexStore(path) as restored:
        assert restored.scan_cursor == 20
        assert {row.signature for row in restored.dirty_window()} == {10, 20, 30}
        restored.mark_derived(20, 1)
    with SignatureIndexStore(path) as restored_again:
        assert not restored_again.get(20).derivation_dirty

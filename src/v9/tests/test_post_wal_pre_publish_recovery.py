from __future__ import annotations

import pytest

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_wal import CanonicalCommitWAL


def test_apply_failure_leaves_durable_transaction_for_recovery(tmp_path) -> None:
    store = CanonicalStore()
    wal = CanonicalCommitWAL(tmp_path / "canonical.wal")
    overlay = store.begin_overlay(1)
    overlay.put("payload", "node", 3)

    def crash(_frame) -> None:
        raise RuntimeError("injected post-fsync crash")

    with pytest.raises(RuntimeError, match="injected"):
        wal.commit_overlay(store, overlay, transaction_id="tx", after_durable=crash)
    assert wal.wal_durable_lsn == 1
    assert store.current_handle.canonical_applied_lsn == 0
    assert tuple(frame.wal_tx_id for frame in wal.recover().frames) == ("tx",)

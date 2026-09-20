from __future__ import annotations

from v9.runtime.canonical_wal import CanonicalCommitWAL


def test_recovery_truncates_only_torn_tail(tmp_path) -> None:
    path = tmp_path / "canonical.wal"
    wal = CanonicalCommitWAL(path)
    wal.append_group(({"transaction_id": "durable"},))
    valid_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b"V9WAL716-partial")
    recovery = wal.recover(truncate=True)
    assert recovery.wal_durable_lsn == 1
    assert recovery.truncated_bytes > 0
    assert path.stat().st_size == valid_size

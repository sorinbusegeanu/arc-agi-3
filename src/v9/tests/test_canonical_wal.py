from __future__ import annotations

from pathlib import Path

import pytest

from v9.runtime.canonical_wal import CanonicalCommitWAL


def test_wal_round_trip_preserves_frames_and_frontier(tmp_path: Path) -> None:
    path = tmp_path / "canonical.wal"
    wal = CanonicalCommitWAL(path)
    committed = wal.append_group(
        (
            {"transaction_id": "tx-1", "mutations": ({"op": "put", "key": "a"},)},
            {"transaction_id": "tx-2", "work_metadata": {"rows": 1, "bytes": 8}},
        )
    )
    assert tuple(row.wal_lsn for row in committed) == (1, 2)
    restored = CanonicalCommitWAL(path)
    recovery = restored.recover()
    assert recovery.wal_durable_lsn == 2
    assert tuple(row.wal_tx_id for row in recovery.frames) == ("tx-1", "tx-2")
    assert recovery.truncated_bytes == 0


def test_durable_frontier_moves_only_after_fsync(tmp_path: Path) -> None:
    observations: list[int] = []
    wal: CanonicalCommitWAL

    def fsync(_descriptor: int) -> None:
        observations.append(wal.wal_durable_lsn)

    wal = CanonicalCommitWAL(tmp_path / "canonical.wal", fsync=fsync)
    wal.append_group(({"transaction_id": "tx"},))
    assert observations == [0]
    assert wal.wal_durable_lsn == 1


def test_failed_fsync_requires_reopen_before_another_append(tmp_path: Path) -> None:
    path = tmp_path / "canonical.wal"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    wal = CanonicalCommitWAL(path, fsync=fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        wal.append_group(({"transaction_id": "indeterminate"},))

    assert wal.wal_durable_lsn == 0
    assert not wal.append_available
    with pytest.raises(RuntimeError, match="reopen and recover"):
        wal.append_group(({"transaction_id": "must-not-branch"},))

    reopened = CanonicalCommitWAL(path)
    assert reopened.append_available
    assert reopened.wal_durable_lsn == 1
    committed = reopened.append_group(({"transaction_id": "after-recovery"},))
    assert committed[0].previous_lsn == 1
    assert committed[0].wal_lsn == 2



def test_wal_recovery_and_reclaim_do_not_read_whole_file(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "streamed.wal"
    wal = CanonicalCommitWAL(path, max_group_frames=1)
    for index in range(3):
        wal.append_group(({"transaction_id": f"tx-{index}"},))
    wal.register_durable_consumer("snapshot", 0)
    wal.update_durable_consumer("snapshot", 2)

    original_read_bytes = Path.read_bytes

    def fail_wal_read_bytes(self: Path):
        if self == path:
            raise AssertionError("WAL recovery materialized the whole file")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fail_wal_read_bytes)
    recovery = wal.recover()
    assert tuple(row.wal_lsn for row in recovery.frames) == (1, 2, 3)
    assert wal.reclaim_prefix() == 2
    recovery = wal.recover()
    assert tuple(row.wal_lsn for row in recovery.frames) == (3,)

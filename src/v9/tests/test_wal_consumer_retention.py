from __future__ import annotations

import pytest

from v9.runtime.canonical_wal import CanonicalCommitWAL


def test_wal_reclaim_frontier_is_minimum_consumer_checkpoint(tmp_path) -> None:
    wal = CanonicalCommitWAL(tmp_path / "wal")
    wal.append_group(tuple({"transaction_id": f"tx-{index}"} for index in range(3)))
    wal.register_durable_consumer("snapshot", 2)
    wal.register_durable_consumer("hgt", 1)
    assert wal.wal_reclaim_lsn == 1
    wal.update_durable_consumer("hgt", 3)
    assert wal.wal_reclaim_lsn == 2


def test_wal_durable_consumer_registry_is_bounded(tmp_path) -> None:
    wal = CanonicalCommitWAL(tmp_path / "wal", max_durable_consumers=1)
    wal.register_durable_consumer("snapshot")
    with pytest.raises(OverflowError, match="consumer count"):
        wal.register_durable_consumer("hgt")


def test_wal_prefix_reclamation_is_physical_and_restart_safe(tmp_path) -> None:
    path = tmp_path / "wal"
    wal = CanonicalCommitWAL(path)
    for index in range(4):
        wal.append_group(
            ({"transaction_id": f"tx-{index}", "mutations": ({"payload": "x" * 512},)},)
        )
    original_bytes = path.stat().st_size
    wal.register_durable_consumer("snapshot", 3)
    wal.register_durable_consumer("hgt", 2)

    assert wal.reclaim_prefix() == 2
    assert path.stat().st_size < original_bytes
    recovery = wal.recover()
    assert tuple(frame.wal_lsn for frame in recovery.frames) == (3, 4)
    assert recovery.frames[0].previous_lsn == 2
    assert recovery.wal_durable_lsn == 4

    reopened = CanonicalCommitWAL(path)
    committed = reopened.append_group(({"transaction_id": "tx-4"},))
    assert committed[0].previous_lsn == 4
    assert committed[0].wal_lsn == 5


def test_wal_reclamation_never_crosses_slowest_consumer(tmp_path) -> None:
    wal = CanonicalCommitWAL(tmp_path / "wal")
    for index in range(4):
        wal.append_group(({"transaction_id": f"tx-{index}"},))
    wal.register_durable_consumer("snapshot", 3)
    wal.register_durable_consumer("hgt", 1)

    assert wal.reclaim_prefix() == 1
    assert tuple(frame.wal_lsn for frame in wal.recover().frames) == (2, 3, 4)


def test_post_replace_reclaim_failure_requires_reopen(tmp_path) -> None:
    path = tmp_path / "wal"
    wal = CanonicalCommitWAL(path)
    for index in range(3):
        wal.append_group(({"transaction_id": f"tx-{index}"},))
    wal.register_durable_consumer("snapshot", 2)
    calls = 0

    def fail_directory_fsync(_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")

    wal._fsync = fail_directory_fsync
    with pytest.raises(OSError, match="directory fsync"):
        wal.reclaim_prefix()
    assert not wal.append_available
    with pytest.raises(RuntimeError, match="reopen and recover"):
        wal.append_group(({"transaction_id": "unsafe"},))

    reopened = CanonicalCommitWAL(path)
    assert tuple(frame.wal_lsn for frame in reopened.recover().frames) == (3,)
    assert reopened.append_group(({"transaction_id": "safe"},))[0].wal_lsn == 4


def test_durable_consumer_checkpoints_survive_restart(tmp_path) -> None:
    path = tmp_path / "wal"
    wal = CanonicalCommitWAL(path)
    wal.append_group(tuple({"transaction_id": f"tx-{index}"} for index in range(3)))
    wal.register_durable_consumer("snapshot", 2)
    wal.register_durable_consumer("hgt", 1)

    restored = CanonicalCommitWAL(path)
    assert restored.wal_reclaim_lsn == 1
    restored.register_durable_consumer("snapshot")
    assert restored.wal_reclaim_lsn == 1

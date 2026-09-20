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

from __future__ import annotations

import pytest

from v9.runtime.canonical_wal import PersistenceFrontiers


def test_persistence_frontier_ordering() -> None:
    frontiers = PersistenceFrontiers(10, 8, 5, 7)
    assert frontiers.wal_durable_lsn >= frontiers.canonical_applied_lsn >= frontiers.snapshot_applied_lsn
    assert frontiers.hgt_checkpoint_lsn <= frontiers.wal_durable_lsn
    with pytest.raises(ValueError, match="invariant"):
        PersistenceFrontiers(5, 6, 1, 0)

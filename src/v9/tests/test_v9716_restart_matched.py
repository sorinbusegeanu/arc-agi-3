from __future__ import annotations

import pytest

from v9.runtime.canonical_store import CanonicalStore


def test_matched_snapshot_cannot_mix_scientific_identity(tmp_path) -> None:
    store = CanonicalStore()
    path = store.write_snapshot(tmp_path / "snapshot", scientific_identity={"experiment_id": "one", "mode": "MATCHED_REASONING", "epoch_view": "view-1"})
    with pytest.raises(ValueError, match="identity mismatch"):
        CanonicalStore.from_snapshot(path, expected_scientific_identity={"experiment_id": "two", "mode": "MATCHED_REASONING", "epoch_view": "view-1"})

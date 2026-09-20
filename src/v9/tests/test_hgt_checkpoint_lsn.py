from __future__ import annotations

import pytest

from v9.runtime.training_evidence import TrainingEvidenceMaterializer


def test_hgt_checkpoint_never_exceeds_durable_wal(tmp_path) -> None:
    materializer = TrainingEvidenceMaterializer(tmp_path)
    with pytest.raises(ValueError, match="durable"):
        materializer.materialize(start_lsn=1, end_lsn=2, records=(), wal_durable_lsn=1)

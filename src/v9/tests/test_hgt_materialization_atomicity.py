from __future__ import annotations

import pytest

from v9.runtime.training_evidence import TrainingEvidenceMaterializer, TrainingEvidenceRecord


def test_post_segment_pre_manifest_crash_keeps_segment_invisible(tmp_path) -> None:
    materializer = TrainingEvidenceMaterializer(tmp_path)
    record = TrainingEvidenceRecord.create(kind="derivation", source_wal_lsn=1, scientific_provenance={"id": 1}, schema_versions={"record": 1}, label_payload={})

    def crash(boundary: str) -> None:
        if boundary == "segment_renamed":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        materializer.materialize(start_lsn=1, end_lsn=1, records=(record,), wal_durable_lsn=1, crash_hook=crash)
    restored = TrainingEvidenceMaterializer(tmp_path)
    assert restored.manifest.hgt_checkpoint_lsn == 0
    assert restored.records() == ()
    assert len(restored.reclaim_orphans()) == 1


def test_durable_segment_directory_entry_is_still_invisible_before_manifest(tmp_path) -> None:
    materializer = TrainingEvidenceMaterializer(tmp_path)
    record = TrainingEvidenceRecord.create(kind="interaction", source_wal_lsn=1, scientific_provenance={"id": 1}, schema_versions={"record": 1}, label_payload={})

    def crash(boundary: str) -> None:
        if boundary == "segment_directory_fsynced":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        materializer.materialize(start_lsn=1, end_lsn=1, records=(record,), wal_durable_lsn=1, crash_hook=crash)

    restored = TrainingEvidenceMaterializer(tmp_path)
    assert restored.manifest.hgt_checkpoint_lsn == 0
    assert restored.records() == ()
    assert len(restored.reclaim_orphans()) == 1

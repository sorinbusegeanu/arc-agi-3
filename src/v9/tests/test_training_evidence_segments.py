from __future__ import annotations

from v9.runtime.training_evidence import TrainingEvidenceMaterializer, TrainingEvidenceRecord


def _record(lsn: int) -> TrainingEvidenceRecord:
    return TrainingEvidenceRecord.create(kind="interaction", source_wal_lsn=lsn, scientific_provenance={"id": lsn}, schema_versions={"record": 1}, label_payload={"action": lsn})


def test_segments_are_immutable_and_contiguous(tmp_path) -> None:
    materializer = TrainingEvidenceMaterializer(tmp_path)
    first = materializer.materialize(start_lsn=1, end_lsn=2, records=(_record(1), _record(2)), wal_durable_lsn=3)
    second = materializer.materialize(start_lsn=3, end_lsn=3, records=(_record(3),), wal_durable_lsn=3)
    assert first.hgt_checkpoint_lsn == 2
    assert second.hgt_checkpoint_lsn == 3
    assert materializer.records() == (_record(1), _record(2), _record(3))

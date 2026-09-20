from __future__ import annotations

from v9.runtime.training_evidence import TrainingEvidenceMaterializer, TrainingEvidenceRecord


def test_replay_cache_can_rebuild_from_manifest(tmp_path) -> None:
    record = TrainingEvidenceRecord.create(kind="reasoning_trace", source_wal_lsn=1, scientific_provenance={"trace": 9}, schema_versions={"record": 1}, label_payload={"score": 3})
    TrainingEvidenceMaterializer(tmp_path).materialize(start_lsn=1, end_lsn=1, records=(record,), wal_durable_lsn=1)
    restored = TrainingEvidenceMaterializer(tmp_path)
    assert restored.records() == (record,)

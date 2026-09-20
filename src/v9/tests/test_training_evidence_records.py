from __future__ import annotations

from v9.runtime.training_evidence import TrainingEvidenceKind, TrainingEvidenceRecord


def test_training_evidence_identity_is_deterministic() -> None:
    kwargs = dict(
        kind=TrainingEvidenceKind.GROUNDING,
        source_wal_lsn=3,
        scientific_provenance={"experiment": "e", "producer": 7},
        schema_versions={"label": 1},
        label_payload={"grounded": True},
    )
    left = TrainingEvidenceRecord.create(**kwargs)
    right = TrainingEvidenceRecord.create(**kwargs)
    assert left == right
    assert TrainingEvidenceRecord.from_dict(left.as_dict()) == left

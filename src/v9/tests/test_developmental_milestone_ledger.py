from __future__ import annotations

from v9.research.prediction_registry import DevelopmentalMilestoneLedger


def test_ledger_records_observed_not_predicted_order(tmp_path) -> None:
    path = tmp_path / "milestones.jsonl"
    ledger = DevelopmentalMilestoneLedger(path)
    first = ledger.record_first(kind="role", scientific_evidence_id="e2", canonical_generation=2, canonical_lsn=2, evidence_artifact="role.json")
    ledger.record_first(kind="contingency", scientific_evidence_id="e3", canonical_generation=3, canonical_lsn=3, evidence_artifact="contingency.json")
    duplicate = ledger.record_first(kind="role", scientific_evidence_id="later", canonical_generation=9, canonical_lsn=9, evidence_artifact="later.json")
    assert duplicate == first
    assert tuple(row.kind for row in DevelopmentalMilestoneLedger(path).records) == ("role", "contingency")

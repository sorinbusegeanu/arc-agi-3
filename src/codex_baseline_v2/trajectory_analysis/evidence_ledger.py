from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from codex_baseline_v2.shared.schemas import EvidenceLedgerEntryV2, SCHEMA_VERSION


def update_evidence_ledger(
    existing_entries: List[EvidenceLedgerEntryV2],
    new_subject_evidence: Iterable[Tuple[str, str, str, List[str], List[str]]],
    round_id: int,
) -> list[EvidenceLedgerEntryV2]:
    ledger: Dict[Tuple[str, str, str], EvidenceLedgerEntryV2] = {
        (row.subject_type, row.subject_id, row.claim_type): row for row in existing_entries
    }
    for subject_type, subject_id, claim_type, positive_refs, negative_refs in new_subject_evidence:
        key = (subject_type, subject_id, claim_type)
        prior = ledger.get(key)
        game_id = prior.game_id if prior is not None else "unknown_game"
        merged_pos = sorted(set((prior.positive_refs if prior is not None else []) + list(positive_refs)))
        merged_neg = sorted(set((prior.negative_refs if prior is not None else []) + list(negative_refs)))
        positive_count = (prior.positive_count if prior is not None else 0) + len(positive_refs)
        negative_count = (prior.negative_count if prior is not None else 0) + len(negative_refs)
        total = positive_count + negative_count
        confidence = (positive_count + 1.0) / float(max(1.0, total + 2.0))
        ledger[key] = EvidenceLedgerEntryV2(
            schema_version=SCHEMA_VERSION,
            game_id=game_id,
            entry_id=prior.entry_id if prior is not None else "ledger:%s:%s:%s" % key,
            subject_type=subject_type,
            subject_id=subject_id,
            claim_type=claim_type,
            positive_refs=merged_pos,
            negative_refs=merged_neg,
            positive_count=positive_count,
            negative_count=negative_count,
            confidence=confidence,
            last_updated_round=round_id,
        )
    return sorted(ledger.values(), key=lambda row: (row.subject_type, row.subject_id, row.claim_type))

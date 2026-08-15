from __future__ import annotations

from dataclasses import dataclass

from v8.evidence import EvidenceLedger, EvidenceRecord


@dataclass(frozen=True, slots=True)
class ReportingCut:
    watermark: int
    generation: int
    graph_digest: str
    evidence: tuple[EvidenceRecord, ...]


def capture_reporting_cut(
    read_view,
    ledger: EvidenceLedger,
    watermark: int,
    *,
    generation: int = 0,
) -> ReportingCut:
    """Capture one reproducible graph generation plus causally bounded evidence cut."""
    return ReportingCut(
        int(watermark),
        int(generation),
        str(read_view.state_digest()),
        ledger.cut(int(watermark)),
    )

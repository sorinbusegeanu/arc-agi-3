from __future__ import annotations

from collections.abc import Iterable, Mapping

from .contracts import ChainStatus, DEFAULT_CAUSAL_CHAIN
from .models import ChainAuditResult, ChainEdgeEvidence


def audit_chain(
    evidence: Mapping[str, ChainEdgeEvidence],
    chain: Iterable[str] = DEFAULT_CAUSAL_CHAIN,
) -> ChainAuditResult:
    """Return ordered chain status and the first causal link that failed."""
    ordered: list[ChainEdgeEvidence] = []
    first_broken: str | None = None
    blocked = False

    for edge in tuple(chain):
        observed = evidence.get(edge)
        if observed is None:
            current = ChainEdgeEvidence(
                edge=edge,
                status=ChainStatus.NOT_REACHED if blocked else ChainStatus.INSUFFICIENT_EVIDENCE,
            )
        elif blocked and observed.status == ChainStatus.INSUFFICIENT_EVIDENCE:
            current = ChainEdgeEvidence(
                edge=edge,
                status=ChainStatus.NOT_REACHED,
                evidence_count=observed.evidence_count,
                evidence_ids=observed.evidence_ids,
                blocker=observed.blocker,
            )
        else:
            current = observed

        ordered.append(current)
        if first_broken is None and current.status == ChainStatus.FAIL:
            first_broken = edge
            blocked = True

    complete = bool(ordered) and all(item.status == ChainStatus.PASS for item in ordered)
    return ChainAuditResult(tuple(ordered), first_broken, complete)

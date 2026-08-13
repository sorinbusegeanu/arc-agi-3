from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

VALID_DECISIONS = frozenset({"VALID", "PARTIALLY_VALID", "INVALID", "INSUFFICIENT_EVIDENCE"})


@dataclass(frozen=True, slots=True)
class EvidenceContract:
    hypothesis_id: str
    required_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HypothesisReport:
    hypothesis_id: str
    raw_decision: str
    quality_gate: str
    dependency_gate: str
    final_decision: str
    evidence: Mapping[str, Any]
    missing_fields: tuple[str, ...]


DEFAULT_CONTRACTS = MappingProxyType({
    hypothesis_id: EvidenceContract(hypothesis_id, ("evidence_rows", "measurement"))
    for hypothesis_id in (f"H{index:02d}" for index in range(1, 13))
})


class StrictHypothesisReporter:
    """Read-only evidence-contract reporter. Proxy/missing evidence never validates."""

    def __init__(self, contracts: Mapping[str, EvidenceContract] | None = None) -> None:
        self.contracts = contracts or DEFAULT_CONTRACTS

    def evaluate(
        self,
        hypothesis_id: str,
        *,
        raw_decision: str,
        evidence: Mapping[str, Any],
        quality_gate: str = "PASS",
        dependency_gate: str = "PASS",
    ) -> HypothesisReport:
        if hypothesis_id not in self.contracts:
            raise KeyError(hypothesis_id)
        if raw_decision not in VALID_DECISIONS:
            raise ValueError("unknown raw decision")
        contract = self.contracts[hypothesis_id]
        missing = tuple(field for field in contract.required_fields if field not in evidence or evidence[field] is None)
        proxy_only = bool(evidence.get("proxy_only", False))
        gates_pass = quality_gate == "PASS" and dependency_gate == "PASS"
        if missing or proxy_only or not gates_pass:
            final = "INSUFFICIENT_EVIDENCE"
        else:
            final = raw_decision
        return HypothesisReport(hypothesis_id, raw_decision, quality_gate, dependency_gate, final, MappingProxyType(dict(evidence)), missing)

    def evaluate_suite(
        self,
        rows: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, HypothesisReport]:
        reports = {}
        for hypothesis_id in sorted(self.contracts):
            row = rows.get(hypothesis_id, {})
            reports[hypothesis_id] = self.evaluate(
                hypothesis_id,
                raw_decision=str(row.get("raw_decision", "INSUFFICIENT_EVIDENCE")),
                quality_gate=str(row.get("quality_gate", "PASS")),
                dependency_gate=str(row.get("dependency_gate", "PASS")),
                evidence=row.get("evidence", {}),
            )
        return MappingProxyType(reports)


def report_as_dict(report: HypothesisReport) -> dict[str, Any]:
    return {
        "hypothesis_id": report.hypothesis_id,
        "raw_decision": report.raw_decision,
        "quality_gate": report.quality_gate,
        "dependency_gate": report.dependency_gate,
        "final_decision": report.final_decision,
        "missing_fields": list(report.missing_fields),
        "evidence": dict(report.evidence),
    }

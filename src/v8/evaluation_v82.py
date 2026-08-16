from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from v8.evaluation import HypothesisDecision, ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.scientific_traceability import TRACEABILITY, ordering_gates


@dataclass(frozen=True, slots=True)
class V82HypothesisDecision:
    hypothesis_id: str
    raw_decision: str
    quality_gate: str
    dependency_gate: str
    final_decision: str
    evidence_count: int
    blocker: str
    paper_claim: str
    required_measurements: tuple[str, ...]
    falsification_measurements: tuple[str, ...]
    ordering_gate: str


_ORDERING_BY_HYPOTHESIS = {
    "H02": "contingency_before_prediction",
    "H04": "family_before_carrier",
    "H05": "carrier_before_role",
    "H07": "role_before_validated_concept",
    "H14": "outcome_before_replanning",
    "H15": "outcome_before_preference",
}


class V82ScientificHypothesisEvaluator(ScientificHypothesisEvaluator):
    """H01-H15 evaluator with explicit paper claim and developmental-order gates."""

    def evaluate(self, evidence: Iterable[EvidenceRecord]) -> tuple[V82HypothesisDecision, ...]:
        rows = tuple(evidence)
        base = super().evaluate(rows)
        trace = {record.hypothesis_id: record for record in TRACEABILITY}
        ordering = ordering_gates(rows)
        result: list[V82HypothesisDecision] = []
        for decision in base:
            record = trace[decision.hypothesis_id]
            gate_name = _ORDERING_BY_HYPOTHESIS.get(decision.hypothesis_id)
            gate = "PASS" if gate_name is None else ordering.get(gate_name, "NOT_REACHED")
            final = decision.final_decision
            blocker = decision.blocker
            if gate == "FAIL":
                # Observing a later capability before its required developmental
                # precursor directly contradicts the ordering claim.
                final = "INVALID" if final == "VALID" else final
                blocker = (blocker + "; " if blocker else "") + f"developmental ordering failed: {gate_name}"
            result.append(
                V82HypothesisDecision(
                    decision.hypothesis_id,
                    decision.raw_decision,
                    decision.quality_gate,
                    decision.dependency_gate,
                    final,
                    decision.evidence_count,
                    blocker,
                    record.paper_claim,
                    record.required_evidence,
                    record.falsification_evidence,
                    gate,
                )
            )
        return tuple(result)

    @staticmethod
    def status_map(decisions: Iterable[V82HypothesisDecision | HypothesisDecision]) -> dict[str, str]:
        return {decision.hypothesis_id: decision.final_decision for decision in decisions}

    def write_report(
        self,
        path: str | Path,
        evidence: Iterable[EvidenceRecord],
    ) -> tuple[V82HypothesisDecision, ...]:
        decisions = self.evaluate(evidence)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([asdict(row) for row in decisions], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return decisions

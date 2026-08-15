from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from v8.evidence import EvidenceRecord


@dataclass(frozen=True, slots=True)
class HypothesisContract:
    hypothesis_id: str
    partial_kinds: tuple[str, ...]
    required_kinds: tuple[str, ...]
    min_required_records: int = 1


@dataclass(frozen=True, slots=True)
class HypothesisDecision:
    hypothesis_id: str
    raw_decision: str
    quality_gate: str
    dependency_gate: str
    final_decision: str
    evidence_count: int
    blocker: str


CONTRACTS: tuple[HypothesisContract, ...] = (
    HypothesisContract("H01", ("contingency_recurrence",), ("contingency_recurrence",), 2),
    HypothesisContract("H02", ("supported_prediction",), ("prediction_violation",), 1),
    HypothesisContract("H03", ("family_recurrence",), ("family_compression",), 1),
    HypothesisContract("H04", ("carrier_candidate",), ("carrier_emergence",), 1),
    HypothesisContract("H05", ("role_candidate",), ("role_emergence",), 1),
    HypothesisContract("H06", ("transfer_structural",), ("transfer_trial_pass",), 1),
    HypothesisContract("H07", ("concept_candidate",), ("concept_transfer_pass",), 1),
    HypothesisContract("H08", ("consequence_structure",), ("consequence_structure",), 2),
    HypothesisContract("H09", ("future_option_estimate",), ("future_option_estimate",), 2),
    HypothesisContract("H10", ("context_refinement",), ("context_refinement_gain",), 1),
    HypothesisContract("H11", ("transfer_trial_pass",), ("transfer_trial_pass",), 2),
    HypothesisContract("H12", ("strategy_reuse",), ("strategy_efficiency",), 1),
    HypothesisContract("H13", ("outcome_equivalence",), ("outcome_equivalence",), 2),
    HypothesisContract("H14", ("alternative_strategy", "replanning_observed"), ("replanning_recovery_trial",), 1),
    HypothesisContract("H15", ("preference_probe",), ("stable_preference_probe",), 1),
)


class ScientificHypothesisEvaluator:
    """Read-only H01-H15 evaluator over one immutable evidence cut."""

    def evaluate(self, evidence: Iterable[EvidenceRecord]) -> tuple[HypothesisDecision, ...]:
        rows = tuple(evidence)
        kinds: dict[str, list[EvidenceRecord]] = {}
        for row in rows:
            kinds.setdefault(row.evidence_kind, []).append(row)
        decisions = []
        for contract in CONTRACTS:
            required = [row for kind in contract.required_kinds for row in kinds.get(kind, ())]
            partial = [row for kind in contract.partial_kinds for row in kinds.get(kind, ())]
            if len(required) >= contract.min_required_records:
                final = "VALID"
                blocker = ""
                raw = "VALID"
            elif partial:
                final = "PARTIALLY_VALID"
                blocker = "missing required evidence: " + ",".join(contract.required_kinds)
                raw = "PARTIALLY_VALID"
            else:
                final = "INSUFFICIENT_EVIDENCE"
                blocker = "missing evidence contract fields: " + ",".join(contract.partial_kinds)
                raw = "INSUFFICIENT_EVIDENCE"
            decisions.append(
                HypothesisDecision(
                    contract.hypothesis_id,
                    raw,
                    "PASS" if rows else "NO_EVIDENCE",
                    "PASS" if not blocker else "BLOCKED",
                    final,
                    len(required) if required else len(partial),
                    blocker,
                )
            )
        return tuple(decisions)

    @staticmethod
    def status_map(decisions: Iterable[HypothesisDecision]) -> dict[str, str]:
        return {decision.hypothesis_id: decision.final_decision for decision in decisions}

    def write_report(self, path: str | Path, evidence: Iterable[EvidenceRecord]) -> tuple[HypothesisDecision, ...]:
        decisions = self.evaluate(evidence)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([asdict(row) for row in decisions], indent=2, sort_keys=True), encoding="utf-8")
        return decisions

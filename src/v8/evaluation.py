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
    negative_kinds: tuple[str, ...] = ()
    causal_required: bool = False
    held_out_required: bool = False
    positive_effect_required: bool = False
    min_distinct_targets: int = 0
    dependencies: tuple[str, ...] = ()


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
    HypothesisContract(
        "H06",
        ("transfer_structural",),
        ("transfer_trial_pass",),
        1,
        negative_kinds=("transfer_trial_fail",),
        causal_required=True,
        held_out_required=True,
        positive_effect_required=True,
        min_distinct_targets=1,
    ),
    HypothesisContract(
        "H07",
        ("concept_candidate",),
        ("concept_transfer_pass",),
        1,
        negative_kinds=("concept_transfer_fail",),
        causal_required=True,
        held_out_required=True,
        positive_effect_required=True,
        min_distinct_targets=1,
        dependencies=("H06",),
    ),
    HypothesisContract("H08", ("consequence_structure",), ("world_model_component",), 1),
    HypothesisContract("H09", ("future_option_estimate",), ("future_option_estimate",), 2),
    HypothesisContract("H10", ("context_refinement",), ("context_refinement_gain",), 1),
    HypothesisContract(
        "H11",
        ("transfer_trial_pass",),
        ("transfer_trial_pass",),
        2,
        negative_kinds=("transfer_trial_fail",),
        causal_required=True,
        held_out_required=True,
        positive_effect_required=True,
        min_distinct_targets=2,
        dependencies=("H06",),
    ),
    HypothesisContract("H12", ("strategy_reuse",), ("strategy_efficiency",), 1),
    HypothesisContract(
        "H13",
        ("outcome_equivalence", "outcome_merge"),
        ("outcome_consistency_holdout",),
        1,
        negative_kinds=("outcome_consistency_fail",),
        held_out_required=True,
        positive_effect_required=True,
        min_distinct_targets=1,
    ),
    HypothesisContract(
        "H14",
        ("alternative_strategy", "replanning_observed"),
        ("replanning_recovery_trial",),
        1,
        negative_kinds=("replanning_recovery_fail",),
        causal_required=True,
        positive_effect_required=True,
        dependencies=("H13",),
    ),
    HypothesisContract(
        "H15",
        ("preference_probe",),
        ("stable_preference_probe",),
        1,
        negative_kinds=("preference_instability",),
        causal_required=True,
        dependencies=("H13",),
    ),
)


class ScientificHypothesisEvaluator:
    """Read-only H01-H15 evaluator over one immutable evidence cut."""

    @staticmethod
    def _admissible(row: EvidenceRecord, contract: HypothesisContract) -> bool:
        if not row.quality_valid():
            return False
        if contract.causal_required and not row.causal_intervention:
            return False
        if contract.held_out_required:
            if row.target_game_hash == 0:
                return False
            if row.target_game_hash in set(row.provenance_games):
                return False
        if contract.positive_effect_required and row.effect_direction <= 0 and row.raw_value <= 0:
            return False
        return True

    @staticmethod
    def _enough(rows: list[EvidenceRecord], contract: HypothesisContract) -> bool:
        if len(rows) < contract.min_required_records:
            return False
        if contract.min_distinct_targets > 0:
            targets = {row.target_game_hash for row in rows if row.target_game_hash != 0}
            if len(targets) < contract.min_distinct_targets:
                return False
        return True

    def evaluate(self, evidence: Iterable[EvidenceRecord]) -> tuple[HypothesisDecision, ...]:
        rows = tuple(evidence)
        kinds: dict[str, list[EvidenceRecord]] = {}
        for row in rows:
            kinds.setdefault(row.evidence_kind, []).append(row)

        decisions: list[HypothesisDecision] = []
        decided: dict[str, str] = {}
        for contract in CONTRACTS:
            required_raw = [row for kind in contract.required_kinds for row in kinds.get(kind, ())]
            partial_raw = [row for kind in contract.partial_kinds for row in kinds.get(kind, ())]
            negative_raw = [row for kind in contract.negative_kinds for row in kinds.get(kind, ())]
            required = [row for row in required_raw if self._admissible(row, contract)]
            negative = [row for row in negative_raw if row.quality_valid()]

            if negative and not self._enough(required, contract):
                raw = "INVALID"
            elif self._enough(required_raw, contract):
                raw = "VALID"
            elif partial_raw:
                raw = "PARTIALLY_VALID"
            else:
                raw = "INSUFFICIENT_EVIDENCE"

            quality_pass = self._enough(required, contract)
            if raw == "INVALID" and negative:
                quality_pass = True
            quality_gate = "PASS" if quality_pass else ("NO_EVIDENCE" if not rows else "FAIL")

            blocked_dependencies = [
                dep for dep in contract.dependencies if decided.get(dep) != "VALID"
            ]
            dependency_gate = "PASS" if not blocked_dependencies else "BLOCKED"

            blockers: list[str] = []
            if raw == "VALID" and not self._enough(required, contract):
                blockers.append("required evidence failed quality/causality/held-out gate")
            elif raw == "PARTIALLY_VALID":
                blockers.append("missing required evidence: " + ",".join(contract.required_kinds))
            elif raw == "INSUFFICIENT_EVIDENCE":
                blockers.append("missing evidence contract fields: " + ",".join(contract.partial_kinds))
            if contract.min_distinct_targets > 0:
                targets = {row.target_game_hash for row in required if row.target_game_hash}
                if len(targets) < contract.min_distinct_targets:
                    blockers.append(
                        f"requires {contract.min_distinct_targets} distinct held-out targets; has {len(targets)}"
                    )
            if blocked_dependencies:
                blockers.append("blocked dependencies: " + ",".join(blocked_dependencies))

            if raw == "INVALID" and quality_gate == "PASS" and dependency_gate == "PASS":
                final = "INVALID"
            elif raw == "VALID" and quality_gate == "PASS" and dependency_gate == "PASS":
                final = "VALID"
            elif partial_raw or required_raw:
                final = "PARTIALLY_VALID"
            else:
                final = "INSUFFICIENT_EVIDENCE"

            decided[contract.hypothesis_id] = final
            decisions.append(
                HypothesisDecision(
                    contract.hypothesis_id,
                    raw,
                    quality_gate,
                    dependency_gate,
                    final,
                    len(required) if required else len(partial_raw),
                    "; ".join(blockers),
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
        target.write_text(
            json.dumps([asdict(row) for row in decisions], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return decisions

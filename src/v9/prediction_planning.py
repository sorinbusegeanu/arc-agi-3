from __future__ import annotations

from dataclasses import dataclass

from v9.grounding import GroundingMaturity, GroundingState


@dataclass(frozen=True, slots=True)
class ShadowPrediction:
    baseline_correct: bool
    symbol_conditioned_correct: bool
    symbol_used: bool
    false_positive_transfer: bool = False


class ShadowPredictionEvaluator:
    def __init__(self) -> None:
        self.rows: list[ShadowPrediction] = []

    def record(self, baseline_correct: bool, symbol_conditioned_correct: bool, *, grounding: GroundingState | None, false_positive_transfer: bool = False) -> ShadowPrediction:
        symbol_used = bool(grounding is not None and grounding.maturity >= GroundingMaturity.G3 and not grounding.suspended)
        row = ShadowPrediction(bool(baseline_correct), bool(symbol_conditioned_correct if symbol_used else baseline_correct), symbol_used, bool(false_positive_transfer and symbol_used)); self.rows.append(row); return row

    def metrics(self) -> dict[str, float | int]:
        if not self.rows: return {"samples": 0, "baseline_accuracy": 0.0, "symbol_accuracy": 0.0, "improvement": 0.0, "false_positive_transfer_rate": 0.0}
        total = len(self.rows); baseline = sum(row.baseline_correct for row in self.rows) / total; symbol = sum(row.symbol_conditioned_correct for row in self.rows) / total; false_rate = sum(row.false_positive_transfer for row in self.rows) / total
        return {"samples": total, "baseline_accuracy": baseline, "symbol_accuracy": symbol, "improvement": symbol - baseline, "false_positive_transfer_rate": false_rate}


@dataclass(frozen=True, slots=True)
class RankedAction:
    action: int
    score: float
    symbol_influenced: bool


class SymbolActionGate:
    def rank(self, baseline_scores: dict[int, float], symbol_score_delta: dict[int, float], *, grounding: GroundingState | None, available_actions: tuple[int, ...] | list[int], higher_memory_validated: bool, cross_environment: bool = False) -> tuple[RankedAction, ...]:
        allowed = {int(action) for action in available_actions}; maturity_required = GroundingMaturity.G5 if cross_environment else GroundingMaturity.G4
        use_symbol = bool(grounding is not None and not grounding.suspended and grounding.maturity >= maturity_required and higher_memory_validated)
        rows = []
        for action in allowed:
            base = float(baseline_scores.get(action, 0.0)); delta = float(symbol_score_delta.get(action, 0.0)) if use_symbol else 0.0; rows.append(RankedAction(action, base + delta, use_symbol and delta != 0.0))
        return tuple(sorted(rows, key=lambda row: (-row.score, row.action)))

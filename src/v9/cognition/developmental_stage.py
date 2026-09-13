from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum


class DevelopmentalStage(IntEnum):
    INTERACTION_SEEDING = 0
    STABLE_CONTINGENCIES = 1
    STRUCTURAL_ABSTRACTION = 2
    HELD_OUT_TRANSFER = 3
    CONSEQUENCE_INTEGRATION = 4
    OUTCOME_AND_PREFERENCE = 5
    ALTERNATIVE_STRATEGIES = 6
    DEMONSTRATED_REPLANNING = 7


@dataclass(frozen=True, slots=True)
class StageEvidence:
    stable_contingencies: int = 0
    structural_abstractions: int = 0
    held_out_transfer_successes: int = 0
    mature_consequences: int = 0
    outcome_equivalences: int = 0
    learned_preferences: int = 0
    alternative_strategies: int = 0
    demonstrated_replans: int = 0
    efficient_replans: int = 0


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    interval_id: int
    evidence_watermark: int
    stage: DevelopmentalStage
    next_stage: DevelopmentalStage
    evidence: StageEvidence


class DevelopmentalStageTracker:
    """Advances capability only between causal evaluation intervals."""

    def __init__(self, *, history_limit: int = 1024) -> None:
        if history_limit <= 0:
            raise ValueError("stage history limit must be positive")
        self.history_limit = int(history_limit)
        self.stage = DevelopmentalStage.INTERACTION_SEEDING
        self.interval_id = 0
        self.history: list[StageSnapshot] = []

    @staticmethod
    def _eligible(stage: DevelopmentalStage, evidence: StageEvidence) -> bool:
        gates = {
            DevelopmentalStage.INTERACTION_SEEDING: evidence.stable_contingencies > 0,
            DevelopmentalStage.STABLE_CONTINGENCIES: evidence.structural_abstractions > 0,
            DevelopmentalStage.STRUCTURAL_ABSTRACTION: evidence.held_out_transfer_successes > 0,
            DevelopmentalStage.HELD_OUT_TRANSFER: evidence.mature_consequences > 0,
            DevelopmentalStage.CONSEQUENCE_INTEGRATION: evidence.outcome_equivalences > 0 and evidence.learned_preferences > 0,
            DevelopmentalStage.OUTCOME_AND_PREFERENCE: evidence.alternative_strategies > 0,
            DevelopmentalStage.ALTERNATIVE_STRATEGIES: evidence.demonstrated_replans > 0 and evidence.efficient_replans > 0,
            DevelopmentalStage.DEMONSTRATED_REPLANNING: False,
        }
        return gates[stage]

    def close_interval(self, evidence: StageEvidence, *, evidence_watermark: int) -> StageSnapshot:
        before = self.stage
        after = DevelopmentalStage(min(7, int(before) + 1)) if self._eligible(before, evidence) else before
        row = StageSnapshot(self.interval_id, int(evidence_watermark), before, after, evidence)
        self.history.append(row)
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        self.interval_id += 1
        self.stage = after
        return row

    def state_dict(self) -> dict[str, object]:
        return {
            "stage": int(self.stage),
            "interval_id": self.interval_id,
            "history_limit": self.history_limit,
            "history": [
                {
                    "interval_id": row.interval_id,
                    "evidence_watermark": row.evidence_watermark,
                    "stage": int(row.stage),
                    "next_stage": int(row.next_stage),
                    "evidence": asdict(row.evidence),
                }
                for row in self.history
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "DevelopmentalStageTracker":
        result = cls(history_limit=int(state.get("history_limit", 1024)))
        result.stage = DevelopmentalStage(int(state.get("stage", 0)))
        result.interval_id = int(state.get("interval_id", 0))
        result.history = [
            StageSnapshot(
                int(raw["interval_id"]),
                int(raw["evidence_watermark"]),
                DevelopmentalStage(int(raw["stage"])),
                DevelopmentalStage(int(raw["next_stage"])),
                StageEvidence(**{key: int(value) for key, value in raw["evidence"].items()}),
            )
            for raw in state.get("history", [])
        ]
        return result

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .developmental_stage import DevelopmentalStage


@dataclass(frozen=True, slots=True)
class ISFComponents:
    pvi: float
    osi: float
    pe: float
    lv: float
    tp: float
    ep: float

    def values(self) -> tuple[float, ...]:
        values = (self.pvi, self.osi, self.pe, self.lv, self.tp, self.ep)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("ISF components must be finite and non-negative")
        return values


@dataclass(frozen=True, slots=True)
class ISFDecision:
    decision_watermark: int
    evidence_availability_watermark: int
    raw_components: ISFComponents
    normalized_components: ISFComponents
    developmental_stage: DevelopmentalStage
    next_developmental_stage: DevelopmentalStage
    score_schema_version: int
    graph_generation: int
    score: float


class InteractionSignificanceFunction:
    """Causally snapshots developmental allocation scores, never utility."""

    def __init__(self, weights_by_stage: tuple[tuple[float, ...], ...], *, schema_version: int, hot_limit: int) -> None:
        if len(weights_by_stage) != 8 or any(len(row) != 6 for row in weights_by_stage):
            raise ValueError("ISF requires six weights for every developmental stage")
        if hot_limit <= 0:
            raise ValueError("ISF hot limit must be positive")
        self.weights_by_stage = tuple(tuple(float(value) for value in row) for row in weights_by_stage)
        self.schema_version = int(schema_version)
        self.hot_limit = int(hot_limit)
        self.decisions: list[ISFDecision] = []
        self.component_maxima = [1.0] * 6

    def score(
        self,
        raw: ISFComponents,
        *,
        decision_watermark: int,
        evidence_availability_watermark: int,
        stage: DevelopmentalStage,
        next_stage: DevelopmentalStage,
        graph_generation: int,
    ) -> ISFDecision:
        values = raw.values()
        normalized_values: list[float] = []
        for index, value in enumerate(values):
            # The maxima used here contain only evidence causally available at
            # this decision. Updating them afterward cannot rewrite the row.
            normalized_values.append(min(1.0, value / self.component_maxima[index]))
        normalized = ISFComponents(*normalized_values)
        weights = self.weights_by_stage[int(stage)]
        score = sum(value * weight for value, weight in zip(normalized.values(), weights)) / sum(weights)
        row = ISFDecision(int(decision_watermark), int(evidence_availability_watermark), raw, normalized, stage, next_stage, self.schema_version, int(graph_generation), score)
        self.decisions.append(row)
        if len(self.decisions) > self.hot_limit:
            del self.decisions[: len(self.decisions) - self.hot_limit]
        for index, value in enumerate(values):
            self.component_maxima[index] = max(self.component_maxima[index], value)
        return row

    def state_dict(self) -> dict[str, object]:
        return {
            "weights_by_stage": [list(row) for row in self.weights_by_stage],
            "schema_version": self.schema_version,
            "hot_limit": self.hot_limit,
            "component_maxima": list(self.component_maxima),
            "decisions": [
                {
                    **asdict(row),
                    "developmental_stage": int(row.developmental_stage),
                    "next_developmental_stage": int(row.next_developmental_stage),
                }
                for row in self.decisions
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "InteractionSignificanceFunction":
        result = cls(tuple(tuple(float(value) for value in row) for row in state["weights_by_stage"]), schema_version=int(state["schema_version"]), hot_limit=int(state["hot_limit"]))
        result.component_maxima = [float(value) for value in state.get("component_maxima", [1.0] * 6)]
        result.decisions = [
            ISFDecision(
                int(raw["decision_watermark"]),
                int(raw["evidence_availability_watermark"]),
                ISFComponents(**{key: float(value) for key, value in raw["raw_components"].items()}),
                ISFComponents(**{key: float(value) for key, value in raw["normalized_components"].items()}),
                DevelopmentalStage(int(raw["developmental_stage"])),
                DevelopmentalStage(int(raw["next_developmental_stage"])),
                int(raw["score_schema_version"]),
                int(raw["graph_generation"]),
                float(raw["score"]),
            )
            for raw in state.get("decisions", [])
        ]
        return result

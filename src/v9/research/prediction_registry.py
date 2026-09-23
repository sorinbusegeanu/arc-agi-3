from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Mapping


class PredictionResult(str, Enum):
    SUPPORTED = "SUPPORTED"
    VIOLATED = "VIOLATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class PredictionDefinition:
    prediction_id: str
    criterion_name: str
    observables: tuple[str, ...]
    controls: tuple[str, ...]
    statistic: str
    threshold: str
    evidence_artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PredictionAssessment:
    prediction_id: str
    result: PredictionResult
    evidence_artifacts: tuple[str, ...]
    detail: str


_CRITERIA = (
    ("F1", "Semantic-Prior Necessity"),
    ("F2", "Object-First Emergence"),
    ("F3", "Appearance-Dominated Transfer"),
    ("F4", "World-Model Necessity"),
    ("F5", "Failure of Prediction-Violation Allocation"),
    ("F6", "Failure of Future-Option Contribution"),
    ("F7", "Failure of Explanatory Reach"),
    ("F8", "Failure of Context Refinement"),
    ("F9", "Failure of Empirical Transfer Validation"),
    ("F10", "Failure of Developmental Ordering"),
    ("F11", "Failure of Outcome/Strategy Separation"),
    ("F12", "Failure of Emergent Target-Like Structure"),
    ("F13", "Failure of Efficiency Emergence"),
    ("F14", "Failure of Conditional Compression"),
    ("F15", "Persistent Architectural Thrashing"),
    ("F16", "Failure of Grounded Symbolic Emergence"),
    ("F17", "Failure of Developmental Stability"),
    ("F18", "Failure or Mischaracterization of Structural-Prior Dependence"),
    ("H19-REJECT", "Learned Relational Reasoning Rejection"),
)


class ResearchPredictionRegistry:
    def __init__(self, definitions: tuple[PredictionDefinition, ...] | None = None) -> None:
        self.definitions = definitions or tuple(
            PredictionDefinition(
                prediction_id,
                name,
                (f"{prediction_id.lower()}_primary_observable",),
                (f"{prediction_id.lower()}_matched_control",),
                "declared preregistered statistic",
                "declared preregistered threshold",
                (f"research/{prediction_id.lower()}/assessment.json",),
            )
            for prediction_id, name in _CRITERIA
        )
        identifiers = tuple(row.prediction_id for row in self.definitions)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("prediction identifiers must be unique")

    def definition(self, prediction_id: str) -> PredictionDefinition:
        try:
            return next(row for row in self.definitions if row.prediction_id == prediction_id)
        except StopIteration as exc:
            raise KeyError(prediction_id) from exc

    def assess(
        self,
        prediction_id: str,
        *,
        observable_values: Mapping[str, float] | None,
        controls_present: bool,
        threshold_passed: bool | None,
        evidence_artifacts: tuple[str, ...] = (),
    ) -> PredictionAssessment:
        definition = self.definition(prediction_id)
        missing = observable_values is None or any(name not in observable_values for name in definition.observables)
        if missing or not controls_present or threshold_passed is None or not evidence_artifacts:
            return PredictionAssessment(
                prediction_id,
                PredictionResult.INSUFFICIENT_EVIDENCE,
                evidence_artifacts,
                "missing observable, matched control, threshold result, or evidence artifact",
            )
        return PredictionAssessment(
            prediction_id,
            PredictionResult.SUPPORTED if threshold_passed else PredictionResult.VIOLATED,
            evidence_artifacts,
            "preregistered threshold evaluated",
        )


MILESTONE_KINDS = (
    "contingency",
    "family",
    "carrier",
    "role",
    "concept_candidate",
    "validated_concept",
    "future_option_motif",
    "planning_replanning_effectiveness",
    "m5_consequence",
    "m6_outcome",
    "m7_strategy",
)


@dataclass(frozen=True, slots=True)
class DevelopmentalMilestone:
    kind: str
    scientific_evidence_id: str
    canonical_generation: int
    canonical_lsn: int
    evidence_artifact: str
    observed_ordinal: int


class DevelopmentalMilestoneLedger:
    """Records first qualified observations; it never imposes an expected order."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self._lock = RLock()
        self._records: dict[str, DevelopmentalMilestone] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = DevelopmentalMilestone(**json.loads(line))
                    self._records.setdefault(row.kind, row)

    @property
    def records(self) -> tuple[DevelopmentalMilestone, ...]:
        return tuple(sorted(self._records.values(), key=lambda row: row.observed_ordinal))

    def record_first(
        self,
        *,
        kind: str,
        scientific_evidence_id: str,
        canonical_generation: int,
        canonical_lsn: int,
        evidence_artifact: str,
    ) -> DevelopmentalMilestone:
        if kind not in MILESTONE_KINDS:
            raise ValueError(f"unknown developmental milestone: {kind}")
        with self._lock:
            existing = self._records.get(kind)
            if existing is not None:
                return existing
            row = DevelopmentalMilestone(
                kind,
                scientific_evidence_id,
                int(canonical_generation),
                int(canonical_lsn),
                evidence_artifact,
                len(self._records) + 1,
            )
            self._records[kind] = row
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            return row


__all__ = [
    "DevelopmentalMilestone",
    "DevelopmentalMilestoneLedger",
    "MILESTONE_KINDS",
    "PredictionAssessment",
    "PredictionDefinition",
    "PredictionResult",
    "ResearchPredictionRegistry",
]

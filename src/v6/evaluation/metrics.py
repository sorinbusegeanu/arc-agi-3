from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from v6.contingency.contingency_learner import Contingency
from v6.transformation.transformation_clusterer import TransformationFamily


@dataclass(frozen=True)
class MetricsSnapshot:
    transformation_family_count: int
    average_family_support: float
    prediction_accuracy: float | None
    stable_contingency_count: int
    contingency_confidences: list[float]


def compute_metrics(
    *,
    families: list[TransformationFamily],
    contingencies: list[Contingency],
    connection: sqlite3.Connection,
) -> MetricsSnapshot:
    family_count = len(families)
    average_support = 0.0
    if families:
        average_support = sum(family.support_count for family in families) / len(families)

    rows = connection.execute(
        """
        SELECT prediction_error
        FROM prediction_results
        WHERE prediction_error IS NOT NULL
        """
    ).fetchall()
    prediction_accuracy: float | None = None
    if rows:
        errors = [int(row[0]) for row in rows]
        prediction_accuracy = 1.0 - (sum(errors) / len(errors))

    return MetricsSnapshot(
        transformation_family_count=family_count,
        average_family_support=float(average_support),
        prediction_accuracy=prediction_accuracy,
        stable_contingency_count=len(contingencies),
        contingency_confidences=[float(contingency.confidence) for contingency in contingencies],
    )

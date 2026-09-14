from __future__ import annotations

from typing import Any, Iterable

from v9.cognition.isf import ISFComponents, ISFDecision


def score_isf_batch(
    runtime: Any,
    rows: Iterable[tuple[ISFComponents, int, int, Any, Any, int]],
) -> None:
    batch = tuple(rows)
    if not batch:
        return
    isf = runtime.isf
    maxima = list(isf.component_maxima)
    decisions: list[ISFDecision] = []
    for raw, decision_watermark, evidence_watermark, stage, next_stage, graph_generation in batch:
        values = raw.values()
        normalized_values = [min(1.0, value / maxima[index]) for index, value in enumerate(values)]
        normalized = ISFComponents(*normalized_values)
        weights = isf.weights_by_stage[int(stage)]
        score = sum(value * weight for value, weight in zip(normalized_values, weights)) / sum(weights)
        decisions.append(
            ISFDecision(
                int(decision_watermark),
                int(evidence_watermark),
                raw,
                normalized,
                stage,
                next_stage,
                int(isf.schema_version),
                int(graph_generation),
                score,
            )
        )
        for index, value in enumerate(values):
            maxima[index] = max(maxima[index], value)
    isf.decisions.extend(decisions)
    overflow = len(isf.decisions) - int(isf.hot_limit)
    if overflow > 0:
        del isf.decisions[:overflow]
    isf.component_maxima = maxima

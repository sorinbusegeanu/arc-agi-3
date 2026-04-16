from __future__ import annotations

from collections import Counter
from typing import Any

from v5_0.contracts.avatar_types import HUDHintSummary


def build_hud_hint_summaries(
    hud_regions,
    hud_mask,
    raw_hud_value_samples=None,
    cross_reset_hud_evidence=None,
) -> tuple[HUDHintSummary, ...]:
    samples_by_region = _samples_by_region(raw_hud_value_samples, hud_regions)
    evidence_hist = _evidence_histogram_by_region(cross_reset_hud_evidence)

    summaries: list[HUDHintSummary] = []
    for region in hud_regions:
        region_id = str(region.hud_region_id)
        histogram = Counter(int(k) for k, v in getattr(region, "value_histogram", {}).items() for _ in range(int(v)))
        histogram.update(samples_by_region.get(region_id, Counter()))
        if region_id in evidence_hist:
            histogram.update(evidence_hist[region_id])
        value_hist = dict(sorted((int(k), int(v)) for k, v in histogram.items() if int(v) > 0))
        dominant = tuple(
            key
            for key, _ in sorted(
                value_hist.items(),
                key=lambda item: (-int(item[1]), int(item[0])),
            )[:3]
        )
        stable_value_count = sum(1 for _, count in value_hist.items() if int(count) >= 2)
        summaries.append(
            HUDHintSummary(
                hud_region_id=region_id,
                bbox=tuple(region.bbox),
                edge_side=str(region.edge_side),
                value_histogram=value_hist,
                dominant_values=dominant,
                stable_value_count=int(stable_value_count),
                confidence=float(getattr(region, "confidence", 0.0)),
            )
        )
    summaries.sort(key=lambda item: (-item.confidence, item.edge_side, item.hud_region_id))
    return tuple(summaries)


def build_internal_hud_hint_summaries(hud_hints: tuple[HUDHintSummary, ...]) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for hint in hud_hints:
        out.append(
            {
                "hud_region_id": str(hint.hud_region_id),
                "bbox": tuple(hint.bbox),
                "edge_side": str(hint.edge_side),
                "value_histogram": dict(hint.value_histogram),
                "dominant_values": tuple(hint.dominant_values),
                "stable_value_count": int(hint.stable_value_count),
                "confidence": float(hint.confidence),
                "value_entropy_proxy": _value_entropy_proxy(hint.value_histogram),
            }
        )
    out.sort(key=lambda item: (-float(item["confidence"]), str(item["edge_side"]), str(item["hud_region_id"])))
    return tuple(out)


def _samples_by_region(raw_hud_value_samples, hud_regions) -> dict[str, Counter[int]]:
    if raw_hud_value_samples is None:
        return {}
    rows_by_region: dict[str, set[int]] = {}
    cols_by_region: dict[str, set[int]] = {}
    for region in hud_regions:
        x0, y0, x1, y1 = region.bbox
        rows_by_region[str(region.hud_region_id)] = set(range(y0, y1 + 1))
        cols_by_region[str(region.hud_region_id)] = set(range(x0, x1 + 1))

    out: dict[str, Counter[int]] = {str(region.hud_region_id): Counter() for region in hud_regions}
    flattened = []
    if isinstance(raw_hud_value_samples, dict):
        for values in raw_hud_value_samples.values():
            flattened.extend(values)
    else:
        flattened.extend(raw_hud_value_samples)
    for sample in flattened:
        row = int(getattr(sample, "row", -1))
        col = int(getattr(sample, "col", -1))
        value = int(getattr(sample, "value", 0))
        for region_id in out:
            if row in rows_by_region[region_id] and col in cols_by_region[region_id]:
                out[region_id][value] += 1
    return out


def _evidence_histogram_by_region(cross_reset_hud_evidence) -> dict[str, Counter[int]]:
    out: dict[str, Counter[int]] = {}
    if not cross_reset_hud_evidence:
        return out
    for item in cross_reset_hud_evidence:
        region_id = str(getattr(item, "canonical_region_id", ""))
        hist = getattr(item, "value_histogram_aggregate", {})
        if not region_id:
            continue
        out[region_id] = Counter({int(k): int(v) for k, v in dict(hist).items()})
    return out


def _value_entropy_proxy(value_histogram: dict[int, int]) -> float:
    total = sum(max(0, int(v)) for v in value_histogram.values())
    if total <= 0:
        return 1.0
    probs = [int(v) / total for v in value_histogram.values() if int(v) > 0]
    if not probs:
        return 1.0
    dominance = max(probs)
    unique = len(probs)
    breadth = min(1.0, max(0.0, (unique - 1) / 6.0))
    # Lower entropy proxy = more specific (single-value hints trend toward 0).
    return float(max(0.0, min(1.0, 0.7 * (1.0 - dominance) + 0.3 * breadth)))

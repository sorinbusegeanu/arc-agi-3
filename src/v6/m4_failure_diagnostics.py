from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


EXPECTED_FAILURE_BUCKETS = [
    "no_source_neighborhoods",
    "no_source_roles",
    "source_role_map_family_id_mismatch",
    "manifest_resolution_failure",
    "insufficient_stable_role_items",
    "subcomposition_generation_failure",
    "no_target_projection",
    "no_lift_vs_best_individual_role",
    "no_lift_vs_unordered_role_bag",
    "no_lift_vs_surface_effect_raw",
    "no_positive_compression_gain",
    "no_future_option_prediction_lift",
    "insufficient_explained_m2_families",
    "insufficient_positive_lift_families",
    "insufficient_games",
    "insufficient_manifest_families",
]


def count_by_reason(rows: Iterable[Mapping[str, Any]], field: str = "rejection_reason") -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        reason = row.get(field)
        if not reason:
            continue
        if isinstance(reason, list):
            for item in reason:
                if item:
                    counter[str(item)] += 1
        else:
            counter[str(reason)] += 1
    return dict(counter)


def merge_reason_counts(*counts: Mapping[str, int]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for count_map in counts:
        for key, value in count_map.items():
            merged[str(key)] += int(value)
    return dict(merged)


def ensure_failure_buckets(counts: Mapping[str, int]) -> dict[str, int]:
    result = {reason: 0 for reason in EXPECTED_FAILURE_BUCKETS}
    result.update({str(key): int(value) for key, value in counts.items()})
    return result

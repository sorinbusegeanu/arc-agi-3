from __future__ import annotations

from collections import Counter

from v5_0.contracts.avatar_types import HUDPOIMatch
from v5_0.hud.hint_summary import build_internal_hud_hint_summaries


def match_hud_hints_to_pois(
    hud_hints,
    ranked_poi_candidates,
    cross_reset_poi_evidence=None,
) -> dict[str, object]:
    support_by_poi = _support_by_poi(cross_reset_poi_evidence)
    matches: list[HUDPOIMatch] = []
    rejected = Counter()

    for internal_hint in build_internal_hud_hint_summaries(tuple(hud_hints)):
        per_region_candidates: list[dict[str, object]] = []
        for poi in ranked_poi_candidates:
            value_overlap_score = _hist_overlap(dict(internal_hint["value_histogram"]), poi.value_histogram)
            dominant_value_match_score = _dominant_overlap(tuple(internal_hint["dominant_values"]), tuple(sorted(poi.value_histogram.keys())))
            if value_overlap_score < 0.05 and dominant_value_match_score < 0.15:
                rejected["no_value_overlap"] += 1
                continue

            value_precision_score = _compute_value_precision_score(dict(internal_hint["value_histogram"]), poi.value_histogram)
            poi_purity_score = _compute_poi_purity_score(poi)
            structural_compatibility_score = _structural_compatibility(
                tuple(internal_hint["bbox"]),
                int(internal_hint["stable_value_count"]),
                poi.bbox,
                int(poi.area),
            )
            support_episode_count = int(support_by_poi.get(poi.poi_id, len(getattr(poi, "support_episode_indices", ()))))
            support_score = min(1.0, support_episode_count / 3.0)
            base_confidence = max(
                0.0,
                min(
                    1.0,
                    0.24 * value_overlap_score
                    + 0.14 * dominant_value_match_score
                    + 0.23 * value_precision_score
                    + 0.17 * structural_compatibility_score
                    + 0.12 * poi_purity_score
                    + 0.10 * support_score,
                ),
            )
            flags: list[str] = []
            if structural_compatibility_score < 0.35:
                flags.append("structural_mismatch")
            if dominant_value_match_score < 0.15:
                flags.append("weak_dominant_match")
            if value_precision_score < 0.2:
                flags.append("low_value_precision")
            per_region_candidates.append(
                {
                    "poi": poi,
                    "base_confidence": float(base_confidence),
                    "value_overlap_score": float(value_overlap_score),
                    "dominant_value_match_score": float(dominant_value_match_score),
                    "value_precision_score": float(value_precision_score),
                    "structural_compatibility_score": float(structural_compatibility_score),
                    "poi_purity_score": float(poi_purity_score),
                    "support_episode_count": int(support_episode_count),
                    "flags": tuple(sorted(flags)),
                }
            )

        specificity_penalties = _compute_region_specificity_score(internal_hint, per_region_candidates)
        for candidate in per_region_candidates:
            poi = candidate["poi"]
            penalty = float(specificity_penalties.get(str(poi.poi_id), 0.0))
            confidence = max(0.0, min(1.0, float(candidate["base_confidence"]) - penalty))
            flags = set(candidate["flags"])
            if penalty > 0.0:
                flags.add("region_specificity_penalty")
            matches.append(
                HUDPOIMatch(
                    hud_region_id=str(internal_hint["hud_region_id"]),
                    poi_id=str(poi.poi_id),
                    hud_values=dict(sorted((int(k), int(v)) for k, v in dict(internal_hint["value_histogram"]).items())),
                    poi_values=dict(sorted((int(k), int(v)) for k, v in poi.value_histogram.items())),
                    value_overlap_score=float(candidate["value_overlap_score"]),
                    dominant_value_match_score=float(candidate["dominant_value_match_score"]),
                    structural_compatibility_score=float(candidate["structural_compatibility_score"]),
                    support_episode_count=int(candidate["support_episode_count"]),
                    confidence=float(confidence),
                    ambiguity_flags=tuple(sorted(flags)),
                    value_precision_score=float(candidate["value_precision_score"]),
                    poi_purity_score=float(candidate["poi_purity_score"]),
                    region_specificity_penalty=float(penalty),
                )
            )

    matches.sort(key=lambda item: (-item.confidence, item.hud_region_id, item.poi_id))
    return {
        "matches": tuple(matches),
        "rejected_match_reasons": dict(sorted((str(k), int(v)) for k, v in rejected.items())),
    }


def _support_by_poi(cross_reset_poi_evidence) -> dict[str, int]:
    out: dict[str, int] = {}
    if not cross_reset_poi_evidence:
        return out
    for item in cross_reset_poi_evidence:
        canonical = str(getattr(item, "canonical_poi_id", ""))
        support = int(getattr(item, "support_episode_count", 0))
        if canonical:
            out[canonical] = max(int(out.get(canonical, 0)), support)
    return out


def _hist_overlap(left: dict[int, int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(int(k) for k in left) | set(int(k) for k in right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return float(overlap / max(total, 1))


def _dominant_overlap(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    l = set(int(v) for v in left[:3])
    r = set(int(v) for v in right[:5])
    overlap = len(l & r)
    return float(overlap / max(1, min(len(l), len(r))))


def _compute_value_precision_score(hud_values, poi_values) -> float:
    if not hud_values or not poi_values:
        return 0.0
    overlap_keys = set(int(k) for k in hud_values) & set(int(k) for k in poi_values)
    if not overlap_keys:
        return 0.0
    hud_total = sum(max(0, int(v)) for v in hud_values.values())
    poi_total = sum(max(0, int(v)) for v in poi_values.values())
    poi_extra = sum(max(0, int(v)) for k, v in poi_values.items() if int(k) not in overlap_keys)
    hud_unique = len([k for k, v in hud_values.items() if int(v) > 0])
    poi_unique = len([k for k, v in poi_values.items() if int(v) > 0])
    overlap_mass = sum(min(int(hud_values.get(k, 0)), int(poi_values.get(k, 0))) for k in overlap_keys)
    overlap_ratio = overlap_mass / max(1.0, float(hud_total))
    dominant_hud = sorted(hud_values.items(), key=lambda item: (-int(item[1]), int(item[0])))[:2]
    dominant_poi = sorted(poi_values.items(), key=lambda item: (-int(item[1]), int(item[0])))[:3]
    dominant_score = _dominant_overlap(tuple(int(k) for k, _ in dominant_hud), tuple(int(k) for k, _ in dominant_poi))
    narrow_hud_bonus = max(0.0, min(1.0, 1.0 - (max(1, hud_unique) - 1) / 5.0))
    extra_penalty = poi_extra / max(1.0, float(poi_total))
    broad_poi_penalty = max(0.0, min(1.0, (max(1, poi_unique) - max(1, hud_unique)) / 8.0))
    return float(
        max(
            0.0,
            min(
                1.0,
                0.45 * overlap_ratio + 0.30 * dominant_score + 0.25 * narrow_hud_bonus - 0.35 * extra_penalty - 0.20 * broad_poi_penalty,
            ),
        )
    )


def _compute_region_specificity_score(hud_hint, candidate_matches_for_same_hud_region) -> dict[str, float]:
    if not candidate_matches_for_same_hud_region:
        return {}
    ordered = sorted(
        candidate_matches_for_same_hud_region,
        key=lambda item: (-float(item["base_confidence"]), str(item["poi"].poi_id)),
    )
    if len(ordered) == 1:
        return {str(ordered[0]["poi"].poi_id): 0.0}
    best = float(ordered[0]["base_confidence"])
    second = float(ordered[1]["base_confidence"])
    closeness = max(0.0, 1.0 - abs(best - second) / 0.20)
    bbox = tuple(hud_hint["bbox"])
    area = max(1, (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
    tiny_region_factor = max(0.0, min(1.0, 1.0 - (area - 1) / 8.0))
    entropy_proxy = float(hud_hint.get("value_entropy_proxy", 0.5))
    specificity = max(0.0, min(1.0, 1.0 - entropy_proxy))
    penalties: dict[str, float] = {str(item["poi"].poi_id): 0.0 for item in ordered}
    for rank, item in enumerate(ordered):
        poi_id = str(item["poi"].poi_id)
        if rank == 0:
            penalties[poi_id] = 0.0
            continue
        base_gap = max(0.0, best - float(item["base_confidence"]))
        weak_diff = max(0.0, 1.0 - base_gap / 0.20)
        penalties[poi_id] = float(max(0.0, min(0.40, 0.30 * closeness * weak_diff * (0.4 + 0.6 * specificity) * (0.4 + 0.6 * tiny_region_factor))))
    return penalties


def _compute_poi_purity_score(poi_candidate) -> float:
    hist = dict(getattr(poi_candidate, "value_histogram", {}))
    if not hist:
        return 0.0
    total = sum(max(0, int(v)) for v in hist.values())
    if total <= 0:
        return 0.0
    dominant = max(max(0, int(v)) for v in hist.values())
    dominance = dominant / max(1.0, float(total))
    unique = len([k for k, v in hist.items() if int(v) > 0])
    compact = _bbox_compactness(tuple(getattr(poi_candidate, "bbox")))
    mixed_penalty = max(0.0, min(1.0, (max(1, unique) - 1) / 10.0))
    return float(max(0.0, min(1.0, 0.45 * dominance + 0.35 * compact + 0.20 * (1.0 - mixed_penalty))))


def _structural_compatibility(
    hud_bbox: tuple[int, int, int, int],
    hud_stable_value_count: int,
    poi_bbox: tuple[int, int, int, int],
    poi_area: int,
) -> float:
    hw = max(1, hud_bbox[2] - hud_bbox[0] + 1)
    hh = max(1, hud_bbox[3] - hud_bbox[1] + 1)
    hud_area = hw * hh
    poi_area = max(1, int(poi_area))
    ratio = max(hud_area, poi_area) / max(1.0, min(hud_area, poi_area))
    ratio_score = max(0.0, min(1.0, 1.0 - (ratio - 1.0) / 12.0))
    hud_compact = max(0.0, min(1.0, min(hw, hh) / max(hw, hh)))
    poi_w = max(1, poi_bbox[2] - poi_bbox[0] + 1)
    poi_h = max(1, poi_bbox[3] - poi_bbox[1] + 1)
    poi_compact = max(0.0, min(1.0, min(poi_w, poi_h) / max(poi_w, poi_h)))
    compact_score = max(0.0, 1.0 - abs(hud_compact - poi_compact))
    stable_bonus = min(1.0, max(0, int(hud_stable_value_count)) / 3.0)
    return float(max(0.0, min(1.0, 0.65 * ratio_score + 0.25 * compact_score + 0.10 * stable_bonus)))


def _bbox_compactness(bbox: tuple[int, int, int, int]) -> float:
    w = max(1, bbox[2] - bbox[0] + 1)
    h = max(1, bbox[3] - bbox[1] + 1)
    return max(0.0, min(1.0, min(w, h) / max(w, h)))

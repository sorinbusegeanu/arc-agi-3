from __future__ import annotations

from collections import defaultdict
from typing import Any

from v5_0.contracts.avatar_types import HUDTargetSelection


def rank_pois_from_hud_matches(
    hud_poi_matches,
    ranked_poi_candidates,
    matches_by_hud_region=None,
    return_details: bool = False,
) -> HUDTargetSelection | tuple[HUDTargetSelection, dict[str, Any]]:
    if not hud_poi_matches:
        selection = HUDTargetSelection(
            selected_poi_id=None,
            ranked_poi_ids=(),
            top_match_hud_region_ids=(),
            ambiguous=False,
            failure_reason="no_hud_poi_match",
        )
        if return_details:
            return selection, {
                "best_match_win_counts": {},
                "aggregate_score_gap": 0.0,
                "selected_match_margin": 0.0,
                "win_gap": 0,
            }
        return selection

    if matches_by_hud_region is None:
        matches_by_hud_region = _group_matches_by_region(hud_poi_matches)

    aggregate = _aggregate_match_evidence_by_unique_hud_region(hud_poi_matches)
    win_counts, near_second_counts = _compute_best_match_win_count(matches_by_hud_region)

    by_poi: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "score": 0.0,
            "matches": [],
        }
    )
    for poi_id, item in aggregate.items():
        wins = int(win_counts.get(poi_id, 0))
        second_penalty = 0.06 * int(near_second_counts.get(poi_id, 0))
        poi_consistency_bonus = float(item["poi_consistency_bonus"])
        best_per_region_bonus = float(item["best_per_region_bonus"])
        score = (
            0.52 * float(item["summed_confidence"])
            + 0.22 * float(item["unique_region_count"])
            + 0.14 * best_per_region_bonus
            + 0.12 * poi_consistency_bonus
            + 0.18 * wins
            - second_penalty
        )
        by_poi[poi_id]["score"] = float(score)

    for match in hud_poi_matches:
        by_poi[str(match.poi_id)]["matches"].append(match)

    candidate_order = {str(candidate.poi_id): index for index, candidate in enumerate(ranked_poi_candidates)}
    scored: list[tuple[str, float]] = []
    for poi_id, slot in by_poi.items():
        scored.append((poi_id, float(slot["score"])))

    scored.sort(key=lambda item: (-item[1], candidate_order.get(item[0], 10**9), item[0]))
    ranked_ids = tuple(poi_id for poi_id, _ in scored)
    selected = ranked_ids[0] if ranked_ids else None
    top_region_ids = ()
    if selected is not None:
        top_matches = sorted(
            by_poi[selected]["matches"],
            key=lambda item: (-item.confidence, item.hud_region_id, item.poi_id),
        )
        top_region_ids = tuple(dict.fromkeys(str(item.hud_region_id) for item in top_matches))

    ambiguous = False
    failure_reason = None
    score_gap, win_gap, combined_margin = _compute_selection_margin(scored, win_counts)
    if len(scored) >= 2:
        if score_gap <= 0.10 and abs(win_gap) <= 1 and combined_margin <= 0.18:
            ambiguous = True
            failure_reason = "ambiguous_target"
    if selected is None:
        failure_reason = "no_hud_poi_match"

    selection = HUDTargetSelection(
        selected_poi_id=selected,
        ranked_poi_ids=ranked_ids,
        top_match_hud_region_ids=top_region_ids,
        ambiguous=bool(ambiguous),
        failure_reason=failure_reason,
    )
    if return_details:
        return selection, {
            "best_match_win_counts": dict(sorted((str(k), int(v)) for k, v in win_counts.items())),
            "aggregate_score_gap": float(score_gap),
            "selected_match_margin": float(combined_margin),
            "win_gap": int(win_gap),
        }
    return selection


def _aggregate_match_evidence_by_unique_hud_region(matches) -> dict[str, dict[str, float]]:
    by_poi: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "regions": set(),
            "summed_confidence": 0.0,
            "best_per_region_bonus": 0.0,
            "consistency_inputs": [],
        }
    )
    by_region = _group_matches_by_region(matches)
    for region_id, region_matches in by_region.items():
        ordered = sorted(region_matches, key=lambda item: (-item.confidence, item.poi_id))
        best_conf = float(ordered[0].confidence) if ordered else 0.0
        for index, match in enumerate(ordered):
            slot = by_poi[str(match.poi_id)]
            slot["regions"].add(region_id)
            slot["summed_confidence"] += float(match.confidence)
            if index == 0:
                slot["best_per_region_bonus"] += float(max(0.0, best_conf))
            slot["consistency_inputs"].append(float(match.value_precision_score) * 0.5 + float(match.poi_purity_score) * 0.5)
    out: dict[str, dict[str, float]] = {}
    for poi_id, item in by_poi.items():
        consistency = 0.0
        if item["consistency_inputs"]:
            mean_val = sum(item["consistency_inputs"]) / len(item["consistency_inputs"])
            spread = max(item["consistency_inputs"]) - min(item["consistency_inputs"])
            consistency = max(0.0, min(1.0, mean_val - 0.2 * spread))
        out[poi_id] = {
            "unique_region_count": float(len(item["regions"])),
            "summed_confidence": float(item["summed_confidence"]),
            "best_per_region_bonus": float(item["best_per_region_bonus"]),
            "poi_consistency_bonus": float(consistency),
        }
    return out


def _compute_best_match_win_count(matches_by_hud_region) -> tuple[dict[str, int], dict[str, int]]:
    wins: dict[str, int] = defaultdict(int)
    near_seconds: dict[str, int] = defaultdict(int)
    for _, region_matches in matches_by_hud_region.items():
        ordered = sorted(region_matches, key=lambda item: (-item.confidence, item.poi_id))
        if not ordered:
            continue
        wins[str(ordered[0].poi_id)] += 1
        if len(ordered) > 1 and abs(float(ordered[0].confidence) - float(ordered[1].confidence)) <= 0.05:
            near_seconds[str(ordered[1].poi_id)] += 1
    return dict(wins), dict(near_seconds)


def _compute_selection_margin(scored: list[tuple[str, float]], win_counts: dict[str, int]) -> tuple[float, int, float]:
    if len(scored) < 2:
        return (0.0 if not scored else float(scored[0][1]), int(win_counts.get(scored[0][0], 0)) if scored else 0, 1.0 if scored else 0.0)
    first_id, first_score = scored[0]
    second_id, second_score = scored[1]
    score_gap = float(first_score - second_score)
    win_gap = int(win_counts.get(first_id, 0) - win_counts.get(second_id, 0))
    combined = float(score_gap + 0.08 * max(0, win_gap))
    return score_gap, win_gap, combined


def _group_matches_by_region(matches) -> dict[str, list]:
    by_region: dict[str, list] = defaultdict(list)
    for match in matches:
        by_region[str(match.hud_region_id)].append(match)
    return dict(by_region)

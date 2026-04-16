from __future__ import annotations

from collections import defaultdict

from v5_0.contracts.avatar_types import POIMechanicEvidence


def build_mechanic_evidence(
    ranked_poi_candidates,
    contact_experiment_report=None,
    hud_targeting_report=None,
    solve_report=None,
) -> tuple[POIMechanicEvidence, ...]:
    by_poi: dict[str, dict] = defaultdict(
        lambda: {
            "episode_index": 0,
            "contact_count": 0,
            "useful_change_count": 0,
            "no_effect_count": 0,
            "reward_change_count": 0,
            "object_removed_count": 0,
            "door_opens_count": 0,
            "level_transition_count": 0,
            "terminal_count": 0,
            "hud_match_confidence": 0.0,
            "support_step_indices": set(),
        }
    )

    for poi in tuple(ranked_poi_candidates or ()):
        slot = by_poi[str(poi.poi_id)]
        slot["episode_index"] = min(slot["episode_index"], min(getattr(poi, "support_episode_indices", (0,)) or (0,)))
        for idx in tuple(getattr(poi, "seen_step_indices", ())):
            slot["support_step_indices"].add(int(idx))

    if contact_experiment_report is not None:
        for tested in tuple(getattr(contact_experiment_report, "tested_pois", ())):
            poi_id = str(getattr(tested, "poi_id", ""))
            if not poi_id:
                continue
            outcome = getattr(tested, "outcome", None)
            slot = by_poi[poi_id]
            slot["episode_index"] = min(slot["episode_index"], int(getattr(tested, "episode_index", 0)))
            slot["contact_count"] += 1
            if outcome is None:
                continue
            outcome_type = str(getattr(outcome, "outcome_type", ""))
            if outcome_type in {"reward_change", "object_removed", "new_object_appeared", "door_opens", "level_transition", "terminal"}:
                slot["useful_change_count"] += 1
            if outcome_type in {"no_effect", "hud_change_only"}:
                slot["no_effect_count"] += 1
            if outcome_type == "reward_change":
                slot["reward_change_count"] += 1
            if outcome_type == "object_removed":
                slot["object_removed_count"] += 1
            if outcome_type == "door_opens":
                slot["door_opens_count"] += 1
            if outcome_type == "level_transition":
                slot["level_transition_count"] += 1
            if outcome_type == "terminal":
                slot["terminal_count"] += 1

    hud_conf = _hud_confidence_by_poi(hud_targeting_report)
    for poi_id, score in hud_conf.items():
        by_poi[poi_id]["hud_match_confidence"] = max(float(by_poi[poi_id]["hud_match_confidence"]), float(score))

    if solve_report is not None:
        for ep in tuple(getattr(solve_report, "episodes", ())):
            for step in tuple(getattr(ep, "steps", ())):
                poi_id = str(getattr(step, "target_poi_id", ""))
                if not poi_id:
                    continue
                slot = by_poi[poi_id]
                slot["episode_index"] = min(slot["episode_index"], int(getattr(ep, "episode_index", 0)))
                slot["support_step_indices"].add(int(getattr(step, "step_index", 0)))
                outcome_type = str(getattr(step, "outcome_type", ""))
                if outcome_type in {"no_effect", "hud_change_only"}:
                    slot["no_effect_count"] += 1
                if outcome_type in {"reward_change", "object_removed", "new_object_appeared", "door_opens", "level_transition", "terminal"}:
                    slot["useful_change_count"] += 1
                if outcome_type == "reward_change":
                    slot["reward_change_count"] += 1
                if outcome_type == "object_removed":
                    slot["object_removed_count"] += 1
                if outcome_type == "door_opens":
                    slot["door_opens_count"] += 1
                if outcome_type == "level_transition":
                    slot["level_transition_count"] += 1
                if outcome_type == "terminal":
                    slot["terminal_count"] += 1
                if bool(getattr(step, "contact_detected", False)):
                    slot["contact_count"] += 1

    out = []
    for poi_id in sorted(by_poi):
        item = by_poi[poi_id]
        out.append(
            POIMechanicEvidence(
                poi_id=poi_id,
                episode_index=int(item["episode_index"]),
                contact_count=int(item["contact_count"]),
                useful_change_count=int(item["useful_change_count"]),
                no_effect_count=int(item["no_effect_count"]),
                reward_change_count=int(item["reward_change_count"]),
                object_removed_count=int(item["object_removed_count"]),
                door_opens_count=int(item["door_opens_count"]),
                level_transition_count=int(item["level_transition_count"]),
                terminal_count=int(item["terminal_count"]),
                hud_match_confidence=float(item["hud_match_confidence"]),
                support_step_indices=tuple(sorted(int(v) for v in item["support_step_indices"])),
            )
        )
    return tuple(out)


def _hud_confidence_by_poi(hud_targeting_report) -> dict[str, float]:
    out: dict[str, float] = {}
    if hud_targeting_report is None:
        return out
    matches = tuple(getattr(hud_targeting_report, "matches", ()) or ())
    if matches:
        for match in matches:
            poi_id = str(getattr(match, "poi_id", ""))
            if not poi_id:
                continue
            out[poi_id] = max(out.get(poi_id, 0.0), float(getattr(match, "confidence", 0.0)))
        return out

    selected = getattr(hud_targeting_report, "selected", None)
    if selected is not None:
        selected_id = getattr(selected, "selected_poi_id", None)
        if selected_id is not None:
            conf = 1.0 if not bool(getattr(selected, "ambiguous", True)) else 0.5
            out[str(selected_id)] = conf
    return out

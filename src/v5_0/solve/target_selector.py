from __future__ import annotations

from collections import Counter

from v5_0.contracts.avatar_types import SolveTargetState
from v5_0.mechanics.decision import select_target_from_mechanic_memory


def select_initial_target(
    hud_targeting_report,
    ranked_poi_candidates,
    contact_experiment_report=None,
    mechanic_report=None,
    adaptive_episode_history=None,
) -> SolveTargetState | None:
    candidates = tuple(_valid_world_pois(ranked_poi_candidates))
    if not candidates:
        return None

    no_effect_counts = _no_effect_counts(contact_experiment_report)
    for step in tuple(adaptive_episode_history or ()):
        if str(getattr(step, "outcome_type", "")) in {"no_effect", "hud_change_only"}:
            pid = str(getattr(step, "target_poi_id", ""))
            if pid:
                no_effect_counts[pid] += 1
    mechanic_memory = getattr(mechanic_report, "memory", None) if mechanic_report is not None else None
    if mechanic_memory is not None:
        decision = select_target_from_mechanic_memory(mechanic_memory, hud_targeting_report, candidates)
        selected_id = decision.selected_poi_id
        if selected_id is not None and no_effect_counts.get(selected_id, 0) < 2:
            return SolveTargetState(
                target_poi_id=str(selected_id),
                source="mechanic_memory",
                confidence=float(decision.confidence),
                attempt_count=0,
                last_outcome_type=None,
                active=True,
                route_feasibility=True,
            )

    selected_id = None
    source = "fallback_ranked_poi"
    confidence = 0.0
    if hud_targeting_report is not None:
        selected = getattr(hud_targeting_report, "selected", None)
        if selected is not None:
            if getattr(selected, "selected_poi_id", None) is not None and not bool(getattr(selected, "ambiguous", True)):
                candidate_id = str(selected.selected_poi_id)
                if candidate_id in {item.poi_id for item in candidates} and no_effect_counts.get(candidate_id, 0) < 2:
                    selected_id = candidate_id
                    source = "hud_selected"
                    confidence = 1.0

    if selected_id is None:
        scored = []
        contact_scores = _contact_scores(contact_experiment_report)
        for index, poi in enumerate(candidates):
            no_effect_penalty = 0.4 * min(1.0, no_effect_counts.get(poi.poi_id, 0) / 3.0)
            contact_bonus = float(contact_scores.get(poi.poi_id, 0.0))
            score = 0.65 * float(poi.confidence) + 0.35 * contact_bonus - no_effect_penalty
            scored.append((score, -index, poi.poi_id, poi))
        scored.sort(key=lambda item: (-item[0], item[2]))
        if not scored:
            return None
        _, _, selected_id, selected_poi = scored[0]
        source = "contact_fallback" if contact_experiment_report is not None else "fallback_ranked_poi"
        confidence = float(max(0.0, min(1.0, selected_poi.confidence)))

    return SolveTargetState(
        target_poi_id=selected_id,
        source=source,
        confidence=float(confidence),
        attempt_count=0,
        last_outcome_type=None,
        active=True,
        route_feasibility=True,
    )


def select_next_target(
    current_target_state,
    ranked_poi_candidates,
    solve_episode_history,
    contact_experiment_report=None,
    mechanic_report=None,
    adaptive_episode_history=None,
) -> SolveTargetState | None:
    candidates = tuple(_valid_world_pois(ranked_poi_candidates))
    if not candidates:
        return None

    history = tuple(solve_episode_history or ()) + tuple(adaptive_episode_history or ())
    if current_target_state is not None and current_target_state.target_poi_id is not None:
        current_id = str(current_target_state.target_poi_id)
        recent = [step for step in history if str(getattr(step, "target_poi_id", "")) == current_id]
        no_effect_streak = _tail_count(recent, lambda step: str(getattr(step, "outcome_type", "")) in {"no_effect", "hud_change_only"})
        blocked_streak = _tail_count(recent, lambda step: bool(getattr(step, "blocked_action", False)))
        last_outcome = str(getattr(history[-1], "outcome_type", "")) if history else ""
        useful_recent = last_outcome in {"reward_change", "object_removed", "new_object_appeared", "door_opens"}
        target_visible = bool(getattr(history[-1], "target_bbox_after", None)) if history else True
        if useful_recent and target_visible:
            return SolveTargetState(
                target_poi_id=current_id,
                source=str(getattr(current_target_state, "source", "retained")),
                confidence=float(getattr(current_target_state, "confidence", 0.0)),
                attempt_count=int(getattr(current_target_state, "attempt_count", 0)) + 1,
                last_outcome_type=last_outcome,
                active=True,
                route_feasibility=getattr(current_target_state, "route_feasibility", None),
            )
        exhausted = (
            no_effect_streak >= 3
            or blocked_streak >= 2
            or not target_visible
            or getattr(current_target_state, "route_feasibility", None) is False
        )
        if not exhausted:
            return SolveTargetState(
                target_poi_id=current_id,
                source=str(getattr(current_target_state, "source", "retained")),
                confidence=float(getattr(current_target_state, "confidence", 0.0)),
                attempt_count=int(getattr(current_target_state, "attempt_count", 0)) + 1,
                last_outcome_type=last_outcome or None,
                active=True,
                route_feasibility=getattr(current_target_state, "route_feasibility", None),
            )

    no_effect_counts = Counter(str(getattr(step, "target_poi_id", "")) for step in history if str(getattr(step, "outcome_type", "")) in {"no_effect", "hud_change_only"})
    failed_target_counts = Counter()
    for step in history:
        pid = str(getattr(step, "target_poi_id", ""))
        if not pid:
            continue
        if str(getattr(step, "outcome_type", "")) in {"no_effect", "hud_change_only"} or bool(getattr(step, "blocked_action", False)):
            failed_target_counts[pid] += 1
    contact_scores = _contact_scores(contact_experiment_report)
    mechanic_memory = getattr(mechanic_report, "memory", None) if mechanic_report is not None else None
    current_id = str(getattr(current_target_state, "target_poi_id", "")) if current_target_state is not None else ""

    scored = []
    states_by_id = {str(item.poi_id): item for item in tuple(getattr(mechanic_memory, "poi_states", ()))}
    retired = set(str(v) for v in tuple(getattr(mechanic_memory, "retired_poi_ids", ()))) if mechanic_memory is not None else set()
    for index, poi in enumerate(candidates):
        if poi.poi_id == current_id:
            continue
        if poi.poi_id in retired:
            continue
        if failed_target_counts.get(poi.poi_id, 0) >= 4:
            continue
        no_effect_penalty = 0.4 * min(1.0, no_effect_counts.get(poi.poi_id, 0) / 3.0)
        contact_bonus = float(contact_scores.get(poi.poi_id, 0.0))
        mech_state = states_by_id.get(str(poi.poi_id))
        mech_label = str(getattr(mech_state, "mechanic_label", "unknown"))
        mech_priority = _mechanic_label_priority(mech_label)
        if mech_label == "decoy":
            continue
        if mech_label == "hazard":
            safer_exists = any(
                _mechanic_label_priority(str(getattr(states_by_id.get(str(other.poi_id)), "mechanic_label", "unknown"))) >= 0.4
                for other in candidates
                if other.poi_id != poi.poi_id and other.poi_id not in retired
            )
            if safer_exists:
                continue
        score = 0.45 * float(poi.confidence) + 0.25 * contact_bonus + 0.30 * mech_priority - no_effect_penalty
        scored.append((score, -index, poi.poi_id, poi))
    scored.sort(key=lambda item: (-item[0], item[2]))
    if not scored:
        return None

    _, _, poi_id, poi = scored[0]
    return SolveTargetState(
        target_poi_id=poi_id,
        source="retargeted",
        confidence=float(max(0.0, min(1.0, poi.confidence))),
        attempt_count=0,
        last_outcome_type=None,
        active=True,
        route_feasibility=True,
    )


def _mechanic_label_priority(label: str) -> float:
    return {
        "exit": 1.0,
        "target": 0.9,
        "collectible": 0.75,
        "door_or_switch": 0.65,
        "unknown": 0.5,
        "hazard": 0.1,
        "decoy": 0.0,
    }.get(label, 0.4)


def _valid_world_pois(ranked_poi_candidates):
    for poi in tuple(ranked_poi_candidates or ()):
        if _is_border_locked_poi(poi):
            continue
        yield poi


def _is_border_locked_poi(poi_candidate) -> bool:
    flags = set(getattr(poi_candidate, "ambiguity_flags", ()))
    if "border_locked" in flags:
        return True
    x0, y0, x1, y1 = tuple(getattr(poi_candidate, "bbox", (0, 0, 0, 0)))
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    tiny_or_strip = int(getattr(poi_candidate, "area", 0)) <= 8 or width <= 2 or height <= 2 or width >= 3 * height or height >= 3 * width
    if not tiny_or_strip:
        return False
    frame_dims = _extract_frame_dims_from_candidate(poi_candidate)
    if frame_dims is None:
        return bool(x0 == 0 or y0 == 0)
    frame_w, frame_h = frame_dims
    touches_left = x0 <= 0
    touches_top = y0 <= 0
    touches_right = x1 >= (int(frame_w) - 1)
    touches_bottom = y1 >= (int(frame_h) - 1)
    return bool(touches_left or touches_top or touches_right or touches_bottom)


def _extract_frame_dims_from_candidate(poi_candidate) -> tuple[int, int] | None:
    for w_key, h_key in (
        ("frame_width", "frame_height"),
        ("grid_width", "grid_height"),
        ("world_width", "world_height"),
    ):
        w = getattr(poi_candidate, w_key, None)
        h = getattr(poi_candidate, h_key, None)
        if w is not None and h is not None:
            try:
                wi = int(w)
                hi = int(h)
            except Exception:
                continue
            if wi > 0 and hi > 0:
                return wi, hi
    metadata = getattr(poi_candidate, "metadata", None)
    if isinstance(metadata, dict):
        w = metadata.get("frame_width") or metadata.get("grid_width")
        h = metadata.get("frame_height") or metadata.get("grid_height")
        if w is not None and h is not None:
            try:
                wi = int(w)
                hi = int(h)
            except Exception:
                return None
            if wi > 0 and hi > 0:
                return wi, hi
    return None


def _tail_count(items, predicate) -> int:
    count = 0
    for item in reversed(tuple(items)):
        if not predicate(item):
            break
        count += 1
    return count


def _no_effect_counts(contact_experiment_report) -> Counter:
    counts = Counter()
    if contact_experiment_report is None:
        return counts
    for item in tuple(getattr(contact_experiment_report, "tested_pois", ())):
        outcome = getattr(item, "outcome", None)
        if outcome is None:
            continue
        if str(getattr(outcome, "outcome_type", "")) == "no_effect":
            counts[str(getattr(item, "poi_id", ""))] += 1
    return counts


def _contact_scores(contact_experiment_report) -> dict[str, float]:
    if contact_experiment_report is None:
        return {}
    scores: dict[str, float] = {}
    for item in tuple(getattr(contact_experiment_report, "tested_pois", ())):
        poi_id = str(getattr(item, "poi_id", ""))
        if not poi_id:
            continue
        outcome = getattr(item, "outcome", None)
        if outcome is None:
            continue
        outcome_type = str(getattr(outcome, "outcome_type", ""))
        confidence = float(getattr(outcome, "confidence", 0.0))
        bonus = 0.0
        if outcome_type in {"reward_change", "object_removed", "new_object_appeared", "door_opens", "level_transition"}:
            bonus = 0.6 + 0.4 * confidence
        elif outcome_type == "no_effect":
            bonus = 0.0
        else:
            bonus = 0.2 * confidence
        scores[poi_id] = max(scores.get(poi_id, 0.0), bonus)
    return scores

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .planner_types import CandidateAction, CandidateMeta, PlannerInputs, PlannerState


def score_candidates(
    candidates: List[CandidateAction],
    meta: Dict[Tuple[Any, ...], CandidateMeta],
    planner_state: PlannerState,
    inputs: PlannerInputs,
    action_schema: Dict[str, Any],
    state_key: str,
    mode: str,
    cfg: Any,
) -> Tuple[List[CandidateAction], List[str]]:
    warnings: List[str] = []
    ranked: List[Tuple[float, CandidateAction]] = []

    for action in candidates:
        key = action.key()
        cand_meta = meta[key]
        cand_meta.novelty = _novelty(action, planner_state, inputs, state_key)
        cand_meta.disambiguation = _disambiguation_gain(action, inputs)
        cand_meta.expected_change = _expected_change(action, inputs, action_schema)
        cand_meta.loop_risk = _loop_risk(action, planner_state, state_key)
        cand_meta.action_cost = _action_cost(action, cfg)
        if mode == "goal_directed":
            cand_meta.expected_progress = _expected_progress(action, inputs)
            cand_meta.hypothesis_align = _hypothesis_alignment(action, inputs)
            cand_meta.score = (
                cfg.w_progress * cand_meta.expected_progress
                + cfg.w_effect * cand_meta.expected_change
                + cfg.w_hypothesis_align * cand_meta.hypothesis_align
                - cfg.w_loop * cand_meta.loop_risk
                - cfg.w_cost * cand_meta.action_cost
            )
        else:
            cand_meta.score = (
                cfg.w_novelty * cand_meta.novelty
                + cfg.w_disambiguation * cand_meta.disambiguation
                + cfg.w_effect * cand_meta.expected_change
                - cfg.w_loop * cand_meta.loop_risk
                - cfg.w_cost * cand_meta.action_cost
            )
        ranked.append((cand_meta.score, action))

    ranked.sort(key=lambda item: (-item[0], _stable_tiebreak(state_key, item[1])))
    return [action for _, action in ranked], warnings


def _novelty(
    action: CandidateAction,
    planner_state: PlannerState,
    inputs: PlannerInputs,
    state_key: str,
) -> float:
    key = action.key()
    if key in planner_state.recent_noop_actions:
        return 0.0
    simple_report = inputs.simple_report or {}
    full_report = inputs.full_report or {}
    if action.type == "simple":
        entry = simple_report.get("frontier", {}).get(state_key, {})
        if action.action_id in entry.get("untried_actions", []):
            return 1.0
    if action.type == "coord":
        entry = full_report.get("frontier", {}).get(state_key, {})
        banlist = entry.get("banlist", [])
        key_list = [action.action_id, action.x, action.y]
        key_tuple = (action.action_id, action.x, action.y)
        if key_list not in banlist and key_tuple not in banlist:
            return 1.0
        return 0.0
    count = planner_state.action_counts.get(action.action_id, 0)
    return 0.5 if count == 0 else 0.0


def _disambiguation_gain(action: CandidateAction, inputs: PlannerInputs) -> float:
    report = inputs.hypotheses_report or {}
    hypotheses = report.get("hypotheses", [])
    competing = [h for h in hypotheses if h.get("confidence", 0.0) > 0]
    if len(competing) <= 1:
        return 0.0
    for hyp in hypotheses:
        for test in hyp.get("tests", []):
            seq = test.get("action_sequence") or []
            if not seq:
                continue
            first = seq[0]
            if _match_action(action, first):
                supports = test.get("supports", [])
                refutes = test.get("refutes", [])
                if supports or refutes:
                    return 1.0
    return 0.0


def _expected_change(
    action: CandidateAction,
    inputs: PlannerInputs,
    action_schema: Dict[str, Any],
) -> float:
    grid = action_schema.get("primary_grid", {}) if isinstance(action_schema, dict) else {}
    area = float(grid.get("width", 1) * grid.get("height", 1)) if grid else 1.0
    if action.type == "simple":
        model = (inputs.simple_report or {}).get("action_effect_model", {})
        stats = model.get(action.action_id)
        if stats and "avg_changed_cells" in stats:
            return min(1.0, float(stats["avg_changed_cells"]) / area)
        return 0.1
    model = (inputs.full_report or {}).get("coord_action_effect_model", {})
    stats = model.get(action.action_id)
    if stats and "avg_changed_cells" in stats:
        return min(1.0, float(stats["avg_changed_cells"]) / area)
    return 0.2


def _loop_risk(action: CandidateAction, planner_state: PlannerState, state_key: str) -> float:
    key = action.key()
    state_action = (state_key, key)
    if state_action in planner_state.recent_state_action_noops:
        return 1.0
    if state_action in planner_state.recent_state_actions:
        return 0.5
    if state_key in planner_state.recent_states:
        return 0.25
    if key in planner_state.recent_noop_actions:
        return 0.5
    return 0.0


def _stable_tiebreak(state_key: str, action: CandidateAction) -> int:
    import hashlib

    y = action.y if action.y is not None else -1
    x = action.x if action.x is not None else -1
    payload = f"{state_key}|{action.action_id}|{y}|{x}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _expected_progress(action: CandidateAction, inputs: PlannerInputs) -> float:
    goal_report = inputs.goal_report or {}
    goal_type = goal_report.get("goal_hints", {}).get("likely_goal_type", "unknown")
    expected_sig = None
    if goal_type == "collect_all":
        expected_sig = "despawn"
    elif goal_type == "paint_to_match":
        expected_sig = "paint"
    elif goal_type == "stabilize_state":
        return 1.0 if action.type == "simple" else 0.5

    if expected_sig:
        model = (inputs.simple_report or {}).get("action_effect_model", {})
        stats = model.get(action.action_id, {})
        dominant = stats.get("dominant_event_signatures", [])
        for sig, _ in dominant:
            if sig == expected_sig:
                return 1.0
    if goal_type == "unknown":
        hints = (inputs.mechanic_prior or {}).get("family_tags", {}).get("constraints", {})
        preferred_actions = hints.get("preferred_action_families", [])
        preferred_coords = hints.get("preferred_coord_selectors", [])
        if preferred_actions and action.type == "simple":
            return 0.5
        if preferred_coords and action.type == "coord":
            return 0.5
    return 0.0


def _hypothesis_alignment(action: CandidateAction, inputs: PlannerInputs) -> float:
    report = inputs.hypotheses_report or {}
    hypotheses = sorted(report.get("hypotheses", []), key=lambda h: (-h.get("confidence", 0.0), h.get("hypothesis_id", "")))
    if not hypotheses:
        return 0.0
    top = hypotheses[0]
    for test in top.get("tests", []):
        seq = test.get("action_sequence") or []
        if not seq:
            continue
        if _match_action(action, seq[0]):
            return 1.0
    return 0.0


def _action_cost(action: CandidateAction, cfg: Any) -> float:
    return cfg.coord_action_cost if action.type == "coord" else 0.0


def _match_action(action: CandidateAction, spec: Dict[str, Any]) -> bool:
    if action.type != spec.get("type"):
        return False
    if action.action_id != spec.get("action_id"):
        return False
    if action.type == "coord":
        return action.x == spec.get("x") and action.y == spec.get("y")
    return True

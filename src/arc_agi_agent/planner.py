from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .action_schema import ActionSchema, parse_action_schema_data
from .grid_utils import grid_hash
from .normalize import normalize_observation
from .planner_candidates import build_candidates
from .planner_config import PlannerConfig
from .planner_scoring import score_candidates
from .planner_types import (
    CandidateAction,
    CandidateMeta,
    DecisionTrace,
    PlannerInputs,
    PlannerState,
)


def plan_next(
    observation: Any,
    planner_state: PlannerState,
    inputs: PlannerInputs,
    action_schema: ActionSchema | Dict[str, Any],
    fp_report_current: Optional[Dict[str, Any]] = None,
    fp_analyst: Optional[Any] = None,
    cfg: Optional[PlannerConfig] = None,
) -> Tuple[Dict[str, Any], PlannerState, DecisionTrace]:
    cfg = cfg or PlannerConfig()
    schema = parse_action_schema_data(action_schema) if isinstance(action_schema, dict) else action_schema
    state_key = _state_key(observation)
    if fp_report_current is None:
        if fp_analyst is None:
            raise ValueError("fp_report_current or fp_analyst is required")
        fp_report_current = fp_analyst.analyze(observation)

    mode, warnings = _select_mode(state_key, planner_state, inputs, cfg)
    candidates, meta, cand_warnings = build_candidates(
        state_key,
        _schema_to_dict(schema),
        planner_state,
        inputs,
        cfg,
        fp_report_current,
    )
    warnings.extend(cand_warnings)

    looped = _loop_detected(planner_state, state_key, cfg)
    if looped:
        mode = "escape_loop"

    if mode == "escape_loop":
        ranked = _escape_loop_choice(
            candidates,
            meta,
            planner_state,
            inputs,
            _schema_to_dict(schema),
            state_key,
            cfg,
        )
    else:
        ranked, _ = score_candidates(
            candidates,
            meta,
            planner_state,
            inputs,
            _schema_to_dict(schema),
            state_key,
            "goal_directed" if mode == "goal_directed" else "info_gain",
            cfg,
        )

    chosen = ranked[0] if ranked else _fallback_action(inputs, schema)
    decision_trace = _build_trace(mode, ranked, meta, chosen, warnings, state_key)
    planner_state_next = _update_state(planner_state, chosen, state_key, fp_report_current, cfg)
    planner_state_next.mode_history.append(mode)

    return _action_to_dict(chosen), planner_state_next, decision_trace


def _select_mode(
    state_key: str,
    planner_state: PlannerState,
    inputs: PlannerInputs,
    cfg: PlannerConfig,
) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    mechanic_prior = inputs.mechanic_prior or {}
    hypotheses = inputs.hypotheses_report or {}
    goal_report = inputs.goal_report or {}
    transition_graph = inputs.transition_graph or {}

    mechanic_max = _max_prior(mechanic_prior)
    top_hyp = _top_hypothesis_conf(hypotheses)
    goal_conf = goal_report.get("progress_estimate", {}).get("confidence", 0.0)
    known_state = state_key in (transition_graph.get("nodes", {}) if isinstance(transition_graph, dict) else {})

    if mechanic_max == 0.0 and not hypotheses and not goal_report:
        warnings.append("missing_strategic_inputs")

    if mechanic_max < cfg.mechanic_conf_threshold or top_hyp < cfg.hypothesis_conf_threshold or goal_conf < cfg.goal_conf_threshold or not known_state:
        return "info_gain", warnings
    return "goal_directed", warnings


def _state_key(observation: Any) -> str:
    grids = normalize_observation(observation, schema_warnings=[]).grids
    return grid_hash(grids)


def _loop_detected(planner_state: PlannerState, state_key: str, cfg: PlannerConfig) -> bool:
    recent = planner_state.recent_states[-cfg.loop_window_N :]
    return recent.count(state_key) >= cfg.loop_repeat_R


def _escape_loop_choice(
    candidates: List[CandidateAction],
    meta: Dict[Tuple[Any, ...], CandidateMeta],
    planner_state: PlannerState,
    inputs: PlannerInputs,
    action_schema: Dict[str, Any],
    state_key: str,
    cfg: PlannerConfig,
) -> List[CandidateAction]:
    if candidates:
        score_candidates(
            candidates,
            meta,
            planner_state,
            inputs,
            action_schema,
            state_key,
            "info_gain",
            cfg,
        )
        recent_ids = {a.action_id for a in planner_state.recent_actions[-cfg.loop_avoid_recent_K :]}
        filtered = [c for c in candidates if c.action_id not in recent_ids] or candidates
        last_coord = _last_coord(planner_state.recent_actions)
        return sorted(
            filtered,
            key=lambda c: (
                -meta[c.key()].novelty,
                -_coord_distance(c, last_coord),
                _stable_tiebreak(state_key, c),
            ),
        )

    heuristics = _fallback_heuristics(action_schema, _last_coord(planner_state.recent_actions))
    if not heuristics:
        return []
    return heuristics


def _fallback_action(inputs: PlannerInputs, action_schema: ActionSchema) -> CandidateAction:
    simple_report = inputs.simple_report or {}
    best = _highest_noop_action(simple_report)
    if best:
        return CandidateAction(type="simple", action_id=best)
    actions = sorted([a.action_id for a in action_schema.actions])
    return CandidateAction(type="simple", action_id=actions[0])


def _highest_noop_action(simple_report: Dict[str, Any]) -> Optional[str]:
    effects = simple_report.get("action_effect_model", {})
    best = None
    best_noop = -1.0
    for action_id, stats in effects.items():
        rate = float(stats.get("no_effect_rate", 0.0))
        if rate > best_noop or (rate == best_noop and (best is None or action_id < best)):
            best = action_id
            best_noop = rate
    return best


def _fallback_heuristics(action_schema: Dict[str, Any], last_coord: Optional[Tuple[int, int]]) -> List[CandidateAction]:
    actions = action_schema.get("actions", [])
    coords = [a for a in actions if a.get("kind") == "coord"]
    simples = [a for a in actions if a.get("kind") == "simple"]
    if coords:
        action_id = coords[0].get("action_id")
        x, y = _farthest_coord(action_schema, last_coord)
        return [CandidateAction(type="coord", action_id=action_id, x=x, y=y)]
    if simples:
        action_id = simples[0].get("action_id")
        return [CandidateAction(type="simple", action_id=action_id)]
    return []


def _update_state(
    planner_state: PlannerState,
    chosen: CandidateAction,
    state_key: str,
    fp_report_current: Dict[str, Any],
    cfg: PlannerConfig,
) -> PlannerState:
    planner_state.recent_states.append(state_key)
    planner_state.recent_states = planner_state.recent_states[-cfg.loop_window_N :]
    planner_state.recent_actions.append(chosen)
    planner_state.action_counts[chosen.action_id] = planner_state.action_counts.get(chosen.action_id, 0) + 1
    planner_state.recent_state_actions.append((state_key, chosen.key()))
    planner_state.recent_state_actions = planner_state.recent_state_actions[-cfg.loop_window_N :]

    diff = fp_report_current.get("diff_summary") or {}
    changed_cells = diff.get("changed_cells_count", 1)
    if isinstance(changed_cells, int) and changed_cells == 0:
        planner_state.recent_noop_actions.append(chosen.key())
        planner_state.recent_noop_actions = planner_state.recent_noop_actions[-cfg.recent_noop_window :]
        planner_state.recent_state_action_noops.append((state_key, chosen.key()))
        planner_state.recent_state_action_noops = planner_state.recent_state_action_noops[-cfg.loop_window_N :]
    if planner_state.pending_tests:
        test_action = planner_state.pending_tests[0].get("action_sequence", [])
        if test_action and _matches_test_action(chosen, test_action[0]):
            planner_state.pending_tests.pop(0)
    return planner_state


def _build_trace(
    mode: str,
    ranked: List[CandidateAction],
    meta: Dict[Tuple[Any, ...], CandidateMeta],
    chosen: CandidateAction,
    warnings: List[str],
    state_key: str,
) -> DecisionTrace:
    candidates = []
    for action in ranked[:10]:
        cand_meta = meta.get(action.key())
        candidates.append(
            {
                "action": _action_to_dict(action),
                "score": cand_meta.score if cand_meta else 0.0,
                "source": cand_meta.source if cand_meta else "unknown",
                "terms": {
                    "novelty": getattr(cand_meta, "novelty", 0.0),
                    "disambiguation": getattr(cand_meta, "disambiguation", 0.0),
                    "expected_change": getattr(cand_meta, "expected_change", 0.0),
                    "loop_risk": getattr(cand_meta, "loop_risk", 0.0),
                    "expected_progress": getattr(cand_meta, "expected_progress", 0.0),
                    "hypothesis_align": getattr(cand_meta, "hypothesis_align", 0.0),
                    "action_cost": getattr(cand_meta, "action_cost", 0.0),
                    "tie_break": _stable_tiebreak(state_key, action),
                },
            }
        )
    chosen_meta = meta.get(chosen.key())
    chosen_entry = {
        "action": _action_to_dict(chosen),
        "score": chosen_meta.score if chosen_meta else 0.0,
        "source": chosen_meta.source if chosen_meta else "fallback",
    }
    return DecisionTrace(mode=mode, candidates=candidates, chosen=chosen_entry, warnings=warnings)


def _schema_to_dict(schema: ActionSchema) -> Dict[str, Any]:
    return {
        "version": schema.version,
        "primary_grid": {"width": schema.primary_grid.width, "height": schema.primary_grid.height},
        "actions": [{"action_id": a.action_id, "kind": a.kind} for a in schema.actions],
    }


def _action_to_dict(action: CandidateAction) -> Dict[str, Any]:
    if action.type == "coord":
        return {"type": "coord", "action_id": action.action_id, "x": action.x, "y": action.y}
    return {"type": "simple", "action_id": action.action_id}


def _matches_test_action(action: CandidateAction, spec: Dict[str, Any]) -> bool:
    if action.type != spec.get("type"):
        return False
    if action.action_id != spec.get("action_id"):
        return False
    if action.type == "coord":
        return action.x == spec.get("x") and action.y == spec.get("y")
    return True


def _last_coord(actions: List[CandidateAction]) -> Optional[Tuple[int, int]]:
    for action in reversed(actions):
        if action.type == "coord" and action.x is not None and action.y is not None:
            return (action.x, action.y)
    return None


def _coord_distance(action: CandidateAction, last_coord: Optional[Tuple[int, int]]) -> int:
    if action.type != "coord" or action.x is None or action.y is None or last_coord is None:
        return 0
    return abs(action.x - last_coord[0]) + abs(action.y - last_coord[1])


def _stable_tiebreak(state_key: str, action: CandidateAction) -> int:
    import hashlib

    y = action.y if action.y is not None else -1
    x = action.x if action.x is not None else -1
    payload = f"{state_key}|{action.action_id}|{y}|{x}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _farthest_coord(action_schema: Dict[str, Any], last_coord: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    grid = action_schema.get("primary_grid", {})
    width = int(grid.get("width", 1))
    height = int(grid.get("height", 1))
    corners = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]
    if last_coord is None:
        return corners[-1]
    corners.sort(key=lambda c: -(abs(c[0] - last_coord[0]) + abs(c[1] - last_coord[1])))
    return corners[0]


def _max_prior(mechanic_prior: Dict[str, Any]) -> float:
    families = mechanic_prior.get("mechanic_prior", {}).get("families", [])
    if not families:
        return 0.0
    return max(float(item.get("prior", 0.0)) for item in families)


def _top_hypothesis_conf(hypotheses_report: Dict[str, Any]) -> float:
    hypotheses = hypotheses_report.get("hypotheses", [])
    if not hypotheses:
        return 0.0
    return max(float(h.get("confidence", 0.0)) for h in hypotheses)

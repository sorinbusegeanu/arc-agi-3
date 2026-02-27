from __future__ import annotations

import os
from dataclasses import asdict
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from .action_schema import ActionSchema, parse_action_schema_data
from .coord_selectors import CoordCandidate, build_coords as select_build_coords
from .frontier_priority import CoordActionCandidate, score_candidates, selector_base_scores
from .full_explorer_config import FullExplorerConfig
from .full_explorer_types import (
    CoordActionEffectStats,
    FullFrontierState,
    FullExplorerReport,
    RunSummary,
    FrontierEntry,
    TransitionNode,
)
from .full_transition_graph import FullTransitionGraphStore
from .fp_analyst import FPAnalyst
from .grid_utils import bbox_area, grid_from_ascii, grid_hash
from .logger import get_logger
from .normalize import normalize_observation
from .trace import TraceWriter

logger = get_logger(__name__)


def _bb_get(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def run(
    env: Any,
    game_id: str,
    seed: int,
    fp_analyst: FPAnalyst,
    cfg: FullExplorerConfig,
    ctx: Optional[Dict[str, Any]] = None,
    start_observation: Any = None,
    simple_explorer_report: Optional[Any] = None,
) -> FullExplorerReport:
    ctx = ctx or {}
    out_dir = ctx.get("output_dir", os.path.join("runs", "full_explorer", f"{game_id}_{seed}"))
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "trace.jsonl")
    trace_writer = TraceWriter(trace_path) if cfg.save_trace else None

    if start_observation is None:
        try:
            obs = env.reset()
        except Exception as e:
            return _empty_report(game_id, seed, f"reset_failed: {e}", out_dir)
    else:
        obs = start_observation

    action_space = [a for a in env.action_space if a.is_complex()]
    if not action_space:
        return _empty_report(game_id, seed, "no_coord_actions", out_dir)

    prev_obs = None
    state_seq: deque[str] = deque(maxlen=cfg.revisit_window_N)
    loops_detected = Counter()

    graph = FullTransitionGraphStore()
    state_frontier: Dict[str, List[CoordActionCandidate]] = {}
    state_banlist: Dict[str, set[Tuple[str, int, int]]] = {}
    state_noop_counts: Dict[str, Dict[Tuple[str, int, int], int]] = {}
    state_action_attempts: Dict[str, Dict[str, int]] = {}
    global_noop_counts: Dict[Tuple[str, int, int], int] = {}
    coord_tried: Dict[Tuple[str, int, int], int] = {}
    action_coord_tried: Dict[str, set[Tuple[int, int]]] = {}
    action_attempts: Dict[str, int] = {a.name: 0 for a in action_space}
    action_selector_counts: Dict[str, Dict[str, int]] = {a.name: {} for a in action_space}
    action_changed_cells: Dict[str, int] = {a.name: 0 for a in action_space}
    action_changed_bbox_area: Dict[str, int] = {a.name: 0 for a in action_space}
    action_event_counts: Dict[str, Counter[str]] = {a.name: Counter() for a in action_space}
    action_coord_counts: Dict[str, Dict[Tuple[int, int], int]] = {a.name: {} for a in action_space}
    action_coord_noops: Dict[str, Dict[Tuple[int, int], int]] = {a.name: {} for a in action_space}

    steps = 0
    termination_reason = "max_steps"

    current_state = _state_key(obs)
    state_seq.append(current_state)

    while steps < cfg.max_steps:
        if len(state_frontier) >= cfg.max_unique_states:
            termination_reason = "max_unique_states"
            break

        if current_state not in state_frontier or not state_frontier[current_state]:
            state_frontier[current_state] = _build_frontier(
                obs,
                prev_obs,
                fp_analyst,
                action_space,
                cfg,
                state_noop_counts.get(current_state, {}),
                global_noop_counts,
                action_coord_tried,
                action_attempts,
                coord_tried,
            )

        if not state_frontier[current_state]:
            next_state = _pick_frontier_state(state_frontier, current_state)
            if next_state is None:
                termination_reason = "frontier_empty"
                break
            if cfg.bfs_route_to_frontier:
                path = graph.bfs_path(current_state, next_state, cfg.bfs_max_depth)
                if path is None:
                    current_state = next_state
                    continue
                for action_id, x, y, target_state in path:
                    action_obj = _action_by_name(action_space, action_id)
                    if action_obj is None:
                        continue
                    action_obj.set_data({"x": x, "y": y})
                    obs, prev_obs, steps, current_state = _step_and_record(
                        env,
                        fp_analyst,
                        obs,
                        prev_obs,
                        action_obj,
                        x,
                        y,
                        "bfs_route",
                        current_state,
                        graph,
                        trace_writer,
                        coord_tried,
                        global_noop_counts,
                        state_noop_counts,
                        state_banlist,
                        action_coord_tried,
                        action_attempts,
                        action_selector_counts,
                        action_changed_cells,
                        action_changed_bbox_area,
                        action_event_counts,
                        action_coord_counts,
                        action_coord_noops,
                        cfg,
                        state_seq,
                        steps,
                        state_action_attempts,
                    )
                    if steps >= cfg.max_steps:
                        break
                if steps >= cfg.max_steps:
                    break
            else:
                current_state = next_state
                continue

        candidate = state_frontier[current_state].pop(0) if state_frontier[current_state] else None
        if candidate is None:
            continue
        if (candidate.action_id, candidate.x, candidate.y) in state_banlist.get(current_state, set()):
            continue
        if coord_tried.get((candidate.action_id, candidate.x, candidate.y), 0) >= cfg.attempts_per_coord_candidate:
            continue
        action_obj = _action_by_name(action_space, candidate.action_id)
        if action_obj is None:
            continue
        action_obj.set_data({"x": candidate.x, "y": candidate.y})

        obs, prev_obs, steps, current_state = _step_and_record(
            env,
            fp_analyst,
            obs,
            prev_obs,
            action_obj,
            candidate.x,
            candidate.y,
            candidate.selector,
            current_state,
            graph,
            trace_writer,
            coord_tried,
            global_noop_counts,
            state_noop_counts,
            state_banlist,
            action_coord_tried,
            action_attempts,
            action_selector_counts,
            action_changed_cells,
            action_changed_bbox_area,
            action_event_counts,
            action_coord_counts,
            action_coord_noops,
            cfg,
            state_seq,
            steps,
            state_action_attempts,
        )

        loop_flags = _detect_loops(state_seq, current_state, cfg)
        for flag in loop_flags:
            loops_detected[flag] += 1

    report = _build_report(
        game_id=game_id,
        seed=seed,
        steps=steps,
        loops_detected=dict(loops_detected),
        termination_reason=termination_reason,
        graph=graph,
        state_frontier=state_frontier,
        state_noop_counts=state_noop_counts,
        state_banlist=state_banlist,
        state_action_attempts=state_action_attempts,
        action_attempts=action_attempts,
        action_coord_tried=action_coord_tried,
        action_selector_counts=action_selector_counts,
        action_changed_cells=action_changed_cells,
        action_changed_bbox_area=action_changed_bbox_area,
        action_event_counts=action_event_counts,
        action_coord_counts=action_coord_counts,
        action_coord_noops=action_coord_noops,
        coord_tried=coord_tried,
        trace_path=trace_path,
        cfg=cfg,
    )
    return report


def build_coords(fp_report: Any, cfg: FullExplorerConfig) -> List[CoordCandidate]:
    if not fp_report.state_summary.grid_summaries:
        return []
    grid_summary = fp_report.state_summary.grid_summaries[0]
    bg_color = grid_summary.bg_candidates[0][0] if grid_summary.bg_candidates else 0
    ascii_grid = fp_report.viz_artifacts.ascii_grid.get(grid_summary.name)
    if ascii_grid is None:
        return []
    grid_arr = _grid_from_ascii(ascii_grid)
    diff_bbox = fp_report.diff_summary.changed_bbox if fp_report.diff_summary else None
    return select_build_coords(grid_arr, bg_color, diff_bbox, cfg)


def build_frontier(
    state_key: str,
    coord_candidates: List[CoordCandidate],
    action_schema: List[Any],
    cfg: FullExplorerConfig,
) -> List[CoordActionCandidate]:
    selector_scores = selector_base_scores(cfg)
    action_attempts = {a.name: 0 for a in action_schema}
    action_coord_tried: Dict[str, set[Tuple[int, int]]] = {}
    global_noop_counts: Dict[Tuple[str, int, int], int] = {}
    state_noop_counts: Dict[Tuple[str, int, int], int] = {}
    bg_penalties: Dict[Tuple[int, int], bool] = {}
    frontier: List[CoordActionCandidate] = []
    for action in action_schema:
        scored = score_candidates(
            action.name,
            coord_candidates,
            selector_scores,
            global_noop_counts,
            state_noop_counts,
            action_coord_tried,
            action_attempts,
            bg_penalties,
            cfg,
        )
        frontier.extend(scored)
    frontier.sort(key=lambda c: (-c.score, c.action_id, c.y, c.x))
    return frontier


def choose_action(
    blackboard: Any,
    action_schema: Any,
    fp_current: Any,
    frontier_state: Optional[FullFrontierState],
    cfg: Optional[FullExplorerConfig],
) -> Optional[Dict[str, Any]]:
    cfg = cfg or FullExplorerConfig()
    schema = _parse_action_schema(action_schema)
    coord_action_ids = _coord_action_ids(schema)
    if not coord_action_ids:
        return None

    frontier_state = frontier_state or FullFrontierState()
    _ensure_action_state(frontier_state, coord_action_ids)
    _update_frontier_state_from_history(frontier_state, blackboard, cfg)
    if not frontier_state.coord_tried:
        _backfill_coord_trials(frontier_state, blackboard, cfg)

    current_state = _bb_get(blackboard, "state_hash")
    if current_state is None:
        return None

    frontier = frontier_state.state_frontier.get(current_state)
    filtered_out: List[Dict[str, str]] = []
    candidates_before: List[Dict[str, Any]] = []
    if not frontier:
        frontier, filtered_out, raw = _build_frontier_with_diagnostics(
            fp_current, coord_action_ids, frontier_state, cfg, current_state
        )
        candidates_before = _frontier_actions(raw)
        frontier_state.state_frontier[current_state] = frontier

    candidate = _pop_coord_candidate(frontier_state, current_state, cfg)
    if candidate is None:
        _record_action_selection(
            blackboard,
            mode="probe_full",
            state_hash_before=current_state,
            candidates_before=candidates_before or _frontier_actions(frontier),
            candidates_after=_frontier_actions(frontier),
            filtered_out=filtered_out,
            scores={},
            selected_action=None,
            selected_reason="none",
        )
        return None
    _record_action_selection(
        blackboard,
        mode="probe_full",
        state_hash_before=current_state,
        candidates_before=candidates_before or _frontier_actions(frontier),
        candidates_after=_frontier_actions(frontier),
        filtered_out=filtered_out,
        scores={},
        selected_action={"type": "coord", "action_id": candidate.action_id, "x": candidate.x, "y": candidate.y},
        selected_reason="frontier",
    )
    return {"type": "coord", "action_id": candidate.action_id, "x": candidate.x, "y": candidate.y}


def build_frontier_report(
    blackboard: Any,
    action_schema: ActionSchema | Dict[str, Any],
    frontier_state: Optional[FullFrontierState],
    cfg: Optional[FullExplorerConfig] = None,
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or FullExplorerConfig()
    frontier_state = frontier_state or FullFrontierState()
    schema = _parse_action_schema(action_schema)
    coord_action_ids = _coord_action_ids(schema)
    if not coord_action_ids:
        return {
            "run_summary": {"steps_executed": getattr(blackboard, "step_idx", 0)},
            "coord_action_effect_model": {},
            "frontier": {},
            "coord_actions_supported": False,
            "diagnostics": {
                "builder": "build_frontier_report",
                "coord_actions_supported": False,
                "coord_action_effect_model_count": 0,
                "reason": "no_coord_actions",
            }
            if debug
            else {},
        }

    current_state = _bb_get(blackboard, "state_hash")
    fp_current = _bb_get(blackboard, "fp_current")
    _ensure_action_state(frontier_state, coord_action_ids)
    _update_frontier_state_from_history(frontier_state, blackboard, cfg)
    frontier = frontier_state.state_frontier.get(current_state)
    if not frontier:
        frontier_state.state_frontier[current_state] = _build_frontier_from_fp(
            fp_current, coord_action_ids, frontier_state, cfg, current_state
        )

    frontier_entries = {}
    coord_action_effect_model: Dict[str, Dict[str, Any]] = {}
    scored_coords: List[Tuple[str, int, int, float]] = []
    for state, candidates in frontier_state.state_frontier.items():
        entry = FrontierEntry(
            state=state,
            pending_candidates=len(candidates),
            attempt_counts_by_action_family={},
            cooldowns={},
            banlist=sorted(frontier_state.state_banlist.get(state, set())),
        )
        frontier_entries[state] = asdict(entry)
        for cand in candidates:
            scored_coords.append((cand.action_id, cand.x, cand.y, float(cand.score)))

    scored_coords.sort(key=lambda item: (-item[3], item[0], item[2], item[1]))
    for action_id, x, y, score in scored_coords:
        stats = coord_action_effect_model.setdefault(action_id, {"hotspots": []})
        if len(stats["hotspots"]) >= cfg.topK_hotspots:
            continue
        stats["hotspots"].append((x, y, score))

    diagnostics = {
        "builder": "build_frontier_report",
        "coord_actions_supported": True,
        "coord_action_effect_model_count": sum(
            len(stats.get("hotspots", [])) for stats in coord_action_effect_model.values()
        ),
        "coord_tried": len(frontier_state.coord_tried),
        "global_noop_counts": len(frontier_state.global_noop_counts),
        "state_frontier_states": len(frontier_state.state_frontier),
        "reason": "stepwise_frontier_only",
    } if debug else {}
    return {
        "run_summary": {"steps_executed": getattr(blackboard, "step_idx", 0)},
        "coord_action_effect_model": coord_action_effect_model,
        "frontier": frontier_entries,
        "coord_actions_supported": True,
        "diagnostics": diagnostics,
    }


def _backfill_coord_trials(frontier_state: FullFrontierState, blackboard: Any, cfg: FullExplorerConfig) -> None:
    history = _bb_get(blackboard, "history", []) or []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action") or {}
        if action.get("type") != "coord":
            continue
        action_id = action.get("action_id")
        x = action.get("x")
        y = action.get("y")
        if not action_id or x is None or y is None:
            continue
        prev_state = entry.get("state_before")
        if prev_state is None:
            continue
        key = (action_id, int(x), int(y))
        frontier_state.coord_tried[key] = frontier_state.coord_tried.get(key, 0) + 1
        frontier_state.action_coord_tried.setdefault(action_id, set()).add((int(x), int(y)))
        frontier_state.action_attempts[action_id] = frontier_state.action_attempts.get(action_id, 0) + 1

        fp_diff = entry.get("fp_diff") or {}
        changed_cells = int(fp_diff.get("changed_cells", 0))
        if changed_cells == 0:
            state_noops = frontier_state.state_noop_counts.setdefault(prev_state, {})
            state_noops[key] = state_noops.get(key, 0) + 1
            frontier_state.global_noop_counts[key] = frontier_state.global_noop_counts.get(key, 0) + 1
            if state_noops[key] >= cfg.ban_noop_after:
                frontier_state.state_banlist.setdefault(prev_state, set()).add(key)


def _parse_action_schema(action_schema: Any) -> ActionSchema:
    if isinstance(action_schema, ActionSchema):
        return action_schema
    if isinstance(action_schema, dict):
        return parse_action_schema_data(action_schema)
    raise ValueError("action_schema must be an ActionSchema or dict")


def _coord_action_ids(schema: ActionSchema) -> List[str]:
    return [a.action_id for a in schema.actions if a.kind == "coord"]


def _ensure_action_state(frontier_state: FullFrontierState, coord_action_ids: List[str]) -> None:
    for action_id in coord_action_ids:
        frontier_state.action_attempts.setdefault(action_id, 0)
        frontier_state.action_coord_tried.setdefault(action_id, set())


def _update_frontier_state_from_history(
    frontier_state: FullFrontierState,
    blackboard: Any,
    cfg: FullExplorerConfig,
) -> None:
    history = _bb_get(blackboard, "history", []) or []
    last_processed = frontier_state.last_processed_step
    entries = [entry for entry in history if isinstance(entry, dict) and entry.get("step_idx", -1) > last_processed]
    entries.sort(key=lambda e: e.get("step_idx", -1))
    for entry in entries:
        action = entry.get("action") or {}
        if action.get("type") != "coord":
            continue
        action_id = action.get("action_id")
        if not action_id:
            continue
        x = action.get("x")
        y = action.get("y")
        if x is None or y is None:
            continue
        prev_state = entry.get("state_before")
        if prev_state is None:
            continue
        key = (action_id, int(x), int(y))
        frontier_state.coord_tried[key] = frontier_state.coord_tried.get(key, 0) + 1
        frontier_state.action_coord_tried.setdefault(action_id, set()).add((int(x), int(y)))
        frontier_state.action_attempts[action_id] = frontier_state.action_attempts.get(action_id, 0) + 1

        fp_diff = entry.get("fp_diff") or {}
        changed_cells = int(fp_diff.get("changed_cells", 0))
        if changed_cells == 0:
            state_noops = frontier_state.state_noop_counts.setdefault(prev_state, {})
            state_noops[key] = state_noops.get(key, 0) + 1
            frontier_state.global_noop_counts[key] = frontier_state.global_noop_counts.get(key, 0) + 1
            if state_noops[key] >= cfg.ban_noop_after:
                frontier_state.state_banlist.setdefault(prev_state, set()).add(key)

        frontier = frontier_state.state_frontier.get(prev_state)
        if frontier:
            frontier_state.state_frontier[prev_state] = [
                cand for cand in frontier if (cand.action_id, cand.x, cand.y) != key
            ]
        frontier_state.last_processed_step = max(frontier_state.last_processed_step, entry.get("step_idx", -1))


def _build_frontier_from_fp(
    fp_current: Any,
    coord_action_ids: List[str],
    frontier_state: FullFrontierState,
    cfg: FullExplorerConfig,
    state_key: str,
) -> List[CoordActionCandidate]:
    frontier, _, _ = _build_frontier_with_diagnostics(
        fp_current, coord_action_ids, frontier_state, cfg, state_key
    )
    return frontier


def _build_frontier_with_diagnostics(
    fp_current: Any,
    coord_action_ids: List[str],
    frontier_state: FullFrontierState,
    cfg: FullExplorerConfig,
    state_key: str,
) -> Tuple[List[CoordActionCandidate], List[Dict[str, str]], List[CoordActionCandidate]]:
    coord_candidates, grid, bg_color, diff_bbox = _coord_candidates_from_fp(fp_current, cfg)
    if grid is None:
        return [], [], []
    bg_penalties = _bg_penalties(grid, bg_color, coord_candidates, cfg)
    selector_scores = selector_base_scores(cfg)

    frontier_raw: List[CoordActionCandidate] = []
    frontier_filtered: List[CoordActionCandidate] = []
    filtered_out: List[Dict[str, str]] = []
    state_noops = frontier_state.state_noop_counts.get(state_key, {})
    for action_id in coord_action_ids:
        scored = score_candidates(
            action_id,
            coord_candidates,
            selector_scores,
            frontier_state.global_noop_counts,
            state_noops,
            frontier_state.action_coord_tried,
            frontier_state.action_attempts,
            bg_penalties,
            cfg,
        )
        frontier_raw.extend(scored)
        for cand in scored:
            key = (cand.action_id, cand.x, cand.y)
            if frontier_state.coord_tried.get(key, 0) >= cfg.attempts_per_coord_candidate:
                filtered_out.append({"action_id": cand.action_id, "reason": "coord_attempt_cap"})
                continue
            if frontier_state.global_noop_counts.get(key, 0) >= cfg.global_ban_noop_after:
                filtered_out.append({"action_id": cand.action_id, "reason": "global_noop_ban"})
                continue
            frontier_filtered.append(cand)
    frontier_filtered.sort(key=lambda c: (-c.score, c.action_id, c.y, c.x))
    return frontier_filtered, filtered_out, frontier_raw


def _coord_candidates_from_fp(
    fp_current: Any,
    cfg: FullExplorerConfig,
) -> Tuple[List[CoordCandidate], Optional[np.ndarray], int, Optional[Tuple[int, int, int, int]]]:
    if fp_current is None:
        return [], None, 0, None
    if hasattr(fp_current, "state_summary"):
        grid_summary = fp_current.state_summary.grid_summaries[0] if fp_current.state_summary.grid_summaries else None
        if grid_summary is None:
            return [], None, 0, None
        bg_color = grid_summary.bg_candidates[0][0] if grid_summary.bg_candidates else 0
        ascii_grid = fp_current.viz_artifacts.ascii_grid.get(grid_summary.name) if fp_current.viz_artifacts else None
        if ascii_grid is None:
            return [], None, bg_color, None
        grid = _grid_from_ascii(ascii_grid)
        diff_bbox = fp_current.diff_summary.changed_bbox if fp_current.diff_summary else None
        return select_build_coords(grid, bg_color, diff_bbox, cfg), grid, bg_color, diff_bbox

    if isinstance(fp_current, dict):
        state_summary = fp_current.get("state_summary") or {}
        grids = state_summary.get("grid_summaries") or []
        if not grids:
            return [], None, 0, None
        grid_summary = grids[0]
        bg_candidates = grid_summary.get("bg_candidates") or []
        bg_color = int(bg_candidates[0][0]) if bg_candidates else 0
        grid_name = grid_summary.get("name")
        ascii_map = fp_current.get("viz_artifacts", {}).get("ascii_grid", {})
        ascii_grid = None
        if isinstance(ascii_map, dict) and ascii_map:
            if grid_name in ascii_map:
                ascii_grid = ascii_map.get(grid_name)
            elif ascii_map:
                ascii_grid = next(iter(ascii_map.values()))
        if ascii_grid is None:
            return [], None, bg_color, None
        grid = _grid_from_ascii(ascii_grid)
        diff_bbox = (fp_current.get("diff_summary") or {}).get("changed_bbox")
        return select_build_coords(grid, bg_color, diff_bbox, cfg), grid, bg_color, diff_bbox
    return [], None, 0, None


def _pop_coord_candidate(
    frontier_state: FullFrontierState,
    current_state: str,
    cfg: FullExplorerConfig,
) -> Optional[CoordActionCandidate]:
    frontier = frontier_state.state_frontier.get(current_state, [])
    while frontier:
        candidate = frontier.pop(0)
        key = (candidate.action_id, candidate.x, candidate.y)
        if key in frontier_state.state_banlist.get(current_state, set()):
            continue
        if frontier_state.coord_tried.get(key, 0) >= cfg.attempts_per_coord_candidate:
            continue
        if frontier_state.global_noop_counts.get(key, 0) >= cfg.global_ban_noop_after:
            continue
        return candidate
    return None


def _frontier_actions(frontier: List[CoordActionCandidate]) -> List[Dict[str, Any]]:
    return [{"type": "coord", "action_id": c.action_id, "x": c.x, "y": c.y} for c in frontier]


def _record_action_selection(
    blackboard: Any,
    *,
    mode: str,
    state_hash_before: str,
    candidates_before: List[Dict[str, Any]],
    candidates_after: List[Dict[str, Any]],
    filtered_out: List[Dict[str, str]],
    scores: Dict[str, float],
    selected_action: Optional[Dict[str, Any]],
    selected_reason: str,
) -> None:
    report = {
        "mode": mode,
        "state_hash_before": state_hash_before,
        "candidates_before_filter": candidates_before,
        "candidates_after_filter": candidates_after,
        "filtered_out": filtered_out,
        "scores": scores,
        "selected_action": selected_action,
        "selected_reason": selected_reason,
    }
    if hasattr(blackboard, "action_selection_report"):
        blackboard.action_selection_report = report
    elif isinstance(blackboard, dict):
        blackboard["action_selection_report"] = report


def _build_frontier(
    obs: Any,
    prev_obs: Any,
    fp_analyst: FPAnalyst,
    action_space: List[Any],
    cfg: FullExplorerConfig,
    state_noop_counts: Dict[Tuple[str, int, int], int],
    global_noop_counts: Dict[Tuple[str, int, int], int],
    action_coord_tried: Dict[str, set[Tuple[int, int]]],
    action_attempts: Dict[str, int],
    coord_tried: Dict[Tuple[str, int, int], int],
) -> List[CoordActionCandidate]:
    report = fp_analyst.analyze(obs, prev_observation=prev_obs)
    if not report.state_summary.grid_summaries:
        return []
    grid_summary = report.state_summary.grid_summaries[0]
    bg_color = grid_summary.bg_candidates[0][0] if grid_summary.bg_candidates else 0
    grid = normalize_observation(obs, schema_warnings=[]).grids[0]
    diff_bbox = report.diff_summary.changed_bbox if report.diff_summary else None
    coord_candidates = select_build_coords(grid, bg_color, diff_bbox, cfg)
    bg_penalties = _bg_penalties(grid, bg_color, coord_candidates, cfg)

    selector_scores = selector_base_scores(cfg)
    frontier: List[CoordActionCandidate] = []
    for action in action_space:
        scored = score_candidates(
            action.name,
            coord_candidates,
            selector_scores,
            global_noop_counts,
            state_noop_counts,
            action_coord_tried,
            action_attempts,
            bg_penalties,
            cfg,
        )
        scored = [
            cand
            for cand in scored
            if coord_tried.get((cand.action_id, cand.x, cand.y), 0) < cfg.attempts_per_coord_candidate
            and global_noop_counts.get((cand.action_id, cand.x, cand.y), 0) < cfg.global_ban_noop_after
        ]
        frontier.extend(scored)
    frontier.sort(key=lambda c: (-c.score, c.action_id, c.y, c.x))
    return frontier


def _step_and_record(
    env: Any,
    fp_analyst: FPAnalyst,
    obs: Any,
    prev_obs: Any,
    action_obj: Any,
    x: int,
    y: int,
    selector: str,
    current_state: str,
    graph: FullTransitionGraphStore,
    trace_writer: Optional[TraceWriter],
    coord_tried: Dict[Tuple[str, int, int], int],
    global_noop_counts: Dict[Tuple[str, int, int], int],
    state_noop_counts: Dict[str, Dict[Tuple[str, int, int], int]],
    state_banlist: Dict[str, set[Tuple[str, int, int]]],
    action_coord_tried: Dict[str, set[Tuple[int, int]]],
    action_attempts: Dict[str, int],
    action_selector_counts: Dict[str, Dict[str, int]],
    action_changed_cells: Dict[str, int],
    action_changed_bbox_area: Dict[str, int],
    action_event_counts: Dict[str, Counter[str]],
    action_coord_counts: Dict[str, Dict[Tuple[int, int], int]],
    action_coord_noops: Dict[str, Dict[Tuple[int, int], int]],
    cfg: FullExplorerConfig,
    state_seq: deque[str],
    steps: int,
    state_action_attempts: Dict[str, Dict[str, int]],
) -> Tuple[Any, Any, int, str]:
    prev_obs = obs
    obs = env.step(action_obj)
    steps += 1

    report = fp_analyst.analyze(obs, prev_observation=prev_obs, action_taken=action_obj)
    prev_report = fp_analyst.analyze(prev_obs) if prev_obs is not None else None
    diff = report.diff_summary
    changed_cells = diff.changed_cells_count if diff else 0
    bbox_area_val = bbox_area(diff.changed_bbox) if diff and diff.changed_bbox else 0
    event_signatures = [sig.kind for sig in diff.event_signatures] if diff else []

    next_state = _state_key(obs)
    if prev_report is not None:
        _register_state(graph, prev_report, current_state)
    else:
        _register_state(graph, report, current_state)
    _register_state(graph, report, next_state)
    graph.add_edge(
        current_state,
        action_obj.name,
        x,
        y,
        next_state,
        changed_cells,
        bbox_area_val,
        event_signatures,
        steps,
    )

    key = (action_obj.name, x, y)
    coord_tried[key] = coord_tried.get(key, 0) + 1
    action_coord_tried.setdefault(action_obj.name, set()).add((x, y))
    action_attempts[action_obj.name] = action_attempts.get(action_obj.name, 0) + 1
    state_action_attempts.setdefault(current_state, {})
    state_action_attempts[current_state][action_obj.name] = state_action_attempts[current_state].get(action_obj.name, 0) + 1
    action_selector_counts[action_obj.name][selector] = action_selector_counts[action_obj.name].get(selector, 0) + 1
    action_changed_cells[action_obj.name] += changed_cells
    action_changed_bbox_area[action_obj.name] += bbox_area_val
    for sig in event_signatures:
        action_event_counts[action_obj.name][sig] += 1
    action_coord_counts[action_obj.name][(x, y)] = action_coord_counts[action_obj.name].get((x, y), 0) + 1

    if changed_cells == 0:
        state_noop_counts.setdefault(current_state, {})[key] = state_noop_counts.get(current_state, {}).get(key, 0) + 1
        global_noop_counts[key] = global_noop_counts.get(key, 0) + 1
        action_coord_noops[action_obj.name][(x, y)] = action_coord_noops[action_obj.name].get((x, y), 0) + 1
        if state_noop_counts[current_state][key] >= cfg.ban_noop_after:
            state_banlist.setdefault(current_state, set()).add(key)

    entry = {
        "step_idx": steps,
        "state_before": current_state,
        "action": {"type": "coord", "action_id": action_obj.name, "x": x, "y": y},
        "state_after": next_state,
        "reward": _reward_value(obs),
        "reward_delta": _reward_delta(prev_obs, obs),
        "terminal": _terminal_bool(obs),
        "info": {"state": _terminal_flag(obs)},
        "counters": _counter_snapshot(obs),
        "fp_diff": {
            "changed_cells": changed_cells,
            "changed_bbox_area": bbox_area_val,
            "event_signatures": event_signatures,
        },
    }
    if trace_writer:
        trace_writer.write(entry)

    state_seq.append(next_state)
    return obs, prev_obs, steps, next_state


def _build_report(
    game_id: str,
    seed: int,
    steps: int,
    loops_detected: Dict[str, int],
    termination_reason: str,
    graph: FullTransitionGraphStore,
    state_frontier: Dict[str, List[CoordActionCandidate]],
    state_noop_counts: Dict[str, Dict[Tuple[str, int, int], int]],
    state_banlist: Dict[str, set[Tuple[str, int, int]]],
    state_action_attempts: Dict[str, Dict[str, int]],
    action_attempts: Dict[str, int],
    action_coord_tried: Dict[str, set[Tuple[int, int]]],
    action_selector_counts: Dict[str, Dict[str, int]],
    action_changed_cells: Dict[str, int],
    action_changed_bbox_area: Dict[str, int],
    action_event_counts: Dict[str, Counter[str]],
    action_coord_counts: Dict[str, Dict[Tuple[int, int], int]],
    action_coord_noops: Dict[str, Dict[Tuple[int, int], int]],
    coord_tried: Dict[Tuple[str, int, int], int],
    trace_path: str,
    cfg: FullExplorerConfig,
) -> FullExplorerReport:
    coord_effect_model = _summarize_coord_effects(
        action_attempts,
        action_selector_counts,
        action_changed_cells,
        action_changed_bbox_area,
        action_event_counts,
        action_coord_counts,
        action_coord_noops,
        cfg,
    )
    frontier = {}
    for state, candidates in state_frontier.items():
        frontier[state] = FrontierEntry(
            state=state,
            pending_candidates=len(candidates),
            attempt_counts_by_action_family=state_action_attempts.get(state, {}),
            cooldowns={},
            banlist=sorted(state_banlist.get(state, set())),
        )

    run_summary = RunSummary(
        game_id=game_id,
        seed=seed,
        steps_executed=steps,
        unique_states=len(graph.nodes),
        unique_transitions=len(graph.edges),
        unique_coord_actions_tried=len(coord_tried),
        loops_detected=loops_detected,
        termination_reason=termination_reason,
    )

    artifacts = {"trace": trace_path}

    return FullExplorerReport(
        run_summary=run_summary,
        coord_action_effect_model=coord_effect_model,
        transition_graph=graph.to_graph(),
        frontier=frontier,
        artifacts=artifacts,
    )


def _summarize_coord_effects(
    action_attempts: Dict[str, int],
    action_selector_counts: Dict[str, Dict[str, int]],
    action_changed_cells: Dict[str, int],
    action_changed_bbox_area: Dict[str, int],
    action_event_counts: Dict[str, Counter[str]],
    action_coord_counts: Dict[str, Dict[Tuple[int, int], int]],
    action_coord_noops: Dict[str, Dict[Tuple[int, int], int]],
    cfg: FullExplorerConfig,
) -> Dict[str, CoordActionEffectStats]:
    effects: Dict[str, CoordActionEffectStats] = {}
    for action_id, attempts in action_attempts.items():
        noop_total = sum(action_coord_noops.get(action_id, {}).values())
        avg_changed_cells = action_changed_cells.get(action_id, 0) / attempts if attempts else 0.0
        avg_changed_bbox_area = action_changed_bbox_area.get(action_id, 0) / attempts if attempts else 0.0

        event_counts = action_event_counts.get(action_id, Counter())
        dominant = [(k, v / attempts) for k, v in event_counts.most_common(3)] if attempts else []

        coord_attempts = action_coord_counts.get(action_id, {})
        coord_noops = action_coord_noops.get(action_id, {})
        hotspots = _rank_coords(coord_attempts, coord_noops, invert=True, top_k=cfg.topK_hotspots)
        negative_zones = _rank_coords(coord_attempts, coord_noops, invert=False, top_k=cfg.topK_negative_zones)

        effects[action_id] = CoordActionEffectStats(
            attempts_total=attempts,
            attempts_by_coord_selector=action_selector_counts.get(action_id, {}),
            no_effect_rate=noop_total / attempts if attempts else 0.0,
            avg_changed_cells=avg_changed_cells,
            avg_changed_bbox_area=avg_changed_bbox_area,
            dominant_event_signatures=dominant,
            hotspots=hotspots,
            negative_zones=negative_zones,
        )
    return effects


def _rank_coords(
    coord_attempts: Dict[Tuple[int, int], int],
    coord_noops: Dict[Tuple[int, int], int],
    invert: bool,
    top_k: int,
) -> List[Tuple[int, int, float]]:
    items = []
    for (x, y), attempts in coord_attempts.items():
        noops = coord_noops.get((x, y), 0)
        if attempts <= 0:
            continue
        effect_rate = (attempts - noops) / attempts
        rate = effect_rate if invert else (noops / attempts)
        items.append((x, y, rate))
    items.sort(key=lambda item: (-item[2], item[1], item[0]) if invert else (-item[2], item[1], item[0]))
    return items[:top_k]


def _state_key(observation: Any) -> str:
    norm = normalize_observation(observation, schema_warnings=[])
    return grid_hash(norm.grids)


def _register_state(graph: FullTransitionGraphStore, report: Any, state_key: str) -> None:
    summaries = report.state_summary.grid_summaries
    if not summaries:
        return
    grid = summaries[0]
    node = TransitionNode(
        state=state_key,
        height=grid.height,
        width=grid.width,
        palette_size=len(grid.palette_sorted),
        object_count=len(report.state_summary.object_catalog),
    )
    graph.add_node(node)


def _pick_frontier_state(frontier: Dict[str, List[CoordActionCandidate]], current_state: str) -> Optional[str]:
    candidates = [(state, items) for state, items in frontier.items() if items]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    return candidates[0][0]


def _action_by_name(action_space: List[Any], name: str) -> Optional[Any]:
    for action in action_space:
        if action.name == name:
            return action
    return None


def _detect_loops(state_seq: deque[str], current_state: str, cfg: FullExplorerConfig) -> List[str]:
    flags: List[str] = []
    if state_seq and state_seq[-1] == current_state:
        flags.append("immediate_repeat")
    seq = list(state_seq)
    for cycle_len in range(2, cfg.short_cycle_max_len + 1):
        if len(seq) >= cycle_len * 2 and seq[-cycle_len:] == seq[-2 * cycle_len : -cycle_len]:
            flags.append(f"short_cycle_{cycle_len}")
            break
    if seq.count(current_state) > cfg.revisit_threshold_R:
        flags.append("state_revisit_flood")
    return flags


def _bbox_area(bbox: Optional[Tuple[int, int, int, int]]) -> int:
    return bbox_area(bbox)


def _palette_delta(prev_obs: Any, obs: Any) -> Dict[str, List[int]]:
    if prev_obs is None or obs is None:
        return {"added": [], "removed": []}
    prev_norm = normalize_observation(prev_obs, schema_warnings=[])
    curr_norm = normalize_observation(obs, schema_warnings=[])
    prev_palette = set()
    curr_palette = set()
    for grid in prev_norm.grids:
        prev_palette.update(int(v) for v in set(grid.flatten()))
    for grid in curr_norm.grids:
        curr_palette.update(int(v) for v in set(grid.flatten()))
    added = sorted(curr_palette - prev_palette)
    removed = sorted(prev_palette - curr_palette)
    return {"added": added, "removed": removed}


def _object_count_delta(diff: Any) -> int:
    if diff is None:
        return 0
    appeared = sum(1 for d in diff.per_object_deltas if d.event == "appeared")
    disappeared = sum(1 for d in diff.per_object_deltas if d.event == "disappeared")
    return appeared - disappeared


def _terminal_flag(obs: Any) -> Optional[str]:
    state = getattr(obs, "state", None)
    if state is None:
        return None
    return state.name if hasattr(state, "name") else str(state)


def _terminal_bool(obs: Any) -> Optional[bool]:
    state = _terminal_flag(obs)
    if state is None:
        return None
    upper = str(state).upper()
    if upper in {"WIN", "WON", "SUCCESS", "GAME_OVER", "LOSE", "LOST", "FAIL"}:
        return True
    return False


def _reward_value(obs: Any) -> Optional[int]:
    if obs is None:
        return None
    score = getattr(obs, "levels_completed", None)
    if score is None:
        return None
    try:
        return int(score)
    except Exception:
        return None


def _counter_snapshot(obs: Any) -> Dict[str, Any]:
    if obs is None:
        return {}
    counters: Dict[str, Any] = {}
    for key in ("levels_completed", "win_levels"):
        val = getattr(obs, key, None)
        if isinstance(val, (int, float)):
            counters[key] = val
    return counters


def _reward_delta(prev_obs: Any, obs: Any) -> Optional[int]:
    if prev_obs is None or obs is None:
        return None
    prev_score = getattr(prev_obs, "levels_completed", None)
    curr_score = getattr(obs, "levels_completed", None)
    if prev_score is None or curr_score is None:
        return None
    try:
        return int(curr_score) - int(prev_score)
    except Exception:
        return None


def _object_delta_counts(diff: Any) -> Dict[str, int]:
    if diff is None:
        return {"moved": 0, "appeared": 0, "disappeared": 0}
    moved = sum(1 for d in diff.per_object_deltas if d.event == "moved")
    appeared = sum(1 for d in diff.per_object_deltas if d.event == "appeared")
    disappeared = sum(1 for d in diff.per_object_deltas if d.event == "disappeared")
    return {"moved": moved, "appeared": appeared, "disappeared": disappeared}


def _grid_from_ascii(ascii_grid: str) -> Any:
    return grid_from_ascii(ascii_grid)


def _bg_penalties(
    grid: Any,
    bg_color: int,
    candidates: List[CoordCandidate],
    cfg: FullExplorerConfig,
) -> Dict[Tuple[int, int], bool]:
    object_cells = _object_cells(grid, bg_color)
    penalties: Dict[Tuple[int, int], bool] = {}
    for cand in candidates:
        if int(grid[cand.y, cand.x]) != bg_color:
            penalties[(cand.x, cand.y)] = False
            continue
        dist = _min_manhattan_distance(cand.x, cand.y, object_cells)
        penalties[(cand.x, cand.y)] = dist >= cfg.far_distance_threshold
    return penalties


def _object_cells(grid: Any, bg_color: int) -> List[Tuple[int, int]]:
    cells = []
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if int(grid[y, x]) != bg_color:
                cells.append((x, y))
    return cells


def _min_manhattan_distance(x: int, y: int, cells: List[Tuple[int, int]]) -> int:
    if not cells:
        return 0
    return min(abs(x - cx) + abs(y - cy) for cx, cy in cells)


def _empty_report(game_id: str, seed: int, reason: str, out_dir: str) -> FullExplorerReport:
    run_summary = RunSummary(
        game_id=game_id,
        seed=seed,
        steps_executed=0,
        unique_states=0,
        unique_transitions=0,
        unique_coord_actions_tried=0,
        loops_detected={},
        termination_reason=reason,
    )
    return FullExplorerReport(
        run_summary=run_summary,
        coord_action_effect_model={},
        transition_graph=FullTransitionGraphStore().to_graph(),
        frontier={},
        artifacts={"trace": os.path.join(out_dir, "trace.jsonl")},
    )

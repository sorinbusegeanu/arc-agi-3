from __future__ import annotations

import os
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

from .action_schema import ActionSchema, parse_action_schema_data
from .fp_analyst import FPAnalyst
from .grid_utils import bbox_area, grid_hash
from .logger import get_logger
from .normalize import normalize_observation
from .simple_explorer_config import SimpleExplorerConfig
from .simple_explorer_types import (
    ActionEffectStats,
    ActionKey,
    FrontierEntry,
    RunSummary,
    SimpleFrontierState,
    SimpleExplorerReport,
    StateKey,
    TransitionNode,
)
from .trace import TraceWriter
from .transition_graph import TransitionGraphStore

logger = get_logger(__name__)


def run(
    env: Any,
    game_id: str,
    seed: int,
    fp_analyst: FPAnalyst,
    cfg: SimpleExplorerConfig,
    ctx: Optional[Dict[str, Any]] = None,
    start_observation: Any = None,
) -> SimpleExplorerReport:
    ctx = ctx or {}
    out_dir = ctx.get("output_dir", os.path.join("runs", "simple_explorer", f"{game_id}_{seed}"))
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "trace.jsonl")
    trace_writer = TraceWriter(trace_path) if cfg.save_trace else None

    loops_detected = Counter()
    errors: List[str] = []

    if start_observation is None:
        try:
            obs = env.reset()
        except Exception as e:
            errors.append(f"reset_failed: {e}")
            return _empty_report(game_id, seed, errors, out_dir)
    else:
        obs = start_observation

    prev_obs = None
    state_seq: deque[StateKey] = deque(maxlen=cfg.revisit_window_N)
    last_seen_step: Dict[StateKey, int] = {}
    state_action_attempts: Dict[StateKey, Dict[ActionKey, int]] = {}
    state_untried: Dict[StateKey, List[ActionKey]] = {}
    state_action_outcomes: Dict[StateKey, Dict[ActionKey, Dict[StateKey, int]]] = {}

    graph = TransitionGraphStore()
    action_attempts_global: Dict[ActionKey, List[Dict[str, Any]]] = {}

    def _state_key(observation: Any) -> StateKey:
        norm = normalize_observation(observation, schema_warnings=[])
        return grid_hash(norm.grids)

    GameAction = type(env.action_space[0]) if env.action_space else None

    def _simple_actions() -> List[Any]:
        actions = [a for a in env.action_space if not a.is_complex()]
        return [a for a in actions if a.name.upper() != "NOOP"]

    action_space = _simple_actions()
    if not action_space:
        errors.append("no_simple_actions")
        return _empty_report(game_id, seed, errors, out_dir)
    action_ids = [a.name for a in action_space]

    steps = 0
    current_state = _state_key(obs)
    state_seq.append(current_state)
    _ensure_state(state_action_attempts, state_untried, state_action_outcomes, current_state, action_ids)

    while steps < cfg.max_steps:
        if len(state_action_attempts) >= cfg.max_unique_states:
            break

        if not state_untried.get(current_state):
            next_state = _pick_frontier(state_untried, last_seen_step)
            if next_state is None:
                break
            if cfg.bfs_route_to_frontier:
                path = graph.bfs_path(current_state, next_state, cfg.bfs_max_depth)
                if path is None:
                    current_state = next_state
                    _ensure_state(state_action_attempts, state_untried, state_action_outcomes, current_state, action_ids)
                    continue
                for action_key, target_state in path:
                    prev_state = current_state
                    action_obj = _action_from_key(action_space, action_key)
                    if action_obj is None:
                        continue
                    obs, prev_obs, steps, current_state = _step_and_record(
                        env,
                        fp_analyst,
                        obs,
                        prev_obs,
                        action_obj,
                        current_state,
                        graph,
                        trace_writer,
                        action_attempts_global,
                        steps,
                        state_seq,
                        cfg,
                    )
                    _ensure_state(state_action_attempts, state_untried, state_action_outcomes, current_state, action_ids)
                    _record_action_outcome(
                        state_action_attempts,
                        state_action_outcomes,
                        state_untried,
                        prev_state,
                        action_key,
                        current_state,
                        cfg,
                    )
                    if steps >= cfg.max_steps:
                        break
                if steps >= cfg.max_steps:
                    break
            else:
                current_state = next_state
                _ensure_state(state_action_attempts, state_untried, state_action_outcomes, current_state, action_ids)
                continue

        action_key = _pop_next_action(state_untried, current_state, action_ids, steps)
        if action_key is None:
            continue
        action_obj = _action_from_key(action_space, action_key)
        if action_obj is None:
            continue

        try:
            prev_state = current_state
            obs, prev_obs, steps, current_state = _step_and_record(
                env,
                fp_analyst,
                obs,
                prev_obs,
                action_obj,
                current_state,
                graph,
                trace_writer,
                action_attempts_global,
                steps,
                state_seq,
                cfg,
            )
        except Exception as e:
            errors.append(f"step_failed: {e}")
            break

        _ensure_state(state_action_attempts, state_untried, state_action_outcomes, current_state, action_ids)

        _record_action_outcome(
            state_action_attempts,
            state_action_outcomes,
            state_untried,
            prev_state,
            action_key,
            current_state,
            cfg,
        )

        last_seen_step[current_state] = steps

        loop_flags = _detect_loops(state_seq, current_state, cfg)
        for flag in loop_flags:
            loops_detected[flag] += 1
        if loop_flags and cfg.noop_edge_deprioritize:
            next_state = _pick_frontier(state_untried, last_seen_step)
            if next_state:
                current_state = next_state

    report = _build_report(
        game_id=game_id,
        seed=seed,
        steps=steps,
        loops_detected=dict(loops_detected),
        errors=errors,
        graph=graph,
        action_attempts_global=action_attempts_global,
        state_untried=state_untried,
        state_action_attempts=state_action_attempts,
        out_dir=out_dir,
    )
    return report


def summarize(
    trace: List[Dict[str, Any]],
    fp_reports: Optional[List[Any]],
    cfg: SimpleExplorerConfig,
) -> Dict[ActionKey, ActionEffectStats]:
    by_action: Dict[ActionKey, List[Dict[str, Any]]] = {}
    for entry in trace:
        action = _action_id_from_entry(entry)
        if not action:
            continue
        by_action.setdefault(action, []).append(entry)

    return _summarize_action_effects(by_action)


def choose_action(
    blackboard: Any,
    action_schema: Any,
    fp_current: Any,
    frontier_state: Optional[SimpleFrontierState],
    cfg: Optional[SimpleExplorerConfig],
) -> Optional[Dict[str, Any]]:
    cfg = cfg or SimpleExplorerConfig()
    schema = _parse_action_schema(action_schema)
    action_ids = _simple_action_ids(schema)
    if not action_ids:
        return None

    frontier_state = frontier_state or SimpleFrontierState()
    _update_frontier_state_from_history(frontier_state, blackboard, action_ids, cfg)

    current_state = getattr(blackboard, "state_hash", None)
    if current_state is None and isinstance(blackboard, dict):
        current_state = blackboard.get("state_hash")
    if current_state is None:
        return None

    _ensure_state(
        frontier_state.state_action_attempts,
        frontier_state.state_untried,
        frontier_state.state_action_outcomes,
        current_state,
        action_ids,
    )
    step_idx = getattr(blackboard, "step_idx", None)
    if step_idx is None and isinstance(blackboard, dict):
        step_idx = blackboard.get("step_idx", 0)
    if step_idx is None:
        step_idx = 0
    action_key = _pop_next_action(frontier_state.state_untried, current_state, action_ids, int(step_idx))
    if action_key is None:
        return None
    _record_action_selection(
        blackboard,
        mode="probe_simple",
        state_hash_before=current_state,
        candidates_before=list(frontier_state.state_untried.get(current_state, [])),
        candidates_after=list(frontier_state.state_untried.get(current_state, [])),
        filtered_out=[],
        scores={},
        selected_action={"type": "simple", "action_id": action_key},
        selected_reason="rotation",
    )
    return {"type": "simple", "action_id": action_key}


def build_frontier_report(
    blackboard: Any,
    action_schema: Any,
    frontier_state: Optional[SimpleFrontierState],
    cfg: Optional[SimpleExplorerConfig],
) -> Dict[str, Any]:
    cfg = cfg or SimpleExplorerConfig()
    schema = _parse_action_schema(action_schema)
    action_ids = _simple_action_ids(schema)
    frontier_state = frontier_state or SimpleFrontierState()
    _update_frontier_state_from_history(frontier_state, blackboard, action_ids, cfg)

    current_state = getattr(blackboard, "state_hash", None) or blackboard.get("state_hash")
    if current_state is not None:
        _ensure_state(
            frontier_state.state_action_attempts,
            frontier_state.state_untried,
            frontier_state.state_action_outcomes,
            current_state,
            action_ids,
        )

    frontier: Dict[StateKey, Dict[str, Any]] = {}
    for state, actions in frontier_state.state_untried.items():
        frontier[state] = {
            "state": state,
            "untried_actions": list(actions),
            "action_attempt_counts": frontier_state.state_action_attempts.get(state, {}),
        }

    if isinstance(blackboard, dict):
        history_len = len(blackboard.get("history", []))
    else:
        history_len = len(getattr(blackboard, "history", []))
    return {
        "run_summary": {"steps_executed": history_len},
        "action_effect_model": {},
        "frontier": frontier,
    }


def _record_action_selection(
    blackboard: Any,
    *,
    mode: str,
    state_hash_before: str,
    candidates_before: List[str],
    candidates_after: List[str],
    filtered_out: List[Dict[str, str]],
    scores: Dict[str, float],
    selected_action: Dict[str, Any],
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


def _parse_action_schema(action_schema: Any) -> ActionSchema:
    if isinstance(action_schema, ActionSchema):
        return action_schema
    if isinstance(action_schema, dict):
        return parse_action_schema_data(action_schema)
    raise ValueError("action_schema must be an ActionSchema or dict")


def _simple_action_ids(schema: ActionSchema) -> List[ActionKey]:
    action_ids = [a.action_id for a in schema.actions if a.kind == "simple"]
    return [action_id for action_id in action_ids if action_id.upper() != "NOOP"]


def _update_frontier_state_from_history(
    frontier_state: SimpleFrontierState,
    blackboard: Any,
    action_ids: List[ActionKey],
    cfg: SimpleExplorerConfig,
) -> None:
    history = getattr(blackboard, "history", None)
    if history is None and isinstance(blackboard, dict):
        history = blackboard.get("history", [])
    if history is None:
        history = []
    last_processed = frontier_state.last_processed_step
    entries = [entry for entry in history if isinstance(entry, dict) and entry.get("step_idx", -1) > last_processed]
    entries.sort(key=lambda e: e.get("step_idx", -1))
    for entry in entries:
        action = entry.get("action") or {}
        if action.get("type") != "simple":
            continue
        action_id = action.get("action_id")
        if action_id not in action_ids:
            continue
        prev_state = entry.get("state_before")
        next_state = entry.get("state_after")
        if prev_state is None or next_state is None:
            continue
        _ensure_state(
            frontier_state.state_action_attempts,
            frontier_state.state_untried,
            frontier_state.state_action_outcomes,
            prev_state,
            action_ids,
        )
        _ensure_state(
            frontier_state.state_action_attempts,
            frontier_state.state_untried,
            frontier_state.state_action_outcomes,
            next_state,
            action_ids,
        )
        _record_action_outcome(
            frontier_state.state_action_attempts,
            frontier_state.state_action_outcomes,
            frontier_state.state_untried,
            prev_state,
            action_id,
            next_state,
            cfg,
        )
        frontier_state.last_processed_step = max(frontier_state.last_processed_step, entry.get("step_idx", -1))


def _action_id_from_entry(entry: Dict[str, Any]) -> Optional[str]:
    action = entry.get("action")
    if isinstance(action, dict):
        return action.get("action_id")
    if isinstance(action, str):
        return action
    return None


def _step_and_record(
    env: Any,
    fp_analyst: FPAnalyst,
    obs: Any,
    prev_obs: Any,
    action_obj: GameAction,
    current_state: StateKey,
    graph: TransitionGraphStore,
    trace_writer: Optional[TraceWriter],
    action_attempts_global: Dict[ActionKey, List[Dict[str, Any]]],
    steps: int,
    state_seq: deque[StateKey],
    cfg: SimpleExplorerConfig,
) -> Tuple[Any, Any, int, StateKey]:
    prev_obs = obs
    obs = env.step(action_obj)
    steps += 1

    report = fp_analyst.analyze(obs, prev_observation=prev_obs, action_taken=action_obj)
    prev_report = fp_analyst.analyze(prev_obs) if prev_obs is not None else None
    diff = report.diff_summary
    changed_cells = diff.changed_cells_count if diff else 0
    bbox_area_val = bbox_area(diff.changed_bbox) if diff and diff.changed_bbox else 0
    event_signatures = [sig.kind for sig in diff.event_signatures] if diff else []
    motion_vectors = _motion_vectors(diff) if diff else []
    palette_delta = _palette_delta(prev_obs, obs)
    object_count_delta = _object_count_delta(diff)
    terminal_flag = _terminal_flag(obs)
    reward_delta = _reward_delta(prev_obs, obs)

    next_state = grid_hash(normalize_observation(obs, schema_warnings=[]).grids)

    if prev_report is not None:
        _register_state(graph, prev_report, current_state)
    else:
        _register_state(graph, report, current_state)
    _register_state(graph, report, next_state)
    graph.add_edge(
        current_state,
        action_obj.name,
        next_state,
        changed_cells,
        bbox_area_val,
        event_signatures,
        steps,
    )

    entry = {
        "step_idx": steps,
        "state_before": current_state,
        "action": {"type": "simple", "action_id": action_obj.name},
        "state_after": next_state,
        "reward": _reward_value(obs),
        "reward_delta": reward_delta,
        "terminal": _terminal_bool(obs),
        "info": {"state": terminal_flag},
        "counters": _counter_snapshot(obs),
        "fp_diff": {
            "changed_cells": changed_cells,
        "changed_bbox_area": bbox_area_val,
            "event_signatures": event_signatures,
        },
    }
    if trace_writer:
        trace_writer.write(entry)

    action_attempts_global.setdefault(action_obj.name, []).append(entry)
    state_seq.append(next_state)
    return obs, prev_obs, steps, next_state


def _register_state(graph: TransitionGraphStore, report: Any, state_key: StateKey) -> None:
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


def _summarize_action_effects(
    by_action: Dict[ActionKey, List[Dict[str, Any]]],
) -> Dict[ActionKey, ActionEffectStats]:
    out: Dict[ActionKey, ActionEffectStats] = {}
    for action, entries in by_action.items():
        attempts = len(entries)
        if attempts == 0:
            continue
        no_effect = sum(1 for e in entries if _diff_value(e, "changed_cells") == 0)
        avg_changed_cells = sum(_diff_value(e, "changed_cells") for e in entries) / attempts
        avg_bbox_area = sum(_diff_value(e, "changed_bbox_area") for e in entries) / attempts
        sig_hist = Counter()
        for e in entries:
            for sig in _diff_list(e, "event_signatures"):
                sig_hist[sig] += 1
        dominant = [(k, v / attempts) for k, v in sig_hist.most_common(3)]
        typical_motion = []
        out[action] = ActionEffectStats(
            attempts=attempts,
            no_effect_rate=no_effect / attempts,
            avg_changed_cells=avg_changed_cells,
            avg_changed_bbox_area=avg_bbox_area,
            dominant_event_signatures=dominant,
            typical_motion_vectors=typical_motion,
            common_block_conditions=[],
        )
    return out


def _diff_value(entry: Dict[str, Any], key: str) -> float:
    fp_diff = entry.get("fp_diff") or {}
    return float(fp_diff.get(key, 0))


def _diff_list(entry: Dict[str, Any], key: str) -> List[Any]:
    fp_diff = entry.get("fp_diff") or {}
    val = fp_diff.get(key, [])
    return val if isinstance(val, list) else []


def _build_report(
    game_id: str,
    seed: int,
    steps: int,
    loops_detected: Dict[str, int],
    errors: List[str],
    graph: TransitionGraphStore,
    action_attempts_global: Dict[ActionKey, List[Dict[str, Any]]],
    state_untried: Dict[StateKey, List[ActionKey]],
    state_action_attempts: Dict[StateKey, Dict[ActionKey, int]],
    out_dir: str,
) -> SimpleExplorerReport:
    run_summary = RunSummary(
        game_id=game_id,
        seed=seed,
        steps_executed=steps,
        unique_states=len(graph.nodes),
        unique_transitions=len(graph.edges),
        loops_detected=loops_detected,
        timeouts_or_errors=errors,
    )
    action_effect_model = _summarize_action_effects(action_attempts_global)
    frontier = {}
    for state, actions in state_untried.items():
        frontier[state] = FrontierEntry(
            state=state,
            untried_actions=actions,
            action_attempt_counts=state_action_attempts.get(state, {}),
        )
    artifacts = {
        "trace": os.path.join(out_dir, "trace.jsonl"),
    }
    return SimpleExplorerReport(
        run_summary=run_summary,
        action_effect_model=action_effect_model,
        transition_graph=graph.to_graph(),
        frontier=frontier,
        artifacts=artifacts,
    )


def _empty_report(game_id: str, seed: int, errors: List[str], out_dir: str) -> SimpleExplorerReport:
    run_summary = RunSummary(
        game_id=game_id,
        seed=seed,
        steps_executed=0,
        unique_states=0,
        unique_transitions=0,
        loops_detected={},
        timeouts_or_errors=errors,
    )
    return SimpleExplorerReport(
        run_summary=run_summary,
        action_effect_model={},
        transition_graph=TransitionGraphStore().to_graph(),
        frontier={},
        artifacts={"trace": os.path.join(out_dir, "trace.jsonl")},
    )


def _ensure_state(
    state_action_attempts: Dict[StateKey, Dict[ActionKey, int]],
    state_untried: Dict[StateKey, List[ActionKey]],
    state_action_outcomes: Dict[StateKey, Dict[ActionKey, Dict[StateKey, int]]],
    state: StateKey,
    action_ids: List[ActionKey],
) -> None:
    if state not in state_action_attempts:
        state_action_attempts[state] = {}
        state_untried[state] = _rotated_actions(action_ids, state)
        state_action_outcomes[state] = {}


def _rotated_actions(action_ids: List[ActionKey], state: StateKey) -> List[ActionKey]:
    if not action_ids:
        return []
    offset = abs(hash(state)) % len(action_ids)
    return action_ids[offset:] + action_ids[:offset]


def _pop_next_action(
    state_untried: Dict[StateKey, List[ActionKey]],
    state: StateKey,
    action_ids: List[ActionKey],
    offset: int = 0,
) -> Optional[ActionKey]:
    actions = state_untried.get(state, [])
    if not actions:
        return None
    if action_ids:
        preferred = action_ids[offset % len(action_ids)]
        if preferred in actions:
            return preferred
        for action_id in action_ids:
            if action_id in actions:
                return action_id
    return actions[0]


def _remove_action(
    state_untried: Dict[StateKey, List[ActionKey]],
    state: StateKey,
    action: ActionKey,
) -> None:
    if state in state_untried and action in state_untried[state]:
        state_untried[state].remove(action)


def _record_action_outcome(
    state_action_attempts: Dict[StateKey, Dict[ActionKey, int]],
    state_action_outcomes: Dict[StateKey, Dict[ActionKey, Dict[StateKey, int]]],
    state_untried: Dict[StateKey, List[ActionKey]],
    prev_state: StateKey,
    action_key: ActionKey,
    next_state: StateKey,
    cfg: SimpleExplorerConfig,
) -> None:
    state_action_attempts[prev_state][action_key] = state_action_attempts[prev_state].get(action_key, 0) + 1
    outcome_counts = state_action_outcomes[prev_state].setdefault(action_key, {})
    outcome_counts[next_state] = outcome_counts.get(next_state, 0) + 1
    if outcome_counts[next_state] >= 2:
        _remove_action(state_untried, prev_state, action_key)
    if state_action_attempts[prev_state][action_key] >= cfg.attempts_per_action_per_state:
        _remove_action(state_untried, prev_state, action_key)


def _pick_frontier(
    state_untried: Dict[StateKey, List[ActionKey]],
    last_seen_step: Dict[StateKey, int],
) -> Optional[StateKey]:
    candidates = [(state, actions) for state, actions in state_untried.items() if actions]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -len(item[1]),
            last_seen_step.get(item[0], -1),
        )
    )
    return candidates[0][0]


def _action_from_key(action_space: List[Any], key: ActionKey) -> Optional[Any]:
    for action in action_space:
        if action.name == key:
            return action
    return None


def _detect_loops(
    state_seq: deque[StateKey],
    current_state: StateKey,
    cfg: SimpleExplorerConfig,
) -> List[str]:
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


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    return bbox_area(bbox)


def _motion_vectors(diff: Any) -> List[Tuple[float, float]]:
    if diff is None:
        return []
    vectors = []
    for delta in diff.per_object_deltas:
        if delta.event == "moved":
            vectors.append((delta.dy, delta.dx))
    return vectors


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
    if hasattr(state, "name"):
        return state.name
    return str(state)


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

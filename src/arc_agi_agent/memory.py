from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from .memory_config import MemoryConfig
from .memory_types import (
    ActionEfficacy,
    CoordStat,
    MemoryState,
    MemoryUpdateInputs,
    StateActionEfficacy,
    TemplateStats,
)
from .memory_store import MemoryStore


def memory_init(
    ctx: Dict[str, Any],
    cfg: MemoryConfig,
    persisted_state: Optional[MemoryState | Dict[str, Any]] = None,
) -> MemoryState:
    if isinstance(persisted_state, dict):
        state = _state_from_dict(persisted_state)
    elif isinstance(persisted_state, MemoryState):
        state = persisted_state
    else:
        state = MemoryState()
    state.last_update_debug = None
    return state


def memory_update(state: MemoryState, inputs: MemoryUpdateInputs, cfg: MemoryConfig) -> MemoryState:
    action = inputs.action or {}
    if action.get("type") == "coord" and (action.get("x") is None or action.get("y") is None):
        alt = _coord_from_planner_decision(inputs.planner_decision)
        if alt:
            action = dict(action)
            action.update(alt)
    action_key = _action_key(action)
    if not action_key:
        state.last_update_debug = {"updated": False, "reason": "missing_action"}
        return state

    diff_summary = inputs.diff_summary or _diff_summary_from_fp(inputs.fp_report_after)
    fp_diff = inputs.fp_diff or {}
    changed_cells = _changed_cells(diff_summary, fp_diff)
    bbox_area = _changed_bbox_area(diff_summary, fp_diff)
    event_sigs = _event_signatures(diff_summary, fp_diff)
    object_deltas = _object_deltas(diff_summary)
    no_effect = changed_cells == 0

    _update_per_action(state, action_key, changed_cells, bbox_area, no_effect, inputs.ctx.get("step_idx", 0))
    _update_action_source_counts(state, action_key, inputs.planner_decision)
    _update_per_state_action(
        state,
        inputs.state_hash_before,
        action_key,
        no_effect,
        inputs.ctx.get("step_idx", 0),
        cfg,
    )
    _update_recent_actions(state, inputs.state_hash_before, action_key, cfg)
    if action.get("type") == "coord":
        _update_coord_heatmap(state, action, changed_cells, no_effect, inputs.ctx.get("step_idx", 0), cfg)

    _update_feature_windows(state, event_sigs, object_deltas, cfg)
    _update_action_signatures(state, action_key, event_sigs)
    _update_mechanic_priors(state, inputs)
    _update_template_stats(state, inputs.rule_proposer, inputs.ctx.get("step_idx", 0))

    state.last_update_debug = _build_debug(state, inputs, action_key, no_effect)
    return state


def memory_snapshot(state: MemoryState) -> Dict[str, Any]:
    return asdict(state)


def memory_save(state: MemoryState, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory_snapshot(state), f, indent=2)


def memory_load(path: str) -> MemoryState:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _state_from_dict(payload)


def memory_save_cross_run(
    state: MemoryState,
    game_id: str,
    run_id: str,
    task_signature: str,
    cfg: MemoryConfig,
    win: bool,
    run_summary: Optional[Dict[str, Any]] = None,
) -> None:
    raise NotImplementedError("Use memory_ingest_run_summary(run_summary_v1) for persistence")


def memory_ingest_run_summary(run_summary: Dict[str, Any], cfg: Optional[MemoryConfig] = None) -> None:
    cfg = cfg or MemoryConfig()
    _validate_run_summary(run_summary)
    store = MemoryStore(cfg.memory_dir, enable_lock=True)
    store.ingest_run_summary(run_summary)


def memory_view(
    state: Optional[MemoryState],
    *,
    state_hash: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if state is None:
        return {}
    view: Dict[str, Any] = {}
    view["noop_rate_by_action"] = _noop_rate_by_action(state)
    view["attempts_by_action"] = _attempts_by_action(state)
    view["avg_changed_cells_by_action"] = _avg_changed_cells_by_action(state)
    view["noop_rate_by_state_action"] = _noop_rate_by_state_action(state, state_hash)
    view["last_k_actions_per_state"] = (
        list(state.recent_actions_by_state.get(state_hash, [])) if state_hash else []
    )
    view["action_effect_signatures_by_action"] = _action_signature_diversity(state)
    view["coord_noop_rate_by_action"] = _coord_noop_rate(state)
    view["coord_effect_score_by_action"] = _coord_effect_scores(state)
    view["event_signature_baseline"] = _event_signature_baseline(state)
    view["object_delta_baseline"] = _object_delta_baseline(state)
    view["mechanic_by_fingerprint"] = dict(state.mechanic_by_fingerprint)
    view["template_stats"] = _template_stats_view(state)
    if evidence:
        _merge_evidence_into_view(view, evidence)
    return view


def memory_query(
    task_signature: str,
    *,
    game_id: Optional[str] = None,
    cfg: Optional[MemoryConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MemoryConfig()
    store = MemoryStore(cfg.memory_dir, enable_lock=False)
    return store.query(task_signature, game_id=game_id)


def memory_query_actions(
    task_signature: str,
    *,
    action_schema: Dict[str, Any],
    k: int = 20,
    cfg: Optional[MemoryConfig] = None,
) -> List[Dict[str, Any]]:
    evidence = memory_query(task_signature, cfg=cfg)
    priors = evidence.get("priors", {}).get("action", {})
    entries = []
    for action_id, stats in priors.items():
        attempts = stats.get("attempts_total", 0)
        if attempts <= 0:
            continue
        effect = stats.get("effect_total", 0)
        rate = effect / float(attempts)
        entries.append({"action_id": action_id, "effect_rate": rate, "attempts": attempts})
    entries.sort(key=lambda item: (-item["effect_rate"], item["action_id"]))
    return entries[:k]


def memory_query_game(game_id: str, *, cfg: Optional[MemoryConfig] = None) -> Dict[str, Any]:
    cfg = cfg or MemoryConfig()
    store = MemoryStore(cfg.memory_dir, enable_lock=False)
    return store.query("", game_id=game_id).get("game", {})


def _noop_rate_by_action(state: MemoryState) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for key, stats in state.per_action.items():
        if stats.attempts <= 0:
            continue
        rates[key] = stats.no_effect_count / float(stats.attempts)
    return rates


def _attempts_by_action(state: MemoryState) -> Dict[str, int]:
    attempts: Dict[str, int] = {}
    for key, stats in state.per_action.items():
        attempts[key] = stats.attempts
    return attempts


def _avg_changed_cells_by_action(state: MemoryState) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, stats in state.per_action.items():
        out[key] = float(stats.avg_changed_cells)
    return out


def _noop_rate_by_state_action(
    state: MemoryState, state_hash: Optional[str]
) -> Dict[str, float]:
    if not state_hash:
        return {}
    rates: Dict[str, float] = {}
    prefix = f"{state_hash}|"
    for key, stats in state.per_state_action.items():
        if not key.startswith(prefix):
            continue
        if stats.attempts <= 0:
            continue
        action_key = key.split("|", 1)[-1]
        rates[action_key] = stats.no_effect_count / float(stats.attempts)
    return rates


def _action_signature_diversity(state: MemoryState) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, stats in state.per_action.items():
        out[key] = len(stats.event_signature_counts)
    return out


def _coord_noop_rate(state: MemoryState) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for action_id, coord_map in state.coord_heatmaps.items():
        per_action: Dict[str, float] = {}
        for coord_key, stats in coord_map.items():
            if stats.attempts <= 0:
                continue
            per_action[coord_key] = stats.no_effect_count / float(stats.attempts)
        out[action_id] = per_action
    return out


def _coord_effect_scores(state: MemoryState) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for action_id, coord_map in state.coord_heatmaps.items():
        per_action: Dict[str, float] = {}
        for coord_key, stats in coord_map.items():
            if stats.attempts <= 0:
                continue
            per_action[coord_key] = stats.avg_changed_cells
        out[action_id] = per_action
    return out


def _event_signature_baseline(state: MemoryState) -> Dict[str, float]:
    if not state.event_sig_window:
        return {}
    counts: Dict[str, int] = {}
    total = 0
    for entry in state.event_sig_window:
        for key, val in entry.items():
            if key == "_total":
                total += int(val)
                continue
            counts[key] = counts.get(key, 0) + int(val)
    baseline: Dict[str, float] = {}
    for key, count in counts.items():
        if total > 0:
            baseline[f"global.event_sig.{key}.rate"] = count / float(total)
    return baseline


def _object_delta_baseline(state: MemoryState) -> Dict[str, float]:
    if not state.object_delta_window:
        return {}
    counts: Dict[str, int] = {}
    total = 0
    for entry in state.object_delta_window:
        for key, val in entry.items():
            counts[key] = counts.get(key, 0) + int(val)
            total += int(val)
    baseline: Dict[str, float] = {}
    for key, count in counts.items():
        if total > 0:
            baseline[f"global.object_tracking.{key}.rate"] = count / float(total)
    return baseline


def _template_stats_view(state: MemoryState) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, stats in state.template_stats.items():
        out[key] = {
            "times_considered": stats.times_considered,
            "times_triggered": stats.times_triggered,
            "times_scored_positive": stats.times_scored_positive,
            "last_step_triggered": stats.last_step_triggered,
        }
    return out


def _merge_and_save(state: MemoryState, path: str) -> None:
    if os.path.exists(path):
        existing = memory_load(path)
        merged = _merge_states(existing, state)
    else:
        merged = state
    memory_save(merged, path)


def _merge_states(base: MemoryState, new: MemoryState) -> MemoryState:
    merged = MemoryState()
    merged.per_action = _merge_action_stats(base.per_action, new.per_action)
    merged.per_state_action = _merge_state_action_stats(base.per_state_action, new.per_state_action)
    merged.coord_heatmaps = _merge_coord_heatmaps(base.coord_heatmaps, new.coord_heatmaps)
    merged.event_sig_window = (base.event_sig_window + new.event_sig_window)[-max(len(base.event_sig_window), len(new.event_sig_window), 1) :]
    merged.object_delta_window = (base.object_delta_window + new.object_delta_window)[-max(len(base.object_delta_window), len(new.object_delta_window), 1) :]
    merged.template_stats = _merge_template_stats(base.template_stats, new.template_stats)
    merged.recent_actions_by_state = _merge_recent_actions(base.recent_actions_by_state, new.recent_actions_by_state)
    merged.mechanic_by_fingerprint = _merge_mechanic_by_fingerprint(
        base.mechanic_by_fingerprint, new.mechanic_by_fingerprint
    )
    return merged


def _merge_action_stats(
    base: Dict[str, ActionEfficacy],
    new: Dict[str, ActionEfficacy],
) -> Dict[str, ActionEfficacy]:
    out: Dict[str, ActionEfficacy] = {}
    keys = sorted(set(base.keys()) | set(new.keys()))
    for key in keys:
        a = base.get(key)
        b = new.get(key)
        if a is None:
            out[key] = b
            continue
        if b is None:
            out[key] = a
            continue
        attempts = a.attempts + b.attempts
        avg_cells = _merge_running_mean(a.avg_changed_cells, a.attempts, b.avg_changed_cells, b.attempts)
        avg_bbox = _merge_running_mean(a.avg_changed_bbox_area, a.attempts, b.avg_changed_bbox_area, b.attempts)
        out[key] = ActionEfficacy(
            attempts=attempts,
            no_effect_count=a.no_effect_count + b.no_effect_count,
            effect_count=a.effect_count + b.effect_count,
            avg_changed_cells=avg_cells,
            avg_changed_bbox_area=avg_bbox,
            last_step_seen=max(a.last_step_seen, b.last_step_seen),
            event_signature_counts=_merge_counts(a.event_signature_counts, b.event_signature_counts),
            source_counts=_merge_counts(a.source_counts, b.source_counts),
        )
    return out


def _merge_state_action_stats(
    base: Dict[str, StateActionEfficacy],
    new: Dict[str, StateActionEfficacy],
) -> Dict[str, StateActionEfficacy]:
    out: Dict[str, StateActionEfficacy] = {}
    keys = sorted(set(base.keys()) | set(new.keys()))
    for key in keys:
        a = base.get(key)
        b = new.get(key)
        if a is None:
            out[key] = b
            continue
        if b is None:
            out[key] = a
            continue
        out[key] = StateActionEfficacy(
            attempts=a.attempts + b.attempts,
            no_effect_count=a.no_effect_count + b.no_effect_count,
            last_effect_step=_max_optional(a.last_effect_step, b.last_effect_step),
            last_step_seen=max(a.last_step_seen, b.last_step_seen),
        )
    return out


def _merge_coord_heatmaps(
    base: Dict[str, Dict[str, CoordStat]],
    new: Dict[str, Dict[str, CoordStat]],
) -> Dict[str, Dict[str, CoordStat]]:
    out: Dict[str, Dict[str, CoordStat]] = {}
    action_ids = sorted(set(base.keys()) | set(new.keys()))
    for action_id in action_ids:
        merged_coords: Dict[str, CoordStat] = {}
        coords = set()
        coords.update(base.get(action_id, {}).keys())
        coords.update(new.get(action_id, {}).keys())
        for key in sorted(coords):
            a = base.get(action_id, {}).get(key)
            b = new.get(action_id, {}).get(key)
            if a is None:
                merged_coords[key] = b
                continue
            if b is None:
                merged_coords[key] = a
                continue
            attempts = a.attempts + b.attempts
            avg_cells = _merge_running_mean(a.avg_changed_cells, a.attempts, b.avg_changed_cells, b.attempts)
            merged_coords[key] = CoordStat(
                attempts=attempts,
                no_effect_count=a.no_effect_count + b.no_effect_count,
                avg_changed_cells=avg_cells,
                last_step_seen=max(a.last_step_seen, b.last_step_seen),
            )
        out[action_id] = merged_coords
    return out


def _merge_template_stats(
    base: Dict[str, TemplateStats],
    new: Dict[str, TemplateStats],
) -> Dict[str, TemplateStats]:
    out: Dict[str, TemplateStats] = {}
    keys = sorted(set(base.keys()) | set(new.keys()))
    for key in keys:
        a = base.get(key)
        b = new.get(key)
        if a is None:
            out[key] = b
            continue
        if b is None:
            out[key] = a
            continue
        merged_support = dict(a.supporting_events)
        for ev, count in b.supporting_events.items():
            merged_support[ev] = merged_support.get(ev, 0) + count
        out[key] = TemplateStats(
            times_considered=a.times_considered + b.times_considered,
            times_triggered=a.times_triggered + b.times_triggered,
            times_scored_positive=a.times_scored_positive + b.times_scored_positive,
            last_step_triggered=_max_optional(a.last_step_triggered, b.last_step_triggered),
            supporting_events=merged_support,
        )
    return out


def _merge_recent_actions(
    base: Dict[str, List[str]],
    new: Dict[str, List[str]],
    limit: int = 25,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    keys = sorted(set(base.keys()) | set(new.keys()))
    for key in keys:
        combined = list(base.get(key, [])) + list(new.get(key, []))
        if len(combined) > limit:
            combined = combined[-limit:]
        out[key] = combined
    return out


def _merge_mechanic_by_fingerprint(
    base: Dict[str, Dict[str, Dict[str, float]]],
    new: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    keys = sorted(set(base.keys()) | set(new.keys()))
    for key in keys:
        out[key] = {}
        fams = set()
        fams.update(base.get(key, {}).keys())
        fams.update(new.get(key, {}).keys())
        for fam in fams:
            a = base.get(key, {}).get(fam, {})
            b = new.get(key, {}).get(fam, {})
            count_a = float(a.get("count", 0.0))
            count_b = float(b.get("count", 0.0))
            if count_a + count_b == 0:
                continue
            avg_a = float(a.get("avg_prior", 0.0))
            avg_b = float(b.get("avg_prior", 0.0))
            avg = (avg_a * count_a + avg_b * count_b) / float(count_a + count_b)
            out[key][fam] = {"count": count_a + count_b, "avg_prior": avg}
    return out


def _merge_evidence_into_view(view: Dict[str, Any], evidence: Dict[str, Any]) -> None:
    priors = evidence.get("priors", {})
    action_priors = priors.get("action", {})
    for action_key, stats in action_priors.items():
        attempts = stats.get("attempts_total", 0)
        if attempts <= 0:
            continue
        noop = stats.get("no_effect_total", 0) / float(attempts)
        view.setdefault("noop_rate_by_action", {}).setdefault(action_key, noop)
        view.setdefault("attempts_by_action", {}).setdefault(action_key, int(attempts))
        view.setdefault("avg_changed_cells_by_action", {}).setdefault(action_key, stats.get("avg_changed_cells", 0.0))
    template_priors = priors.get("templates", {})
    if template_priors:
        tmpl_view = view.setdefault("template_stats", {})
        for template_id, stats in template_priors.items():
            tmpl_view.setdefault(
                template_id,
                {
                    "times_considered": int(stats.get("times_considered", 0)),
                    "times_triggered": 0,
                    "times_scored_positive": int(stats.get("times_accepted", 0)),
                    "last_step_triggered": None,
                    "supporting_events": {},
                },
            )


def _validate_run_summary(run_summary: Dict[str, Any]) -> None:
    if run_summary.get("schema_version") != "RUN_SUMMARY_V1":
        raise ValueError("run_summary_v1 schema_version mismatch")
    required = [
        "task_signature_v1",
        "game_id",
        "run_id",
        "action_efficacy",
        "hypothesis_outcomes",
        "mechanic_posterior_evolution",
        "failure_labels",
        "progress_metrics",
    ]
    for key in required:
        if key not in run_summary:
            raise ValueError(f"run_summary_v1 missing required field: {key}")


def _merge_counts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = dict(a or {})
    for key, val in (b or {}).items():
        out[key] = out.get(key, 0) + int(val)
    return out


def _state_from_dict(payload: Dict[str, Any]) -> MemoryState:
    state = MemoryState()
    state.version = str(payload.get("version", "1.0"))
    for key, val in (payload.get("per_action") or {}).items():
        state.per_action[key] = ActionEfficacy(**val)
    for key, val in (payload.get("per_state_action") or {}).items():
        state.per_state_action[key] = StateActionEfficacy(**val)
    for action_id, coord_map in (payload.get("coord_heatmaps") or {}).items():
        state.coord_heatmaps[action_id] = {k: CoordStat(**v) for k, v in coord_map.items()}
    state.event_sig_window = list(payload.get("event_sig_window") or [])
    state.object_delta_window = list(payload.get("object_delta_window") or [])
    for key, val in (payload.get("template_stats") or {}).items():
        state.template_stats[key] = TemplateStats(**val)
    state.recent_actions_by_state = dict(payload.get("recent_actions_by_state") or {})
    state.mechanic_by_fingerprint = dict(payload.get("mechanic_by_fingerprint") or {})
    return state


def _action_key(action: Dict[str, Any]) -> Optional[str]:
    if not action:
        return None
    if action.get("type") == "coord":
        return f"{action.get('action_id')}@{action.get('x')},{action.get('y')}"
    return str(action.get("action_id")) if action.get("action_id") else None


def _coord_from_planner_decision(planner_decision: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(planner_decision, dict):
        return None
    action = planner_decision.get("selected_action")
    if isinstance(action, dict) and action.get("type") == "coord":
        return {"x": action.get("x"), "y": action.get("y")}
    key = planner_decision.get("selected_action_key")
    if isinstance(key, str) and "@" in key:
        try:
            _, coord = key.split("@", 1)
            x_str, y_str = coord.split(",", 1)
            return {"x": int(x_str), "y": int(y_str)}
        except Exception:
            return None
    return None


def task_signature_v1(fp_report: Dict[str, Any], action_schema: Dict[str, Any]) -> str:
    debug = fp_report.get("debug", {}) if isinstance(fp_report, dict) else {}
    fingerprint = debug.get("grid_fingerprint", "")
    grid = action_schema.get("primary_grid", {}) if isinstance(action_schema, dict) else {}
    width = grid.get("width", "")
    height = grid.get("height", "")
    actions = action_schema.get("actions", []) if isinstance(action_schema, dict) else []
    kinds = ",".join(sorted({a.get("kind", "") for a in actions if isinstance(a, dict)}))
    return f"v1|{width}x{height}|{kinds}|{fingerprint}"


def state_signature_v1(fp_report: Dict[str, Any]) -> str:
    debug = fp_report.get("debug", {}) if isinstance(fp_report, dict) else {}
    fingerprint = debug.get("grid_fingerprint", "")
    state = fp_report.get("state_summary", {}) if isinstance(fp_report, dict) else {}
    objs = state.get("object_catalog", []) if isinstance(state, dict) else []
    counts: Dict[int, int] = {}
    for obj in objs:
        color = obj.get("color")
        if color is None:
            continue
        counts[int(color)] = counts.get(int(color), 0) + 1
    color_counts = ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    return f"v1|{fingerprint}|{color_counts}"


def _update_per_action(
    state: MemoryState,
    action_key: str,
    changed_cells: int,
    bbox_area: int,
    no_effect: bool,
    step_idx: int,
) -> None:
    stats = state.per_action.get(action_key)
    if stats is None:
        stats = ActionEfficacy()
        state.per_action[action_key] = stats
    stats.attempts += 1
    if no_effect:
        stats.no_effect_count += 1
    else:
        stats.effect_count += 1
    stats.avg_changed_cells = _running_mean(stats.avg_changed_cells, stats.attempts, float(changed_cells))
    stats.avg_changed_bbox_area = _running_mean(stats.avg_changed_bbox_area, stats.attempts, float(bbox_area))
    stats.last_step_seen = step_idx


def _update_per_state_action(
    state: MemoryState,
    state_hash: str,
    action_key: str,
    no_effect: bool,
    step_idx: int,
    cfg: MemoryConfig,
) -> None:
    key = f"{state_hash}|{action_key}"
    stats = state.per_state_action.get(key)
    if stats is None:
        stats = StateActionEfficacy()
        state.per_state_action[key] = stats
    stats.attempts += 1
    if no_effect:
        stats.no_effect_count += 1
    else:
        stats.last_effect_step = step_idx
    stats.last_step_seen = step_idx
    _evict_state_action(state, cfg)


def _update_coord_heatmap(
    state: MemoryState,
    action: Dict[str, Any],
    changed_cells: int,
    no_effect: bool,
    step_idx: int,
    cfg: MemoryConfig,
) -> None:
    action_id = action.get("action_id")
    x = action.get("x")
    y = action.get("y")
    if action_id is None or x is None or y is None:
        return
    coord_key = f"{int(x)},{int(y)}"
    table = state.coord_heatmaps.setdefault(str(action_id), {})
    stats = table.get(coord_key)
    if stats is None:
        stats = CoordStat()
        table[coord_key] = stats
    stats.attempts += 1
    if no_effect:
        stats.no_effect_count += 1
    stats.avg_changed_cells = _running_mean(stats.avg_changed_cells, stats.attempts, float(changed_cells))
    stats.last_step_seen = step_idx
    _evict_coord_table(table, cfg.coord_table_max_per_action)


def _update_feature_windows(
    state: MemoryState,
    event_sigs: Dict[str, int],
    object_deltas: Dict[str, int],
    cfg: MemoryConfig,
) -> None:
    if event_sigs:
        state.event_sig_window.append(event_sigs)
        state.event_sig_window = state.event_sig_window[-cfg.K_long :]
    if object_deltas:
        state.object_delta_window.append(object_deltas)
        state.object_delta_window = state.object_delta_window[-cfg.K_long :]


def _update_action_signatures(
    state: MemoryState,
    action_key: str,
    event_sigs: Dict[str, int],
) -> None:
    if not event_sigs:
        return
    stats = state.per_action.get(action_key)
    if stats is None:
        return
    for sig, count in event_sigs.items():
        if sig == "_total":
            continue
        stats.event_signature_counts[sig] = stats.event_signature_counts.get(sig, 0) + int(count)


def _update_action_source_counts(
    state: MemoryState,
    action_key: str,
    planner_decision: Optional[Dict[str, Any]],
) -> None:
    if not planner_decision:
        return
    selected = planner_decision.get("selected_action_key")
    if selected and selected != action_key:
        return
    source = None
    for cand in planner_decision.get("candidates", []):
        if cand.get("action_key") == action_key:
            source = cand.get("source")
            break
    if not source:
        return
    stats = state.per_action.get(action_key)
    if stats is None:
        return
    stats.source_counts[source] = stats.source_counts.get(source, 0) + 1


def _update_recent_actions(
    state: MemoryState,
    state_hash: str,
    action_key: str,
    cfg: MemoryConfig,
) -> None:
    if not state_hash:
        return
    series = state.recent_actions_by_state.get(state_hash, [])
    series = list(series)
    series.append(action_key)
    if len(series) > cfg.K_short:
        series = series[-cfg.K_short :]
    state.recent_actions_by_state[state_hash] = series


def _update_mechanic_priors(state: MemoryState, inputs: MemoryUpdateInputs) -> None:
    report = inputs.mechanic_classifier
    if not isinstance(report, dict):
        return
    fp_after = inputs.fp_report_after or {}
    debug = fp_after.get("debug") if isinstance(fp_after, dict) else None
    fingerprint = None
    if isinstance(debug, dict):
        fingerprint = debug.get("grid_fingerprint")
    if not fingerprint:
        return
    mech_prior = report.get("mechanic_prior", {})
    families = mech_prior.get("families", []) if isinstance(mech_prior, dict) else []
    if not families:
        return
    bucket = state.mechanic_by_fingerprint.setdefault(str(fingerprint), {})
    for fam in families:
        fam_id = fam.get("family_id")
        if not fam_id:
            continue
        prior_val = float(fam.get("prior", 0.0))
        entry = bucket.get(fam_id)
        if not entry:
            bucket[fam_id] = {"count": 1.0, "avg_prior": prior_val}
        else:
            count = entry.get("count", 0.0) + 1.0
            avg = entry.get("avg_prior", 0.0)
            entry["avg_prior"] = avg + (prior_val - avg) / float(count)
            entry["count"] = count


def _update_template_stats(state: MemoryState, report: Optional[Dict[str, Any]], step_idx: int) -> None:
    if not isinstance(report, dict):
        return
    for hyp in report.get("hypotheses", []):
        hyp_id = hyp.get("hypothesis_id")
        if not hyp_id:
            continue
        stats = state.template_stats.get(hyp_id)
        if stats is None:
            stats = TemplateStats()
            state.template_stats[hyp_id] = stats
        stats.times_considered += 1
        status = _hypothesis_trigger_status(hyp)
        trigger_failed = _hypothesis_trigger_failed(hyp)
        triggered = status in {"pass", "pass_memory"}
        if triggered:
            stats.times_triggered += 1
            stats.last_step_triggered = step_idx
        if float(hyp.get("confidence", 0.0)) > 0:
            stats.times_scored_positive += 1


def _build_debug(
    state: MemoryState,
    inputs: MemoryUpdateInputs,
    action_key: str,
    no_effect: bool,
    top_k: int = 5,
) -> Dict[str, Any]:
    state_action_key = f"{inputs.state_hash_before}|{action_key}"
    state_stats = state.per_state_action.get(state_action_key)
    no_effect_rate = None
    if state_stats and state_stats.attempts:
        no_effect_rate = state_stats.no_effect_count / float(state_stats.attempts)

    top_actions = _top_effective_actions(state.per_action, top_k)
    top_coords = _top_effective_coords(state.coord_heatmaps, top_k)
    return {
        "updated": True,
        "action_key": action_key,
        "no_effect": no_effect,
        "state_action_no_effect_rate": no_effect_rate,
        "top_actions": top_actions,
        "top_coords": top_coords,
    }


def _hypothesis_trigger_failed(hyp: Dict[str, Any]) -> bool:
    evidence = hyp.get("evidence") or []
    for ev in evidence:
        if isinstance(ev, dict) and ev.get("trigger_failed"):
            return True
    return False


def _hypothesis_trigger_status(hyp: Dict[str, Any]) -> Optional[str]:
    evidence = hyp.get("evidence") or []
    for ev in evidence:
        if isinstance(ev, dict) and "trigger_status" in ev:
            return ev.get("trigger_status")
    return None


def _top_effective_actions(per_action: Dict[str, ActionEfficacy], top_k: int) -> List[Dict[str, Any]]:
    items: List[Tuple[float, str, ActionEfficacy]] = []
    for key, stats in per_action.items():
        if stats.attempts <= 0:
            continue
        rate = stats.effect_count / float(stats.attempts)
        items.append((rate, key, stats))
    items.sort(key=lambda item: (-item[0], item[1]))
    out = []
    for rate, key, stats in items[:top_k]:
        out.append({"action_key": key, "effect_rate": rate, "attempts": stats.attempts})
    return out


def _top_effective_coords(coord_heatmaps: Dict[str, Dict[str, CoordStat]], top_k: int) -> List[Dict[str, Any]]:
    entries: List[Tuple[float, str, str, CoordStat]] = []
    for action_id, coord_map in coord_heatmaps.items():
        for coord_key, stats in coord_map.items():
            if stats.attempts <= 0:
                continue
            rate = 1.0 - (stats.no_effect_count / float(stats.attempts))
            entries.append((rate, action_id, coord_key, stats))
    entries.sort(key=lambda item: (-item[0], item[1], item[2]))
    out = []
    for rate, action_id, coord_key, stats in entries[:top_k]:
        out.append(
            {
                "action_id": action_id,
                "coord": coord_key,
                "effect_rate": rate,
                "attempts": stats.attempts,
            }
        )
    return out


def _diff_summary_from_fp(fp_report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(fp_report, dict):
        return None
    return fp_report.get("diff_summary")


def _changed_cells(diff_summary: Optional[Dict[str, Any]], fp_diff: Dict[str, Any]) -> int:
    if isinstance(diff_summary, dict):
        val = diff_summary.get("changed_cells_count")
        if isinstance(val, int):
            return val
    val = fp_diff.get("changed_cells")
    if isinstance(val, int):
        return val
    return 0


def _changed_bbox_area(diff_summary: Optional[Dict[str, Any]], fp_diff: Dict[str, Any]) -> int:
    if isinstance(diff_summary, dict):
        bbox = diff_summary.get("changed_bbox")
        if bbox and len(bbox) == 4:
            y0, x0, y1, x1 = bbox
            try:
                return max(0, y1 - y0 + 1) * max(0, x1 - x0 + 1)
            except Exception:
                pass
    val = fp_diff.get("changed_bbox_area")
    if isinstance(val, int):
        return val
    return 0


def _event_signatures(diff_summary: Optional[Dict[str, Any]], fp_diff: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    sigs: List[Any] = []
    if isinstance(diff_summary, dict):
        sigs = diff_summary.get("event_signatures") or []
    elif fp_diff:
        sigs = fp_diff.get("event_signatures") or []
    for sig in sigs:
        kind = None
        if isinstance(sig, dict):
            kind = sig.get("kind")
        elif isinstance(sig, str):
            kind = sig
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    if counts:
        total = sum(counts.values())
        counts["_total"] = total
    return counts


def _object_deltas(diff_summary: Optional[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(diff_summary, dict):
        return counts
    deltas = diff_summary.get("per_object_deltas") or []
    for delta in deltas:
        event = None
        if isinstance(delta, dict):
            event = delta.get("event")
        if not event:
            continue
        counts[event] = counts.get(event, 0) + 1
    return counts


def _running_mean(prev: float, attempts: int, new_value: float) -> float:
    if attempts <= 1:
        return new_value
    return prev + (new_value - prev) / float(attempts)


def _merge_running_mean(a_mean: float, a_n: int, b_mean: float, b_n: int) -> float:
    total = a_n + b_n
    if total == 0:
        return 0.0
    return (a_mean * a_n + b_mean * b_n) / float(total)


def _max_optional(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _evict_state_action(state: MemoryState, cfg: MemoryConfig) -> None:
    limit = cfg.state_action_table_max
    if limit <= 0 or len(state.per_state_action) <= limit:
        return
    items = list(state.per_state_action.items())
    items.sort(key=lambda item: (item[1].last_step_seen, item[0]))
    to_evict = len(items) - limit
    for idx in range(to_evict):
        key, _ = items[idx]
        state.per_state_action.pop(key, None)


def _evict_coord_table(table: Dict[str, CoordStat], limit: int) -> None:
    if limit <= 0 or len(table) <= limit:
        return
    items = list(table.items())
    items.sort(key=lambda item: (item[1].last_step_seen, item[0]))
    to_evict = len(items) - limit
    for idx in range(to_evict):
        key, _ = items[idx]
        table.pop(key, None)

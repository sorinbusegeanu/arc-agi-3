from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .feature_aggregate import aggregate_features
from .grid_utils import grid_from_ascii
from .summary_export import write_json, write_markdown
from .trace_reader import read_trace
from .trajectory_summarizer_config import TrajectorySummarizerConfig
from .trajectory_summarizer_types import TrajectorySummaryReport


def summarize(
    planner_trace: Optional[str] = None,
    simple_trace: Optional[str] = None,
    full_trace: Optional[str] = None,
    fp_dir: Optional[str] = None,
    action_schema: Optional[Dict[str, Any]] = None,
    proposer: Optional[Dict[str, Any]] = None,
    classifier: Optional[Dict[str, Any]] = None,
    goal: Optional[Dict[str, Any]] = None,
    cfg: Optional[TrajectorySummarizerConfig] = None,
    ctx: Optional[Dict[str, Any]] = None,
    outdir: Optional[str] = None,
) -> TrajectorySummaryReport:
    cfg = cfg or TrajectorySummarizerConfig()
    traces = []
    traces.extend(read_trace(planner_trace))
    traces.extend(read_trace(simple_trace))
    traces.extend(read_trace(full_trace))

    step_records = _normalize_steps(traces)
    if not step_records:
        raise ValueError("no step records found")

    fp_reports = _load_fp_reports(fp_dir, step_records) if fp_dir else {}
    warnings = _hash_warnings(step_records, fp_reports)
    if action_schema is None:
        warnings.append("action_schema_missing")
    run_summary = _run_summary(step_records, fp_reports, ctx, warnings)
    if action_schema is not None:
        run_summary["never_used_actions"] = _never_used_actions(step_records, action_schema)
    action_efficacy = _action_efficacy(step_records, cfg)
    loops = detect_loops(step_records, cfg)
    invariants = extract_invariants(step_records, fp_reports, action_schema, cfg)
    keyframes = _keyframes(step_records, loops, goal, cfg)
    hypothesis_outcomes = _hypothesis_outcomes(step_records, proposer)
    mechanic_outcomes, run_features = _mechanic_outcomes(fp_reports, classifier)

    lessons = {
        "action_efficacy": action_efficacy,
        "loop_analysis": {"loops": loops},
        "discovered_invariants": invariants,
        "state_keyframes": {"keyframes": keyframes},
        "hypothesis_outcomes": hypothesis_outcomes,
        "mechanic_outcomes": mechanic_outcomes,
        "run_features": run_features,
    }

    export_artifacts = {}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        lessons_path = os.path.join(outdir, "lessons.json")
        write_json(lessons_path, {"run_summary": run_summary, "lessons": lessons})
        export_artifacts["lessons.json"] = lessons_path
        if cfg.export_markdown:
            md_path = os.path.join(outdir, "summary.md")
            write_markdown(md_path, {"run_summary": run_summary, "lessons": lessons})
            export_artifacts["summary.md"] = md_path

    return TrajectorySummaryReport(
        run_summary=run_summary,
        lessons=lessons,
        export_artifacts=export_artifacts,
    )


def detect_loops(step_records: List[Dict[str, Any]], cfg: TrajectorySummarizerConfig) -> List[Dict[str, Any]]:
    loops: List[Dict[str, Any]] = []
    state_seq = [rec["state_after"] for rec in step_records]
    transitions = _transition_map(step_records)

    for idx, rec in enumerate(step_records):
        if rec["state_before"] == rec["state_after"]:
            loops.append(
                _loop_entry(
                    loop_id=f"self_loop_{idx}",
                    loop_type="self_loop",
                    states=[rec["state_before"]],
                    actions=[_action_key(rec["action"])],
                    start_step=rec["step_idx"],
                    end_step=rec["step_idx"],
                    transitions=transitions,
                    steps=step_records,
                )
            )

    for cycle_len in range(2, cfg.short_cycle_max_len + 1):
        for idx in range(cycle_len * 2, len(state_seq) + 1):
            seq = state_seq[idx - cycle_len : idx]
            prev = state_seq[idx - cycle_len * 2 : idx - cycle_len]
            if seq == prev:
                start = idx - cycle_len * 2
                end = idx - 1
                loop_steps = step_records[start : end + 1]
                loops.append(
                    _loop_entry(
                        loop_id=f"short_cycle_{cycle_len}_{start}",
                        loop_type="short_cycle",
                        states=[rec["state_after"] for rec in loop_steps],
                        actions=[_action_key(rec["action"]) for rec in loop_steps],
                        start_step=loop_steps[0]["step_idx"],
                        end_step=loop_steps[-1]["step_idx"],
                        transitions=transitions,
                        steps=loop_steps,
                    )
                )
                break

    window = cfg.revisit_window_N
    for idx in range(len(state_seq)):
        window_seq = state_seq[max(0, idx - window + 1) : idx + 1]
        current = state_seq[idx]
        if window_seq.count(current) >= cfg.revisit_threshold_R:
            loops.append(
                _loop_entry(
                    loop_id=f"revisit_flood_{idx}",
                    loop_type="revisit_flood",
                    states=[current],
                    actions=[],
                    start_step=step_records[max(0, idx - window + 1)]["step_idx"],
                    end_step=step_records[idx]["step_idx"],
                    transitions=transitions,
                    steps=step_records[max(0, idx - window + 1) : idx + 1],
                )
            )
            break

    return loops


def extract_invariants(
    step_records: List[Dict[str, Any]],
    fp_reports: Dict[int, Dict[str, Any]],
    action_schema: Optional[Dict[str, Any]],
    cfg: TrajectorySummarizerConfig,
) -> Dict[str, Any]:
    static_cells = []
    if fp_reports:
        grids = []
        for step_idx in sorted(fp_reports.keys()):
            report = fp_reports[step_idx]
            viz = report.get("viz_artifacts", {})
            ascii_grids = viz.get("ascii_grid", {})
            if not ascii_grids:
                continue
            name = sorted(ascii_grids.keys())[0]
            grid = grid_from_ascii(ascii_grids[name])
            grids.append((name, grid))
        if grids:
            name, base = grids[0]
            mask = base == base
            for _, grid in grids[1:]:
                if grid.shape != base.shape:
                    continue
                mask = (grid == base) & mask
            ys, xs = mask.nonzero()
            static_cells = [(int(x), int(y), int(base[y, x])) for y, x in zip(ys, xs)]
            static_cells.sort(key=lambda c: (c[1], c[0]))
            static_cells = static_cells[: cfg.max_static_cells]

    always_present_objects = _always_present_objects(fp_reports)
    never_used_actions = _never_used_actions(step_records, action_schema)
    return {
        "static_cells": static_cells,
        "always_present_objects": always_present_objects,
        "never_used_actions": never_used_actions,
    }


def _normalize_steps(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for entry in traces:
        if not entry:
            continue
        if "state_before" not in entry or "state_after" not in entry:
            raise ValueError("trace entries must include state_before and state_after")
        records.append(entry)
    records.sort(key=lambda r: r.get("step_idx", 0))
    return records


def _load_fp_reports(fp_dir: str, step_records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    reports = {}
    for rec in step_records:
        step_idx = rec.get("step_idx")
        if step_idx is None:
            continue
        path = os.path.join(fp_dir, f"fp_step_{step_idx}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reports[int(step_idx)] = json.load(f)
    return reports


def _run_summary(
    step_records: List[Dict[str, Any]],
    fp_reports: Dict[int, Dict[str, Any]],
    ctx: Optional[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    states = {rec["state_after"] for rec in step_records}
    transitions = _transition_map(step_records)
    terminal = any(rec.get("terminal") is True for rec in step_records)
    reward_values = [rec.get("reward_delta") for rec in step_records if rec.get("reward_delta") is not None]
    reward_total = sum(reward_values) if reward_values else None
    summary = {
        "game_id": ctx.get("game_id") if ctx else None,
        "seed": ctx.get("seed") if ctx else None,
        "run_id": ctx.get("run_id") if ctx else None,
        "steps": len(step_records),
        "unique_states": len(states),
        "unique_transitions": len(transitions),
        "terminal_reached": terminal if step_records else None,
        "reward_total": reward_total,
        "warnings": warnings,
    }
    return summary


def _hash_warnings(step_records: List[Dict[str, Any]], fp_reports: Dict[int, Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    for rec in step_records:
        step_idx = rec.get("step_idx")
        if step_idx is None:
            continue
        report = fp_reports.get(int(step_idx))
        if not report:
            continue
        report_hash = report.get("debug", {}).get("grid_hash")
        if report_hash and report_hash != rec.get("state_after"):
            warnings.append(f"hash_mismatch_step_{step_idx}")
    return warnings


def _action_efficacy(step_records: List[Dict[str, Any]], cfg: TrajectorySummarizerConfig) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, Any]] = {}
    coord_stats: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    for rec in step_records:
        action = rec.get("action") or {}
        action_id = action.get("action_id")
        if not action_id:
            continue
        fp_diff = rec.get("fp_diff") or {}
        changed_cells = fp_diff.get("changed_cells", 0)
        bbox_area = fp_diff.get("changed_bbox_area", 0)
        event_signatures = fp_diff.get("event_signatures", [])
        if action.get("type") == "coord":
            key = (action_id, int(action.get("x", 0)), int(action.get("y", 0)))
            _accumulate_stats(coord_stats, key, changed_cells, bbox_area, event_signatures)
        _accumulate_stats(stats, action_id, changed_cells, bbox_area, event_signatures)

    efficacy = {
        "per_action": _finalize_stats(stats),
        "per_coord": _finalize_stats(coord_stats),
    }
    best_actions, worst_actions = _best_worst_actions(stats)
    efficacy["best_actions"] = best_actions
    efficacy["worst_actions"] = worst_actions
    efficacy["top_effective_coords"] = _top_coords(coord_stats, cfg.topK_coords, effective=True)
    efficacy["top_noop_coords"] = _top_coords(coord_stats, cfg.topK_coords, effective=False)
    return efficacy


def _accumulate_stats(
    stats: Dict[Any, Dict[str, Any]],
    key: Any,
    changed_cells: int,
    bbox_area: int,
    event_signatures: List[str],
) -> None:
    entry = stats.setdefault(
        key,
        {"attempts": 0, "noops": 0, "changed_cells": 0, "changed_bbox_area": 0, "events": {}},
    )
    entry["attempts"] += 1
    if changed_cells == 0:
        entry["noops"] += 1
    entry["changed_cells"] += changed_cells
    entry["changed_bbox_area"] += bbox_area
    for sig in event_signatures:
        entry["events"][sig] = entry["events"].get(sig, 0) + 1


def _finalize_stats(stats: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, entry in stats.items():
        attempts = entry["attempts"]
        output[str(key)] = {
            "attempts": attempts,
            "no_effect_rate": entry["noops"] / attempts if attempts else 0.0,
            "avg_changed_cells": entry["changed_cells"] / attempts if attempts else 0.0,
            "avg_changed_bbox_area": entry["changed_bbox_area"] / attempts if attempts else 0.0,
            "dominant_event_signatures": sorted(entry["events"].items(), key=lambda kv: (-kv[1], kv[0]))[:3],
        }
    return output


def _best_worst_actions(stats: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    scored = []
    for action_id, entry in stats.items():
        attempts = entry["attempts"]
        no_effect = entry["noops"] / attempts if attempts else 1.0
        scored.append((1.0 - no_effect, action_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return [], []
    if len(scored) == 1:
        return [scored[0][1]], []
    best = [item[1] for item in scored[:3]]
    worst = [item[1] for item in scored[-3:]]
    return best, worst


def _top_coords(
    coord_stats: Dict[Tuple[str, int, int], Dict[str, Any]],
    top_k: int,
    *,
    effective: bool,
) -> List[Tuple[str, int, int, float]]:
    scored = []
    for key, entry in coord_stats.items():
        attempts = entry["attempts"]
        no_effect = entry["noops"] / attempts if attempts else 1.0
        score = (1.0 - no_effect) if effective else no_effect
        scored.append((score, key))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(k[0], k[1], k[2], s) for s, k in scored[:top_k]]


def _transition_map(step_records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], int]:
    transitions: Dict[Tuple[str, str, str], int] = {}
    for rec in step_records:
        action_key = _action_key(rec.get("action"))
        key = (rec["state_before"], action_key, rec["state_after"])
        transitions[key] = transitions.get(key, 0) + 1
    return transitions


def _action_key(action: Dict[str, Any]) -> str:
    if not action:
        return "unknown"
    if action.get("type") == "coord":
        return f"{action.get('action_id')}@{action.get('x')},{action.get('y')}"
    return str(action.get("action_id"))


def _loop_entry(
    loop_id: str,
    loop_type: str,
    states: List[str],
    actions: List[str],
    start_step: int,
    end_step: int,
    transitions: Dict[Tuple[str, str, str], int],
    steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    likely_cause = _loop_cause(steps, transitions)
    escape_actions = _escape_actions(states, transitions)
    return {
        "loop_id": loop_id,
        "type": loop_type,
        "states": states,
        "actions": actions,
        "start_step": start_step,
        "end_step": end_step,
        "likely_cause": likely_cause,
        "escape_actions": escape_actions,
    }


def _loop_cause(
    steps: List[Dict[str, Any]],
    transitions: Dict[Tuple[str, str, str], int],
) -> str:
    if not steps:
        return "unknown"
    noops = sum(1 for rec in steps if (rec.get("fp_diff", {}).get("changed_cells") or 0) == 0)
    if noops / len(steps) >= 0.8:
        return "repeated_noop"
    if _inconsistent_transitions(transitions):
        return "stochastic_env"
    return "unknown"


def _inconsistent_transitions(transitions: Dict[Tuple[str, str, str], int]) -> bool:
    seen: Dict[Tuple[str, str], set[str]] = {}
    for (state, action, next_state) in transitions.keys():
        seen.setdefault((state, action), set()).add(next_state)
        if len(seen[(state, action)]) > 1:
            return True
    return False


def _escape_actions(states: List[str], transitions: Dict[Tuple[str, str, str], int]) -> List[str]:
    escapes = set()
    for (state, action, next_state) in transitions.keys():
        if state in states and next_state not in states:
            escapes.add(action)
    return sorted(escapes)


def _always_present_objects(fp_reports: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not fp_reports:
        return []
    common: Optional[Dict[str, int]] = None
    for report in fp_reports.values():
        state = report.get("state_summary") or {}
        comps = state.get("object_catalog", [])
        signatures = {}
        for comp in comps:
            color = comp.get("color")
            bbox = comp.get("bbox") or [0, 0, 0, 0]
            if color is None:
                continue
            key = f"{color}:{bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}"
            signatures[key] = signatures.get(key, 0) + 1
        if common is None:
            common = signatures
        else:
            common = {k: v for k, v in common.items() if k in signatures}
    return [{"signature": k} for k in (common or {}).keys()]


def _never_used_actions(step_records: List[Dict[str, Any]], action_schema: Optional[Dict[str, Any]]) -> List[str]:
    if action_schema is None:
        return []
    used = {_action_key(rec.get("action")) for rec in step_records}
    actions = action_schema.get("actions", []) if isinstance(action_schema, dict) else []
    all_actions = [a.get("action_id") for a in actions if a.get("action_id")]
    return sorted([action_id for action_id in all_actions if action_id not in used])


def _keyframes(
    step_records: List[Dict[str, Any]],
    loops: List[Dict[str, Any]],
    goal: Optional[Dict[str, Any]],
    cfg: TrajectorySummarizerConfig,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    if not step_records:
        return frames
    first = step_records[0]
    frames.append(
        {"state_hash": first["state_after"], "step_idx": first["step_idx"], "why_selected": "first_state"}
    )
    max_change = max(step_records, key=lambda r: r.get("fp_diff", {}).get("changed_cells", 0))
    if max_change["step_idx"] != first["step_idx"]:
        frames.append(
            {"state_hash": max_change["state_after"], "step_idx": max_change["step_idx"], "why_selected": "max_change"}
        )
    if loops:
        loop = loops[0]
        frames.append({"state_hash": loop["states"][0], "step_idx": loop["start_step"], "why_selected": "first_loop"})
    terminal_steps = [r for r in step_records if r.get("terminal") is True]
    if terminal_steps:
        last_terminal = terminal_steps[-1]
        frames.append({"state_hash": last_terminal["state_after"], "step_idx": last_terminal["step_idx"], "why_selected": "pre_terminal"})
    if goal:
        best = max(step_records, key=lambda r: (r.get("reward_delta") or 0))
        frames.append({"state_hash": best["state_after"], "step_idx": best["step_idx"], "why_selected": "best_progress"})
    frames.append(
        {"state_hash": step_records[-1]["state_after"], "step_idx": step_records[-1]["step_idx"], "why_selected": "final_step"}
    )
    unique = []
    seen = set()
    for frame in frames:
        key = (frame["state_hash"], frame["why_selected"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame)
        if len(unique) >= cfg.keyframes_max:
            break
    return unique


def _hypothesis_outcomes(step_records: List[Dict[str, Any]], proposer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not proposer:
        return {}
    outcomes: Dict[str, Any] = {}
    for hyp in proposer.get("hypotheses", []):
        hyp_id = hyp.get("hypothesis_id")
        supported = None
        refuted = None
        evidence = []
        for test in hyp.get("tests", []):
            seq = test.get("action_sequence") or []
            if not seq:
                continue
            for rec in step_records:
                if _action_matches_seq(rec.get("action"), seq[0]):
                    sigs = (rec.get("fp_diff") or {}).get("event_signatures", [])
                    expected = test.get("expected_signature", [])
                    if any(sig in sigs for sig in expected):
                        supported = True
                        evidence.append({"test_id": test.get("test_id"), "result": "supported"})
                    else:
                        refuted = True
                        evidence.append({"test_id": test.get("test_id"), "result": "refuted"})
        outcomes[hyp_id] = {
            "supported": supported,
            "refuted": refuted,
            "evidence": evidence,
            "confidence_update": 0.1 if supported else (-0.1 if refuted else 0.0),
        }
    return outcomes


def _action_matches_seq(action: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    if not action or not spec:
        return False
    if action.get("type") != spec.get("type"):
        return False
    if action.get("action_id") != spec.get("action_id"):
        return False
    if action.get("type") == "coord":
        return action.get("x") == spec.get("x") and action.get("y") == spec.get("y")
    return True


def _mechanic_outcomes(
    fp_reports: Dict[int, Dict[str, Any]],
    classifier: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    outcomes: Dict[str, Any] = {}
    run_features: Dict[str, Any] = {}
    if classifier:
        families = classifier.get("mechanic_prior", {}).get("families", [])
        outcomes["prior_start"] = families[:3]
    if fp_reports:
        run_features = aggregate_features(list(fp_reports.values()), None, None)
        outcomes["prior_end"] = classifier.get("mechanic_prior", {}).get("families", []) if classifier else []
        outcomes["shift_summary"] = "unchanged"
    return outcomes, run_features

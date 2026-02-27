from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .action_schema import ActionSchema, parse_action_schema_data
from .executable_hypothesis_engine_config import ExecutableHypothesisEngineConfig
from .executable_hypothesis_engine_types import (
    ExecutableHypothesisV1,
    HypothesisEngineReport,
    TransitionEventV1,
)
from .grid_utils import bbox_area
from .primitive_program_v1 import default_hypotheses


def seed_hypotheses(
    rule_proposer_report: Optional[Dict[str, Any]],
    cfg: Optional[ExecutableHypothesisEngineConfig] = None,
) -> List[ExecutableHypothesisV1]:
    cfg = cfg or ExecutableHypothesisEngineConfig()
    hypotheses = default_hypotheses()
    if not rule_proposer_report:
        return hypotheses
    seeds = rule_proposer_report.get("hypotheses", []) if isinstance(rule_proposer_report, dict) else []
    seed_ids = {h.get("hypothesis_id") for h in seeds if isinstance(h, dict)}
    for hyp in hypotheses:
        if hyp.hypothesis_id in seed_ids:
            hyp.confidence = max(hyp.confidence, cfg.seed_boost)
    return hypotheses


def update(
    hypotheses: List[ExecutableHypothesisV1],
    transition_events: List[TransitionEventV1],
    cfg: Optional[ExecutableHypothesisEngineConfig] = None,
) -> List[ExecutableHypothesisV1]:
    cfg = cfg or ExecutableHypothesisEngineConfig()
    if not transition_events:
        return hypotheses
    window = transition_events[-cfg.window_N :]
    for hyp in hypotheses:
        scores: List[float] = []
        falsified = False
        for event in window:
            likelihood, hard_violation = _likelihood(hyp, event, cfg)
            if hard_violation and cfg.hard_falsify and hyp.hypothesis_id != "unknown.mechanic":
                falsified = True
            scores.append(likelihood)
        if scores:
            avg = sum(scores) / float(len(scores))
        else:
            avg = 0.0
        hyp.fit_stats = {
            "transitions_scored": len(scores),
            "avg_likelihood": avg,
            "falsified": falsified,
        }
        if falsified:
            hyp.confidence = 0.0
        else:
            hyp.confidence = max(0.0, min(1.0, avg))
    return hypotheses


def predict_all(
    hypotheses: List[ExecutableHypothesisV1],
    state_features: Dict[str, Any],
    candidate_actions: Iterable[Any],
) -> Dict[str, List[Dict[str, Any]]]:
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    for action in candidate_actions:
        action_key = _normalize_action_key(action)
        preds = []
        for hyp in hypotheses:
            pred = _predict(hyp, state_features, action_key)
            if pred:
                preds.append(pred)
        predictions[action_key] = preds
    return predictions


def _predict(hyp: ExecutableHypothesisV1, state_features: Dict[str, Any], action_key: str) -> Dict[str, Any]:
    if hyp.hypothesis_id == "unknown.mechanic":
        return {}
    program = hyp.program_v1
    gates = program.get("gates") if isinstance(program, dict) else program.gates
    for gate in gates or []:
        if gate.get("requires_coord") and "@" not in action_key:
            return {}
        if gate.get("requires_simple") and "@" in action_key:
            return {}
    effects = program.get("effects") if isinstance(program, dict) else program.effects
    if not effects:
        return {}
    primary = effects[0]
    return {
        "signature": (primary.get("event_signatures") or [None])[0],
        "delta_bin": (primary.get("delta_bins") or [None])[0],
        "noop": primary.get("noop"),
    }


def compile_transition_events(
    fp_reports: List[Dict[str, Any]],
    action_schema: ActionSchema | Dict[str, Any],
    simple_report: Optional[Dict[str, Any]] = None,
    full_report: Optional[Dict[str, Any]] = None,
    transition_records: Optional[List[Dict[str, Any]]] = None,
) -> List[TransitionEventV1]:
    schema = parse_action_schema_data(action_schema) if isinstance(action_schema, dict) else action_schema
    events: List[TransitionEventV1] = []

    if transition_records:
        for rec in transition_records:
            event = _transition_event_from_record(rec)
            if event:
                events.append(event)
        return events

    for rep in fp_reports:
        debug = rep.get("debug", {}) if isinstance(rep, dict) else {}
        diff = rep.get("diff_summary") or {}
        if not diff:
            continue
        action_key = _action_key_from_report(rep, simple_report, full_report)
        events.append(
            TransitionEventV1(
                state_hash_before=str(debug.get("grid_hash", "")),
                state_hash_after=str(debug.get("grid_hash", "")),
                action_key=action_key or "UNKNOWN",
                event_signature_histogram=_event_signature_hist(diff),
                delta_metrics=_delta_metrics(diff),
                meta_delta=_meta_delta(rep, schema),
            )
        )
    return events


def run_engine(
    fp_reports: List[Dict[str, Any]],
    action_schema: ActionSchema | Dict[str, Any],
    ctx: Dict[str, Any],
    simple_report: Optional[Dict[str, Any]] = None,
    full_report: Optional[Dict[str, Any]] = None,
    rule_proposer_report: Optional[Dict[str, Any]] = None,
    transition_records: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional[ExecutableHypothesisEngineConfig] = None,
) -> HypothesisEngineReport:
    cfg = cfg or ExecutableHypothesisEngineConfig()
    hypotheses = seed_hypotheses(rule_proposer_report, cfg)
    events = compile_transition_events(fp_reports, action_schema, simple_report, full_report, transition_records)
    hypotheses = update(hypotheses, events, cfg)

    ranked = sorted(
        hypotheses,
        key=lambda h: (-h.confidence, h.hypothesis_id != "unknown.mechanic", h.hypothesis_id),
    )
    if cfg.topK_hypotheses and len(ranked) > cfg.topK_hypotheses:
        ranked = ranked[: cfg.topK_hypotheses]
    run_summary = {
        "window_used": min(len(events), cfg.window_N),
        "transitions_available": len(events),
        "weights": {
            "w_sig": cfg.w_sig,
            "w_noop": cfg.w_noop,
            "w_delta": cfg.w_delta,
            "w_meta": cfg.w_meta,
        },
        "hard_falsify": cfg.hard_falsify,
        "ctx": ctx,
    }
    return HypothesisEngineReport(hypotheses=ranked, run_summary=run_summary)


def _transition_event_from_record(rec: Dict[str, Any]) -> Optional[TransitionEventV1]:
    try:
        return TransitionEventV1(
            state_hash_before=str(rec.get("state_hash_before", "")),
            state_hash_after=str(rec.get("state_hash_after", "")),
            action_key=str(rec.get("action_key", "UNKNOWN")),
            event_signature_histogram=rec.get("event_signature_histogram", {}) or {},
            delta_metrics=rec.get("delta_metrics", {}) or {},
            meta_delta=rec.get("meta_delta", {}) or {},
        )
    except Exception:
        return None


def transition_events_from_trace(trace_path: str, max_events: int = 50) -> List[TransitionEventV1]:
    events: List[TransitionEventV1] = []
    if not trace_path:
        return events
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                event = _transition_event_from_trace_record(record)
                if event:
                    events.append(event)
    except Exception:
        return events
    if max_events > 0 and len(events) > max_events:
        events = events[-max_events:]
    return events


def _transition_event_from_trace_record(rec: Dict[str, Any]) -> Optional[TransitionEventV1]:
    try:
        fp_diff = rec.get("fp_diff") or {}
        return TransitionEventV1(
            state_hash_before=str(rec.get("state_before", "")),
            state_hash_after=str(rec.get("state_after", "")),
            action_key=_action_key_from_trace(rec),
            event_signature_histogram=_event_signature_hist_from_list(fp_diff.get("event_signatures", [])),
            delta_metrics={
                "changed_cells": int(fp_diff.get("changed_cells", 0)),
                "changed_bbox_area": int(fp_diff.get("changed_bbox_area", 0)),
                "palette_added": 0,
                "palette_removed": 0,
            },
            meta_delta={"reward": rec.get("reward"), "terminal": rec.get("terminal")},
        )
    except Exception:
        return None


def _action_key_from_trace(rec: Dict[str, Any]) -> str:
    action = rec.get("action") or {}
    if isinstance(action, dict):
        action_id = action.get("action_id") or action.get("id")
        if action.get("type") == "coord" and action.get("x") is not None and action.get("y") is not None:
            return f"{action_id}@{action.get('x')},{action.get('y')}"
        if action_id:
            return str(action_id)
    return "UNKNOWN"


def _event_signature_hist_from_list(sigs: List[Any]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for sig in sigs or []:
        if isinstance(sig, str):
            hist[sig] = hist.get(sig, 0) + 1
        elif isinstance(sig, dict):
            kind = sig.get("kind")
            if kind:
                hist[str(kind)] = hist.get(str(kind), 0) + 1
    return hist


def _event_signature_hist(diff: Dict[str, Any]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for entry in diff.get("event_signatures", []) or []:
        kind = entry.get("kind") if isinstance(entry, dict) else None
        if not kind:
            continue
        hist[kind] = hist.get(kind, 0) + 1
    return hist


def _delta_metrics(diff: Dict[str, Any]) -> Dict[str, Any]:
    changed_cells = int(diff.get("changed_cells_count", 0))
    bbox = diff.get("changed_bbox")
    metrics = {
        "changed_cells": changed_cells,
        "changed_bbox_area": bbox_area(bbox) if bbox else 0,
        "palette_added": 0,
        "palette_removed": 0,
    }
    added, removed = _palette_delta(diff)
    metrics["palette_added"] = added
    metrics["palette_removed"] = removed
    return metrics


def _palette_delta(diff: Dict[str, Any]) -> Tuple[int, int]:
    changed = diff.get("changed_colors") or {}
    added = set()
    removed = set()
    for key in changed.keys():
        if not isinstance(key, str) or "->" not in key:
            continue
        left, right = key.split("->", 1)
        try:
            removed.add(int(left))
            added.add(int(right))
        except Exception:
            continue
    return len(added), len(removed)


def _meta_delta(rep: Dict[str, Any], schema: ActionSchema) -> Dict[str, Any]:
    return {
        "available_actions_before": len(schema.actions) if schema else None,
        "available_actions_after": len(schema.actions) if schema else None,
        "reward": rep.get("reward"),
        "terminal": rep.get("terminal"),
    }


def _action_key_from_report(
    rep: Dict[str, Any],
    simple_report: Optional[Dict[str, Any]],
    full_report: Optional[Dict[str, Any]],
) -> str:
    action = rep.get("action") or rep.get("action_taken") or {}
    if isinstance(action, dict):
        action_id = action.get("action_id") or action.get("id")
        if action.get("type") == "coord" and action.get("x") is not None and action.get("y") is not None:
            return f"{action_id}@{action.get('x')},{action.get('y')}"
        if action_id:
            return str(action_id)
    action_key = rep.get("action_key")
    if action_key:
        return str(action_key)
    return "UNKNOWN"


def _normalize_action_key(action: Any) -> str:
    if isinstance(action, dict):
        action_id = action.get("action_id") or action.get("id")
        if action.get("type") == "coord":
            return f"{action_id}@{action.get('x')},{action.get('y')}"
        return str(action_id)
    return str(action)


def _likelihood(
    hyp: ExecutableHypothesisV1,
    event: TransitionEventV1,
    cfg: ExecutableHypothesisEngineConfig,
) -> Tuple[float, bool]:
    if hyp.hypothesis_id == "unknown.mechanic":
        return 0.01, False

    action_key = event.action_key
    hard_violation = False
    for gate in hyp.program_v1.gates:
        if gate.get("requires_coord") and "@" not in action_key and action_key != "UNKNOWN":
            hard_violation = True
        if gate.get("requires_simple") and "@" in action_key:
            hard_violation = True

    obs_sig = _dominant_signature(event.event_signature_histogram)
    pred_sig = _predicted_signatures(hyp)
    sig_score = 0.0
    if pred_sig:
        sig_score = 1.0 if obs_sig in pred_sig else 0.0
    else:
        sig_score = 0.5

    changed_cells = int(event.delta_metrics.get("changed_cells", 0))
    pred_bins = _predicted_delta_bins(hyp)
    obs_bin = _bin_changed_cells(changed_cells)
    delta_score = 1.0 if obs_bin in pred_bins else 0.0

    pred_noop = _predicted_noop(hyp)
    obs_noop = changed_cells == 0
    noop_score = 1.0 if pred_noop is None or pred_noop == obs_noop else 0.0

    meta_score, meta_weight = _meta_agreement(event.meta_delta)
    total_weight = cfg.w_sig + cfg.w_noop + cfg.w_delta + (cfg.w_meta if meta_weight else 0.0)
    if total_weight <= 0:
        return 0.0, hard_violation
    score = (
        cfg.w_sig * sig_score
        + cfg.w_noop * noop_score
        + cfg.w_delta * delta_score
        + (cfg.w_meta * meta_score if meta_weight else 0.0)
    ) / total_weight
    return score, hard_violation


def _meta_agreement(meta: Dict[str, Any]) -> Tuple[float, bool]:
    if not meta:
        return 0.0, False
    if meta.get("terminal") is None and meta.get("reward") is None:
        return 0.0, False
    return 0.5, True


def _dominant_signature(hist: Dict[str, int]) -> Optional[str]:
    if not hist:
        return None
    return max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _predicted_signatures(hyp: ExecutableHypothesisV1) -> List[str]:
    sigs: List[str] = []
    for eff in hyp.program_v1.effects:
        for sig in eff.get("event_signatures", []) or []:
            if sig not in sigs:
                sigs.append(sig)
    return sigs


def _predicted_delta_bins(hyp: ExecutableHypothesisV1) -> List[str]:
    bins: List[str] = []
    for eff in hyp.program_v1.effects:
        for b in eff.get("delta_bins", []) or []:
            if b not in bins:
                bins.append(b)
    return bins or ["tiny", "small", "medium", "large"]


def _predicted_noop(hyp: ExecutableHypothesisV1) -> Optional[bool]:
    for eff in hyp.program_v1.effects:
        if "noop" in eff:
            return bool(eff.get("noop"))
    return None


def _bin_changed_cells(changed_cells: int) -> str:
    if changed_cells <= 0:
        return "tiny"
    if changed_cells <= 4:
        return "small"
    if changed_cells <= 20:
        return "medium"
    return "large"


def asdict_report(report: HypothesisEngineReport) -> Dict[str, Any]:
    return {
        "hypotheses": [asdict(hyp) for hyp in report.hypotheses],
        "run_summary": report.run_summary,
    }

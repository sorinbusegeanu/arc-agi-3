from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .action_schema import ActionSchema, ActionSpec, parse_action_schema_data
from .feature_aggregate import aggregate_features
from .hypothesis_catalog import HYPOTHESES
from .rule_proposer_config import RuleProposerConfig
from .rule_proposer_types import Hypothesis, RuleProposerReport, TestActionSpec, TestSpec
from .score_utils import eval_predicate, feature_value, transform_value


def propose(
    initial_fp_reports: List[Dict[str, Any]],
    simple_report: Optional[Dict[str, Any]] = None,
    full_report: Optional[Dict[str, Any]] = None,
    action_schema: Optional[ActionSchema | Dict[str, Any]] = None,
    cfg: Optional[RuleProposerConfig] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> RuleProposerReport:
    if not initial_fp_reports:
        raise ValueError("initial_fp_reports is required and must be non-empty")
    if action_schema is None:
        raise ValueError("action_schema is required")

    cfg = cfg or RuleProposerConfig()
    if isinstance(action_schema, dict):
        action_schema = parse_action_schema_data(action_schema)

    window = min(len(initial_fp_reports), cfg.initial_T)
    reports_window = initial_fp_reports[:window]

    features = aggregate_features(reports_window, simple_report, full_report)
    availability = _build_availability(reports_window, simple_report, full_report, action_schema)

    diagnostics: Dict[str, Any] = {
        "templates_total": len(HYPOTHESES),
        "requires_blocked": 0,
        "trigger_failed": 0,
        "trigger_insufficient_window": 0,
        "score_nonpositive": 0,
        "tests_total": 0,
        "tests_generated": 0,
        "tests_kept": 0,
        "hypotheses_final": 0,
        "early_exit": None,
    }
    for template in HYPOTHESES:
        gate = _template_gate_reason(template, reports_window, features, availability, cfg)
        if gate and gate.startswith("requires_"):
            diagnostics["requires_blocked"] += 1
        elif gate == "trigger_not_met":
            diagnostics["trigger_failed"] += 1
        elif gate == "trigger_insufficient_window":
            diagnostics["trigger_insufficient_window"] += 1
    if ctx and ctx.get("debug"):
        diagnostics["trigger_evals"] = _trigger_evals(reports_window, features, availability, cfg)
        diagnostics["feature_audit"] = _feature_audit(reports_window)
        diagnostics["score_audit"] = _score_audit(features, availability, top_n=10)
        diagnostics["trigger_thresholds"] = _trigger_thresholds(reports_window)

    hypotheses = score_hypotheses(reports_window, features, availability, cfg)
    ranked = _rank_hypotheses(hypotheses)

    if not availability["has_simple_report"] and not availability["has_full_report"]:
        diagnostics["early_exit"] = "explorer_reports_missing"
        return RuleProposerReport(
            hypotheses=[],
            run_summary=_build_run_summary(window, availability, cfg, ctx, diagnostics=diagnostics),
        )

    if _all_nonpositive(ranked):
        if window < 2:
            diagnostics["early_exit"] = "insufficient_temporal_window"
            diagnostics["fallback_used"] = False
            diagnostics["fallback_block_reason"] = "insufficient_temporal_window"
            return RuleProposerReport(
                hypotheses=[],
                run_summary=_build_run_summary(window, availability, cfg, ctx, diagnostics=diagnostics),
            )
        diagnostics["early_exit"] = "all_nonpositive"
        diagnostics["score_nonpositive"] = len(ranked)
        allow_fallback, block_reason = _fallback_gate(
            availability,
            full_report,
            diagnostics,
            features,
            reports_window,
            cfg,
        )
        diagnostics["fallback_used"] = False
        diagnostics["fallback_block_reason"] = block_reason
        if allow_fallback:
            fallback = _fallback_hypotheses(reports_window, features, availability, action_schema, cfg, full_report)
            diagnostics["fallback_used"] = bool(fallback)
            diagnostics["hypotheses_final"] = len(fallback) if fallback else 0
            if fallback:
                return RuleProposerReport(
                    hypotheses=fallback,
                    run_summary=_build_run_summary(window, availability, cfg, ctx, diagnostics=diagnostics),
                )
        diagnostics["early_exit"] = "all_nonpositive_no_fallback"
        return RuleProposerReport(
            hypotheses=[],
            run_summary=_build_run_summary(window, availability, cfg, ctx, diagnostics=diagnostics),
        )

    ranked = _apply_max_hypotheses(ranked, cfg.max_hypotheses)
    diagnostics["score_nonpositive"] = len([h for h in ranked if h.confidence <= 0])

    tests_total = 0
    final_hypotheses: List[Hypothesis] = []
    active_ids = [h.hypothesis_id for h in ranked if h.hypothesis_id != "unknown.mechanic" and h.confidence > 0]

    for hyp in ranked:
        if hyp.hypothesis_id != "unknown.mechanic" and hyp.confidence <= 0:
            continue
        if hyp.hypothesis_id == "unknown.mechanic" and active_ids:
            pass
        tests = build_tests(hyp, features, action_schema, cfg, reports_window, full_report)
        diagnostics["tests_generated"] += len(tests)
        tests = _ensure_discriminating_tests(tests, hyp.hypothesis_id, active_ids)
        diagnostics["tests_kept"] += len(tests)
        if tests:
            tests_total += len(tests)
        hyp.tests = tests
        final_hypotheses.append(hyp)
        if tests_total >= cfg.max_total_tests:
            break

    final_hypotheses = _apply_total_test_cap(final_hypotheses, cfg.max_total_tests)
    diagnostics["tests_total"] = tests_total
    diagnostics["hypotheses_final"] = len(final_hypotheses)
    if diagnostics["early_exit"] is None and not final_hypotheses:
        diagnostics["early_exit"] = "no_hypotheses_emitted"

    return RuleProposerReport(
        hypotheses=final_hypotheses,
        run_summary=_build_run_summary(window, availability, cfg, ctx, diagnostics=diagnostics),
    )


def score_hypotheses(
    reports_window: List[Dict[str, Any]],
    features: Dict[str, Any],
    availability: Dict[str, bool],
    cfg: RuleProposerConfig,
) -> List[Hypothesis]:
    hypotheses: List[Hypothesis] = []
    for template in HYPOTHESES:
        hypothesis = _build_hypothesis_from_template(template, reports_window, features, availability, cfg)
        hypotheses.append(hypothesis)
    return hypotheses


def build_tests(
    hypothesis: Hypothesis,
    features: Dict[str, Any],
    action_schema: ActionSchema,
    cfg: RuleProposerConfig,
    fp_reports: List[Dict[str, Any]],
    full_report: Optional[Dict[str, Any]],
) -> List[TestSpec]:
    template = _find_template(hypothesis.hypothesis_id)
    if template is None:
        return []
    if hypothesis.confidence <= 0 and hypothesis.hypothesis_id != "unknown.mechanic":
        if not any(ev.get("trigger_failed") for ev in (hypothesis.evidence or []) if isinstance(ev, dict)):
            return []

    simple_actions = _sorted_actions(action_schema.actions, "simple")
    coord_actions = _sorted_actions(action_schema.actions, "coord")
    coord_sources = _build_coord_sources(fp_reports, full_report, action_schema)

    tests: List[TestSpec] = []
    for tmpl in template.tests_builder.get("test_templates", []):
        generated = _expand_test_template(
            tmpl,
            hypothesis.hypothesis_id,
            simple_actions,
            coord_actions,
            coord_sources,
            cfg,
        )
        tests.extend(generated)

    tests = sorted(tests, key=lambda t: t.test_id)
    if any(ev.get("trigger_failed") for ev in (hypothesis.evidence or []) if isinstance(ev, dict)):
        return tests[: cfg.trigger_failed_tests_max]
    return tests[: cfg.tests_per_hypothesis]


def _build_hypothesis_from_template(
    template: Any,
    reports_window: List[Dict[str, Any]],
    features: Dict[str, Any],
    availability: Dict[str, bool],
    cfg: RuleProposerConfig,
) -> Hypothesis:
    requires = template.requires
    if requires.get("needs_coord_actions") and not availability.get("has_coord_actions", False):
        return _empty_hypothesis(template, confidence=0.0)
    if requires.get("needs_simple_actions") and not availability.get("has_simple_actions", False):
        return _empty_hypothesis(template, confidence=0.0)
    if requires.get("needs_object_tracking") and not availability.get("has_object_tracking", False):
        return _empty_hypothesis(template, confidence=0.0)
    if requires.get("needs_reward_signal") and not availability.get("has_reward_signal", False):
        return _empty_hypothesis(template, confidence=0.0)

    trigger_status = _trigger_status(template, reports_window, features, availability, cfg)
    score = _score_template(template, features)
    trigger_failed = False
    if trigger_status == "fail":
        score = max(0.0, score - cfg.trigger_fail_penalty)
        trigger_failed = True
    if trigger_status == "insufficient_window":
        trigger_failed = True

    score = max(0.0, min(1.0, score)) if template.scoring_function.get("clamp", True) else score

    evidence = _build_evidence(template, features)
    if trigger_failed:
        evidence = list(evidence) + [{"trigger_failed": True, "status": trigger_status}]
    predictions = _build_predictions(template, features)
    expected_observations = _build_expected_observations(predictions)
    return Hypothesis(
        hypothesis_id=template.hypothesis_id,
        name=template.name,
        description=_describe_hypothesis(template, features),
        confidence=score,
        evidence=evidence,
        predictions=predictions,
        tests=[],
        expected_observations=expected_observations,
        dependencies=_dependencies_from_requires(template.requires),
    )


def _fallback_hypotheses(
    reports_window: List[Dict[str, Any]],
    features: Dict[str, Any],
    availability: Dict[str, bool],
    action_schema: ActionSchema,
    cfg: RuleProposerConfig,
    full_report: Optional[Dict[str, Any]],
) -> List[Hypothesis]:
    fallback: List[Hypothesis] = []
    for template in HYPOTHESES:
        if template.hypothesis_id == "unknown.mechanic":
            continue
        if _requires_gate_reason(template, availability) is not None:
            continue
        hyp = _build_hypothesis_from_template(template, reports_window, features, availability, cfg)
        hyp.confidence = cfg.fallback_confidence
        tests = build_tests(hyp, features, action_schema, cfg, reports_window, full_report)
        hyp.tests = tests[: cfg.tests_per_hypothesis]
        fallback.append(hyp)
        if len(fallback) >= cfg.fallback_max_hypotheses:
            break
    return fallback


def _fallback_gate(
    availability: Dict[str, bool],
    full_report: Optional[Dict[str, Any]],
    diagnostics: Dict[str, Any],
    features: Dict[str, Any],
    reports_window: List[Dict[str, Any]],
    cfg: RuleProposerConfig,
) -> Tuple[bool, str]:
    if availability.get("has_simple_report") is not True:
        return False, "missing_simple_report"
    if availability.get("has_full_report") is not True:
        return False, "missing_full_report"
    if not _has_triggered_template(reports_window, features, availability, cfg):
        return False, "no_triggered_template"
    if _coord_trials(full_report) <= 0:
        return False, "coord_trials_missing"
    return True, ""


def _has_triggered_template(
    reports_window: List[Dict[str, Any]],
    features: Dict[str, Any],
    availability: Dict[str, bool],
    cfg: RuleProposerConfig,
) -> bool:
    for template in HYPOTHESES:
        if template.hypothesis_id == "unknown.mechanic":
            continue
        if _requires_gate_reason(template, availability) is not None:
            continue
        if _trigger_status(template, reports_window, features, availability, cfg) == "pass":
            return True
    return False


def _coord_trials(full_report: Optional[Dict[str, Any]]) -> int:
    if not isinstance(full_report, dict):
        return 0
    diagnostics = full_report.get("diagnostics")
    if isinstance(diagnostics, dict):
        val = diagnostics.get("coord_tried", 0)
        if isinstance(val, int):
            return val
    return 0


def _template_gate_reason(
    template: Any,
    reports_window: List[Dict[str, Any]],
    features: Dict[str, Any],
    availability: Dict[str, bool],
    cfg: RuleProposerConfig,
) -> Optional[str]:
    requires_reason = _requires_gate_reason(template, availability)
    if requires_reason is not None:
        return requires_reason
    trigger_status = _trigger_status(template, reports_window, features, availability, cfg)
    if trigger_status == "insufficient_window":
        return "trigger_insufficient_window"
    if trigger_status == "fail":
        return "trigger_not_met"
    return None


def _requires_gate_reason(template: Any, availability: Dict[str, bool]) -> Optional[str]:
    requires = template.requires
    if requires.get("needs_coord_actions") and not availability.get("has_coord_actions", False):
        return "requires_coord_actions"
    if requires.get("needs_simple_actions") and not availability.get("has_simple_actions", False):
        return "requires_simple_actions"
    if requires.get("needs_object_tracking") and not availability.get("has_object_tracking", False):
        return "requires_object_tracking"
    if requires.get("needs_reward_signal") and not availability.get("has_reward_signal", False):
        return "requires_reward_signal"
    return None


def _trigger_evals(
    reports_window: List[Dict[str, Any]],
    features: Dict[str, Any],
    availability: Dict[str, bool],
    cfg: RuleProposerConfig,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    window_ok = _diff_window_count(reports_window) >= cfg.trigger_window_min
    for template in HYPOTHESES:
        if template.hypothesis_id == "unknown.mechanic":
            continue
        gate = _template_gate_reason(template, reports_window, features, availability, cfg)
        if gate != "trigger_not_met":
            continue
        preds = []
        for predicate in template.trigger_features:
            key = predicate.get("feature_key", "")
            op = predicate.get("op", "")
            target = predicate.get("value")
            mode = _predicate_mode(key, predicate)
            if mode == "rate":
                actual = feature_value(features, key)
                passed = eval_predicate(actual, op, target)
                preds.append(
                    {
                        "feature_key": key,
                        "mode": "rate",
                        "op": op,
                        "target": target,
                        "actual": actual,
                        "passed": passed,
                    }
                )
            else:
                values = _per_report_feature_values(reports_window, key)
                hits = sum(1 for v in values if eval_predicate(v, op, target))
                passed = window_ok and hits >= cfg.trigger_n_of_k
                preds.append(
                    {
                        "feature_key": key,
                        "mode": "hits",
                        "op": op,
                        "target": target,
                        "actual_hits": hits,
                        "target_hits": cfg.trigger_n_of_k,
                        "window_used": len(values),
                        "passed": passed,
                    }
                )
        details.append(
            {
                "hypothesis_id": template.hypothesis_id,
                "predicates": preds,
            }
        )
        if len(details) >= limit:
            break
    return details


def _feature_audit(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    trigger_keys = _trigger_feature_keys()
    per_report = [_event_signature_counts(rep) for rep in reports]
    total_reports_with_diff = sum(1 for rep in reports if rep.get("diff_summary"))
    audit: Dict[str, Any] = {
        "window_used": len(reports),
        "total_reports_with_diff": total_reports_with_diff,
        "features": {},
    }
    for key in sorted(trigger_keys):
        if key.startswith("global.event_sig."):
            sig = key.split(".")[2]
            numerator = sum(r.get(sig, 0) for r in per_report)
            denominator = sum(r.get("_total", 0) for r in per_report) or 1
            values = [
                (r.get(sig, 0) / r.get("_total", 1)) if r.get("_total", 0) > 0 else 0.0
                for r in per_report
            ]
            audit["features"][key] = {
                "numerator": numerator,
                "denominator": denominator,
                "per_report_values": values,
            }
        elif key.startswith("global.object_tracking."):
            kind = key.split(".")[2]
            numerator = sum(r.get(kind, 0) for r in per_report)
            denominator = total_reports_with_diff or 1
            values = [r.get(kind, 0) for r in per_report]
            audit["features"][key] = {
                "numerator": numerator,
                "denominator": denominator,
                "per_report_values": values,
            }
    return audit


def _score_audit(
    features: Dict[str, Any],
    availability: Dict[str, bool],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    audits: List[Dict[str, Any]] = []
    for template in HYPOTHESES:
        if template.hypothesis_id == "unknown.mechanic":
            continue
        if _requires_gate_reason(template, availability) is not None:
            continue
        terms = []
        score = 0.0
        for term in template.scoring_function.get("terms", []):
            key = term.get("feature_ref", "")
            val = feature_value(features, key)
            contrib = term.get("weight", 0.0) * transform_value(val, term.get("transform", "identity"), term.get("params"))
            score += contrib
            terms.append({"type": "term", "feature_key": key, "value": val, "weight": term.get("weight", 0.0), "contribution": contrib})
        for term in template.scoring_function.get("penalties", []):
            key = term.get("feature_ref", "")
            val = feature_value(features, key)
            contrib = term.get("weight", 0.0) * transform_value(val, term.get("transform", "identity"), term.get("params"))
            score -= contrib
            terms.append({"type": "penalty", "feature_key": key, "value": val, "weight": term.get("weight", 0.0), "contribution": -contrib})
        audits.append({"hypothesis_id": template.hypothesis_id, "raw_score": score, "components": terms})
    audits.sort(key=lambda item: (-item.get("raw_score", 0.0), item.get("hypothesis_id", "")))
    return audits[:top_n]


def _trigger_thresholds(
    reports: List[Dict[str, Any]],
    window_k: int = 5,
) -> List[Dict[str, Any]]:
    keys = _trigger_feature_keys()
    per_report = [_event_signature_counts(rep) for rep in reports]
    total_reports_with_diff = sum(1 for rep in reports if rep.get("diff_summary"))
    per_report_values: Dict[str, List[float]] = {}
    for key in keys:
        if key.startswith("global.event_sig."):
            sig = key.split(".")[2]
            per_report_values[key] = [
                (r.get(sig, 0) / r.get("_total", 1)) if r.get("_total", 0) > 0 else 0.0
                for r in per_report
            ]
        elif key.startswith("global.object_tracking."):
            kind = key.split(".")[2]
            per_report_values[key] = [float(r.get(kind, 0)) for r in per_report]

    summaries: List[Dict[str, Any]] = []
    for template in HYPOTHESES:
        if template.hypothesis_id == "unknown.mechanic":
            continue
        for predicate in template.trigger_features:
            key = predicate.get("feature_key", "")
            op = predicate.get("op", "")
            target = predicate.get("value")
            values = per_report_values.get(key, [])
            tail = values[-window_k:] if window_k > 0 else values
            fails = sum(1 for v in tail if not eval_predicate(v, op, target))
            summaries.append(
                {
                    "feature_key": key,
                    "op": op,
                    "target": target,
                    "window": len(tail),
                    "failures": fails,
                    "min": min(tail) if tail else None,
                    "max": max(tail) if tail else None,
                }
            )
    return summaries


def _trigger_feature_keys() -> List[str]:
    keys = set()
    for template in HYPOTHESES:
        for predicate in template.trigger_features:
            key = predicate.get("feature_key")
            if key:
                keys.add(key)
    return sorted(keys)


def _event_signature_counts(rep: Dict[str, Any]) -> Dict[str, int]:
    diff = rep.get("diff_summary") or {}
    counts: Dict[str, int] = {"_total": 0}
    for sig in diff.get("event_signatures", []):
        kind = sig.get("kind") if isinstance(sig, dict) else None
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        counts["_total"] += 1
    return counts


def _per_report_feature_values(reports: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for rep in reports:
        diff = rep.get("diff_summary") or {}
        if key.startswith("global.event_sig."):
            sig = key.split(".")[2]
            total = 0
            count = 0
            for entry in diff.get("event_signatures", []):
                kind = entry.get("kind") if isinstance(entry, dict) else None
                if kind:
                    total += 1
                    if kind == sig:
                        count += 1
            values.append((count / total) if total > 0 else 0.0)
        elif key.startswith("global.object_tracking."):
            kind = key.split(".")[2]
            count = 0
            for entry in diff.get("event_signatures", []):
                ev = entry.get("kind") if isinstance(entry, dict) else None
                if ev == kind:
                    count += 1
            values.append(float(count))
        else:
            values.append(0.0)
    return values


def _trigger_status(
    template: Any,
    reports_window: List[Dict[str, Any]],
    features: Dict[str, Any],
    availability: Dict[str, bool],
    cfg: RuleProposerConfig,
) -> str:
    if not template.trigger_features:
        return "pass"
    if _diff_window_count(reports_window) < cfg.trigger_window_min:
        return "insufficient_window"

    for predicate in template.trigger_features:
        key = predicate.get("feature_key", "")
        op = predicate.get("op", "")
        target = predicate.get("value")
        mode = _predicate_mode(key, predicate)
        if mode == "rate":
            actual = feature_value(features, key)
            if not eval_predicate(actual, op, target):
                return "fail"
        else:
            values = _per_report_feature_values(reports_window, key)
            hits = sum(1 for v in values if eval_predicate(v, op, target))
            if hits < cfg.trigger_n_of_k:
                return "fail"
    return "pass"


def _predicate_mode(feature_key: str, predicate: Dict[str, Any]) -> str:
    mode = predicate.get("mode")
    if mode in ("rate", "hits"):
        return mode
    if feature_key.endswith(".rate"):
        return "rate"
    return "hits"


def _diff_window_count(reports_window: List[Dict[str, Any]]) -> int:
    return sum(1 for rep in reports_window if rep.get("diff_summary"))


def _score_template(template: Any, features: Dict[str, Any]) -> float:
    score = 0.0
    for term in template.scoring_function.get("terms", []):
        val = feature_value(features, term.get("feature_ref", ""))
        score += term.get("weight", 0.0) * transform_value(val, term.get("transform", "identity"), term.get("params"))
    for term in template.scoring_function.get("penalties", []):
        val = feature_value(features, term.get("feature_ref", ""))
        score -= term.get("weight", 0.0) * transform_value(val, term.get("transform", "identity"), term.get("params"))
    return score


def _build_predictions(template: Any, features: Dict[str, Any]) -> List[Dict[str, Any]]:
    preds = []
    for pred in template.predictions_builder.get("prediction_templates", []):
        entry = dict(pred)
        entry["feature_values"] = {key: feature_value(features, key) for key in pred.get("feature_refs", [])}
        preds.append(entry)
    return preds


def _build_expected_observations(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations = []
    for pred in predictions:
        observations.append(
            {
                "prediction_id": pred.get("prediction_id"),
                "expected_signatures": pred.get("expected_signatures", []),
                "expected_metrics": pred.get("expected_metrics", {}),
            }
        )
    return observations


def _build_evidence(template: Any, features: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for trigger in template.trigger_features:
        key = trigger.get("feature_key", "")
        if key:
            evidence.append({"type": "feature_value", "feature": key, "value": feature_value(features, key)})
    for term in template.scoring_function.get("terms", []):
        key = term.get("feature_ref", "")
        if key:
            evidence.append({"type": "feature_value", "feature": key, "value": feature_value(features, key)})
    evidence.extend(_evidence_from_features(features))
    return _dedupe_evidence(evidence)


def _evidence_from_features(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for key, value in features.items():
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        action_id = None
        feature = key
        if key.startswith("per_action[") and "]." in key:
            action_id = key.split("]", 1)[0].split("[", 1)[-1]
            feature = key.split("].", 1)[-1]
        if "event_sig." in feature:
            parts = feature.split(".")
            try:
                idx = parts.index("event_sig")
                sig = parts[idx + 1]
                evidence.append(
                    {"type": "event_signature", "action": action_id or "global", "sig": sig, "rate": value}
                )
            except Exception:
                pass
        if feature == "noop.rate":
            evidence.append({"type": "no_effect_rate", "action": action_id or "global", "rate": value})
        if feature == "hotspot.non_noop_rate_top1":
            evidence.append({"type": "hotspot_rate", "action": action_id or "global", "rate": value})
        if feature == "negative_zone.noop_rate_top1":
            evidence.append({"type": "negative_zone_rate", "action": action_id or "global", "rate": value})
    return evidence


def _build_coord_sources(
    fp_reports: List[Dict[str, Any]],
    full_report: Optional[Dict[str, Any]],
    action_schema: ActionSchema,
) -> Dict[str, Any]:
    width = action_schema.primary_grid.width
    height = action_schema.primary_grid.height
    sources: Dict[str, Any] = {
        "hotspot": [],
        "hotspot_by_action": {},
        "object_centroid": [],
        "diff_bbox_center": [],
        "grid_edges_midpoints": [],
        "none": [],
    }

    sources["grid_edges_midpoints"] = _grid_edge_midpoints(width, height)
    sources["diff_bbox_center"] = _diff_bbox_center(fp_reports, width, height)
    sources["object_centroid"] = _object_centroids(fp_reports, width, height)
    sources["hotspot_by_action"] = _hotspot_coords_by_action(full_report)
    sources["hotspot"] = _flatten_hotspots(sources["hotspot_by_action"])
    return sources


def _expand_test_template(
    tmpl: Dict[str, Any],
    hypothesis_id: str,
    simple_actions: List[ActionSpec],
    coord_actions: List[ActionSpec],
    coord_sources: Dict[str, Any],
    cfg: RuleProposerConfig,
) -> List[TestSpec]:
    action_family = tmpl.get("action_family")
    sequence_len = int(tmpl.get("sequence_len", 1))
    sequence_len = min(sequence_len, cfg.max_action_sequence_len)
    selector = tmpl.get("selector", "none")
    coord_policy = tmpl.get("coord_policy", {})
    needs_coord = bool(coord_policy.get("needs_coord", False))
    top_k = int(coord_policy.get("top_k_coords", 1))
    fallback_to_noncoord = bool(coord_policy.get("fallback_to_noncoord", True))

    actions = _resolve_action_family(action_family, simple_actions, coord_actions)
    if not actions:
        return []

    tests: List[TestSpec] = []
    max_generated = int(tmpl.get("max_generated_tests", 1))
    for action in actions:
        coords = _select_coords(selector, coord_sources, action.action_id)
        if needs_coord and action.kind != "coord":
            continue
        if needs_coord and not coords:
            if fallback_to_noncoord and simple_actions:
                action = simple_actions[0]
                coords = []
            else:
                continue
        if action.kind == "coord":
            coords = coords[:top_k] if coords else []
            if not coords:
                continue
            for coord in coords:
                tests.append(
                    _build_test_spec(tmpl, hypothesis_id, action, coord, sequence_len)
                )
                if len(tests) >= max_generated:
                    break
        else:
            tests.append(_build_test_spec(tmpl, hypothesis_id, action, None, sequence_len))

        if len(tests) >= max_generated:
            break
    return tests


def _build_test_spec(
    tmpl: Dict[str, Any],
    hypothesis_id: str,
    action: ActionSpec,
    coord: Optional[Tuple[int, int]],
    sequence_len: int,
) -> TestSpec:
    action_seq: List[TestActionSpec] = []
    for _ in range(sequence_len):
        if action.kind == "coord" and coord is not None:
            action_seq.append(
                TestActionSpec(type="coord", action_id=action.action_id, x=coord[0], y=coord[1])
            )
        else:
            action_seq.append(TestActionSpec(type="simple", action_id=action.action_id))

    return TestSpec(
        test_id=tmpl.get("test_id", f"{hypothesis_id}.test"),
        purpose=tmpl.get("purpose", f"Probe {hypothesis_id}"),
        action_sequence=action_seq,
        target_state="any",
        expected_signature=list(tmpl.get("expected_signatures", [])),
        pass_criteria=dict(tmpl.get("pass_criteria", {})),
        fail_criteria=dict(tmpl.get("fail_criteria", {})),
        supports=list(tmpl.get("supports", [])),
        refutes=list(tmpl.get("refutes", [])),
    )


def _resolve_action_family(
    action_family: Any,
    simple_actions: List[ActionSpec],
    coord_actions: List[ActionSpec],
) -> List[ActionSpec]:
    if action_family == "simple":
        return simple_actions[:1]
    if action_family == "coord":
        return coord_actions[:1]
    if isinstance(action_family, list):
        actions = simple_actions + coord_actions
        chosen = [a for a in actions if a.action_id in action_family]
        return chosen[:1]
    return []


def _build_availability(
    fp_reports: List[Dict[str, Any]],
    simple_report: Optional[Dict[str, Any]],
    full_report: Optional[Dict[str, Any]],
    action_schema: ActionSchema,
) -> Dict[str, bool]:
    has_object_tracking = False
    has_reward_signal = False
    for rep in fp_reports:
        diff = rep.get("diff_summary") or {}
        deltas = diff.get("per_object_deltas", [])
        if deltas:
            has_object_tracking = True
        if _reward_present(rep):
            has_reward_signal = True
    return {
        "has_object_tracking": has_object_tracking,
        "has_reward_signal": has_reward_signal,
        "has_simple_actions": any(a.kind == "simple" for a in action_schema.actions),
        "has_coord_actions": any(a.kind == "coord" for a in action_schema.actions),
        "has_simple_report": simple_report is not None,
        "has_full_report": full_report is not None,
    }


def _reward_present(rep: Dict[str, Any]) -> bool:
    for key in ("reward_delta", "reward_change", "reward", "prev_reward"):
        if key in rep:
            return True
    if isinstance(rep.get("meta"), dict):
        meta = rep["meta"]
        if any(k in meta for k in ("reward_delta", "reward_change", "reward", "prev_reward", "terminal", "done", "state")):
            return True
    if isinstance(rep.get("debug"), dict):
        debug = rep["debug"]
        if any(k in debug for k in ("reward_delta", "reward_change", "terminal", "done")):
            return True
    return False




def _rank_hypotheses(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    unknown = [h for h in hypotheses if h.hypothesis_id == "unknown.mechanic"]
    others = [h for h in hypotheses if h.hypothesis_id != "unknown.mechanic"]
    others.sort(key=lambda h: (-h.confidence, h.hypothesis_id))
    if any(h.confidence > 0 for h in others):
        if unknown:
            return others + unknown
        return others
    return others + unknown


def _apply_max_hypotheses(hypotheses: List[Hypothesis], max_hypotheses: int) -> List[Hypothesis]:
    if max_hypotheses <= 0 or len(hypotheses) <= max_hypotheses:
        return hypotheses
    unknown = [h for h in hypotheses if h.hypothesis_id == "unknown.mechanic"]
    others = [h for h in hypotheses if h.hypothesis_id != "unknown.mechanic"]
    limit = max_hypotheses - (1 if unknown else 0)
    trimmed = others[: max(0, limit)]
    if unknown:
        trimmed += unknown[:1]
    return trimmed


def _all_nonpositive(hypotheses: List[Hypothesis]) -> bool:
    return not any(h.confidence > 0 for h in hypotheses if h.hypothesis_id != "unknown.mechanic")


def _build_unknown_only(
    features: Dict[str, Any],
    action_schema: ActionSchema,
    cfg: RuleProposerConfig,
) -> Hypothesis:
    unknown_template = _find_template("unknown.mechanic")
    if unknown_template is None:
        raise ValueError("unknown.mechanic template missing")
    hyp = _build_hypothesis_from_template(
        unknown_template,
        [],
        features,
        _build_availability([], None, None, action_schema),
        cfg,
    )
    hyp.confidence = 0.5
    hyp.tests = build_tests(hyp, features, action_schema, cfg, [], None)
    return hyp


def _empty_hypothesis(template: Any, confidence: float) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=template.hypothesis_id,
        name=template.name,
        description=_describe_hypothesis(template, {}),
        confidence=confidence,
        evidence=[],
        predictions=[],
        tests=[],
        expected_observations=[],
        dependencies=_dependencies_from_requires(template.requires),
    )


def _describe_hypothesis(template: Any, features: Dict[str, Any]) -> str:
    _ = features
    return template.name


def _dependencies_from_requires(requires: Dict[str, bool]) -> List[str]:
    deps = []
    if requires.get("needs_object_tracking"):
        deps.append("requires_object_tracking")
    if requires.get("needs_coord_actions"):
        deps.append("requires_coord_actions")
    if requires.get("needs_simple_actions"):
        deps.append("requires_simple_actions")
    if requires.get("needs_reward_signal"):
        deps.append("requires_reward_signal")
    return deps


def _sorted_actions(actions: List[ActionSpec], kind: str) -> List[ActionSpec]:
    return sorted([a for a in actions if a.kind == kind], key=lambda a: a.action_id)


def _find_template(hypothesis_id: str) -> Optional[Any]:
    for template in HYPOTHESES:
        if template.hypothesis_id == hypothesis_id:
            return template
    return None


def _grid_edge_midpoints(width: int, height: int) -> List[Tuple[int, int]]:
    mid_x = width // 2
    mid_y = height // 2
    coords = [(mid_x, 0), (mid_x, height - 1), (0, mid_y), (width - 1, mid_y)]
    return _unique_coords(coords)


def _diff_bbox_center(fp_reports: List[Dict[str, Any]], width: int, height: int) -> List[Tuple[int, int]]:
    for rep in fp_reports:
        diff = rep.get("diff_summary")
        if not diff:
            continue
        bbox = diff.get("changed_bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            y0, x0, y1, x1 = bbox
            cx = int((x0 + x1) // 2)
            cy = int((y0 + y1) // 2)
            return [(max(0, min(cx, width - 1)), max(0, min(cy, height - 1)))]
    return [(width // 2, height // 2)]


def _object_centroids(fp_reports: List[Dict[str, Any]], width: int, height: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for rep in fp_reports:
        state = rep.get("state_summary") or {}
        for comp in state.get("object_catalog", []):
            centroid = comp.get("centroid")
            if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
                continue
            y, x = centroid
            try:
                cx = int(round(float(x)))
                cy = int(round(float(y)))
            except Exception:
                continue
            if "bbox" in comp and isinstance(comp["bbox"], (list, tuple)) and len(comp["bbox"]) == 4:
                y0, x0, _, _ = comp["bbox"]
                if (cx, cy) in coords:
                    cx, cy = int(x0), int(y0)
            coords.append((max(0, min(cx, width - 1)), max(0, min(cy, height - 1))))
    return _unique_coords(sorted(coords, key=lambda c: (c[1], c[0])))


def _hotspot_coords_by_action(full_report: Optional[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    if not full_report:
        return {}
    model = full_report.get("coord_action_effect_model", {})
    hotspots: Dict[str, List[Tuple[int, int]]] = {}
    for action_id, stats in model.items():
        coords = []
        for entry in stats.get("hotspots", []):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                x, y = int(entry[0]), int(entry[1])
                coords.append((x, y))
        hotspots[action_id] = _unique_coords(coords)
    return hotspots


def _flatten_hotspots(hotspots_by_action: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for action_id in sorted(hotspots_by_action.keys()):
        coords.extend(hotspots_by_action[action_id])
    return _unique_coords(coords)


def _select_coords(
    selector: str,
    coord_sources: Dict[str, Any],
    action_id: str,
) -> List[Tuple[int, int]]:
    if selector == "hotspot":
        by_action = coord_sources.get("hotspot_by_action", {})
        if action_id in by_action:
            return by_action[action_id]
        return coord_sources.get("hotspot", [])
    return coord_sources.get(selector, [])


def _unique_coords(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out = []
    for x, y in coords:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        out.append((x, y))
    return out


def _ensure_discriminating_tests(
    tests: List[TestSpec],
    hypothesis_id: str,
    other_ids: List[str],
) -> List[TestSpec]:
    if not other_ids or not tests:
        return tests
    for test in tests:
        if any(refute in other_ids for refute in test.refutes):
            return tests
    tests[0].refutes = sorted(set(tests[0].refutes + other_ids))
    return tests


def _apply_total_test_cap(hypotheses: List[Hypothesis], max_total: int) -> List[Hypothesis]:
    total = 0
    final: List[Hypothesis] = []
    for hyp in hypotheses:
        if total >= max_total:
            break
        if not hyp.tests:
            final.append(hyp)
            continue
        remaining = max_total - total
        hyp.tests = hyp.tests[:remaining]
        total += len(hyp.tests)
        final.append(hyp)
    return final


def _dedupe_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in evidence:
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _build_run_summary(
    window: int,
    availability: Dict[str, bool],
    cfg: RuleProposerConfig,
    ctx: Optional[Dict[str, Any]],
    diagnostics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    summary = {
        "initial_T": cfg.initial_T,
        "window_used": window,
        "has_simple_report": availability["has_simple_report"],
        "has_full_report": availability["has_full_report"],
        "evidence_quality": "low" if not availability["has_simple_report"] and not availability["has_full_report"] else "high",
        "weights": {
            "w_event_match": cfg.w_event_match,
            "w_motion_consistency": cfg.w_motion_consistency,
            "w_hotspot_support": cfg.w_hotspot_support,
            "w_noop_penalty": cfg.w_noop_penalty,
        },
    }
    if diagnostics and ctx and ctx.get("debug"):
        summary["diagnostics"] = diagnostics
    if ctx:
        summary["ctx"] = ctx
    return summary

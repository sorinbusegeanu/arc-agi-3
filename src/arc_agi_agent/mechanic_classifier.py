from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .action_schema import ActionSchema, parse_action_schema_data
from .feature_aggregate import aggregate_features
from .mechanic_classifier_config import MechanicClassifierConfig
from .mechanic_classifier_types import (
    FamilyPrior,
    FamilyTags,
    MechanicClassifierReport,
    MechanicPrior,
)
from .mechanic_family_catalog import MECHANIC_FAMILIES
from .score_utils import feature_value, transform_value, triggers_met
from .memory import memory_view


def classify(
    fp_reports: List[Dict[str, Any]],
    simple_report: Optional[Dict[str, Any]] = None,
    full_report: Optional[Dict[str, Any]] = None,
    action_schema: Optional[ActionSchema | Dict[str, Any]] = None,
    memory: Optional[Any] = None,
    memory_evidence: Optional[Dict[str, Any]] = None,
    cfg: Optional[MechanicClassifierConfig] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> MechanicClassifierReport:
    if not fp_reports:
        raise ValueError("fp_reports is required and must be non-empty")

    cfg = cfg or MechanicClassifierConfig()
    schema = None
    if action_schema is not None:
        schema = parse_action_schema_data(action_schema) if isinstance(action_schema, dict) else action_schema

    window = min(len(fp_reports), cfg.initial_T)
    reports_window = fp_reports[:window]
    if len(fp_reports) < 2:
        availability = _availability(fp_reports, simple_report, full_report, schema)
        run_summary = _build_run_summary(window, availability, cfg, ctx)
        if ctx and ctx.get("debug"):
            run_summary["diagnostics"] = {
                "availability": {
                    "has_simple_report": availability.get("has_simple_report"),
                    "has_full_report": availability.get("has_full_report"),
                    "has_object_tracking": False,
                    "has_reward_signal": False,
                    "reports_with_diff": availability.get("reports_with_diff", 0),
                    "reports_with_object_deltas": availability.get("reports_with_object_deltas", 0),
                    "object_delta_total": availability.get("object_delta_total", 0),
                },
                "raw_scores_top3": [],
                "priors_top3": [],
                "families_emitted": 0,
                "fallback_reason": "insufficient_temporal_window",
            }
        return MechanicClassifierReport(
            mechanic_prior=MechanicPrior(families=[], normalization={"sum_raw": 0.0, "sum_prior": 0.0}),
            family_tags=FamilyTags(
                required_capabilities={"needs_coord_actions": False, "needs_object_tracking": False},
                constraints={},
            ),
            run_summary=run_summary,
        )

    features = aggregate_features(reports_window, simple_report, full_report)
    availability = _availability(reports_window, simple_report, full_report, schema)
    mem_view = memory_view(memory, evidence=memory_evidence) if memory is not None else {}
    features = _blend_memory_features(features, availability, mem_view, cfg)

    raw_scores, evidence = _score_families(features, availability, cfg)
    if mem_view:
        raw_scores = _apply_memory_priors(raw_scores, mem_view, reports_window)
    priors = _normalize_scores(raw_scores, cfg)
    priors_sorted = sorted(priors.items(), key=lambda kv: (-kv[1], kv[0]))

    families = _build_family_priors(priors_sorted, evidence, cfg)
    family_tags = _build_family_tags(priors_sorted, availability)
    diagnostics = {
        "availability": availability,
        "raw_scores_top3": sorted(raw_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3],
        "priors_top3": priors_sorted[:3],
        "families_emitted": len(families),
        "fallback_reason": None,
        "memory_evidence_used": {
            "event_signature_baseline": len(mem_view.get("event_signature_baseline", {})),
            "object_delta_baseline": len(mem_view.get("object_delta_baseline", {})),
            "mechanic_by_fingerprint": len(mem_view.get("mechanic_by_fingerprint", {})),
        }
        if mem_view
        else {},
    }
    if raw_scores and all(val <= 0 for val in raw_scores.values()):
        diagnostics["fallback_reason"] = "all_zero_scores"
    elif not raw_scores:
        diagnostics["fallback_reason"] = "no_families_scored"

    run_summary = _build_run_summary(window, availability, cfg, ctx)
    if ctx and ctx.get("debug"):
        run_summary["diagnostics"] = diagnostics

    return MechanicClassifierReport(
        mechanic_prior=MechanicPrior(
            families=families,
            normalization={
                "sum_raw": sum(raw_scores.values()),
                "sum_prior": sum(prior for _, prior in priors.items()),
            },
        ),
        family_tags=family_tags,
        run_summary=run_summary,
    )


def _availability(
    fp_reports: List[Dict[str, Any]],
    simple_report: Optional[Dict[str, Any]],
    full_report: Optional[Dict[str, Any]],
    action_schema: Optional[ActionSchema],
) -> Dict[str, Any]:
    has_object_tracking = False
    has_reward_signal = False
    reports_with_object_deltas = 0
    object_delta_total = 0
    reports_with_diff = 0
    for rep in fp_reports:
        diff = rep.get("diff_summary") or {}
        if diff:
            reports_with_diff += 1
        if diff.get("per_object_deltas"):
            has_object_tracking = True
            reports_with_object_deltas += 1
            try:
                object_delta_total += len(diff.get("per_object_deltas", []))
            except Exception:
                pass
        if _reward_present(rep):
            has_reward_signal = True

    has_coord_actions = None
    coord_action_ids: List[str] = []
    simple_action_ids: List[str] = []
    if action_schema is not None:
        has_coord_actions = any(a.kind == "coord" for a in action_schema.actions)
        coord_action_ids = sorted([a.action_id for a in action_schema.actions if a.kind == "coord"])
        simple_action_ids = sorted([a.action_id for a in action_schema.actions if a.kind == "simple"])

    return {
        "has_object_tracking": has_object_tracking,
        "has_reward_signal": has_reward_signal,
        "has_simple_report": simple_report is not None,
        "has_full_report": full_report is not None,
        "has_coord_actions": has_coord_actions,
        "coord_action_ids": coord_action_ids,
        "simple_action_ids": simple_action_ids,
        "reports_with_object_deltas": reports_with_object_deltas,
        "object_delta_total": object_delta_total,
        "reports_with_diff": reports_with_diff,
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


def _score_families(
    features: Dict[str, Any],
    availability: Dict[str, Any],
    cfg: MechanicClassifierConfig,
) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
    raw_scores: Dict[str, float] = {}
    evidence: Dict[str, List[Dict[str, Any]]] = {}
    for family in MECHANIC_FAMILIES:
        if family.requires.get("needs_object_tracking") and not availability["has_object_tracking"]:
            raw_scores[family.family_id] = 0.0
            evidence[family.family_id] = []
            continue
        if family.requires.get("needs_reward_signal") and not availability["has_reward_signal"]:
            raw_scores[family.family_id] = 0.0
            evidence[family.family_id] = []
            continue
        if availability["has_coord_actions"] is False and family.requires.get("needs_coord_actions"):
            raw_scores[family.family_id] = 0.0
            evidence[family.family_id] = []
            continue

        if not triggers_met(family.trigger_features, features):
            raw_scores[family.family_id] = 0.0
            evidence[family.family_id] = []
            continue

        score = 0.0
        contributions: List[Dict[str, Any]] = []
        for term in family.score_terms:
            key = term.get("feature_ref", "")
            val = feature_value(features, key)
            contrib = term.get("weight", 0.0) * transform_value(val, term.get("transform", "identity"), term.get("params"))
            score += contrib
            contributions.append(
                {"type": "feature", "key": key, "value": float(val), "weight": term.get("weight", 0.0), "contribution": contrib}
            )
        for term in family.penalties:
            key = term.get("feature_ref", "")
            val = feature_value(features, key)
            contrib = term.get("weight", 0.0) * transform_value(val, term.get("transform", "identity"), term.get("params"))
            score -= contrib
            contributions.append(
                {"type": "feature", "key": key, "value": float(val), "weight": -term.get("weight", 0.0), "contribution": -contrib}
            )
        raw_scores[family.family_id] = max(0.0, score)
        evidence[family.family_id] = _top_evidence(contributions, cfg.evidence_per_family)
    return raw_scores, evidence


def _top_evidence(contributions: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    sorted_items = sorted(contributions, key=lambda item: (-abs(item.get("contribution", 0.0)), item.get("key", "")))
    return sorted_items[:limit]


def _normalize_scores(scores: Dict[str, float], cfg: MechanicClassifierConfig) -> Dict[str, float]:
    if not scores:
        return {}
    if all(val <= 0 for val in scores.values()):
        return {family_id: (1.0 if family_id == "unknown.mechanic" else 0.0) for family_id in scores}

    unknown = scores.get("unknown.mechanic", 0.0)
    if unknown < cfg.unknown_floor:
        scores = dict(scores)
        scores["unknown.mechanic"] = cfg.unknown_floor

    total = sum(scores.values())
    if total <= 0:
        return {family_id: (1.0 if family_id == "unknown.mechanic" else 0.0) for family_id in scores}
    return {family_id: val / total for family_id, val in scores.items()}


def _blend_memory_features(
    features: Dict[str, Any],
    availability: Dict[str, Any],
    mem_view: Dict[str, Any],
    cfg: MechanicClassifierConfig,
) -> Dict[str, Any]:
    if not mem_view:
        return features
    blended = dict(features)
    reports_with_diff = availability.get("reports_with_diff", 0)
    weight = min(1.0, reports_with_diff / float(max(cfg.initial_T, 1)))
    baselines = {}
    baselines.update(mem_view.get("event_signature_baseline", {}))
    baselines.update(mem_view.get("object_delta_baseline", {}))
    for key, mem_val in baselines.items():
        cur_val = blended.get(key)
        if cur_val is None:
            blended[key] = mem_val
        else:
            blended[key] = weight * float(cur_val) + (1.0 - weight) * float(mem_val)
    return blended


def _apply_memory_priors(
    raw_scores: Dict[str, float],
    mem_view: Dict[str, Any],
    reports_window: List[Dict[str, Any]],
) -> Dict[str, float]:
    if not mem_view:
        return raw_scores
    fp_last = reports_window[-1] if reports_window else {}
    debug = fp_last.get("debug") if isinstance(fp_last, dict) else None
    fingerprint = debug.get("grid_fingerprint") if isinstance(debug, dict) else None
    if not fingerprint:
        return raw_scores
    prior_bucket = mem_view.get("mechanic_by_fingerprint", {}).get(str(fingerprint))
    if not prior_bucket:
        return raw_scores
    adjusted = dict(raw_scores)
    for fam_id, entry in prior_bucket.items():
        avg_prior = float(entry.get("avg_prior", 0.0))
        adjusted[fam_id] = adjusted.get(fam_id, 0.0) + 0.25 * avg_prior
    return adjusted


def _build_family_priors(
    priors_sorted: List[Tuple[str, float]],
    evidence: Dict[str, List[Dict[str, Any]]],
    cfg: MechanicClassifierConfig,
) -> List[FamilyPrior]:
    families: List[FamilyPrior] = []
    for family_id, prior in priors_sorted:
        if prior < cfg.score_threshold and family_id != "unknown.mechanic":
            continue
        families.append(
            FamilyPrior(
                family_id=family_id,
                prior=prior,
                evidence=evidence.get(family_id, [])[: cfg.evidence_per_family],
            )
        )
        if len(families) >= cfg.max_families_emitted:
            break
    return families


def _build_family_tags(
    priors_sorted: List[Tuple[str, float]],
    availability: Dict[str, Any],
) -> FamilyTags:
    dominant = priors_sorted[0][0] if priors_sorted else "unknown.mechanic"
    needs_coord = dominant in {
        "paint.fill_connected_until_boundary",
        "toggle.cell_state",
        "teleport.portal",
        "line_draw",
        "ray_cast",
        "flood_spread",
    }
    needs_tracking = dominant in {
        "move.avatar_4dir",
        "push.sokoban_like",
        "gravity.fall_down",
        "wraparound.torus_edges",
        "swap.objects",
        "collect.target_on_contact",
        "teleport.portal",
    }
    likely_avatar = dominant in {"move.avatar_4dir", "push.sokoban_like"}

    preferred_action_families: List[str] = []
    preferred_coord_selectors: List[str] = []
    deprioritized_actions: List[str] = []

    if dominant.startswith("paint.") or dominant in {"toggle.cell_state", "flood_spread"}:
        preferred_coord_selectors = ["hotspot", "object_centroid", "region_frontier"]
        if availability.get("simple_action_ids"):
            deprioritized_actions = availability["simple_action_ids"]
    if dominant in {"line_draw", "ray_cast"}:
        preferred_coord_selectors = ["grid_edges_midpoints", "hotspot", "object_centroid"]
    if dominant in {"move.avatar_4dir", "push.sokoban_like", "wraparound.torus_edges", "swap.objects", "collect.target_on_contact"}:
        preferred_action_families = ["simple:movement_like"]
    if dominant == "gravity.fall_down":
        preferred_action_families = ["simple:minimal_interference"]

    if availability.get("has_coord_actions") and preferred_action_families and availability.get("coord_action_ids"):
        deprioritized_actions = availability["coord_action_ids"]

    return FamilyTags(
        required_capabilities={
            "needs_coord_actions": needs_coord,
            "needs_object_tracking": needs_tracking,
            "likely_avatar_present": likely_avatar,
        },
        constraints={
            "preferred_action_families": preferred_action_families,
            "preferred_coord_selectors": preferred_coord_selectors,
            "deprioritized_actions": deprioritized_actions,
        },
    )


def _build_run_summary(
    window: int,
    availability: Dict[str, Any],
    cfg: MechanicClassifierConfig,
    ctx: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    warnings: List[str] = []
    if not availability.get("has_simple_report") and not availability.get("has_full_report"):
        warnings.append("explorer_reports_missing")
    if availability.get("has_coord_actions") is False:
        warnings.append("coord_actions_missing")
    if availability.get("has_coord_actions") is None:
        warnings.append("action_schema_missing")

    summary = {
        "window_used": window,
        "evidence_quality": "low"
        if not availability.get("has_simple_report") and not availability.get("has_full_report")
        else "high",
        "warnings": warnings,
        "config": {
            "max_families_emitted": cfg.max_families_emitted,
            "evidence_per_family": cfg.evidence_per_family,
            "unknown_floor": cfg.unknown_floor,
            "score_threshold": cfg.score_threshold,
        },
    }
    if ctx:
        summary["ctx"] = ctx
    return summary

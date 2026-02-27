from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .action_schema import ActionSchema, parse_action_schema_data
from .discriminating_test_selector_config import DiscriminatingTestSelectorConfig
from .discriminating_test_selector_types import CandidateAction, CoordProposal, TestSelectionReport
from .executable_hypothesis_engine_types import ExecutableHypothesisV1


def select_test(
    hypotheses: List[ExecutableHypothesisV1] | List[Dict[str, Any]],
    fp_current: Dict[str, Any],
    action_schema: ActionSchema | Dict[str, Any],
    cfg: Optional[DiscriminatingTestSelectorConfig] = None,
    ctx: Optional[Dict[str, Any]] = None,
    simple_report: Optional[Dict[str, Any]] = None,
    full_report: Optional[Dict[str, Any]] = None,
) -> TestSelectionReport:
    cfg = cfg or DiscriminatingTestSelectorConfig()
    schema = parse_action_schema_data(action_schema) if isinstance(action_schema, dict) else action_schema
    hyp_list = _normalize_hypotheses(hypotheses)
    top_hyps = _top_hypotheses(hyp_list, cfg.topK_hypotheses_used)

    candidates = _generate_candidates(schema, fp_current, cfg, full_report)
    if not candidates:
        fallback = _fallback_action(schema)
        return TestSelectionReport(
            selected_test={"action_sequence": [fallback]},
            score_breakdown={
                "disagreement_score": 0.0,
                "elimination_score": 0.0,
                "loop_risk_penalty": 0.0,
            },
            alternatives_topM=[],
            run_summary={"warnings": ["no_candidates"], "ctx": ctx or {}},
        )

    scored: List[Tuple[float, CandidateAction]] = []
    for cand in candidates:
        preds = [_predict_signature(hyp, cand["action_key"], cand["action"]) for hyp in top_hyps]
        disagreement = _disagreement_score(preds, cfg)
        elimination = _elimination_score(top_hyps, cand["action"], cand["action_key"])
        loop_penalty = _loop_risk(cand, fp_current, simple_report, full_report)
        total = disagreement + elimination - loop_penalty
        breakdown = {
            "disagreement_score": disagreement,
            "elimination_score": elimination,
            "loop_risk_penalty": loop_penalty,
        }
        scored.append((total, CandidateAction(action=cand["action"], score_breakdown=breakdown, action_key=cand["action_key"])))

    scored.sort(key=lambda item: _candidate_sort_key(item[1], item[0]))
    best = scored[0][1]
    alternatives = [item[1] for item in scored[: cfg.alternatives_topM]]
    report = TestSelectionReport(
        selected_test={"action_sequence": [best.action]},
        score_breakdown=best.score_breakdown,
        alternatives_topM=alternatives,
        run_summary={
            "hypotheses_used": len(top_hyps),
            "candidates_considered": len(scored),
            "ctx": ctx or {},
        },
    )
    return report


def propose_coords(fp_current: Dict[str, Any], cfg: Optional[DiscriminatingTestSelectorConfig] = None) -> List[CoordProposal]:
    cfg = cfg or DiscriminatingTestSelectorConfig()
    proposals = _coord_candidates(fp_current)
    proposals.sort(key=lambda p: (p.y, p.x, p.source))
    return proposals[: cfg.coord_topK]


def _generate_candidates(
    schema: ActionSchema,
    fp_current: Dict[str, Any],
    cfg: DiscriminatingTestSelectorConfig,
    full_report: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    actions = []
    coord_action_ids = [a.action_id for a in schema.actions if a.kind == "coord"]
    simple_action_ids = [a.action_id for a in schema.actions if a.kind == "simple"]

    for action_id in simple_action_ids:
        actions.append({"action": {"type": "simple", "action_id": action_id}, "action_key": action_id})

    if coord_action_ids:
        coords = _coord_candidates(fp_current)
        if full_report:
            coords = _merge_full_report_coords(coords, full_report, cfg.coord_topK)
        coords = coords[: cfg.coord_topK]
        for action_id in coord_action_ids:
            for coord in coords:
                actions.append(
                    {
                        "action": {
                            "type": "coord",
                            "action_id": action_id,
                            "x": coord.x,
                            "y": coord.y,
                        },
                        "action_key": f"{action_id}@{coord.x},{coord.y}",
                    }
                )
    return actions


def _merge_full_report_coords(
    base: List[CoordProposal],
    full_report: Dict[str, Any],
    coord_topK: int,
) -> List[CoordProposal]:
    coords = list(base)
    proposals = full_report.get("coord_proposals", []) if isinstance(full_report, dict) else []
    for entry in proposals[:coord_topK]:
        try:
            x = int(entry.get("x"))
            y = int(entry.get("y"))
            coords.append(CoordProposal(x=x, y=y, source=str(entry.get("tag", "full_proposal"))))
        except Exception:
            continue
    model = full_report.get("coord_action_effect_model", {}) if isinstance(full_report, dict) else {}
    for action_id, stats in model.items():
        for entry in stats.get("hotspots", [])[:coord_topK]:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                x, y = int(entry[0]), int(entry[1])
                coords.append(CoordProposal(x=x, y=y, source=f"full:{action_id}"))
    return coords


def _coord_candidates(fp_current: Dict[str, Any]) -> List[CoordProposal]:
    coords: List[CoordProposal] = []
    grid_summary = _primary_grid_summary(fp_current)
    if grid_summary:
        width = int(grid_summary.get("width", 0))
        height = int(grid_summary.get("height", 0))
        if width > 0 and height > 0:
            corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
            for y, x in corners:
                coords.append(CoordProposal(x=x, y=y, source="corner"))
            edges = [(0, width // 2), (height - 1, width // 2), (height // 2, 0), (height // 2, width - 1)]
            for y, x in edges:
                coords.append(CoordProposal(x=x, y=y, source="edge"))

        for comp in grid_summary.get("connected_components", []) or []:
            centroid = comp.get("centroid") if isinstance(comp, dict) else None
            bbox = comp.get("bbox") if isinstance(comp, dict) else None
            if centroid and len(centroid) >= 2:
                cy, cx = int(round(float(centroid[0]))), int(round(float(centroid[1])))
                coords.append(CoordProposal(x=cx, y=cy, source="centroid"))
            if bbox and len(bbox) == 4:
                y0, x0, y1, x1 = [int(v) for v in bbox]
                coords.extend(
                    [
                        CoordProposal(x=x0, y=y0, source="bbox_corner"),
                        CoordProposal(x=x1, y=y0, source="bbox_corner"),
                        CoordProposal(x=x0, y=y1, source="bbox_corner"),
                        CoordProposal(x=x1, y=y1, source="bbox_corner"),
                    ]
                )

    diff = fp_current.get("diff_summary") or {}
    bbox = diff.get("changed_bbox")
    if bbox and len(bbox) == 4:
        y0, x0, y1, x1 = [int(v) for v in bbox]
        coords.append(CoordProposal(x=(x0 + x1) // 2, y=(y0 + y1) // 2, source="change_bbox"))

    dedup = {}
    for coord in coords:
        key = (coord.x, coord.y)
        if key not in dedup:
            dedup[key] = coord
    return list(dedup.values())


def _primary_grid_summary(fp_current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    state = fp_current.get("state_summary") or {}
    grids = state.get("grid_summaries") or []
    if grids:
        return grids[0] if isinstance(grids[0], dict) else None
    return None


def _disagreement_score(preds: List[Dict[str, Any]], cfg: DiscriminatingTestSelectorConfig) -> float:
    if not preds:
        return 0.0
    sigs = [p.get("signature") for p in preds]
    sig_entropy = _entropy(sigs)
    noop_split = _binary_split([p.get("noop") for p in preds])
    delta_var = _variance([_bin_index(p.get("delta_bin")) for p in preds])
    meta_disagree = _binary_split([p.get("meta") for p in preds if p.get("meta") is not None])
    return (
        cfg.w_sig_entropy * sig_entropy
        + cfg.w_noop_split * noop_split
        + cfg.w_delta_var * delta_var
        + cfg.w_meta_disagree * meta_disagree
    )


def _elimination_score(
    hypotheses: List[ExecutableHypothesisV1],
    action: Dict[str, Any],
    action_key: str,
) -> float:
    if not hypotheses:
        return 0.0
    eliminated = 0
    for hyp in hypotheses:
        if hyp.hypothesis_id == "unknown.mechanic":
            continue
        if _gate_violation(hyp, action_key):
            eliminated += 1
    return eliminated / float(max(1, len(hypotheses)))


def _loop_risk(
    cand: Dict[str, Any],
    fp_current: Dict[str, Any],
    simple_report: Optional[Dict[str, Any]],
    full_report: Optional[Dict[str, Any]],
) -> float:
    state_hash = fp_current.get("debug", {}).get("grid_hash") if isinstance(fp_current, dict) else None
    if not state_hash:
        return 0.0
    if cand["action"].get("type") == "simple" and simple_report:
        frontier = (simple_report.get("frontier") or {}).get(state_hash) or {}
        untried = frontier.get("untried_actions", [])
        return 0.0 if cand["action"].get("action_id") in untried else 1.0
    if cand["action"].get("type") == "coord" and full_report:
        frontier = (full_report.get("frontier") or {}).get(state_hash) or {}
        banlist = frontier.get("banlist", [])
        key_list = [cand["action"].get("action_id"), cand["action"].get("x"), cand["action"].get("y")]
        if key_list in banlist:
            return 1.0
    return 0.0


def _candidate_sort_key(candidate: CandidateAction, total: float) -> Tuple[Any, ...]:
    action = candidate.action
    y = action.get("y") if action.get("y") is not None else -1
    x = action.get("x") if action.get("x") is not None else -1
    return (
        -total,
        -candidate.score_breakdown.get("disagreement_score", 0.0),
        -candidate.score_breakdown.get("elimination_score", 0.0),
        candidate.score_breakdown.get("loop_risk_penalty", 0.0),
        str(candidate.action_key),
        y,
        x,
    )


def _fallback_action(schema: ActionSchema) -> Dict[str, Any]:
    simples = [a.action_id for a in schema.actions if a.kind == "simple"]
    if simples:
        return {"type": "simple", "action_id": simples[0]}
    coords = [a.action_id for a in schema.actions if a.kind == "coord"]
    if coords:
        return {"type": "coord", "action_id": coords[0], "x": 0, "y": 0}
    return {"type": "simple", "action_id": "ACTION1"}


def _normalize_hypotheses(hypotheses: List[ExecutableHypothesisV1] | List[Dict[str, Any]]) -> List[ExecutableHypothesisV1]:
    if not hypotheses:
        return []
    if isinstance(hypotheses[0], ExecutableHypothesisV1):
        return hypotheses  # type: ignore[return-value]
    out: List[ExecutableHypothesisV1] = []
    for entry in hypotheses:
        if not isinstance(entry, dict):
            continue
        program = entry.get("program_v1") or {}
        out.append(
            ExecutableHypothesisV1(
                hypothesis_id=str(entry.get("hypothesis_id")),
                name=str(entry.get("name", "")),
                description=str(entry.get("description", "")),
                program_v1=program,
                params=entry.get("params", {}),
                confidence=float(entry.get("confidence", 0.0)),
                fit_stats=entry.get("fit_stats", {}),
                predictions=entry.get("predictions", []),
            )
        )
    return out


def _top_hypotheses(hypotheses: List[ExecutableHypothesisV1], k: int) -> List[ExecutableHypothesisV1]:
    ranked = sorted(hypotheses, key=lambda h: (-h.confidence, h.hypothesis_id))
    return ranked[:k]


def _predict_signature(hyp: ExecutableHypothesisV1, action_key: str, action: Dict[str, Any]) -> Dict[str, Any]:
    if hyp.hypothesis_id == "unknown.mechanic":
        return {"signature": None, "noop": None, "delta_bin": None, "meta": None}
    if _gate_violation(hyp, action_key):
        return {"signature": None, "noop": None, "delta_bin": None, "meta": None}
    program = hyp.program_v1
    effects = program.get("effects") if isinstance(program, dict) else program.effects
    if not effects:
        return {"signature": None, "noop": None, "delta_bin": None, "meta": None}
    first = effects[0]
    sigs = first.get("event_signatures", []) if isinstance(first, dict) else []
    delta_bins = first.get("delta_bins", []) if isinstance(first, dict) else []
    return {
        "signature": sigs[0] if sigs else None,
        "noop": first.get("noop") if isinstance(first, dict) else None,
        "delta_bin": delta_bins[0] if delta_bins else None,
        "meta": None,
    }


def _gate_violation(hyp: ExecutableHypothesisV1, action_key: str) -> bool:
    program = hyp.program_v1
    gates = program.get("gates") if isinstance(program, dict) else program.gates
    for gate in gates or []:
        if gate.get("requires_coord") and "@" not in action_key:
            return True
        if gate.get("requires_simple") and "@" in action_key:
            return True
    return False


def _entropy(values: List[Any]) -> float:
    if not values:
        return 0.0
    counts: Dict[Any, int] = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / float(total)
        ent -= p * math.log(p)
    return ent / math.log(len(counts)) if len(counts) > 1 else 0.0


def _binary_split(values: List[Any]) -> float:
    if not values:
        return 0.0
    true_count = sum(1 for v in values if v)
    false_count = sum(1 for v in values if v is False)
    total = true_count + false_count
    if total == 0:
        return 0.0
    p_true = true_count / float(total)
    p_false = false_count / float(total)
    return 1.0 - abs(p_true - p_false)


def _variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / float(len(values))
    var = sum((v - mean) ** 2 for v in values) / float(len(values))
    return min(1.0, var / 2.25)


def _bin_index(bin_name: Optional[str]) -> float:
    mapping = {"tiny": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}
    return mapping.get(bin_name or "", 0.0)


def asdict_report(report: TestSelectionReport) -> Dict[str, Any]:
    return {
        "selected_test": report.selected_test,
        "score_breakdown": report.score_breakdown,
        "alternatives_topM": [
            {"action": alt.action, "score_breakdown": alt.score_breakdown, "action_key": alt.action_key}
            for alt in report.alternatives_topM
        ],
        "run_summary": report.run_summary,
    }

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .planner_types import CandidateAction, CandidateMeta, PlannerInputs, PlannerState

_MAX_COORD_CANDIDATES = 64
_TOPK_OBJECTS = 3
_NEIGHBORHOOD_RADIUS = 1
_MAX_MEMORY_CANDIDATES = 16


def build_candidates(
    state_key: str,
    action_schema: Dict[str, Any],
    planner_state: PlannerState,
    inputs: PlannerInputs,
    cfg: Any,
    fp_report_current: Dict[str, Any],
) -> Tuple[List[CandidateAction], Dict[Tuple[Any, ...], CandidateMeta], List[str]]:
    warnings: List[str] = []
    candidates: List[CandidateAction] = []
    meta: Dict[Tuple[Any, ...], CandidateMeta] = {}

    _populate_pending_tests(planner_state, inputs, cfg)
    tests = planner_state.pending_tests[: cfg.max_tests_considered]
    for test in tests:
        action = _test_to_action(test)
        if action is None:
            continue
        _add_candidate(action, "from_test", test, candidates, meta)

    frontier_candidates = _frontier_candidates(state_key, inputs, action_schema, cfg)
    for action, data in frontier_candidates:
        _add_candidate(action, "from_frontier", data, candidates, meta)

    heuristic_candidates = _heuristic_candidates(inputs, action_schema, fp_report_current)
    for action, data in heuristic_candidates:
        _add_candidate(action, "from_heuristic", data, candidates, meta)

    memory_candidates = _memory_candidates(inputs, action_schema, state_key)
    for action, data in memory_candidates:
        _add_candidate(action, "from_memory", data, candidates, meta)

    if not candidates:
        warnings.append("no_candidates")
    return candidates[: cfg.max_candidates], meta, warnings


def _populate_pending_tests(planner_state: PlannerState, inputs: PlannerInputs, cfg: Any) -> None:
    if planner_state.pending_tests:
        return
    report = inputs.hypotheses_report or {}
    hypotheses = report.get("hypotheses", [])
    collected = []
    for hyp in hypotheses:
        for test in hyp.get("tests", []):
            collected.append(test)
    planner_state.pending_tests = collected[: cfg.max_tests_considered]


def _test_to_action(test: Dict[str, Any]) -> CandidateAction | None:
    sequence = test.get("action_sequence")
    if not sequence:
        return None
    action = sequence[0]
    if not isinstance(action, dict):
        return None
    if action.get("type") == "coord":
        return CandidateAction(
            type="coord",
            action_id=action.get("action_id"),
            x=action.get("x"),
            y=action.get("y"),
        )
    return CandidateAction(type="simple", action_id=action.get("action_id"))


def _frontier_candidates(
    state_key: str,
    inputs: PlannerInputs,
    action_schema: Dict[str, Any],
    cfg: Any,
) -> List[Tuple[CandidateAction, Dict[str, Any]]]:
    candidates: List[Tuple[CandidateAction, Dict[str, Any]]] = []
    simple_report = inputs.simple_report or {}
    full_report = inputs.full_report or {}
    simple_frontier = simple_report.get("frontier", {})
    full_frontier = full_report.get("frontier", {})

    if state_key in simple_frontier:
        entry = simple_frontier[state_key]
        for action_id in entry.get("untried_actions", [])[: cfg.max_frontier_considered]:
            candidates.append((CandidateAction(type="simple", action_id=action_id), {"frontier": "simple"}))

    coord_actions = _coord_actions(action_schema)
    if coord_actions and state_key in full_frontier:
        model = full_report.get("coord_action_effect_model", {})
        for action_id in coord_actions:
            stats = model.get(action_id, {})
            hotspots = stats.get("hotspots", [])
            if not hotspots:
                continue
            x, y, *_ = hotspots[0]
            candidates.append(
                (CandidateAction(type="coord", action_id=action_id, x=int(x), y=int(y)), {"frontier": "full"})
            )
    return candidates[: cfg.max_frontier_considered]


def _heuristic_candidates(
    inputs: PlannerInputs,
    action_schema: Dict[str, Any],
    fp_report_current: Dict[str, Any],
) -> List[Tuple[CandidateAction, Dict[str, Any]]]:
    candidates: List[Tuple[CandidateAction, Dict[str, Any]]] = []
    simple_report = inputs.simple_report or {}
    full_report = inputs.full_report or {}

    best_simple = _best_simple_action(simple_report)
    if best_simple:
        candidates.append((CandidateAction(type="simple", action_id=best_simple), {"heuristic": "best_simple"}))

    coord_actions = _coord_actions(action_schema)
    if coord_actions:
        for action in _object_centroid_neighborhoods(fp_report_current, coord_actions, action_schema):
            candidates.append((action, {"heuristic": "object_centroid"}))
        for action in _hotspot_candidates(full_report, coord_actions):
            candidates.append((action, {"heuristic": "best_hotspot"}))

    if len(candidates) > _MAX_COORD_CANDIDATES:
        candidates = candidates[:_MAX_COORD_CANDIDATES]
    return candidates


def _memory_candidates(
    inputs: PlannerInputs,
    action_schema: Dict[str, Any],
    state_key: str,
) -> List[Tuple[CandidateAction, Dict[str, Any]]]:
    view = inputs.memory_view or {}
    candidates: List[Tuple[CandidateAction, Dict[str, Any]]] = []
    noop_by_action = view.get("noop_rate_by_action", {})
    attempts_by_action = view.get("attempts_by_action", {})
    recent_actions = set(view.get("last_k_actions_per_state", []))
    simple_actions = _simple_actions(action_schema)
    ranked_simple = []
    for action_id in simple_actions:
        noop = float(noop_by_action.get(action_id, 0.0))
        attempts = int(attempts_by_action.get(action_id, 0))
        ranked_simple.append((noop, attempts, action_id))
    ranked_simple.sort(key=lambda item: (item[0], item[1], item[2]))
    for noop, attempts, action_id in ranked_simple[:_MAX_MEMORY_CANDIDATES]:
        if action_id in recent_actions and noop >= 0.9:
            continue
        candidates.append(
            (
                CandidateAction(type="simple", action_id=action_id),
                {"memory": "action_prior", "noop_rate": noop, "attempts": attempts},
            )
        )

    coord_actions = _coord_actions(action_schema)
    coord_scores = view.get("coord_effect_score_by_action", {})
    for action_id in coord_actions:
        coords = coord_scores.get(action_id, {})
        ranked = sorted(coords.items(), key=lambda kv: (-kv[1], kv[0]))
        for coord_key, score in ranked[:_MAX_MEMORY_CANDIDATES]:
            x_str, y_str = coord_key.split(",", 1)
            candidates.append(
                (
                    CandidateAction(type="coord", action_id=action_id, x=int(x_str), y=int(y_str)),
                    {"memory": "coord_prior", "score": float(score)},
                )
            )
    return candidates


def _best_simple_action(simple_report: Dict[str, Any]) -> str | None:
    effects = simple_report.get("action_effect_model", {})
    best = None
    best_score = -1.0
    for action_id, stats in effects.items():
        no_effect = float(stats.get("no_effect_rate", 1.0))
        score = 1.0 - no_effect
        if score > best_score or (score == best_score and (best is None or action_id < best)):
            best = action_id
            best_score = score
    return best


def _best_hotspot_coord(full_report: Dict[str, Any], coord_actions: List[str]) -> CandidateAction | None:
    model = full_report.get("coord_action_effect_model", {})
    best = None
    best_rate = -1.0
    for action_id in coord_actions:
        stats = model.get(action_id, {})
        for entry in stats.get("hotspots", []):
            if len(entry) < 3:
                continue
            x, y, rate = entry[0], entry[1], entry[2]
            if rate > best_rate:
                best_rate = rate
                best = CandidateAction(type="coord", action_id=action_id, x=int(x), y=int(y))
    return best


def _object_centroid_coord(fp_report: Dict[str, Any], action_id: str) -> CandidateAction | None:
    state = fp_report.get("state_summary") or {}
    objs = state.get("object_catalog") or []
    for obj in objs:
        centroid = obj.get("centroid")
        if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            y, x = centroid
            return CandidateAction(type="coord", action_id=action_id, x=int(round(x)), y=int(round(y)))
    return None


def _object_centroid_neighborhoods(
    fp_report: Dict[str, Any],
    coord_actions: List[str],
    action_schema: Dict[str, Any],
) -> List[CandidateAction]:
    state = fp_report.get("state_summary") or {}
    objs = state.get("object_catalog") or []
    width, height = _grid_bounds(action_schema)
    selected = objs[:_TOPK_OBJECTS]
    coords: List[Tuple[int, int]] = []
    for obj in selected:
        centroid = obj.get("centroid")
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            continue
        y, x = centroid
        cx, cy = int(round(x)), int(round(y))
        for dy in (-_NEIGHBORHOOD_RADIUS, 0, _NEIGHBORHOOD_RADIUS):
            for dx in (-_NEIGHBORHOOD_RADIUS, 0, _NEIGHBORHOOD_RADIUS):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    coords.append((nx, ny))
    ordered = []
    seen = set()
    for x, y in coords:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        ordered.append((x, y))

    actions: List[CandidateAction] = []
    for action_id in coord_actions:
        for x, y in ordered:
            actions.append(CandidateAction(type="coord", action_id=action_id, x=x, y=y))
    return actions


def _hotspot_candidates(full_report: Dict[str, Any], coord_actions: List[str]) -> List[CandidateAction]:
    model = full_report.get("coord_action_effect_model", {})
    entries: List[Tuple[float, str, int, int]] = []
    for action_id in coord_actions:
        stats = model.get(action_id, {})
        for entry in stats.get("hotspots", []):
            if len(entry) < 3:
                continue
            x, y, rate = entry[0], entry[1], entry[2]
            entries.append((float(rate), action_id, int(x), int(y)))
    entries.sort(key=lambda item: (-item[0], item[2], item[3], item[1]))
    actions: List[CandidateAction] = []
    for _, action_id, x, y in entries:
        actions.append(CandidateAction(type="coord", action_id=action_id, x=x, y=y))
    return actions


def _simple_actions(action_schema: Dict[str, Any]) -> List[str]:
    actions = action_schema.get("actions", []) if isinstance(action_schema, dict) else []
    return sorted([a.get("action_id") for a in actions if a.get("kind") == "simple" and a.get("action_id")])


def _grid_bounds(action_schema: Dict[str, Any]) -> Tuple[int, int]:
    grid = action_schema.get("primary_grid", {}) if isinstance(action_schema, dict) else {}
    width = int(grid.get("width", 1))
    height = int(grid.get("height", 1))
    return width, height


def _coord_actions(action_schema: Dict[str, Any]) -> List[str]:
    actions = action_schema.get("actions", []) if isinstance(action_schema, dict) else []
    return sorted([a.get("action_id") for a in actions if a.get("kind") == "coord"])


def _add_candidate(
    action: CandidateAction,
    source: str,
    data: Dict[str, Any],
    candidates: List[CandidateAction],
    meta: Dict[Tuple[Any, ...], CandidateMeta],
) -> None:
    if not action.action_id:
        return
    source_label = source
    if isinstance(data, dict):
        if "heuristic" in data:
            source_label = f"{source}:{data.get('heuristic')}"
        elif "frontier" in data:
            source_label = f"{source}:{data.get('frontier')}"
        elif "test_id" in data:
            source_label = f"{source}:{data.get('test_id')}"
    key = action.key()
    if key in meta:
        return
    candidates.append(action)
    meta[key] = CandidateMeta(
        source=source_label,
        expected_signatures=list(data.get("expected_signature", [])) if isinstance(data, dict) else [],
        supports=list(data.get("supports", [])) if isinstance(data, dict) else [],
        refutes=list(data.get("refutes", [])) if isinstance(data, dict) else [],
    )

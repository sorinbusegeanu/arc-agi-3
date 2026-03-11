from __future__ import annotations

from v3_1.utils.ids import stable_digest


def _candidate_id(prefix: str, payload: dict) -> str:
    return f"{prefix}:{stable_digest(payload)}"


def _target_candidate(target: dict, *, candidate_class: str, action_type: str, rationale: str) -> dict:
    entity_id = str(target["entity_id"])
    centroid = list(target.get("centroid", [0, 0]))
    return {
        "candidate_id": _candidate_id(candidate_class, {"entity_id": entity_id, "action_type": action_type}),
        "candidate_class": candidate_class,
        "target_entity_id": entity_id,
        "target_area_id": target.get("area_id"),
        "action": {
            "type": action_type,
            "target": entity_id,
            "centroid": centroid,
        },
        "confidence": float(target.get("confidence", 0.0)),
        "utility": float(target.get("utility", 0.0)),
        "novelty": float(target.get("novelty", 0.0)),
        "reachable_now": bool(target.get("reachable_now")),
        "reachable_later": bool(target.get("reachable_later")),
        "rationale": rationale,
        "blocked_reasons": [],
    }


def generate_candidates(skill_library: dict[str, dict], belief: dict, limit: int) -> list[dict]:
    del skill_library
    candidates: list[dict] = []

    for target in belief.get("reachable_targets", []):
        candidates.append(_target_candidate(target, candidate_class="target", action_type="inspect", rationale="reachable_target"))
        trigger_rows = belief.get("trigger_support", {}).get(str(target["entity_id"]), [])
        if trigger_rows:
            candidates.append(_target_candidate(target, candidate_class="trigger_probe", action_type="interact", rationale="trigger_supported_target"))

    for target in belief.get("frontier_targets", []):
        candidate = _target_candidate(target, candidate_class="frontier_move", action_type="move_to_frontier", rationale="frontier_target")
        candidate["subgoal"] = "expand_frontier"
        candidates.append(candidate)

    current_area_id = belief.get("current_area_id")
    for target in belief.get("local_pois", []):
        if target.get("area_id") != current_area_id:
            continue
        candidate = _target_candidate(target, candidate_class="local_probe", action_type="inspect_local", rationale="area_local_probe")
        candidates.append(candidate)

    for target in belief.get("blocked_targets", []):
        candidate = _target_candidate(target, candidate_class="recovery_move", action_type="reposition", rationale="recovery_candidate")
        candidate["subgoal"] = "recover_route"
        candidates.append(candidate)

    for action_text, consequence_rows in belief.get("consequence_support", {}).items():
        if not consequence_rows:
            continue
        candidate_id = _candidate_id("route_probe", {"action": action_text, "count": len(consequence_rows)})
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_class": "route_probe",
                "target_entity_id": None,
                "target_area_id": current_area_id,
                "action": {"type": "probe_route", "action_hint": action_text},
                "confidence": 0.25,
                "utility": min(0.5, 0.1 * len(consequence_rows)),
                "novelty": 0.1,
                "reachable_now": True,
                "reachable_later": False,
                "rationale": "consequence_supported_route_probe",
                "blocked_reasons": [],
            }
        )

    deduped: dict[tuple[str, str | None, str], dict] = {}
    for row in candidates:
        key = (str(row["candidate_class"]), row.get("target_entity_id"), str(row.get("action", {}).get("type")))
        if key not in deduped or float(row.get("confidence", 0.0)) > float(deduped[key].get("confidence", 0.0)):
            deduped[key] = row

    ranked = sorted(
        deduped.values(),
        key=lambda row: (
            not bool(row.get("reachable_now")),
            not bool(row.get("reachable_later")),
            -float(row.get("utility", 0.0)),
            -float(row.get("novelty", 0.0)),
            -float(row.get("confidence", 0.0)),
            row["candidate_id"],
        ),
    )
    return ranked[:limit]

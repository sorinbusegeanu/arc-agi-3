from __future__ import annotations

from v3_1.planning.belief_builder import normalized_route_signature, normalized_target_key, normalized_trigger_zone_key
from v3_1.utils.ids import stable_digest

MAX_SUPPORTING_EVIDENCE_REFS = 12
GENERATION_QUOTAS = {
    "target": 6,
    "click_target": 3,
    "local_probe": 4,
    "frontier_move": 4,
    "route_probe": 3,
    "trigger_probe": 4,
    "recovery_move": 3,
    "fallback_action": 1,
}


def _candidate_id(prefix: str, payload: dict) -> str:
    return f"{prefix}:{stable_digest(payload)}"


def _available_action_families(belief: dict) -> set[str]:
    families = set(belief.get("available_action_families", []) or [])
    if not families:
        families = {"move"}
    return families


def _match_skill_id(skill_library: dict[str, dict], *, entity_id: str, area_id: str | None, required_action_family: str, candidate_class: str) -> tuple[str | None, str | None]:
    preferred_skill_types = (
        ("trigger_probe", "inspect_target") if required_action_family in {"interact", "click_at"} and candidate_class == "trigger_probe"
        else ("inspect_target", "trigger_probe") if required_action_family in {"interact", "click_at"}
        else ("recover_route",) if required_action_family == "move"
        else ()
    )
    for skill_type in preferred_skill_types:
        for skill_id, skill in skill_library.items():
            if str(skill.get("skill_type") or "") != skill_type:
                continue
            if entity_id and str(skill.get("entity_id") or "") == entity_id:
                return str(skill_id), skill_type
            if area_id is not None and str(skill.get("target_area_id") or "") == str(area_id):
                return str(skill_id), skill_type
    return None, None


def _compress_supporting_evidence_refs(refs: list[str] | tuple[str, ...]) -> tuple[list[str], dict]:
    ordered = [str(ref) for ref in refs if ref]
    unique = list(dict.fromkeys(ordered))
    sample = unique[:MAX_SUPPORTING_EVIDENCE_REFS]
    return sample, {
        "supporting_evidence_ref_count": len(unique),
        "supporting_evidence_ref_sample": sample,
        "supporting_evidence_signature": stable_digest(unique),
        "supporting_evidence_truncated": len(unique) > len(sample),
    }


def _candidate_schema(*, candidate_class: str, action_type: str, target: dict | None, required_action_family: str, rationale: str, generation_source: str, belief: dict, route_required: bool, route_signature: str | None, expected_progress_type: str, trigger_zone_id: str | None = None, subgoal: str | None = None, action_overrides: dict | None = None, support_refs: list[str] | None = None) -> dict:
    target = dict(target or {})
    centroid = list(target.get("centroid", [0, 0]))
    target_entity_id = str(target.get("entity_id")) if target.get("entity_id") is not None else None
    target_area_id = str(target.get("area_id")) if target.get("area_id") is not None else belief.get("current_area_id")
    target_entity_class = str(target.get("kind") or target.get("poi_class") or "unknown")
    target_key = normalized_target_key(target_entity_id, target_area_id, target_class=target_entity_class)
    supporting_refs, supporting_meta = _compress_supporting_evidence_refs(list(support_refs or target.get("evidence_refs", [])))
    stable_key = {
        "candidate_class": candidate_class,
        "target_entity_id": target_entity_id or "none",
        "target_area_id": target_area_id or "none",
        "route_signature": route_signature or "none",
        "trigger_zone_id": trigger_zone_id or "none",
        "action_type": action_type,
        "target_entity_class": target_entity_class,
    }
    utility = float(target.get("utility", 0.0))
    confidence = float(target.get("confidence", 0.0))
    novelty = float(target.get("novelty", 0.0))
    direct_support = min(1.0, confidence + (0.1 * len(supporting_refs)))
    indirect_support = min(1.0, utility + (0.25 if trigger_zone_id else 0.0))
    prior_row = dict(belief.get("durable_prior_merge", {}).get("per_target", {}).get(target_entity_id or "", {}))
    prior_support = min(1.0, float(prior_row.get("poi_pattern", {}).get("observations", 0) or 0) / 10.0) if prior_row else 0.0
    contradiction_flags = dict(target.get("contradiction_markers", {}))
    stale_support_flags = {
        "support_refs_missing": bool(supporting_refs) and len(supporting_refs) < int(supporting_meta.get("supporting_evidence_ref_count", 0)),
        "target_stale": contradiction_flags.get("stale_target", False),
    }
    action = {
        "type": action_type,
        "target": target_entity_id,
        "centroid": centroid,
        "skill_id": target.get("skill_id"),
    }
    if action_overrides:
        action.update(action_overrides)
    return {
        "candidate_id": _candidate_id(candidate_class, stable_key),
        "candidate_class": candidate_class,
        "target_entity_id": target_entity_id,
        "target_area_id": target_area_id,
        "target_key": target_key,
        "skill_id": target.get("skill_id"),
        "skill_type": target.get("skill_type"),
        "required_action_family": required_action_family,
        "effect_action_family": str(target.get("candidate_effect_mode") or required_action_family),
        "expected_progress_type": expected_progress_type,
        "route_required": bool(route_required),
        "route_signature": route_signature,
        "trigger_zone_id": trigger_zone_id,
        "target_entity_class": target_entity_class,
        "candidate_context": {
            "avatar_area": belief.get("local_context", {}).get("current_area_id"),
            "local_area": belief.get("current_area_id"),
            "route_signature": route_signature,
            "trigger_zone_id": trigger_zone_id,
            "target_entity_class": target_entity_class,
        },
        "expected_outcomes": {
            "expected_state_change": float(target.get("candidate_effect_score", target.get("interact_effect_score", 0.0))),
            "expected_evidence_gain": min(1.0, novelty + float(target.get("motion_score", 0.0))),
            "expected_route_progress": float(target.get("distance_score", 0.0)) if route_required else 0.0,
        },
        "support_strength": {
            "direct_support": direct_support,
            "indirect_support": indirect_support,
            "prior_support": prior_support,
        },
        "contradiction_flags": contradiction_flags,
        "stale_support_flags": stale_support_flags,
        "supporting_evidence_refs": supporting_refs,
        "generation_source": generation_source,
        **supporting_meta,
        "action": action,
        "confidence": confidence,
        "utility": utility,
        "novelty": novelty,
        "movement_effect_score": float(target.get("movement_effect_score", 0.0)),
        "interact_effect_score": float(target.get("interact_effect_score", 0.0)),
        "click_effect_score": float(target.get("click_effect_score", 0.0)),
        "candidate_effect_score": float(target.get("candidate_effect_score", 0.0)),
        "distance_from_avatar": float(target.get("distance_from_avatar", 0.0)),
        "distance_score": float(target.get("distance_score", 0.0)),
        "motion_variance": float(target.get("motion_variance", 0.0)),
        "motion_score": float(target.get("motion_score", 0.0)),
        "reachable_now": bool(target.get("reachable_now")),
        "reachable_later": bool(target.get("reachable_later")),
        "rationale": rationale,
        "subgoal": subgoal,
        "blocked_reasons": [],
        "blocked_reason_details": [],
    }


def _target_candidate(skill_library: dict[str, dict], belief: dict, target: dict, *, candidate_class: str, action_type: str, required_action_family: str, rationale: str) -> dict:
    entity_id = str(target["entity_id"])
    area_id = str(target.get("area_id")) if target.get("area_id") is not None else None
    skill_id, skill_type = _match_skill_id(
        skill_library,
        entity_id=entity_id,
        area_id=area_id,
        required_action_family=required_action_family,
        candidate_class=candidate_class,
    )
    effect_action_family = str(target.get("candidate_effect_mode") or required_action_family)
    candidate_effect_score = (
        float(target.get("movement_effect_score", 0.0)) if effect_action_family == "move"
        else float(target.get("interact_effect_score", 0.0)) if effect_action_family == "interact"
        else float(target.get("click_effect_score", 0.0)) if effect_action_family == "click_at"
        else 0.0
    )
    target_payload = dict(target)
    target_payload["skill_id"] = skill_id
    target_payload["skill_type"] = skill_type
    target_payload["candidate_effect_score"] = candidate_effect_score
    return _candidate_schema(
        candidate_class=candidate_class,
        action_type=action_type,
        target=target_payload,
        required_action_family=required_action_family,
        rationale=rationale,
        generation_source="belief_target",
        belief=belief,
        route_required=bool(required_action_family == "move" or not bool(target.get("reachable_now"))),
        route_signature=normalized_route_signature(candidate_class=candidate_class, area_id=area_id, target_entity_id=entity_id, centroid=list(target.get("centroid", [0, 0]))),
        expected_progress_type="interaction" if required_action_family in {"interact", "click_at"} else "movement",
    )


def generate_candidates(skill_library: dict[str, dict], belief: dict, limit: int) -> list[dict]:
    candidates: list[dict] = []
    available_families = _available_action_families(belief)
    diagnostics = {"count_by_class": {}, "dropped_during_generation": 0, "unsupported_template_count": 0}
    class_counts: dict[str, int] = {}

    def _admit(row: dict | None) -> None:
        if row is None:
            diagnostics["unsupported_template_count"] += 1
            return
        candidate_class = str(row.get("candidate_class") or "unknown")
        quota = GENERATION_QUOTAS.get(candidate_class, limit)
        count = class_counts.get(candidate_class, 0)
        if count >= quota:
            diagnostics["dropped_during_generation"] += 1
            return
        class_counts[candidate_class] = count + 1
        diagnostics["count_by_class"][candidate_class] = class_counts[candidate_class]
        candidates.append(row)

    for target in belief.get("reachable_targets", []):
        if "interact" in available_families:
            candidate = _target_candidate(skill_library, belief, target, candidate_class="target", action_type="interact", required_action_family="interact", rationale="reachable_target")
            candidate["generation_source"] = "belief_reachable_target"
            _admit(candidate)
        if "click_at" in available_families:
            click_candidate = _target_candidate(skill_library, belief, target, candidate_class="click_target", action_type="click_at", required_action_family="click_at", rationale="reachable_click_target")
            click_candidate["action"]["click_target_coordinates"] = list(target.get("centroid", [0, 0]))
            click_candidate["expected_progress_type"] = "interaction"
            click_candidate["generation_source"] = "belief_click_target"
            _admit(click_candidate)
        trigger_rows = belief.get("trigger_support", {}).get(str(target["entity_id"]), [])
        if trigger_rows and "interact" in available_families:
            trigger_candidate = _target_candidate(skill_library, belief, target, candidate_class="trigger_probe", action_type="interact", required_action_family="interact", rationale="trigger_supported_target")
            trigger_candidate["trigger_zone_id"] = normalized_trigger_zone_key(entity_id=str(target.get("entity_id")), area_id=str(target.get("area_id")) if target.get("area_id") is not None else None)
            trigger_candidate["candidate_context"]["trigger_zone_id"] = trigger_candidate["trigger_zone_id"]
            trigger_candidate["generation_source"] = "belief_trigger_support"
            _admit(trigger_candidate)

    if "move" in available_families:
        for target in belief.get("frontier_targets", []):
            candidate = _target_candidate(skill_library, belief, target, candidate_class="frontier_move", action_type="move_to_frontier", required_action_family="move", rationale="frontier_target")
            candidate["subgoal"] = "expand_frontier"
            candidate["generation_source"] = "belief_frontier"
            candidate["expected_outcomes"]["expected_evidence_gain"] = float(target.get("expected_information_gain", candidate["expected_outcomes"]["expected_evidence_gain"]))
            _admit(candidate)

    current_area_id = belief.get("current_area_id")
    for target in belief.get("local_pois", []):
        if target.get("area_id") != current_area_id:
            continue
        if "interact" in available_families:
            candidate = _target_candidate(skill_library, belief, target, candidate_class="local_probe", action_type="interact", required_action_family="interact", rationale="area_local_probe")
            candidate["generation_source"] = "belief_local_poi"
            _admit(candidate)

    if "move" in available_families:
        for target in belief.get("blocked_targets", []):
            candidate = _target_candidate(skill_library, belief, target, candidate_class="recovery_move", action_type="reposition", required_action_family="move", rationale="recovery_candidate")
            candidate["subgoal"] = "recover_route"
            candidate["generation_source"] = "belief_recovery"
            _admit(candidate)

    if "move" in available_families:
        for action_text, consequence_rows in belief.get("consequence_support", {}).items():
            if not consequence_rows:
                continue
            supporting_refs, supporting_meta = _compress_supporting_evidence_refs(
                [str(row.get("consequence_id")) for row in consequence_rows if row.get("consequence_id")]
            )
            route_signature = normalized_route_signature(candidate_class="route_probe", area_id=str(current_area_id) if current_area_id is not None else None, action_hint=action_text)
            _admit(
                {
                    "candidate_id": _candidate_id("route_probe", {"route_signature": route_signature}),
                    "candidate_class": "route_probe",
                    "target_entity_id": None,
                    "target_area_id": current_area_id,
                    "target_key": normalized_target_key(None, str(current_area_id) if current_area_id is not None else None, target_class="route_probe"),
                    "skill_id": None,
                    "skill_type": None,
                    "required_action_family": "move",
                    "effect_action_family": "move",
                    "expected_progress_type": "route_probe",
                    "route_required": True,
                    "route_signature": route_signature,
                    "trigger_zone_id": None,
                    "target_entity_class": "route_probe",
                    "candidate_context": {
                        "avatar_area": belief.get("local_context", {}).get("current_area_id"),
                        "local_area": current_area_id,
                        "route_signature": route_signature,
                        "trigger_zone_id": None,
                        "target_entity_class": "route_probe",
                    },
                    "expected_outcomes": {
                        "expected_state_change": min(1.0, 0.1 * len(consequence_rows)),
                        "expected_evidence_gain": min(1.0, 0.08 * len(consequence_rows)),
                        "expected_route_progress": 0.4,
                    },
                    "support_strength": {
                        "direct_support": min(1.0, 0.1 * len(supporting_refs)),
                        "indirect_support": min(1.0, 0.05 * len(consequence_rows)),
                        "prior_support": 0.0,
                    },
                    "contradiction_flags": {},
                    "stale_support_flags": {"support_refs_missing": False, "target_stale": False},
                    "supporting_evidence_refs": supporting_refs,
                    "generation_source": "belief_consequence_support",
                    **supporting_meta,
                    "action": {"type": "probe_route", "action_hint": action_text, "skill_id": None},
                    "confidence": 0.25,
                    "utility": min(0.5, 0.1 * len(consequence_rows)),
                    "novelty": 0.1,
                    "movement_effect_score": 0.0,
                    "interact_effect_score": 0.0,
                    "click_effect_score": 0.0,
                    "candidate_effect_score": 0.0,
                    "distance_from_avatar": 0.0,
                    "distance_score": 0.0,
                    "motion_variance": 0.0,
                    "motion_score": 0.0,
                    "reachable_now": True,
                    "reachable_later": False,
                    "rationale": "consequence_supported_route_probe",
                    "blocked_reasons": [],
                    "blocked_reason_details": [],
                }
            )

    _admit(
        {
            "candidate_id": _candidate_id("fallback_action", {"area_id": current_area_id or "none"}),
            "candidate_class": "fallback_action",
            "target_entity_id": None,
            "target_area_id": current_area_id,
            "target_key": normalized_target_key(None, str(current_area_id) if current_area_id is not None else None, target_class="fallback"),
            "skill_id": None,
            "skill_type": None,
            "required_action_family": "move",
            "effect_action_family": "move",
            "expected_progress_type": "fallback",
            "route_required": False,
            "route_signature": normalized_route_signature(candidate_class="fallback", area_id=str(current_area_id) if current_area_id is not None else None),
            "trigger_zone_id": None,
            "target_entity_class": "fallback",
            "candidate_context": {
                "avatar_area": belief.get("local_context", {}).get("current_area_id"),
                "local_area": current_area_id,
                "route_signature": normalized_route_signature(candidate_class="fallback", area_id=str(current_area_id) if current_area_id is not None else None),
                "trigger_zone_id": None,
                "target_entity_class": "fallback",
            },
            "expected_outcomes": {"expected_state_change": 0.0, "expected_evidence_gain": 0.1, "expected_route_progress": 0.0},
            "support_strength": {"direct_support": 0.0, "indirect_support": 0.0, "prior_support": 0.0},
            "contradiction_flags": {},
            "stale_support_flags": {},
            "supporting_evidence_refs": [],
            "generation_source": "fallback_template",
            "supporting_evidence_ref_count": 0,
            "supporting_evidence_ref_sample": [],
            "supporting_evidence_signature": stable_digest([]),
            "supporting_evidence_truncated": False,
            "action": {"type": "hold_position", "area_id": current_area_id},
            "confidence": 0.0,
            "utility": 0.0,
            "novelty": 0.0,
            "movement_effect_score": 0.0,
            "interact_effect_score": 0.0,
            "click_effect_score": 0.0,
            "candidate_effect_score": 0.0,
            "distance_from_avatar": 0.0,
            "distance_score": 0.0,
            "motion_variance": 0.0,
            "motion_score": 0.0,
            "reachable_now": True,
            "reachable_later": False,
            "rationale": "always_available_fallback",
            "blocked_reasons": [],
            "blocked_reason_details": [],
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
    trimmed = ranked[:limit]
    for row in trimmed:
        row["generation_diagnostics"] = dict(diagnostics)
    return trimmed

from __future__ import annotations

from v3_1.utils.ids import stable_digest


def _selected_action_family(selected: dict) -> str:
    family = str(selected.get("required_action_family") or "").strip().lower()
    if family:
        return family
    action = dict(selected.get("action", {})) if isinstance(selected.get("action"), dict) else {}
    action_type = str(action.get("type") or "").strip().lower()
    if action_type in {"move", "move_to_frontier", "reposition", "probe_route"}:
        return "move"
    if action_type in {"interact", "inspect"}:
        return "interact"
    if action_type == "click_at":
        return "click_at"
    return action_type or "unknown"


def _merge_skill(prior: dict | None, incoming: dict) -> dict:
    if prior is None:
        payload = dict(incoming)
        payload["execution_stats"] = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "last_termination_reason": None,
        }
        payload["prior_stats"] = {"attempts": 0, "successes": 0, "failures": 0, "usefulness": 0.0, "confidence": 0.0}
        return payload
    payload = dict(prior)
    payload.update(incoming)
    payload["execution_stats"] = dict(prior.get("execution_stats", {}))
    payload["prior_stats"] = dict(prior.get("prior_stats", {}))
    return payload


def _persistent_skill_prior(skill_id: str, persistent_priors: dict[str, dict] | None) -> dict:
    persistent_priors = persistent_priors or {}
    row = dict(persistent_priors.get(skill_id, {}))
    return {
        "attempts": int(row.get("attempts", 0)),
        "successes": int(row.get("successes", 0)),
        "failures": int(row.get("failures", 0)),
        "usefulness": float(row.get("usefulness", 0.0) or row.get("usefulness_total", 0.0)),
        "confidence": float(row.get("confidence", 0.0) or row.get("confidence_total", 0.0)),
    }


def rebuild_skill_library(entities: dict[str, dict], triggers: dict[str, dict] | None = None, prior_library: dict[str, dict] | None = None, persistent_priors: dict[str, dict] | None = None) -> dict[str, dict]:
    triggers = triggers or {}
    prior_library = prior_library or {}
    library: dict[str, dict] = {}

    for entity_id, entity in entities.items():
        base = {
            "entity_id": entity_id,
            "target_area_id": entity.get("area_id"),
            "target_centroid": entity.get("centroid"),
            "confidence": float(entity.get("confidence", 0.0)),
            "utility": float(entity.get("utility", 0.0)),
            "novelty": float(entity.get("novelty", 0.0)),
        }
        inspect_id = f"skill:inspect:{stable_digest({'entity_id': entity_id})}"
        inspect_skill = {
            "skill_id": inspect_id,
            "skill_type": "inspect_target",
            "action": {"type": "inspect", "target": entity_id, "centroid": entity.get("centroid")},
            **base,
        }
        library[inspect_id] = _merge_skill(prior_library.get(inspect_id), inspect_skill)
        library[inspect_id]["prior_stats"] = _persistent_skill_prior(inspect_id, persistent_priors)
        library[inspect_id]["confidence"] = max(float(library[inspect_id].get("confidence", 0.0)), min(1.0, library[inspect_id]["prior_stats"]["confidence"]))
        library[inspect_id]["usefulness"] = max(float(library[inspect_id].get("utility", 0.0)), library[inspect_id]["prior_stats"]["usefulness"])

        if entity.get("reachable_later"):
            move_id = f"skill:recover:{stable_digest({'entity_id': entity_id, 'area': entity.get('area_id')})}"
            move_skill = {
                "skill_id": move_id,
                "skill_type": "recover_route",
                "action": {"type": "reposition", "target": entity_id, "centroid": entity.get("centroid")},
                **base,
            }
            library[move_id] = _merge_skill(prior_library.get(move_id), move_skill)
            library[move_id]["prior_stats"] = _persistent_skill_prior(move_id, persistent_priors)

    for trigger_id, trigger in triggers.items():
        target_entity_id = trigger.get("entity_id")
        interact_id = f"skill:trigger:{stable_digest({'entity_id': target_entity_id, 'centroid': trigger.get('centroid')})}"
        interact_skill = {
            "skill_id": interact_id,
            "skill_type": "trigger_probe",
            "entity_id": target_entity_id,
            "target_area_id": None,
            "target_centroid": trigger.get("centroid"),
            "confidence": float(trigger.get("confidence", 0.0)),
            "utility": 0.35,
            "novelty": 0.15,
            "action": {"type": "interact", "target": target_entity_id, "centroid": trigger.get("centroid")},
            "source_trigger_zone_id": trigger_id,
        }
        library[interact_id] = _merge_skill(prior_library.get(interact_id), interact_skill)
        library[interact_id]["prior_stats"] = _persistent_skill_prior(interact_id, persistent_priors)
    return library


def update_skill_execution_stats(skill_library: dict[str, dict], *, decision: dict | None, outcome: dict | None) -> dict[str, dict]:
    updated = {skill_id: dict(skill) for skill_id, skill in skill_library.items()}
    metadata = dict(decision.get("metadata", {})) if isinstance(decision, dict) else {}
    selected_raw = metadata.get("selected_candidate", {})
    selected = dict(selected_raw) if isinstance(selected_raw, dict) else {}
    selected_skill_id = selected.get("skill_id")
    target_entity_id = selected.get("target_entity_id")
    target_area_id = selected.get("target_area_id")
    action_family = _selected_action_family(selected)
    termination_reason = None
    success = False
    if isinstance(outcome, dict):
        success = bool(outcome.get("success") or outcome.get("outcome", {}).get("success"))
        termination_reason = outcome.get("termination_reason") or outcome.get("outcome", {}).get("termination_reason")
    matched_skill_id = None
    if selected_skill_id and selected_skill_id in updated:
        matched_skill_id = str(selected_skill_id)
    elif action_family == "move":
        for skill_id, skill in updated.items():
            if str(skill.get("skill_type") or "") != "recover_route":
                continue
            if target_entity_id and str(skill.get("entity_id") or "") == str(target_entity_id):
                matched_skill_id = str(skill_id)
                break
            if target_area_id and str(skill.get("target_area_id") or "") == str(target_area_id):
                matched_skill_id = str(skill_id)
                break
    elif action_family in {"interact", "click_at"}:
        preferred_skill_types = ("trigger_probe", "inspect_target") if selected.get("candidate_class") == "trigger_probe" else ("inspect_target", "trigger_probe")
        for skill_type in preferred_skill_types:
            for skill_id, skill in updated.items():
                if str(skill.get("skill_type") or "") != skill_type:
                    continue
                if target_entity_id and str(skill.get("entity_id") or "") == str(target_entity_id):
                    matched_skill_id = str(skill_id)
                    break
            if matched_skill_id is not None:
                break
    if matched_skill_id is None:
        return updated
    for skill_id, skill in updated.items():
        if str(skill_id) != matched_skill_id:
            continue
        stats = dict(skill.get("execution_stats", {}))
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        if success:
            stats["successes"] = int(stats.get("successes", 0)) + 1
        else:
            stats["failures"] = int(stats.get("failures", 0)) + 1
        stats["last_termination_reason"] = termination_reason
        skill["execution_stats"] = stats
    return updated

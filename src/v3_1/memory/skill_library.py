from __future__ import annotations

from v3_1.utils.ids import stable_digest


def _merge_skill(prior: dict | None, incoming: dict) -> dict:
    if prior is None:
        payload = dict(incoming)
        payload["execution_stats"] = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "last_termination_reason": None,
        }
        return payload
    payload = dict(prior)
    payload.update(incoming)
    payload["execution_stats"] = dict(prior.get("execution_stats", {}))
    return payload


def rebuild_skill_library(entities: dict[str, dict], triggers: dict[str, dict] | None = None, prior_library: dict[str, dict] | None = None) -> dict[str, dict]:
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

        if entity.get("reachable_later"):
            move_id = f"skill:recover:{stable_digest({'entity_id': entity_id, 'area': entity.get('area_id')})}"
            move_skill = {
                "skill_id": move_id,
                "skill_type": "recover_route",
                "action": {"type": "reposition", "target": entity_id, "centroid": entity.get("centroid")},
                **base,
            }
            library[move_id] = _merge_skill(prior_library.get(move_id), move_skill)

    for trigger_id, trigger in triggers.items():
        target_entity_id = trigger.get("entity_id")
        interact_id = f"skill:trigger:{stable_digest({'trigger_id': trigger_id, 'target': target_entity_id})}"
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
        }
        library[interact_id] = _merge_skill(prior_library.get(interact_id), interact_skill)
    return library


def update_skill_execution_stats(skill_library: dict[str, dict], *, decision: dict | None, outcome: dict | None) -> dict[str, dict]:
    updated = {skill_id: dict(skill) for skill_id, skill in skill_library.items()}
    selected = dict(decision.get("metadata", {}).get("selected_candidate", {})) if isinstance(decision, dict) else {}
    target_entity_id = selected.get("target_entity_id")
    action_type = selected.get("action", {}).get("type") if isinstance(selected.get("action"), dict) else None
    termination_reason = None
    success = False
    if isinstance(outcome, dict):
        success = bool(outcome.get("success") or outcome.get("outcome", {}).get("success"))
        termination_reason = outcome.get("termination_reason") or outcome.get("outcome", {}).get("termination_reason")
    for skill_id, skill in updated.items():
        if skill.get("entity_id") != target_entity_id:
            continue
        if action_type and skill.get("action", {}).get("type") != action_type:
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

from __future__ import annotations

from typing import Iterable

from v3_1.utils.ids import stable_digest


def _bbox_iou(lhs: dict | None, rhs: dict | None) -> float:
    if not isinstance(lhs, dict) or not isinstance(rhs, dict):
        return 0.0
    x1 = max(int(lhs["x1"]), int(rhs["x1"]))
    y1 = max(int(lhs["y1"]), int(rhs["y1"]))
    x2 = min(int(lhs["x2"]), int(rhs["x2"]))
    y2 = min(int(lhs["y2"]), int(rhs["y2"]))
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1 + 1) * (y2 - y1 + 1)
    lhs_area = (int(lhs["x2"]) - int(lhs["x1"]) + 1) * (int(lhs["y2"]) - int(lhs["y1"]) + 1)
    rhs_area = (int(rhs["x2"]) - int(rhs["x1"]) + 1) * (int(rhs["y2"]) - int(rhs["y1"]) + 1)
    denom = max(1, lhs_area + rhs_area - inter)
    return inter / float(denom)


def _centroid_distance(lhs: list | tuple | None, rhs: list | tuple | None) -> float:
    if not isinstance(lhs, (list, tuple)) or not isinstance(rhs, (list, tuple)) or len(lhs) != 2 or len(rhs) != 2:
        return 9999.0
    return abs(float(lhs[0]) - float(rhs[0])) + abs(float(lhs[1]) - float(rhs[1]))


def _match_score(existing: dict, incoming: dict) -> float:
    score = 0.0
    if existing.get("signature") and existing.get("signature") == incoming.get("signature"):
        score += 0.55
    if existing.get("canonical_descriptor") and existing.get("canonical_descriptor") == incoming.get("canonical_descriptor"):
        score += 0.2
    score += 0.25 * _bbox_iou(existing.get("bbox"), incoming.get("bbox"))
    score += max(0.0, 0.2 - (_centroid_distance(existing.get("centroid"), incoming.get("centroid")) / 20.0))
    if existing.get("kind") == incoming.get("kind"):
        score += 0.05
    return score


def _merge_bbox(lhs: dict | None, rhs: dict | None) -> dict | None:
    if not isinstance(lhs, dict):
        return rhs
    if not isinstance(rhs, dict):
        return lhs
    return {
        "x1": min(int(lhs["x1"]), int(rhs["x1"])),
        "y1": min(int(lhs["y1"]), int(rhs["y1"])),
        "x2": max(int(lhs["x2"]), int(rhs["x2"])),
        "y2": max(int(lhs["y2"]), int(rhs["y2"])),
    }


def _stable_entity_id(incoming: dict) -> str:
    basis = {
        "signature": incoming.get("signature"),
        "descriptor": incoming.get("canonical_descriptor") or incoming.get("stable_descriptor"),
        "kind": incoming.get("kind"),
        "primary_color": incoming.get("primary_color"),
    }
    return f"entity:{stable_digest(basis)}"


def merge_entities(existing: dict[str, dict], incoming: Iterable[dict]) -> dict[str, dict]:
    merged = {entity_id: dict(row) for entity_id, row in existing.items()}
    seen_this_merge: set[str] = set()

    for row in incoming:
        incoming_row = dict(row)
        match_id = None
        best_score = -1.0
        hinted_id = str(incoming_row.get("stable_entity_id_hint") or "")
        if hinted_id and hinted_id in merged:
            match_id = hinted_id
            best_score = 999.0
        for entity_id, candidate in merged.items():
            if match_id is not None and best_score >= 999.0:
                break
            score = _match_score(candidate, incoming_row)
            if score > best_score:
                best_score = score
                match_id = entity_id
        if match_id is None or best_score < 0.65:
            match_id = _stable_entity_id(incoming_row)
        prior = merged.get(match_id, {})
        history = list(prior.get("history", []))
        evidence = sorted(set(prior.get("evidence_refs", [])) | set(incoming_row.get("evidence_refs", [])))
        history.append(
            {
                "episode_id": incoming_row.get("episode_id"),
                "step_idx": incoming_row.get("step_idx"),
                "centroid": incoming_row.get("centroid"),
                "confidence": incoming_row.get("confidence"),
            }
        )
        payload = dict(prior)
        payload.update(incoming_row)
        payload["entity_id"] = match_id
        payload["stable_entity_id"] = match_id
        payload["bbox"] = _merge_bbox(prior.get("bbox"), incoming_row.get("bbox"))
        payload["centroid"] = incoming_row.get("centroid") or prior.get("centroid")
        payload["confidence"] = max(float(prior.get("confidence", 0.0)), float(incoming_row.get("confidence", 0.0)))
        payload["observations"] = int(prior.get("observations", 0)) + int(incoming_row.get("observations", 1))
        payload["first_seen_round"] = prior.get("first_seen_round", incoming_row.get("round_id"))
        payload["last_seen_round"] = incoming_row.get("round_id", prior.get("last_seen_round"))
        payload["first_seen_episode"] = prior.get("first_seen_episode", incoming_row.get("episode_id"))
        payload["last_seen_episode"] = incoming_row.get("episode_id", prior.get("last_seen_episode"))
        payload["history"] = history[-20:]
        payload["evidence_refs"] = evidence[-32:]
        payload["merge_matches"] = int(prior.get("merge_matches", 0)) + (1 if prior else 0)
        payload["lifecycle_state"] = "active"
        payload["movement_attempts"] = int(prior.get("movement_attempts", 0) or 0) + int(incoming_row.get("movement_attempts", 0) or 0)
        payload["interact_attempts"] = int(prior.get("interact_attempts", 0) or 0) + int(incoming_row.get("interact_attempts", 0) or 0)
        payload["click_attempts"] = int(prior.get("click_attempts", 0) or 0) + int(incoming_row.get("click_attempts", 0) or 0)
        payload["movement_effect_sum"] = int(prior.get("movement_effect_sum", 0) or 0) + int(incoming_row.get("movement_effect_sum", 0) or 0)
        payload["interact_effect_sum"] = int(prior.get("interact_effect_sum", 0) or 0) + int(incoming_row.get("interact_effect_sum", 0) or 0)
        payload["click_effect_sum"] = int(prior.get("click_effect_sum", 0) or 0) + int(incoming_row.get("click_effect_sum", 0) or 0)
        payload["movement_effect_score"] = max(float(prior.get("movement_effect_score", 0.0) or 0.0), float(incoming_row.get("movement_effect_score", 0.0) or 0.0))
        payload["interact_effect_score"] = max(float(prior.get("interact_effect_score", 0.0) or 0.0), float(incoming_row.get("interact_effect_score", 0.0) or 0.0))
        payload["click_effect_score"] = max(float(prior.get("click_effect_score", 0.0) or 0.0), float(incoming_row.get("click_effect_score", 0.0) or 0.0))
        if float(incoming_row.get("candidate_effect_score", 0.0) or 0.0) >= float(prior.get("candidate_effect_score", 0.0) or 0.0):
            payload["candidate_effect_mode"] = incoming_row.get("candidate_effect_mode", prior.get("candidate_effect_mode"))
        else:
            payload["candidate_effect_mode"] = prior.get("candidate_effect_mode", incoming_row.get("candidate_effect_mode"))
        payload["candidate_effect_score"] = max(float(prior.get("candidate_effect_score", 0.0) or 0.0), float(incoming_row.get("candidate_effect_score", 0.0) or 0.0))
        merged[match_id] = payload
        seen_this_merge.add(match_id)

    for entity_id, payload in merged.items():
        if entity_id in seen_this_merge:
            continue
        stale_steps = int(payload.get("stale_steps", 0)) + 1
        payload["stale_steps"] = stale_steps
        payload["lifecycle_state"] = "stale" if stale_steps >= 2 else payload.get("lifecycle_state", "active")
    return merged

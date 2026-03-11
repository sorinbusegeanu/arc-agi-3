from __future__ import annotations

from v3_1.utils.ids import stable_digest


def propose_trigger_zones(*, entities: dict[str, dict], consequences: dict[str, dict]) -> list[dict]:
    proposals = []
    for entity in entities.values():
        if not entity.get("reachable_now") and not entity.get("reachable_later"):
            continue
        if entity.get("kind") != "poi":
            continue
        related = [
            row for row in consequences.values()
            if row.get("action_effect_near_avatar") and row.get("local_change_area", 0) > 0
        ]
        confidence = min(1.0, 0.2 + (0.1 * len(related)) + (0.2 * float(entity.get("confidence", 0.0))))
        proposals.append(
            {
                "trigger_zone_id": f"trigger:{stable_digest({'entity_id': entity['entity_id'], 'centroid': entity.get('centroid')})}",
                "entity_id": entity["entity_id"],
                "centroid": entity.get("centroid"),
                "bbox": entity.get("bbox"),
                "confidence": confidence,
                "observations": len(related) + int(entity.get("observations", 1)),
                "evidence_refs": list(entity.get("evidence_refs", [])),
            }
        )
    return proposals


def merge_trigger_zones(existing: dict[str, dict], incoming: list[dict]) -> dict[str, dict]:
    merged = {zone_id: dict(row) for zone_id, row in existing.items()}
    for zone in incoming:
        zone_id = str(zone.get("trigger_zone_id") or zone.get("zone_id") or zone.get("id"))
        prior = merged.get(zone_id, {})
        payload = dict(prior)
        payload.update(zone)
        payload["trigger_zone_id"] = zone_id
        payload["observations"] = int(prior.get("observations", 0)) + int(zone.get("observations", 1))
        payload["confidence"] = min(1.0, max(float(prior.get("confidence", 0.0)), float(zone.get("confidence", 0.0))) + 0.05 * int(bool(prior)))
        payload["evidence_refs"] = sorted(set(prior.get("evidence_refs", [])) | set(zone.get("evidence_refs", [])))[-32:]
        merged[zone_id] = payload
    return merged

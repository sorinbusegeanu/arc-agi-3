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
                "support_count": len(related),
                "support_history": [
                    {
                        "support_count": len(related),
                        "confidence": confidence,
                        "evidence_count": len(entity.get("evidence_refs", [])),
                    }
                ],
                "contradiction_count": 0,
                "decay_steps": 0,
                "merge_lineage": [str(entity["entity_id"])],
                "confidence_history": [confidence],
                "evidence_refs": list(entity.get("evidence_refs", [])),
            }
        )
    return proposals


def merge_trigger_zones(existing: dict[str, dict], incoming: list[dict]) -> dict[str, dict]:
    merged = {zone_id: dict(row) for zone_id, row in existing.items()}
    incoming_ids = {str(zone.get("trigger_zone_id") or zone.get("zone_id") or zone.get("id")) for zone in incoming}
    for zone_id, row in merged.items():
        if zone_id in incoming_ids:
            continue
        row["decay_steps"] = int(row.get("decay_steps", 0)) + 1
        row["confidence"] = max(0.0, float(row.get("confidence", 0.0)) * 0.9)
    for zone in incoming:
        zone_id = str(zone.get("trigger_zone_id") or zone.get("zone_id") or zone.get("id"))
        prior = merged.get(zone_id, {})
        payload = dict(prior)
        payload.update(zone)
        payload["trigger_zone_id"] = zone_id
        payload["observations"] = int(prior.get("observations", 0)) + int(zone.get("observations", 1))
        payload["confidence"] = min(1.0, max(float(prior.get("confidence", 0.0)), float(zone.get("confidence", 0.0))) + 0.05 * int(bool(prior)))
        payload["support_count"] = int(prior.get("support_count", 0)) + int(zone.get("support_count", 0))
        payload["support_history"] = (list(prior.get("support_history", [])) + list(zone.get("support_history", [])))[-12:]
        payload["contradiction_count"] = int(prior.get("contradiction_count", 0))
        payload["decay_steps"] = 0
        payload["merge_lineage"] = (list(prior.get("merge_lineage", [])) + list(zone.get("merge_lineage", [])))[-12:]
        payload["confidence_history"] = (list(prior.get("confidence_history", [])) + [float(zone.get("confidence", 0.0))])[-12:]
        payload["evidence_refs"] = sorted(set(prior.get("evidence_refs", [])) | set(zone.get("evidence_refs", [])))[-32:]
        merged[zone_id] = payload
    return merged

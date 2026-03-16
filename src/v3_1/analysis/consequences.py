from __future__ import annotations


def trigger_contact_evidence(step_rows: list[dict]) -> list[dict]:
    rows = []
    for step in list(step_rows or []):
        if str(step.get("action_family") or "") != "interact":
            continue
        target_entity_id = step.get("target_entity_id")
        if not target_entity_id:
            continue
        rows.append(
            {
                "step_idx": int(step.get("step_idx", 0) or 0),
                "target_entity_id": str(target_entity_id),
                "evidence_ref": f"step:{int(step.get('step_idx', 0) or 0)}",
                "changed_cells": int(step.get("changed_cells", 0) or 0),
            }
        )
    return rows


def remote_region_change_evidence(step_rows: list[dict]) -> list[dict]:
    rows = []
    for step in list(step_rows or []):
        if int(step.get("changed_cells", 0) or 0) <= 0:
            continue
        telemetry = dict(step.get("telemetry", {}) or {})
        effect_region = dict(telemetry.get("effect_region", {}) or {})
        bbox = effect_region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        rows.append(
            {
                "step_idx": int(step.get("step_idx", 0) or 0),
                "bbox": list(bbox),
                "evidence_ref": f"step:{int(step.get('step_idx', 0) or 0)}",
                "changed_cells": int(step.get("changed_cells", 0) or 0),
            }
        )
    return rows


def delayed_change_evidence(step_rows: list[dict], *, lag: int = 1) -> list[dict]:
    rows = []
    ordered = list(step_rows or [])
    for index, step in enumerate(ordered):
        if str(step.get("action_family") or "") not in {"interact", "click_at"}:
            continue
        later = ordered[index + int(lag)] if index + int(lag) < len(ordered) else None
        if later is None or int(later.get("changed_cells", 0) or 0) <= 0:
            continue
        rows.append(
            {
                "cause_step_idx": int(step.get("step_idx", 0) or 0),
                "effect_step_idx": int(later.get("step_idx", 0) or 0),
                "target_entity_id": step.get("target_entity_id"),
                "evidence_ref": f"step:{int(step.get('step_idx', 0) or 0)}",
            }
        )
    return rows


def repeated_contact_to_change_support(step_rows: list[dict]) -> list[dict]:
    counts: dict[str, dict] = {}
    for row in trigger_contact_evidence(step_rows):
        key = str(row.get("target_entity_id") or "")
        if not key:
            continue
        bucket = counts.setdefault(key, {"target_entity_id": key, "support_count": 0, "changed_support_count": 0, "evidence_refs": []})
        bucket["support_count"] += 1
        bucket["changed_support_count"] += 1 if int(row.get("changed_cells", 0) or 0) > 0 else 0
        bucket["evidence_refs"].append(str(row.get("evidence_ref") or ""))
    return [value for value in counts.values() if int(value.get("support_count", 0) or 0) > 0]

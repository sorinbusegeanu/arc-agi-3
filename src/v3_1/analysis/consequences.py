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


def bounded_lag_change_support(step_rows: list[dict], *, max_lag: int = 3) -> list[dict]:
    rows = []
    for lag in range(1, max(1, int(max_lag)) + 1):
        for row in delayed_change_evidence(step_rows, lag=lag):
            rows.append({**row, "lag_steps": lag})
    return rows


def local_vs_remote_change_separation(step_rows: list[dict]) -> dict:
    local_count = 0
    remote_count = 0
    for row in list(step_rows or []):
        telemetry = dict(row.get("telemetry", {}) or {})
        if bool(telemetry.get("action_effect_near_avatar")):
            local_count += 1
        if isinstance(telemetry.get("effect_region"), dict):
            remote_count += 1
    return {"local_change_count": local_count, "remote_change_count": remote_count}


def repeated_effect_region_support(step_rows: list[dict]) -> list[dict]:
    counts: dict[str, dict] = {}
    for row in remote_region_change_evidence(step_rows):
        bbox = list(row.get("bbox", []) or [])
        key = ":".join(str(value) for value in bbox)
        bucket = counts.setdefault(key, {"bbox": bbox, "support_count": 0, "evidence_refs": []})
        bucket["support_count"] += 1
        bucket["evidence_refs"].append(str(row.get("evidence_ref") or ""))
    return [row for row in counts.values() if int(row.get("support_count", 0) or 0) > 0]


def directed_outcome_backed_consequence_support(step_rows: list[dict]) -> list[dict]:
    rows = []
    for row in list(step_rows or []):
        if str(row.get("analysis_mode") or "") != "directed_outcome":
            continue
        changed_cells = int(row.get("changed_cells", 0) or 0)
        if changed_cells <= 0:
            continue
        rows.append(
            {
                "step_idx": int(row.get("step_idx", 0) or 0),
                "target_entity_id": row.get("target_entity_id"),
                "changed_cells": changed_cells,
                "support_ref": f"step:{int(row.get('step_idx', 0) or 0)}",
            }
        )
    return rows


def bounded_lag_change_support(step_rows: list[dict], *, max_lag: int = 3) -> list[dict]:
    evidence = []
    for lag in range(1, max(1, int(max_lag)) + 1):
        for row in delayed_change_evidence(step_rows, lag=lag):
            evidence.append({**row, "lag_steps": int(lag)})
    return evidence


def local_vs_remote_change_separation(step_rows: list[dict]) -> dict[str, list[dict]]:
    local_rows = []
    remote_rows = []
    for row in list(step_rows or []):
        payload = dict(row)
        telemetry = dict(payload.get("telemetry", {}) or {})
        near_avatar = bool(payload.get("action_effect_near_avatar") or telemetry.get("action_effect_near_avatar"))
        if int(payload.get("changed_cells", 0) or 0) <= 0:
            continue
        if near_avatar:
            local_rows.append(payload)
        else:
            remote_rows.append(payload)
    return {"local": local_rows, "remote": remote_rows}


def repeated_effect_region_support(step_rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in remote_region_change_evidence(step_rows):
        bbox = list(row.get("bbox", []) or [])
        key = ",".join(str(value) for value in bbox)
        bucket = grouped.setdefault(key, {"bbox": bbox, "support_count": 0, "changed_cells_sum": 0, "evidence_refs": []})
        bucket["support_count"] += 1
        bucket["changed_cells_sum"] += int(row.get("changed_cells", 0) or 0)
        bucket["evidence_refs"].append(str(row.get("evidence_ref") or ""))
    return [row for row in grouped.values() if int(row.get("support_count", 0) or 0) > 0]


def directed_outcome_backed_consequence_support(step_rows: list[dict], *, analysis_mode: str | None = None) -> list[dict]:
    if str(analysis_mode or "") != "directed_outcome":
        return []
    support = []
    for row in list(step_rows or []):
        payload = dict(row)
        if int(payload.get("changed_cells", 0) or 0) <= 0:
            continue
        if not payload.get("target_entity_id"):
            continue
        support.append(
            {
                "target_entity_id": str(payload.get("target_entity_id")),
                "changed_cells": int(payload.get("changed_cells", 0) or 0),
                "step_idx": int(payload.get("step_idx", 0) or 0),
                "evidence_ref": f"step:{int(payload.get('step_idx', 0) or 0)}",
                "analysis_mode": "directed_outcome",
            }
        )
    return support

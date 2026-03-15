from __future__ import annotations

from v3_1.utils.ids import stable_digest


def consequence_key(row: dict) -> str:
    basis = {
        "step_idx": row.get("step_idx"),
        "action": row.get("action"),
        "state_hash_before": row.get("state_hash_before"),
        "state_hash_after": row.get("state_hash_after"),
        "change_signature": row.get("change_signature"),
    }
    return f"consequence:{stable_digest(basis)}"


def normalized_consequence_action_key(consequence: dict) -> str:
    action_family = str(consequence.get("action_family") or "").strip().lower()
    if action_family:
        return action_family
    action_name = str(consequence.get("action_name") or "").strip().lower()
    if action_name:
        return action_name
    action_value = consequence.get("action")
    if isinstance(action_value, str):
        return action_value.strip().lower()
    return "unknown"


def extract_consequence_records(delta: dict) -> list[dict]:
    records = []
    metadata = dict(delta.get("metadata", {}))
    step_rows = list(metadata.get("step_rows", []))
    consequences = list(delta.get("consequences", ()))
    by_step = {row.get("step_idx"): row for row in step_rows}
    for row in consequences:
        step_idx = row.get("step_idx")
        step = by_step.get(step_idx, {})
        evidence_refs = list(row.get("evidence_refs", [])) or [f"{delta.get('episode_id')}:{step_idx}"] if step_idx is not None else []
        records.append(
            {
                "consequence_id": row.get("consequence_id") or consequence_key(row),
                "step_idx": step_idx,
                "action": step.get("action_name") or row.get("action"),
                "action_id": step.get("action_id", row.get("action_id")),
                "action_name": step.get("action_name", row.get("action_name")),
                "action_family": step.get("action_family", row.get("action_family", "unknown")),
                "state_hash_before": step_rows[step_idx - 1]["state_hash"] if isinstance(step_idx, int) and step_idx > 0 and step_idx - 1 < len(step_rows) else None,
                "state_hash_after": step.get("state_hash"),
                "change_signature": stable_digest(
                    {
                        "local_change_area": row.get("local_change_area"),
                        "blocked": row.get("blocked"),
                        "reward": row.get("reward"),
                        "done": row.get("done"),
                    }
                ),
                "reward": row.get("reward"),
                "done": row.get("done"),
                "blocked": row.get("blocked"),
                "local_change_area": row.get("local_change_area", 0),
                "action_effect_near_avatar": row.get("action_effect_near_avatar", False),
                "evidence_count": row.get("evidence_count", 1),
                "evidence_refs": evidence_refs,
            }
        )
    return records


def merge_consequences(existing: dict[str, dict], incoming: list[dict]) -> dict[str, dict]:
    merged = {consequence_id: dict(row) for consequence_id, row in existing.items()}
    for row in incoming:
        consequence_id = str(row.get("consequence_id") or consequence_key(row))
        prior = merged.get(consequence_id, {})
        payload = dict(prior)
        payload.update(row)
        payload["consequence_id"] = consequence_id
        payload["evidence_count"] = int(prior.get("evidence_count", 0)) + int(row.get("evidence_count", 1))
        payload["evidence_refs"] = sorted(set(prior.get("evidence_refs", [])) | set(row.get("evidence_refs", [])))[-32:]
        payload["occurrence_count"] = int(prior.get("occurrence_count", 0)) + 1
        payload["mean_local_change_area"] = (
            float(prior.get("mean_local_change_area", row.get("local_change_area", 0.0))) * max(0, int(prior.get("occurrence_count", 0)))
            + float(row.get("local_change_area", 0.0))
        ) / float(max(1, int(prior.get("occurrence_count", 0)) + 1))
        merged[consequence_id] = payload
    return merged

from __future__ import annotations

from typing import Iterable

from v3_1.utils.ids import stable_digest


def area_signature(summary: dict) -> str:
    basis = {
        "area_signature": summary.get("area_signature"),
        "state_hash": summary.get("state_hash"),
        "background_color": summary.get("background_color"),
        "palette": tuple(sorted(int(value) for value in summary.get("palette", []))),
        "width": int(summary.get("width", 0)),
        "height": int(summary.get("height", 0)),
    }
    return stable_digest(basis)


def _match_score(existing: dict, incoming: dict) -> float:
    score = 0.0
    if existing.get("state_hash") and existing.get("state_hash") == incoming.get("state_hash"):
        score += 0.55
    if existing.get("area_signature") and existing.get("area_signature") == incoming.get("area_signature"):
        score += 0.25
    lhs_palette = set(existing.get("palette", []))
    rhs_palette = set(incoming.get("palette", []))
    score += 0.1 * (len(lhs_palette & rhs_palette) / float(max(1, len(lhs_palette | rhs_palette))))
    if existing.get("background_color") == incoming.get("background_color"):
        score += 0.1
    return score


def merge_areas(existing: dict[str, dict], incoming: Iterable[dict]) -> dict[str, dict]:
    merged = {area_id: dict(row) for area_id, row in existing.items()}
    for area in incoming:
        incoming_row = dict(area)
        best_id = None
        best_score = -1.0
        for area_id, candidate in merged.items():
            score = _match_score(candidate, incoming_row)
            if score > best_score:
                best_id = area_id
                best_score = score
        if best_id is None or best_score < 0.7:
            best_id = str(incoming_row.get("area_id") or f"area:{area_signature(incoming_row)}")
        prior = merged.get(best_id, {})
        topology_cells = sorted(set(tuple(cell) for cell in prior.get("topology_cells", [])) | set(tuple(cell) for cell in incoming_row.get("topology_cells", [])))
        payload = dict(prior)
        payload.update(incoming_row)
        payload["area_id"] = best_id
        payload["stable_area_id"] = best_id
        payload["area_signature"] = incoming_row.get("area_signature") or prior.get("area_signature") or area_signature(incoming_row)
        payload["visit_count"] = int(prior.get("visit_count", 0)) + int(incoming_row.get("visit_count", 1))
        payload["first_seen_round"] = prior.get("first_seen_round", incoming_row.get("round_id"))
        payload["last_seen_round"] = incoming_row.get("round_id", prior.get("last_seen_round"))
        payload["background_color"] = incoming_row.get("background_color", prior.get("background_color"))
        payload["palette"] = sorted(set(prior.get("palette", [])) | set(incoming_row.get("palette", [])))
        payload["topology_cells"] = [list(cell) for cell in topology_cells]
        payload["entry_edges"] = int(prior.get("entry_edges", 0)) + int(incoming_row.get("entry_edges", 0))
        payload["exit_edges"] = int(prior.get("exit_edges", 0)) + int(incoming_row.get("exit_edges", 0))
        merged[best_id] = payload
    return merged

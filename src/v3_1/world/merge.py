from __future__ import annotations

from v3_1.world.areas import merge_areas
from v3_1.world.consequences import extract_consequence_records, merge_consequences, normalized_consequence_action_key
from v3_1.world.entities import merge_entities
from v3_1.world.indexes import build_indexes
from v3_1.world.reachability import reachable_entities
from v3_1.world.topology import merge_topology
from v3_1.world.trigger_zones import merge_trigger_zones, propose_trigger_zones


def _ensure_state(state: dict) -> dict:
    payload = dict(state)
    payload.setdefault("areas", {})
    payload.setdefault("entities", {})
    payload.setdefault("consequences", {})
    payload.setdefault("trigger_zones", {})
    payload.setdefault("topology_nodes", {})
    payload.setdefault("topology_edges", {})
    payload.setdefault("observed_entities", {})
    payload.setdefault("hypothesized_entities", {})
    payload.setdefault("observed_consequences", {})
    payload.setdefault("hypothesized_consequences", {})
    payload.setdefault("observed_trigger_zones", {})
    payload.setdefault("hypothesized_trigger_zones", {})
    payload.setdefault("observed_topology", {"nodes": {}, "edges": {}})
    payload.setdefault("hypothesized_topology", {"nodes": {}, "edges": {}})
    payload.setdefault("indexes", {})
    payload.setdefault("split_indexes", {"observed": {}, "hypothesized": {}})
    return payload


def _merge_area_topology_metadata(areas: dict[str, dict], topology_nodes: dict[str, dict]) -> dict[str, dict]:
    updated = {area_id: dict(area) for area_id, area in areas.items()}
    for area_id, area in updated.items():
        cells = [tuple(cell) for cell in area.get("topology_cells", [])]
        existing_cells = set(cells)
        for node in topology_nodes.values():
            if node.get("area_id") == area_id and tuple(node.get("cell", ())) not in existing_cells:
                cells.append(tuple(node["cell"]))
        area["topology_cells"] = [list(cell) for cell in sorted(set(cells))]
    return updated


def _consequence_transport_complete(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    if bool(row.get("supports_exit_attempt_relation", False)) or bool(row.get("supports_counterfactual_relation", False)) or str(row.get("support_family") or "") in {"exit_attempt", "counterfactual"}:
        return True
    action_name = str(row.get("action_name") or "").strip()
    action_family = str(row.get("action_family") or "").strip()
    evidence_refs = list(row.get("evidence_refs", []))
    return bool(action_name and action_family and evidence_refs)


def _normalize_consequence_rows(raw_rows: list[dict], delta: dict) -> tuple[list[dict], dict]:
    metadata = dict(delta.get("metadata", {}) or {})
    step_rows = list(metadata.get("step_rows", []) or [])
    by_step = {int(dict(row).get("step_idx", -1) or -1): dict(row) for row in step_rows if isinstance(row, dict)}
    normalized: list[dict] = []
    diagnostics = {
        "exit_attempt_family_markers_present_on_raw_delta_count": 0,
        "exit_attempt_family_markers_present_after_normalization_count": 0,
        "exit_attempt_family_markers_lost_in_normalization_count": 0,
        "counterfactual_family_markers_present_on_raw_delta_count": 0,
        "counterfactual_family_markers_present_after_normalization_count": 0,
        "counterfactual_family_markers_lost_in_normalization_count": 0,
        "normalization_drop_reason_codes": [],
    }
    allowed_keys = {
        "consequence_id",
        "step_idx",
        "action",
        "action_id",
        "action_name",
        "action_family",
        "state_hash_before",
        "state_hash_after",
        "change_signature",
        "reward",
        "done",
        "blocked",
        "local_change_area",
        "action_effect_near_avatar",
        "evidence_count",
        "evidence_refs",
        "support_family",
        "supports_exit_attempt_relation",
        "exit_attempt_support_count",
        "supports_counterfactual_relation",
        "counterfactual_support_count",
        "supports_directed_outcome_relation",
        "directed_outcome_support_count",
        "source_stage",
        "source_pass_id",
        "source_episode_id",
        "source_round_id",
        "analysis_mode",
        "confidence",
        "inference_method",
        "factual_observation",
        "direct_evidence_present",
        "direct_evidence_fields",
        "observation_support_span",
        "analysis_objective",
        "last_supported_round_by_family",
        "last_supported_pass_id_by_family",
        "target_entity_id",
        "area_id",
        "telemetry",
    }
    for row in list(raw_rows or []):
        payload = dict(row or {})
        family = str(payload.get("support_family") or "")
        is_exit = bool(payload.get("supports_exit_attempt_relation", False) or family == "exit_attempt")
        is_cf = bool(payload.get("supports_counterfactual_relation", False) or family == "counterfactual")
        if is_exit:
            diagnostics["exit_attempt_family_markers_present_on_raw_delta_count"] += 1
        if is_cf:
            diagnostics["counterfactual_family_markers_present_on_raw_delta_count"] += 1
        step_idx = int(payload.get("step_idx", -1) or -1)
        step = dict(by_step.get(step_idx, {}) or {})
        normalized_row = {key: payload.get(key) for key in allowed_keys if key in payload}
        normalized_row["step_idx"] = step_idx if step_idx >= 0 else None
        normalized_row["action_name"] = str(normalized_row.get("action_name") or step.get("action_name") or step.get("action_type") or payload.get("action") or "")
        normalized_row["action"] = normalized_row.get("action") or normalized_row["action_name"]
        normalized_row["action_family"] = str(normalized_row.get("action_family") or step.get("action_family") or "unknown")
        evidence_refs = list(normalized_row.get("evidence_refs", []) or [])
        if not evidence_refs and step_idx >= 0:
            evidence_refs = [f"{delta.get('episode_id')}:{step_idx}"]
        normalized_row["evidence_refs"] = evidence_refs
        normalized_row["consequence_id"] = str(normalized_row.get("consequence_id") or "") or ""
        if family in {"exit_attempt", "counterfactual"} and not normalized_row["consequence_id"]:
            normalized_row["consequence_id"] = f"{family}:{step_idx if step_idx >= 0 else 'unknown'}"
        normalized_row["source_stage"] = str(normalized_row.get("source_stage") or "analysis")
        normalized_row["source_pass_id"] = int(normalized_row.get("source_pass_id", delta.get("pass_id", 0)) or 0)
        normalized_row["source_episode_id"] = str(normalized_row.get("source_episode_id") or delta.get("episode_id") or "")
        normalized_row["analysis_mode"] = str(normalized_row.get("analysis_mode") or metadata.get("analysis_mode") or "")
        if family == "exit_attempt":
            normalized_row["support_family"] = "exit_attempt"
            normalized_row["supports_exit_attempt_relation"] = True
            normalized_row["exit_attempt_support_count"] = int(normalized_row.get("exit_attempt_support_count", 0) or 0) or 1
        if family == "counterfactual":
            normalized_row["support_family"] = "counterfactual"
            normalized_row["supports_counterfactual_relation"] = True
            normalized_row["counterfactual_support_count"] = int(normalized_row.get("counterfactual_support_count", 0) or 0) or 1
        if family in {"exit_attempt", "counterfactual"} and not normalized_row.get("evidence_refs"):
            diagnostics["normalization_drop_reason_codes"].append("normalized_row_missing_store_key_basis")
            continue
        if family in {"exit_attempt", "counterfactual"} and not (
            normalized_row.get("action_name")
            or normalized_row.get("target_entity_id")
            or normalized_row.get("area_id")
            or dict(normalized_row.get("telemetry", {}) or {}).get("attempted_escape_direction")
            or dict(normalized_row.get("telemetry", {}) or {}).get("expected_target_id")
        ):
            diagnostics["normalization_drop_reason_codes"].append("normalized_row_missing_relation_signature")
            continue
        if is_exit and not (normalized_row.get("support_family") == "exit_attempt" or normalized_row.get("supports_exit_attempt_relation") is True):
            diagnostics["normalization_drop_reason_codes"].append("family_fields_not_in_allowed_keys")
        if is_cf and not (normalized_row.get("support_family") == "counterfactual" or normalized_row.get("supports_counterfactual_relation") is True):
            diagnostics["normalization_drop_reason_codes"].append("family_fields_not_in_allowed_keys")
        if is_exit and (normalized_row.get("support_family") == "exit_attempt" or normalized_row.get("supports_exit_attempt_relation") is True):
            diagnostics["exit_attempt_family_markers_present_after_normalization_count"] += 1
        if is_cf and (normalized_row.get("support_family") == "counterfactual" or normalized_row.get("supports_counterfactual_relation") is True):
            diagnostics["counterfactual_family_markers_present_after_normalization_count"] += 1
        normalized.append(normalized_row)
    diagnostics["exit_attempt_family_markers_lost_in_normalization_count"] = max(
        0,
        int(diagnostics["exit_attempt_family_markers_present_on_raw_delta_count"])
        - int(diagnostics["exit_attempt_family_markers_present_after_normalization_count"]),
    )
    diagnostics["counterfactual_family_markers_lost_in_normalization_count"] = max(
        0,
        int(diagnostics["counterfactual_family_markers_present_on_raw_delta_count"])
        - int(diagnostics["counterfactual_family_markers_present_after_normalization_count"]),
    )
    return normalized, diagnostics


def _has_direct_fields(row: dict, required_fields: set[str]) -> bool:
    for field in required_fields:
        value = row.get(field)
        if value is None:
            return False
        if isinstance(value, str) and value == "":
            return False
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            return False
    return True


def _base_observed_checks(kind: str, row: dict, *, required_fields: set[str], allowed_objectives: set[str] | None = None) -> tuple[bool, str]:
    if not bool(row.get("factual_observation")):
        return False, "missing_factual_observation"
    if not bool(row.get("direct_evidence_present")):
        return False, "missing_direct_evidence"
    if bool(row.get("contradiction_flag")):
        return False, "contradiction_flag_present"
    if not _has_direct_fields(row, required_fields):
        return False, "missing_required_direct_fields"
    if allowed_objectives is not None and str(row.get("analysis_objective") or "") not in allowed_objectives:
        return False, "analysis_objective_not_admissible"
    if str(row.get("source_stage") or "analysis") not in {"analysis"}:
        return False, "source_stage_not_admissible"
    return True, "base_checks_passed"


def _validate_observed_area(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("areas", row, required_fields={"area_id", "area_signature", "state_hash"})
    if not ok:
        return ok, reason
    return (bool(row.get("width")) and bool(row.get("height"))), ("area_dimensions_present" if bool(row.get("width")) and bool(row.get("height")) else "missing_area_dimensions")


def _validate_observed_entity(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("entities", row, required_fields={"entity_id"}, allowed_objectives={"discovery", "terminal_attribution"})
    if not ok:
        return ok, reason
    evidence_fields = set(str(field) for field in list(row.get("direct_evidence_fields", []) or []))
    valid = bool({"bbox", "centroid"} & evidence_fields or row.get("bbox") or row.get("centroid"))
    return valid, ("entity_localization_present" if valid else "missing_entity_localization")


def _validate_observed_consequence(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks(
        "consequences",
        row,
        required_fields={"consequence_id", "action_name"},
        allowed_objectives={"terminal_attribution", "broad_trigger_suspicion", "discovery"},
    )
    if not ok:
        return ok, reason
    valid = bool(row.get("reward") not in {None, ""} or row.get("done") is not None or int(row.get("local_change_area", 0) or 0) > 0)
    return valid, ("consequence_effect_present" if valid else "missing_consequence_effect")


def _validate_observed_trigger_zone(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("trigger_zones", row, required_fields={"trigger_id"}, allowed_objectives={"broad_trigger_suspicion", "route_progress_attribution", "terminal_attribution"})
    if not ok:
        return ok, reason
    evidence_fields = set(str(field) for field in list(row.get("direct_evidence_fields", []) or []))
    valid = bool({"effect_region", "supporting_steps", "zone_bbox"} & evidence_fields or row.get("zone_bbox") or row.get("supporting_steps"))
    return valid, ("trigger_zone_local_support_present" if valid else "missing_trigger_zone_local_support")


def _validate_observed_topology_node(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("topology_nodes", row, required_fields={"node_id"}, allowed_objectives={"discovery", "route_progress_attribution"})
    if not ok:
        return ok, reason
    return (bool(row.get("cell"))), ("topology_node_cell_present" if bool(row.get("cell")) else "missing_topology_node_cell")


def _validate_observed_topology_edge(row: dict) -> tuple[bool, str]:
    ok, reason = _base_observed_checks("topology_edges", row, required_fields={"edge_id"}, allowed_objectives={"discovery", "route_progress_attribution"})
    if not ok:
        return ok, reason
    valid = bool(row.get("src") and row.get("dst") and row.get("action_key"))
    return valid, ("topology_edge_transition_present" if valid else "missing_topology_edge_transition")


OBSERVED_VALIDATORS = {
    "areas": ("validate_observed_area", _validate_observed_area),
    "entities": ("validate_observed_entity", _validate_observed_entity),
    "consequences": ("validate_observed_consequence", _validate_observed_consequence),
    "trigger_zones": ("validate_observed_trigger_zone", _validate_observed_trigger_zone),
    "topology_nodes": ("validate_observed_topology_node", _validate_observed_topology_node),
    "topology_edges": ("validate_observed_topology_edge", _validate_observed_topology_edge),
}


def _row_key(kind: str, row: dict) -> str:
    if kind == "entities":
        return str(row.get("entity_id") or row.get("poi_id") or row.get("signature") or "")
    if kind == "consequences":
        return str(row.get("consequence_id") or "")
    if kind == "trigger_zones":
        return str(row.get("trigger_id") or row.get("trigger_zone_id") or "")
    if kind == "topology_nodes":
        return str(row.get("node_id") or "")
    if kind == "topology_edges":
        return str(row.get("edge_id") or "")
    if kind == "areas":
        return str(row.get("area_id") or "")
    return ""


def _classify_row(kind: str, row: dict, delta: dict) -> dict:
    payload = dict(row)
    inference_method = str(payload.get("inference_method") or "").strip().lower()
    validator_name, validator = OBSERVED_VALIDATORS.get(kind, ("validate_observed_default_reject", lambda _row: (False, "row_kind_not_observed_eligible")))
    validator_ok, validator_result = validator(payload)
    if validator_ok:
        evidence_tier = "observed"
        observed_admission_reason = "direct_fact_allowlisted"
    else:
        evidence_tier = "hypothesized"
        if inference_method == "direct_observation":
            observed_admission_reason = "rejected_direct_observation_not_sufficient"
        else:
            observed_admission_reason = f"rejected_{validator_result}"
    payload["evidence_tier"] = evidence_tier
    payload["observed_admission_reason"] = observed_admission_reason
    payload["observed_validator_name"] = validator_name
    payload["observed_validator_result"] = validator_result
    payload["source_stage"] = str(payload.get("source_stage") or "analysis")
    payload["source_pass_id"] = int(payload.get("source_pass_id", delta.get("pass_id", 0)) or 0)
    payload["source_episode_id"] = str(payload.get("source_episode_id") or delta.get("episode_id") or "")
    payload["inference_method"] = str(payload.get("inference_method") or ("direct_observation" if evidence_tier == "observed" else f"{kind}_inference"))
    payload["confidence"] = float(payload.get("confidence", 1.0 if evidence_tier == "observed" else 0.5) or 0.0)
    return payload


def _combine_rows(*stores: dict[str, dict]) -> dict[str, dict]:
    combined: dict[str, dict] = {}
    for store in stores:
        for row_id, row in store.items():
            combined[str(row_id)] = dict(row)
    return combined


def _accumulate_support_family_diagnostics(existing: dict, incoming: dict) -> dict:
    merged = dict(existing or {})
    for key, value in dict(incoming or {}).items():
        if isinstance(value, int):
            merged[key] = int(merged.get(key, 0) or 0) + value
        elif isinstance(value, list):
            current = list(merged.get(key, []) or [])
            for item in value:
                if item not in current:
                    current.append(item)
            merged[key] = current
        else:
            merged[key] = value
    return merged


def _fill_family_fields_from_sources(combined: dict[str, dict], *source_stores: dict[str, dict]) -> tuple[dict[str, dict], int]:
    family_fields = {
        "supports_exit_attempt_relation",
        "exit_attempt_support_count",
        "supports_counterfactual_relation",
        "counterfactual_support_count",
        "supports_directed_outcome_relation",
        "directed_outcome_support_count",
        "support_family",
        "last_supported_round_by_family",
        "last_supported_pass_id_by_family",
    }
    rich_sources: dict[str, dict] = {}
    for store in source_stores:
        for row_id, row in dict(store or {}).items():
            payload = dict(row or {})
            if any(field in payload for field in family_fields):
                current = dict(rich_sources.get(str(row_id), {}) or {})
                current.update({field: payload.get(field) for field in family_fields if field in payload})
                rich_sources[str(row_id)] = current
    fillthrough_applied = 0
    next_rows: dict[str, dict] = {}
    for row_id, row in dict(combined or {}).items():
        payload = dict(row or {})
        source = dict(rich_sources.get(str(row_id), {}) or {})
        if source:
            before = dict(payload)
            for field in family_fields:
                if field not in payload and field in source:
                    payload[field] = source[field]
            if payload != before:
                fillthrough_applied += 1
        next_rows[str(row_id)] = payload
    return next_rows, fillthrough_applied


def _fill_entity_identity_fields_from_sources(combined: dict[str, dict], *source_stores: dict[str, dict]) -> tuple[dict[str, dict], int]:
    identity_fields = {
        "identity_status",
        "identity_support_count",
        "identity_contradiction_count",
        "identity_cross_round_stability",
        "identity_last_confirmed_round",
        "identity_match_provenance",
        "identity_aliases",
    }
    rich_sources: dict[str, dict] = {}
    for store in source_stores:
        for row_id, row in dict(store or {}).items():
            payload = dict(row or {})
            if any(field in payload for field in identity_fields):
                current = dict(rich_sources.get(str(row_id), {}) or {})
                rich_sources[str(row_id)] = _merge_entity_identity_fields(current, payload)
    fillthrough_applied = 0
    next_rows: dict[str, dict] = {}
    for row_id, row in dict(combined or {}).items():
        payload = dict(row or {})
        source = dict(rich_sources.get(str(row_id), {}) or {})
        if source and any(field not in payload for field in identity_fields):
            before_missing = any(field not in payload for field in identity_fields)
            payload = _merge_entity_identity_fields(source, payload)
            if before_missing:
                fillthrough_applied += 1
        next_rows[str(row_id)] = payload
    return next_rows, fillthrough_applied


def _stable_edge_identity(row: dict) -> str:
    payload = dict(row or {})
    src = str(payload.get("src") or payload.get("source_node_id") or payload.get("source_entity_id") or payload.get("source_area_id") or "")
    dst = str(payload.get("dst") or payload.get("target_node_id") or payload.get("target_entity_id") or payload.get("target_area_id") or "")
    relation = str(payload.get("edge_kind") or payload.get("relation_type") or payload.get("transition_type") or payload.get("action_key") or "")
    directionality = str(payload.get("directionality") or ("directed" if src or dst else "unknown"))
    return "|".join((src, dst, relation, directionality))


def _stronger_identity_status(left: str, right: str) -> str:
    ranking = {
        "unknown": 0,
        "new_entity": 1,
        "ambiguous_match": 2,
        "split_candidate": 2,
        "merge_candidate": 2,
        "probable": 3,
        "match_existing": 4,
        "confirmed": 5,
    }
    return left if ranking.get(str(left or "unknown"), 0) >= ranking.get(str(right or "unknown"), 0) else right


def _merge_entity_identity_fields(existing_row: dict, incoming_row: dict) -> dict:
    existing = dict(existing_row or {})
    incoming = dict(incoming_row or {})
    if not any(key in incoming for key in {
        "identity_status",
        "identity_support_count",
        "identity_contradiction_count",
        "identity_cross_round_stability",
        "identity_last_confirmed_round",
        "identity_match_provenance",
        "identity_aliases",
    }):
        return existing
    existing["identity_status"] = _stronger_identity_status(
        str(existing.get("identity_status") or "unknown"),
        str(incoming.get("identity_status") or "unknown"),
    )
    existing["identity_support_count"] = int(existing.get("identity_support_count", 0) or 0) + int(incoming.get("identity_support_count", 0) or 0)
    existing["identity_contradiction_count"] = int(existing.get("identity_contradiction_count", 0) or 0) + int(incoming.get("identity_contradiction_count", 0) or 0)
    existing["identity_cross_round_stability"] = max(
        int(existing.get("identity_cross_round_stability", 0) or 0),
        int(incoming.get("identity_cross_round_stability", 0) or 0),
    )
    existing["identity_last_confirmed_round"] = max(
        int(existing.get("identity_last_confirmed_round", 0) or 0),
        int(incoming.get("identity_last_confirmed_round", 0) or 0),
    )
    existing["identity_match_provenance"] = sorted(
        set(list(existing.get("identity_match_provenance", []) or []) + list(incoming.get("identity_match_provenance", []) or []))
    )
    existing["identity_aliases"] = sorted(
        set(list(existing.get("identity_aliases", []) or []) + list(incoming.get("identity_aliases", []) or []))
    )
    return existing


def _merge_topology_edge_support(existing_row: dict, incoming_row: dict) -> dict:
    existing = dict(existing_row or {})
    incoming = dict(incoming_row or {})
    merged = dict(existing)
    merged.update({k: v for k, v in incoming.items() if k not in {
        "display_support_count",
        "match_support_count",
        "counterfactual_support_count",
        "directed_outcome_support_count",
        "exit_attempt_support_count",
        "last_supported_round_by_family",
        "last_supported_pass_id_by_family",
        "source_episode_ids",
        "confidence",
        "evidence_tier",
    }})
    family_markers = {
        "display_support_count": bool(incoming.get("supports_display_relation", False)),
        "match_support_count": bool(incoming.get("supports_match_relation", False)),
        "counterfactual_support_count": bool(incoming.get("supports_counterfactual_relation", False)),
        "directed_outcome_support_count": bool(incoming.get("supports_directed_outcome_relation", False)),
        "exit_attempt_support_count": bool(incoming.get("supports_exit_attempt_relation", False)),
    }
    relation_kind = str(incoming.get("edge_kind") or incoming.get("relation_type") or "")
    if not any(family_markers.values()):
        if relation_kind == "displays":
            family_markers["display_support_count"] = True
        elif relation_kind == "matches":
            family_markers["match_support_count"] = True
    last_rounds = dict(existing.get("last_supported_round_by_family", {}) or {})
    last_passes = dict(existing.get("last_supported_pass_id_by_family", {}) or {})
    incoming_round = int(incoming.get("source_round_id", incoming.get("round_id", 0)) or 0)
    incoming_pass = int(incoming.get("source_pass_id", 0) or 0)
    for family in (
        "display_support_count",
        "match_support_count",
        "counterfactual_support_count",
        "directed_outcome_support_count",
        "exit_attempt_support_count",
    ):
        existing_count = int(existing.get(family, 0) or 0)
        incoming_count = int(incoming.get(family, 0) or 0)
        increment = incoming_count
        if increment <= 0 and family_markers.get(family, False):
            increment = 1
        merged[family] = existing_count + max(0, increment)
        if max(0, increment) > 0:
            family_name = family.removesuffix("_count")
            last_rounds[family_name] = max(int(last_rounds.get(family_name, 0) or 0), incoming_round)
            last_passes[family_name] = max(int(last_passes.get(family_name, 0) or 0), incoming_pass)
    merged["last_supported_round_by_family"] = last_rounds
    merged["last_supported_pass_id_by_family"] = last_passes
    merged["source_episode_ids"] = sorted(set(list(existing.get("source_episode_ids", []) or []) + [str(incoming.get("source_episode_id") or "")] + list(incoming.get("source_episode_ids", []) or [])))
    merged["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), float(incoming.get("confidence", 0.0) or 0.0))
    merged["evidence_tier"] = "observed" if "observed" in {str(existing.get("evidence_tier") or ""), str(incoming.get("evidence_tier") or "")} else str(existing.get("evidence_tier") or incoming.get("evidence_tier") or "hypothesized")
    merged["support_count"] = max(
        int(existing.get("support_count", 0) or 0),
        int(merged.get("display_support_count", 0) or 0)
        + int(merged.get("match_support_count", 0) or 0)
        + int(merged.get("counterfactual_support_count", 0) or 0)
        + int(merged.get("directed_outcome_support_count", 0) or 0)
        + int(merged.get("exit_attempt_support_count", 0) or 0),
    )
    return merged


def _merge_consequence_family_support(existing_row: dict, incoming_row: dict) -> dict:
    existing = dict(existing_row or {})
    incoming = dict(incoming_row or {})
    merged = dict(existing)
    merged.update(incoming)
    support_family = str(incoming.get("support_family") or existing.get("support_family") or "")
    merged["supports_exit_attempt_relation"] = bool(existing.get("supports_exit_attempt_relation", False) or incoming.get("supports_exit_attempt_relation", False) or support_family == "exit_attempt")
    merged["supports_counterfactual_relation"] = bool(existing.get("supports_counterfactual_relation", False) or incoming.get("supports_counterfactual_relation", False) or support_family == "counterfactual")
    merged["supports_directed_outcome_relation"] = bool(existing.get("supports_directed_outcome_relation", False) or incoming.get("supports_directed_outcome_relation", False) or support_family == "directed_outcome")
    for count_field, marker_field in (
        ("exit_attempt_support_count", "supports_exit_attempt_relation"),
        ("counterfactual_support_count", "supports_counterfactual_relation"),
        ("directed_outcome_support_count", "supports_directed_outcome_relation"),
    ):
        increment = int(incoming.get(count_field, 0) or 0)
        if increment <= 0 and bool(incoming.get(marker_field, False)):
            increment = 1
        merged[count_field] = int(existing.get(count_field, 0) or 0) + max(0, increment)
    last_rounds = dict(existing.get("last_supported_round_by_family", {}) or {})
    last_passes = dict(existing.get("last_supported_pass_id_by_family", {}) or {})
    source_round = int(incoming.get("source_round_id", incoming.get("round_id", 0)) or 0)
    source_pass = int(incoming.get("source_pass_id", 0) or 0)
    for family_name, marker in (
        ("exit_attempt", merged.get("supports_exit_attempt_relation")),
        ("counterfactual", merged.get("supports_counterfactual_relation")),
        ("directed_outcome", merged.get("supports_directed_outcome_relation")),
    ):
        if marker:
            last_rounds[family_name] = max(int(last_rounds.get(family_name, 0) or 0), source_round)
            last_passes[family_name] = max(int(last_passes.get(family_name, 0) or 0), source_pass)
    merged["last_supported_round_by_family"] = last_rounds
    merged["last_supported_pass_id_by_family"] = last_passes
    if not support_family:
        support_family = (
            "exit_attempt" if merged.get("supports_exit_attempt_relation")
            else "counterfactual" if merged.get("supports_counterfactual_relation")
            else "directed_outcome" if merged.get("supports_directed_outcome_relation")
            else ""
        )
    if support_family:
        merged["support_family"] = support_family
    return merged


def _synthesize_family_consequences_from_delta(delta: dict) -> list[dict]:
    metadata = dict(delta.get("metadata", {}) or {})
    debug = dict(metadata.get("support_family_emit_debug", {}) or {})
    families = dict(debug.get("families", {}) or {})
    step_rows = list(metadata.get("step_rows", []) or [])
    if not step_rows:
        return []
    first_step = dict(step_rows[0] or {})
    telemetry = dict(first_step.get("telemetry", {}) or {})
    synthesized: list[dict] = []
    if bool(dict(families.get("exit_attempt", {}) or {}).get("row_emitted", False)):
        synthesized.append(
            {
                "consequence_id": f"exit_attempt:{str(telemetry.get('exit_attempt_target_id') or first_step.get('area_id') or telemetry.get('attempted_escape_direction') or 'unknown')}",
                "step_idx": int(first_step.get("step_idx", 0) or 0),
                "action": str(first_step.get("action_name") or first_step.get("action_type") or "unknown"),
                "action_id": first_step.get("action_id"),
                "action_name": str(first_step.get("action_name") or first_step.get("action_type") or "unknown"),
                "action_family": str(first_step.get("action_family") or "unknown"),
                "reward": 0.0,
                "done": False,
                "blocked": bool(telemetry.get("attempted_boundary_contact")),
                "local_change_area": 0,
                "action_effect_near_avatar": False,
                "evidence_count": 1,
                "evidence_refs": [f"{delta.get('episode_id')}:{int(first_step.get('step_idx', 0) or 0)}:exit_attempt"],
                "support_family": "exit_attempt",
                "supports_exit_attempt_relation": True,
                "exit_attempt_support_count": 1,
                "last_supported_round_by_family": {"exit_attempt": int(delta.get("round_id", 0) or 0)},
                "last_supported_pass_id_by_family": {"exit_attempt": int(delta.get("pass_id", 0) or 0)},
                "source_stage": "analysis",
                "source_pass_id": int(delta.get("pass_id", 0) or 0),
                "source_episode_id": str(delta.get("episode_id") or ""),
                "source_round_id": int(delta.get("round_id", 0) or 0),
                "analysis_objective": "broad_trigger_suspicion",
                "direct_evidence_present": False,
                "direct_evidence_fields": ["supports_exit_attempt_relation"],
                "factual_observation": False,
                "confidence": 0.5,
            }
        )
    if bool(dict(families.get("counterfactual", {}) or {}).get("row_emitted", False)):
        synthesized.append(
            {
                "consequence_id": f"counterfactual:{str(telemetry.get('expected_target_id') or first_step.get('area_id') or 'unknown')}",
                "step_idx": int(first_step.get("step_idx", 0) or 0),
                "action": str(first_step.get("action_name") or first_step.get("action_type") or "unknown"),
                "action_id": first_step.get("action_id"),
                "action_name": str(first_step.get("action_name") or first_step.get("action_type") or "unknown"),
                "action_family": str(first_step.get("action_family") or "unknown"),
                "reward": 0.0,
                "done": False,
                "blocked": bool(telemetry.get("observed_effect_absent")),
                "local_change_area": 0,
                "action_effect_near_avatar": False,
                "evidence_count": 1,
                "evidence_refs": [f"{delta.get('episode_id')}:{int(first_step.get('step_idx', 0) or 0)}:counterfactual"],
                "support_family": "counterfactual",
                "supports_counterfactual_relation": True,
                "counterfactual_support_count": 1,
                "last_supported_round_by_family": {"counterfactual": int(delta.get("round_id", 0) or 0)},
                "last_supported_pass_id_by_family": {"counterfactual": int(delta.get("pass_id", 0) or 0)},
                "source_stage": "analysis",
                "source_pass_id": int(delta.get("pass_id", 0) or 0),
                "source_episode_id": str(delta.get("episode_id") or ""),
                "source_round_id": int(delta.get("round_id", 0) or 0),
                "analysis_objective": "broad_trigger_suspicion",
                "direct_evidence_present": False,
                "direct_evidence_fields": ["supports_counterfactual_relation"],
                "factual_observation": False,
                "confidence": 0.5,
            }
        )
    return synthesized


def _remove_collapsed_pois(store: dict[str, dict], collapsed_poi_ids: list[str] | set[str] | tuple[str, ...]) -> dict[str, dict]:
    collapsed = {str(value) for value in list(collapsed_poi_ids or []) if value}
    if not collapsed:
        return {str(row_id): dict(row) for row_id, row in dict(store or {}).items()}
    cleaned: dict[str, dict] = {}
    for row_id, row in dict(store or {}).items():
        payload = dict(row)
        entity_id = str(payload.get("entity_id") or row_id or "")
        poi_id = str(payload.get("poi_id") or "")
        if entity_id in collapsed or poi_id in collapsed:
            continue
        cleaned[str(row_id)] = payload
    return cleaned


def _remove_subsumed_poi_rows(store: dict[str, dict]) -> dict[str, dict]:
    rows = {str(row_id): dict(row) for row_id, row in dict(store or {}).items()}
    subsumed_ids: set[str] = set()
    for row_id, row in rows.items():
        if str(row.get("kind") or "") != "poi":
            continue
        self_id = str(row.get("entity_id") or row_id or "")
        for merged_id in list(row.get("merged_input_poi_ids", []) or []):
            merged_id = str(merged_id or "")
            if merged_id and merged_id != self_id and merged_id.startswith("entity:"):
                subsumed_ids.add(merged_id)
    if not subsumed_ids:
        return rows
    cleaned: dict[str, dict] = {}
    for row_id, row in rows.items():
        entity_id = str(row.get("entity_id") or row_id or "")
        poi_id = str(row.get("poi_id") or "")
        if entity_id in subsumed_ids or poi_id in subsumed_ids:
            continue
        cleaned[str(row_id)] = dict(row)
    return cleaned


def _apply_step_target_effects(entities: dict[str, dict], delta: dict) -> dict[str, dict]:
    updated = {entity_id: dict(row) for entity_id, row in dict(entities or {}).items()}
    metadata = dict(delta.get("metadata", {})) if isinstance(delta.get("metadata"), dict) else {}
    step_rows = list(metadata.get("step_rows", []) or [])
    grouped: dict[str, dict[str, int]] = {}
    for row in step_rows:
        if not isinstance(row, dict):
            continue
        target_id = row.get("target_entity_id")
        if not target_id or str(target_id) not in updated:
            continue
        family = str(row.get("action_family") or "unknown").strip().lower()
        changed_cells = int(row.get("changed_cells", 0) or 0)
        stats = grouped.setdefault(
            str(target_id),
            {
                "movement_attempts": 0,
                "interact_attempts": 0,
                "click_attempts": 0,
                "movement_effect_sum": 0,
                "interact_effect_sum": 0,
                "click_effect_sum": 0,
            },
        )
        if family == "move":
            stats["movement_attempts"] += 1
            stats["movement_effect_sum"] += changed_cells
        elif family == "interact":
            stats["interact_attempts"] += 1
            stats["interact_effect_sum"] += changed_cells
        elif family == "click_at":
            stats["click_attempts"] += 1
            stats["click_effect_sum"] += changed_cells
    for target_id, stats in grouped.items():
        entity = updated.get(target_id)
        if not entity:
            continue
        movement_attempts = int(stats["movement_attempts"])
        interact_attempts = int(stats["interact_attempts"])
        click_attempts = int(stats["click_attempts"])
        movement_effect_sum = int(stats["movement_effect_sum"])
        interact_effect_sum = int(stats["interact_effect_sum"])
        click_effect_sum = int(stats["click_effect_sum"])
        movement_effect_score = min(1.0, (movement_effect_sum / float(movement_attempts) / 50.0) if movement_attempts > 0 else 0.0)
        interact_effect_score = min(1.0, (interact_effect_sum / float(interact_attempts) / 50.0) if interact_attempts > 0 else 0.0)
        click_effect_score = min(1.0, (click_effect_sum / float(click_attempts) / 50.0) if click_attempts > 0 else 0.0)
        if interact_attempts > 0:
            candidate_effect_mode = "interact"
            candidate_effect_score = interact_effect_score
        elif click_attempts > 0:
            candidate_effect_mode = "click_at"
            candidate_effect_score = click_effect_score
        else:
            candidate_effect_mode = "move"
            candidate_effect_score = movement_effect_score
        entity["movement_attempts"] = int(entity.get("movement_attempts", 0) or 0) + movement_attempts
        entity["interact_attempts"] = int(entity.get("interact_attempts", 0) or 0) + interact_attempts
        entity["click_attempts"] = int(entity.get("click_attempts", 0) or 0) + click_attempts
        entity["movement_effect_sum"] = int(entity.get("movement_effect_sum", 0) or 0) + movement_effect_sum
        entity["interact_effect_sum"] = int(entity.get("interact_effect_sum", 0) or 0) + interact_effect_sum
        entity["click_effect_sum"] = int(entity.get("click_effect_sum", 0) or 0) + click_effect_sum
        entity["movement_effect_score"] = max(float(entity.get("movement_effect_score", 0.0) or 0.0), movement_effect_score)
        entity["interact_effect_score"] = max(float(entity.get("interact_effect_score", 0.0) or 0.0), interact_effect_score)
        entity["click_effect_score"] = max(float(entity.get("click_effect_score", 0.0) or 0.0), click_effect_score)
        if float(candidate_effect_score) >= float(entity.get("candidate_effect_score", 0.0) or 0.0):
            entity["candidate_effect_mode"] = candidate_effect_mode
        entity["candidate_effect_score"] = max(float(entity.get("candidate_effect_score", 0.0) or 0.0), float(candidate_effect_score))
    return updated


def _merge_topology_store(existing: dict[str, dict], incoming_nodes: list[dict], incoming_edges: list[dict]) -> dict[str, dict]:
    nodes, _ = merge_topology(
        dict(existing.get("nodes", {})),
        dict(existing.get("edges", {})),
        {row["node_id"]: row for row in incoming_nodes if row.get("node_id")},
        {},
    )
    edges = {str(edge_id): dict(row) for edge_id, row in dict(existing.get("edges", {})).items()}
    edge_identity_to_id = {_stable_edge_identity(row): str(edge_id) for edge_id, row in edges.items()}
    for row in list(incoming_edges or []):
        incoming = dict(row)
        stable_identity = _stable_edge_identity(incoming)
        edge_id = edge_identity_to_id.get(stable_identity) or str(incoming.get("edge_id") or "")
        existing_row = dict(edges.get(edge_id, {}) or {})
        merged_edge = _merge_topology_edge_support(existing_row, incoming)
        if not edge_id:
            edge_id = str(merged_edge.get("edge_id") or stable_identity)
        merged_edge["edge_id"] = edge_id
        merged_edge["stable_edge_identity"] = stable_identity
        edges[edge_id] = merged_edge
        edge_identity_to_id[stable_identity] = edge_id
    return {"nodes": nodes, "edges": edges}


def _merge_entity_store(existing: dict[str, dict], incoming_rows: list[dict]) -> dict[str, dict]:
    merged = merge_entities(existing, incoming_rows)
    incoming_by_id = {
        str(row.get("entity_id") or row.get("poi_id") or ""): dict(row)
        for row in list(incoming_rows or [])
        if str(row.get("entity_id") or row.get("poi_id") or "")
    }
    updated: dict[str, dict] = {}
    for row_id, row in dict(merged or {}).items():
        payload = dict(row)
        incoming = incoming_by_id.get(str(payload.get("entity_id") or row_id or ""), {})
        payload = _merge_entity_identity_fields(payload, incoming)
        updated[str(row_id)] = payload
    return updated


def _assert_rich_fields_preserved(*, split_store: dict[str, dict], combined_store: dict[str, dict], field_names: set[str], label: str) -> None:
    for row_id, row in dict(split_store or {}).items():
        if not any(field in row for field in field_names):
            continue
        combined = dict(combined_store.get(str(row_id), {}) or {})
        missing = [field for field in field_names if field in row and field not in combined]
        if missing:
            raise AssertionError(f"{label}: fields {missing} present before rebuild and missing after rebuild for {row_id}")


def _consequence_sort_key(row: dict) -> tuple:
    return (
        int(row.get("last_seen_round", row.get("round_id", row.get("source_round_id", 0))) or 0),
        int(row.get("step_idx", -1) or -1),
        str(row.get("consequence_id") or ""),
    )


def _prune_consequence_store(store: dict[str, dict], *, limit: int) -> dict[str, dict]:
    if limit <= 0 or len(store) <= limit:
        return {str(row_id): dict(row) for row_id, row in store.items()}
    rows = [dict(row) for row in store.values()]
    family_rows = [
        row for row in rows
        if str(row.get("support_family") or "") in {"exit_attempt", "counterfactual", "directed_outcome"}
        or bool(row.get("supports_exit_attempt_relation", False))
        or bool(row.get("supports_counterfactual_relation", False))
        or bool(row.get("supports_directed_outcome_relation", False))
    ]
    family_ids = {str(row.get("consequence_id") or "") for row in family_rows if row.get("consequence_id")}
    ordered = sorted(
        (row for row in rows if str(row.get("consequence_id") or "") not in family_ids),
        key=_consequence_sort_key,
        reverse=True,
    )
    keep_budget = max(0, limit - len(family_rows))
    kept = family_rows + ordered[:keep_budget]
    return {str(row.get("consequence_id")): row for row in kept if row.get("consequence_id")}


def _enrich_indexes_with_evidence_tiers(next_state: dict) -> dict:
    combined_state = {
        "areas": dict(next_state.get("areas", {})),
        "entities": dict(next_state.get("entities", {})),
        "consequences": dict(next_state.get("consequences", {})),
        "trigger_zones": dict(next_state.get("trigger_zones", {})),
        "topology_nodes": dict(next_state.get("topology_nodes", {})),
        "topology_edges": dict(next_state.get("topology_edges", {})),
    }
    indexes = build_indexes(combined_state)

    def entity_row(entity_id: str) -> dict:
        entity = dict(next_state.get("entities", {}).get(entity_id, {}))
        return {
            "entity_id": entity_id,
            "area_id": entity.get("area_id"),
            "evidence_tier": entity.get("evidence_tier", "hypothesized"),
            "identity_status": str(entity.get("identity_status") or "unknown"),
            "identity_strength_profile": {
                "identity_status": str(entity.get("identity_status") or "unknown"),
                "support_count": int(entity.get("identity_support_count", 0) or 0),
                "contradiction_count": int(entity.get("identity_contradiction_count", 0) or 0),
                "cross_round_stability": int(entity.get("identity_cross_round_stability", 0) or 0),
            },
        }

    def consequence_row(consequence_id: str) -> dict:
        consequence = dict(next_state.get("consequences", {}).get(consequence_id, {}))
        return {
            "consequence_id": consequence_id,
            "action_key": normalized_consequence_action_key(consequence),
            "evidence_tier": consequence.get("evidence_tier", "hypothesized"),
        }

    indexes["entities_by_area_rows"] = {
        area_id: [entity_row(entity_id) for entity_id in entity_ids]
        for area_id, entity_ids in dict(indexes.get("entities_by_area", {})).items()
    }
    indexes["pois_by_area_rows"] = {
        area_id: [entity_row(entity_id) for entity_id in entity_ids]
        for area_id, entity_ids in dict(indexes.get("pois_by_area", {})).items()
    }
    indexes["reachable_targets_rows"] = [entity_row(entity_id) for entity_id in list(indexes.get("reachable_targets", []))]
    indexes["blocked_targets_rows"] = [entity_row(entity_id) for entity_id in list(indexes.get("blocked_targets", []))]
    indexes["frontier_candidates_rows"] = [entity_row(entity_id) for entity_id in list(indexes.get("frontier_candidates", []))]
    indexes["consequence_by_action_rows"] = {
        action_key: [consequence_row(consequence_id) for consequence_id in consequence_ids]
        for action_key, consequence_ids in dict(indexes.get("consequence_by_action", {})).items()
    }
    indexes["evidence_index_rows"] = {
        evidence_ref: [
            {
                "row_id": row_id,
                "evidence_tier": (
                    next_state.get("entities", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("consequences", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("trigger_zones", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("topology_nodes", {}).get(row_id, {}).get("evidence_tier")
                    or next_state.get("topology_edges", {}).get(row_id, {}).get("evidence_tier")
                    or "hypothesized"
                ),
            }
            for row_id in row_ids
        ]
        for evidence_ref, row_ids in dict(indexes.get("evidence_index", {})).items()
    }
    indexes["topology_edge_support_profile_rows"] = {
        str(edge_id): {
            "edge_id": str(edge_id),
            "evidence_tier": str(dict(edge).get("evidence_tier") or "hypothesized"),
            "evidence_support_profile": {
                "display": int(dict(edge).get("display_support_count", 0) or 0),
                "match": int(dict(edge).get("match_support_count", 0) or 0),
                "counterfactual": int(dict(edge).get("counterfactual_support_count", 0) or 0),
                "directed_outcome": int(dict(edge).get("directed_outcome_support_count", 0) or 0),
                "exit_attempt": int(dict(edge).get("exit_attempt_support_count", 0) or 0),
            },
        }
        for edge_id, edge in dict(next_state.get("topology_edges", {}) or {}).items()
    }
    return indexes


def _build_strict_split_indexes(*, areas: dict, entities: dict, consequences: dict, trigger_zones: dict, topology_nodes: dict, topology_edges: dict) -> dict:
    entities_by_area_rows: dict[str, list[dict]] = {}
    pois_by_area_rows: dict[str, list[dict]] = {}
    pois_by_type_rows: dict[str, list[dict]] = {}
    reachable_targets_rows: list[dict] = []
    blocked_targets_rows: list[dict] = []
    frontier_candidates_rows: list[dict] = []
    consequence_by_action_rows: dict[str, list[dict]] = {}
    evidence_index_rows: dict[str, list[dict]] = {}
    topology_lookup = {"node_ids_by_cell": {}, "out_edges_by_src": {}}
    for area_id, area in dict(areas).items():
        entities_by_area_rows.setdefault(str(area_id), [])
        pois_by_area_rows.setdefault(str(area_id), [])
        for cell in list(area.get("topology_cells", []) or []):
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                topology_lookup["node_ids_by_cell"][f"{int(cell[0])}:{int(cell[1])}"] = f"cell:{int(cell[0])}:{int(cell[1])}"
    for entity_id, entity in dict(entities).items():
        row = {
            "entity_id": str(entity_id),
            "area_id": str(entity.get("area_id") or "global"),
            "evidence_tier": str(entity.get("evidence_tier") or "hypothesized"),
            "identity_status": str(entity.get("identity_status") or "unknown"),
            "identity_strength_profile": {
                "identity_status": str(entity.get("identity_status") or "unknown"),
                "support_count": int(entity.get("identity_support_count", 0) or 0),
                "contradiction_count": int(entity.get("identity_contradiction_count", 0) or 0),
                "cross_round_stability": int(entity.get("identity_cross_round_stability", 0) or 0),
            },
        }
        entities_by_area_rows.setdefault(row["area_id"], []).append(row)
        if entity.get("kind") == "poi":
            pois_by_area_rows.setdefault(row["area_id"], []).append(row)
            poi_type = str(entity.get("canonical_descriptor", {}).get("kind") or entity.get("kind") or "unknown")
            pois_by_type_rows.setdefault(poi_type, []).append(row)
        if entity.get("reachable_now"):
            reachable_targets_rows.append(row)
        elif entity.get("reachable_later"):
            frontier_candidates_rows.append(row)
        else:
            blocked_targets_rows.append(row)
        for evidence_ref in list(entity.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(entity_id), "evidence_tier": row["evidence_tier"]})
    for consequence_id, consequence in dict(consequences).items():
        action_key = normalized_consequence_action_key(consequence)
        row = {"consequence_id": str(consequence_id), "action_key": action_key, "evidence_tier": str(consequence.get("evidence_tier") or "hypothesized")}
        consequence_by_action_rows.setdefault(action_key, []).append(row)
        for evidence_ref in list(consequence.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(consequence_id), "evidence_tier": row["evidence_tier"]})
    for trigger_id, trigger in dict(trigger_zones).items():
        for evidence_ref in list(trigger.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(trigger_id), "evidence_tier": str(trigger.get("evidence_tier") or "hypothesized")})
    for edge_id, edge in dict(topology_edges).items():
        topology_lookup["out_edges_by_src"].setdefault(str(edge.get("src")), []).append(str(edge_id))
        for evidence_ref in list(edge.get("evidence_refs", []) or []):
            evidence_index_rows.setdefault(str(evidence_ref), []).append({"row_id": str(edge_id), "evidence_tier": str(edge.get("evidence_tier") or "hypothesized")})
    return {
        "entities_by_area_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in entities_by_area_rows.items()},
        "pois_by_area_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in pois_by_area_rows.items()},
        "pois_by_type_rows": {key: sorted(value, key=lambda row: row["entity_id"]) for key, value in pois_by_type_rows.items()},
        "reachable_targets_rows": sorted(reachable_targets_rows, key=lambda row: row["entity_id"]),
        "blocked_targets_rows": sorted(blocked_targets_rows, key=lambda row: row["entity_id"]),
        "frontier_candidates_rows": sorted(frontier_candidates_rows, key=lambda row: row["entity_id"]),
        "consequence_by_action_rows": {key: sorted(value, key=lambda row: row["consequence_id"]) for key, value in consequence_by_action_rows.items()},
        "evidence_index_rows": {key: sorted(value, key=lambda row: row["row_id"]) for key, value in evidence_index_rows.items()},
        "topology_lookup": topology_lookup,
        "topology_edge_support_profile_rows": {
            str(edge_id): {
                "edge_id": str(edge_id),
                "evidence_tier": str(dict(edge).get("evidence_tier") or "hypothesized"),
                "evidence_support_profile": {
                    "display": int(dict(edge).get("display_support_count", 0) or 0),
                    "match": int(dict(edge).get("match_support_count", 0) or 0),
                    "counterfactual": int(dict(edge).get("counterfactual_support_count", 0) or 0),
                    "directed_outcome": int(dict(edge).get("directed_outcome_support_count", 0) or 0),
                    "exit_attempt": int(dict(edge).get("exit_attempt_support_count", 0) or 0),
                },
            }
            for edge_id, edge in dict(topology_edges).items()
        },
        "entity_count": len(dict(entities)),
        "area_count": len(dict(areas)),
        "topology_node_count": len(dict(topology_nodes)),
        "trigger_count": len(dict(trigger_zones)),
        "consequence_count": len(dict(consequences)),
    }


def _split_index_state(*, areas: dict, entities: dict, consequences: dict, trigger_zones: dict, topology_nodes: dict, topology_edges: dict) -> dict:
    return _build_strict_split_indexes(
        areas=dict(areas),
        entities=dict(entities),
        consequences=dict(consequences),
        trigger_zones=dict(trigger_zones),
        topology_nodes=dict(topology_nodes),
        topology_edges=dict(topology_edges),
    )


def apply_delta(state: dict, delta: dict, *, max_consequences: int = 100) -> tuple[dict, bool]:
    state = _ensure_state(state)
    next_state = dict(state)
    classified_areas = [_classify_row("areas", row, delta) for row in list(delta.get("areas", ()) or [])]
    classified_entities = [_classify_row("entities", row, delta) for row in list(delta.get("entities", ()) or [])]
    prepopulated_consequences = list(delta.get("consequences", ()))
    exit_attempt_rows_seen_on_raw_delta = int(
        sum(
            1
            for row in prepopulated_consequences
            if bool(dict(row).get("supports_exit_attempt_relation", False)) or str(dict(row).get("support_family") or "") == "exit_attempt"
        )
    )
    raw_consequences = prepopulated_consequences if prepopulated_consequences else extract_consequence_records(delta)
    raw_consequences, normalization_diagnostics = _normalize_consequence_rows(raw_consequences, delta)
    if not raw_consequences and prepopulated_consequences and not all(_consequence_transport_complete(row) for row in prepopulated_consequences):
        normalization_diagnostics["normalization_drop_reason_codes"] = list(normalization_diagnostics.get("normalization_drop_reason_codes", []) or []) + ["family_row_kind_unrecognized"]
        raw_consequences, fallback_diagnostics = _normalize_consequence_rows(extract_consequence_records(delta), delta)
        normalization_diagnostics = _accumulate_support_family_diagnostics(normalization_diagnostics, fallback_diagnostics)
    exit_attempt_rows_seen_after_row_normalization = int(
        normalization_diagnostics.get("exit_attempt_family_markers_present_after_normalization_count", 0) or 0
    )
    classified_consequences = [_classify_row("consequences", row, delta) for row in raw_consequences]
    if not any(bool(dict(row).get("supports_exit_attempt_relation", False)) or str(dict(row).get("support_family") or "") == "exit_attempt" for row in classified_consequences):
        synthesized_family_consequences = [
            _classify_row("consequences", row, delta)
            for row in _synthesize_family_consequences_from_delta(delta)
        ]
        classified_consequences.extend(synthesized_family_consequences)
    classified_trigger_zones = [_classify_row("trigger_zones", row, delta) for row in list(delta.get("trigger_zones", ()) or [])]
    classified_topology_nodes = [_classify_row("topology_nodes", row, delta) for row in list(delta.get("topology_nodes", ()) or [])]
    classified_topology_edges = [_classify_row("topology_edges", row, delta) for row in list(delta.get("topology_edges", ()) or [])]

    exit_attempt_ingress_rows = [
        dict(row) for row in classified_consequences
        if bool(dict(row).get("supports_exit_attempt_relation", False)) or str(dict(row).get("support_family") or "") == "exit_attempt"
    ]
    exit_attempt_merge_diagnostics = {
        "exit_attempt_rows_seen_on_raw_delta": int(exit_attempt_rows_seen_on_raw_delta),
        "exit_attempt_rows_seen_after_row_normalization": int(exit_attempt_rows_seen_after_row_normalization),
        "counterfactual_rows_seen_on_raw_delta": int(normalization_diagnostics.get("counterfactual_family_markers_present_on_raw_delta_count", 0) or 0),
        "counterfactual_rows_seen_after_row_normalization": int(normalization_diagnostics.get("counterfactual_family_markers_present_after_normalization_count", 0) or 0),
        "exit_attempt_family_markers_present_on_raw_delta_count": int(normalization_diagnostics.get("exit_attempt_family_markers_present_on_raw_delta_count", 0) or 0),
        "exit_attempt_family_markers_present_after_normalization_count": int(normalization_diagnostics.get("exit_attempt_family_markers_present_after_normalization_count", 0) or 0),
        "exit_attempt_family_markers_lost_in_normalization_count": int(normalization_diagnostics.get("exit_attempt_family_markers_lost_in_normalization_count", 0) or 0),
        "counterfactual_family_markers_present_on_raw_delta_count": int(normalization_diagnostics.get("counterfactual_family_markers_present_on_raw_delta_count", 0) or 0),
        "counterfactual_family_markers_present_after_normalization_count": int(normalization_diagnostics.get("counterfactual_family_markers_present_after_normalization_count", 0) or 0),
        "counterfactual_family_markers_lost_in_normalization_count": int(normalization_diagnostics.get("counterfactual_family_markers_lost_in_normalization_count", 0) or 0),
        "exit_attempt_rows_seen_at_merge_ingress": int(len(exit_attempt_ingress_rows)),
        "exit_attempt_rows_classified_observed": int(sum(1 for row in exit_attempt_ingress_rows if str(row.get("evidence_tier") or "") == "observed")),
        "exit_attempt_rows_classified_hypothesized": int(sum(1 for row in exit_attempt_ingress_rows if str(row.get("evidence_tier") or "") != "observed")),
        "exit_attempt_rows_dropped_before_store": 0,
        "exit_attempt_drop_reason_codes": list(normalization_diagnostics.get("normalization_drop_reason_codes", []) or []),
    }

    merged_areas = merge_areas(state.get("areas", {}), classified_areas)
    collapsed_poi_ids = list(dict(delta.get("metadata", {})).get("collapsed_poi_ids", []) or [])
    observed_entities = _merge_entity_store(
        state.get("observed_entities", {}),
        [row for row in classified_entities if row.get("evidence_tier") == "observed"],
    )
    observed_entities = _remove_collapsed_pois(observed_entities, collapsed_poi_ids)
    observed_entities = _remove_subsumed_poi_rows(observed_entities)
    hypothesized_entities = _merge_entity_store(
        state.get("hypothesized_entities", {}),
        [row for row in classified_entities if row.get("evidence_tier") != "observed"],
    )
    hypothesized_entities = _remove_collapsed_pois(hypothesized_entities, collapsed_poi_ids)
    hypothesized_entities = _remove_subsumed_poi_rows(hypothesized_entities)
    observed_consequences = merge_consequences(
        state.get("observed_consequences", {}),
        [row for row in classified_consequences if row.get("evidence_tier") == "observed"],
    )
    hypothesized_consequences = merge_consequences(
        state.get("hypothesized_consequences", {}),
        [row for row in classified_consequences if row.get("evidence_tier") != "observed"],
    )
    observed_consequences = _prune_consequence_store(observed_consequences, limit=max_consequences)
    hypothesized_consequences = _prune_consequence_store(hypothesized_consequences, limit=max_consequences)
    observed_topology = _merge_topology_store(
        dict(state.get("observed_topology", {})),
        [row for row in classified_topology_nodes if row.get("evidence_tier") == "observed"],
        [row for row in classified_topology_edges if row.get("evidence_tier") == "observed"],
    )
    hypothesized_topology = _merge_topology_store(
        dict(state.get("hypothesized_topology", {})),
        [row for row in classified_topology_nodes if row.get("evidence_tier") != "observed"],
        [row for row in classified_topology_edges if row.get("evidence_tier") != "observed"],
    )

    merged_entities = _combine_rows(hypothesized_entities, observed_entities)
    merged_entities, identity_fillthrough_applied_count = _fill_entity_identity_fields_from_sources(
        merged_entities,
        hypothesized_entities,
        observed_entities,
    )
    merged_consequences = _combine_rows(hypothesized_consequences, observed_consequences)
    merged_consequences, consequence_family_fillthrough_applied_count = _fill_family_fields_from_sources(
        merged_consequences,
        hypothesized_consequences,
        observed_consequences,
    )
    merged_consequences = _prune_consequence_store(merged_consequences, limit=max_consequences)
    merged_ids = set(merged_consequences.keys())
    observed_consequences = {row_id: row for row_id, row in observed_consequences.items() if row_id in merged_ids}
    hypothesized_consequences = {row_id: row for row_id, row in hypothesized_consequences.items() if row_id in merged_ids}
    split_written_ids = {
        str(row_id)
        for store in (observed_consequences, hypothesized_consequences)
        for row_id, row in dict(store or {}).items()
        if bool(dict(row).get("supports_exit_attempt_relation", False)) or str(dict(row).get("support_family") or "") == "exit_attempt"
    }
    ingress_ids = {
        str(row.get("consequence_id") or "")
        for row in exit_attempt_ingress_rows
        if str(row.get("consequence_id") or "")
    }
    missing_ingress_ids = sorted(ingress_ids - split_written_ids)
    exit_attempt_merge_diagnostics["exit_attempt_rows_dropped_before_store"] = int(len(missing_ingress_ids))
    if missing_ingress_ids:
        exit_attempt_merge_diagnostics["exit_attempt_drop_reason_codes"] = ["store_key_resolution_failed"]
    if exit_attempt_rows_seen_on_raw_delta > 0 and exit_attempt_rows_seen_after_row_normalization <= 0:
        exit_attempt_merge_diagnostics["exit_attempt_drop_reason_codes"] = list(exit_attempt_merge_diagnostics.get("exit_attempt_drop_reason_codes", []) or []) + ["normalization_dropped_family_fields"]
    elif exit_attempt_rows_seen_after_row_normalization > 0 and len(exit_attempt_ingress_rows) <= 0:
        exit_attempt_merge_diagnostics["exit_attempt_drop_reason_codes"] = list(exit_attempt_merge_diagnostics.get("exit_attempt_drop_reason_codes", []) or []) + ["classification_rejected_family_row"]
    elif exit_attempt_rows_seen_on_raw_delta <= 0 and bool(dict(delta.get("metadata", {}) or {}).get("support_family_emit_debug", {})):
        exit_attempt_merge_diagnostics["exit_attempt_drop_reason_codes"] = list(exit_attempt_merge_diagnostics.get("exit_attempt_drop_reason_codes", []) or []) + ["missing_row_kind"]
    if ingress_ids and not split_written_ids and not exit_attempt_merge_diagnostics["exit_attempt_drop_reason_codes"]:
        exit_attempt_merge_diagnostics["exit_attempt_drop_reason_codes"] = ["silent_disappearance_prevented"]
    if ingress_ids and not split_written_ids and not exit_attempt_merge_diagnostics["exit_attempt_rows_dropped_before_store"]:
        raise AssertionError("exit_attempt ingress rows seen but neither written nor dropped with reason")
    topology_nodes = _combine_rows(
        dict(hypothesized_topology.get("nodes", {})),
        dict(observed_topology.get("nodes", {})),
    )
    topology_edges = _combine_rows(
        dict(hypothesized_topology.get("edges", {})),
        dict(observed_topology.get("edges", {})),
    )
    merged_entities = reachable_entities(merged_entities, topology_nodes, topology_edges)
    merged_entities = _apply_step_target_effects(merged_entities, delta)
    proposed_trigger_zones = classified_trigger_zones + [
        _classify_row("trigger_zones", row, delta)
        for row in propose_trigger_zones(
            entities=merged_entities,
            consequences=merged_consequences,
        )
    ]
    observed_trigger_zones = merge_trigger_zones(
        state.get("observed_trigger_zones", {}),
        [row for row in proposed_trigger_zones if row.get("evidence_tier") == "observed"],
    )
    hypothesized_trigger_zones = merge_trigger_zones(
        state.get("hypothesized_trigger_zones", {}),
        [row for row in proposed_trigger_zones if row.get("evidence_tier") != "observed"],
    )
    merged_trigger_zones = _combine_rows(hypothesized_trigger_zones, observed_trigger_zones)
    merged_areas = _merge_area_topology_metadata(merged_areas, topology_nodes)

    next_state["areas"] = merged_areas
    next_state["observed_entities"] = observed_entities
    next_state["hypothesized_entities"] = hypothesized_entities
    next_state["entities"] = merged_entities
    next_state["identity_fillthrough_applied_count"] = int(identity_fillthrough_applied_count)
    next_state["observed_consequences"] = observed_consequences
    next_state["hypothesized_consequences"] = hypothesized_consequences
    next_state["consequences"] = merged_consequences
    next_state["consequence_family_fillthrough_applied_count"] = int(consequence_family_fillthrough_applied_count)
    next_state["merge_support_family_diagnostics"] = _accumulate_support_family_diagnostics(
        dict(state.get("merge_support_family_diagnostics", {}) or {}),
        exit_attempt_merge_diagnostics,
    )
    next_state["observed_trigger_zones"] = observed_trigger_zones
    next_state["hypothesized_trigger_zones"] = hypothesized_trigger_zones
    next_state["trigger_zones"] = merged_trigger_zones
    next_state["observed_topology"] = observed_topology
    next_state["hypothesized_topology"] = hypothesized_topology
    next_state["topology_nodes"] = topology_nodes
    next_state["topology_edges"] = topology_edges
    _assert_rich_fields_preserved(
        split_store=dict(observed_topology.get("edges", {})),
        combined_store=topology_edges,
        field_names={
            "display_support_count",
            "match_support_count",
            "counterfactual_support_count",
            "directed_outcome_support_count",
            "exit_attempt_support_count",
            "last_supported_round_by_family",
            "last_supported_pass_id_by_family",
        },
        label="topology_edge_support_rebuild",
    )
    _assert_rich_fields_preserved(
        split_store=observed_entities,
        combined_store=merged_entities,
        field_names={
            "identity_status",
            "identity_support_count",
            "identity_contradiction_count",
            "identity_cross_round_stability",
            "identity_last_confirmed_round",
        },
        label="entity_identity_rebuild",
    )
    _assert_rich_fields_preserved(
        split_store=observed_consequences,
        combined_store=merged_consequences,
        field_names={
            "supports_exit_attempt_relation",
            "exit_attempt_support_count",
            "last_supported_round_by_family",
            "last_supported_pass_id_by_family",
            "support_family",
        },
        label="consequence_exit_attempt_rebuild",
    )
    next_state["indexes"] = _enrich_indexes_with_evidence_tiers(next_state)
    next_state["split_indexes"] = {
        "observed": _split_index_state(
            areas=merged_areas,
            entities=observed_entities,
            consequences=observed_consequences,
            trigger_zones=observed_trigger_zones,
            topology_nodes=dict(observed_topology.get("nodes", {})),
            topology_edges=dict(observed_topology.get("edges", {})),
        ),
        "hypothesized": _split_index_state(
            areas=merged_areas,
            entities=hypothesized_entities,
            consequences=hypothesized_consequences,
            trigger_zones=hypothesized_trigger_zones,
            topology_nodes=dict(hypothesized_topology.get("nodes", {})),
            topology_edges=dict(hypothesized_topology.get("edges", {})),
        ),
    }
    material_change = bool(
        delta.get("material_change", True)
        or delta.get("entities")
        or delta.get("areas")
        or raw_consequences
        or delta.get("topology_edges")
    )
    return next_state, material_change

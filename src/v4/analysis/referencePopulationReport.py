from __future__ import annotations


def _reference_summary(row: dict[str, object]) -> dict[str, object]:
    return {
        "belief_reference_present": bool(row.get("belief_reference_present", False)),
        "belief_unknown_cell_count": int(row.get("belief_unknown_cell_count", 0)),
        "belief_frontier_cell_count": int(row.get("belief_frontier_cell_count", 0)),
        "hypothesis_reference_present": bool(row.get("hypothesis_reference_present", False)),
        "hypothesis_count": int(row.get("hypothesis_count", 0)),
        "temporal_reference_present": bool(row.get("temporal_reference_present", False)),
        "safe_horizon_steps": int(row.get("safe_horizon_steps", 0)),
        "hazard_window_remaining": row.get("hazard_window_remaining"),
        "composition_reference_present": bool(row.get("composition_reference_present", False)),
        "composition_domain_count": int(row.get("composition_domain_count", 0)),
        "composition_present_domains": tuple(row.get("composition_present_domains", ())),
        "composition_cross_domain_effect_count": int(row.get("composition_cross_domain_effect_count", 0)),
    }


def _builder_summary(row: dict[str, object]) -> dict[str, object]:
    return {
        "ms01_builder_ok": bool(row.get("ms01_builder_ok", False)),
        "ms01_builder_error": row.get("ms01_builder_error"),
        "ms01_builder_summary": dict(row.get("ms01_builder_summary", {}) or {}),
        "rs01_builder_ok": bool(row.get("rs01_builder_ok", False)),
        "rs01_builder_error": row.get("rs01_builder_error"),
        "rs01_builder_summary": dict(row.get("rs01_builder_summary", {}) or {}),
        "pt01_phase_detector_ok": bool(row.get("pt01_phase_detector_ok", False)),
        "pt01_phase_detector_error": row.get("pt01_phase_detector_error"),
        "pt01_phase_detector_summary": dict(row.get("pt01_phase_detector_summary", {}) or {}),
        "sv01_builder_ok": bool(row.get("sv01_builder_ok", False)),
        "sv01_builder_error": row.get("sv01_builder_error"),
        "sv01_builder_summary": dict(row.get("sv01_builder_summary", {}) or {}),
        "tb01_builder_ok": bool(row.get("tb01_builder_ok", False)),
        "tb01_builder_error": row.get("tb01_builder_error"),
        "tb01_builder_summary": dict(row.get("tb01_builder_summary", {}) or {}),
    }


def build_reference_population_report(
    step8_trace_rows: tuple[dict[str, object], ...],
    success: bool,
    game_id: str,
) -> dict[str, object]:
    total_steps = len(step8_trace_rows)
    return {
        "game_id": game_id,
        "success": bool(success),
        "total_steps": total_steps,
        "belief_reference_present_count": sum(1 for row in step8_trace_rows if bool(row.get("belief_reference_present", False))),
        "hypothesis_reference_present_count": sum(1 for row in step8_trace_rows if bool(row.get("hypothesis_reference_present", False))),
        "temporal_reference_present_count": sum(1 for row in step8_trace_rows if bool(row.get("temporal_reference_present", False))),
        "composition_reference_present_count": sum(1 for row in step8_trace_rows if bool(row.get("composition_reference_present", False))),
        "max_belief_unknown_cell_count": max((int(row.get("belief_unknown_cell_count", 0)) for row in step8_trace_rows), default=0),
        "max_belief_frontier_cell_count": max((int(row.get("belief_frontier_cell_count", 0)) for row in step8_trace_rows), default=0),
        "max_hypothesis_count": max((int(row.get("hypothesis_count", 0)) for row in step8_trace_rows), default=0),
        "max_safe_horizon_steps": max((int(row.get("safe_horizon_steps", 0)) for row in step8_trace_rows), default=0),
        "max_composition_domain_count": max((int(row.get("composition_domain_count", 0)) for row in step8_trace_rows), default=0),
        "max_composition_cross_domain_effect_count": max((int(row.get("composition_cross_domain_effect_count", 0)) for row in step8_trace_rows), default=0),
        "belief_unknown_ever_positive": any(int(row.get("belief_unknown_cell_count", 0)) > 0 for row in step8_trace_rows),
        "belief_frontier_ever_positive": any(int(row.get("belief_frontier_cell_count", 0)) > 0 for row in step8_trace_rows),
        "hypothesis_ever_positive": any(int(row.get("hypothesis_count", 0)) > 0 for row in step8_trace_rows),
        "temporal_safe_horizon_ever_positive": any(int(row.get("safe_horizon_steps", 0)) > 0 for row in step8_trace_rows),
        "composition_domain_ever_positive": any(int(row.get("composition_domain_count", 0)) > 0 for row in step8_trace_rows),
        "composition_cross_domain_effect_ever_positive": any(int(row.get("composition_cross_domain_effect_count", 0)) > 0 for row in step8_trace_rows),
        "ms01_grounded_hidden_signal_positive": game_id == "ms01" and any(int(row.get("belief_unknown_cell_count", 0)) > 0 for row in step8_trace_rows),
        "rs01_grounded_rule_signal_positive": game_id == "rs01" and any(int(row.get("hypothesis_count", 0)) > 0 for row in step8_trace_rows),
        "pt01_grounded_phase_signal_positive": game_id == "pt01" and any(int(row.get("hypothesis_count", 0)) > 0 for row in step8_trace_rows),
        "pt01_grounded_phase_hypothesis_positive": any(
            int(row.get("emitted_hypothesis_count", 0)) > 0
            and any(str(value).startswith("pt01_") for value in tuple(row.get("emitted_hypothesis_candidate_values", ())))
            for row in step8_trace_rows
        ),
        "rs01_grounded_rule_hypothesis_positive": any(
            int(row.get("emitted_hypothesis_count", 0)) > 0
            and any(isinstance(value, int) for value in tuple(row.get("emitted_hypothesis_candidate_values", ())))
            for row in step8_trace_rows
        ),
        "sv01_grounded_temporal_signal_positive": game_id == "sv01" and any(int(row.get("safe_horizon_steps", 0)) > 0 for row in step8_trace_rows),
        "tb01_grounded_construction_signal_positive": game_id == "tb01" and any(int(row.get("composition_cross_domain_effect_count", 0)) > 0 for row in step8_trace_rows),
        "emitted_hypothesis_count_total": sum(int(row.get("emitted_hypothesis_count", 0)) for row in step8_trace_rows),
        "registry_hypothesis_count_after_update_max": max((int(row.get("registry_hypothesis_count_after_update", 0)) for row in step8_trace_rows), default=0),
        "registry_hypothesis_positive_after_update": any(int(row.get("registry_hypothesis_count_after_update", 0)) > 0 for row in step8_trace_rows),
        "emitted_hypothesis_positive": any(int(row.get("emitted_hypothesis_count", 0)) > 0 for row in step8_trace_rows),
        "hypothesis_update_debug_would_emit_count": sum(1 for row in step8_trace_rows if bool(row.get("hypothesis_update_debug_would_emit", False))),
        "hypothesis_update_debug_error_count": sum(1 for row in step8_trace_rows if row.get("hypothesis_update_debug_error") is not None),
        "hypothesis_update_debug_positive": any(bool(row.get("hypothesis_update_debug_would_emit", False)) for row in step8_trace_rows),
        "ms01_builder_ok_count": sum(1 for row in step8_trace_rows if bool(row.get("ms01_builder_ok", False))),
        "rs01_builder_ok_count": sum(1 for row in step8_trace_rows if bool(row.get("rs01_builder_ok", False))),
        "pt01_phase_detector_ok_count": sum(1 for row in step8_trace_rows if bool(row.get("pt01_phase_detector_ok", False))),
        "sv01_builder_ok_count": sum(1 for row in step8_trace_rows if bool(row.get("sv01_builder_ok", False))),
        "tb01_builder_ok_count": sum(1 for row in step8_trace_rows if bool(row.get("tb01_builder_ok", False))),
        "ms01_builder_hidden_positive": any(int((row.get("ms01_builder_summary", {}) or {}).get("unrevealed_frontier_count", 0)) > 0 for row in step8_trace_rows),
        "rs01_builder_rule_positive": any(bool(tuple((row.get("rs01_builder_summary", {}) or {}).get("safe_color_cycle", ()))) for row in step8_trace_rows),
        "pt01_detector_phase_positive": any(bool(str((row.get("pt01_phase_detector_summary", {}) or {}).get("phase", "") or "").strip()) for row in step8_trace_rows),
        "sv01_builder_temporal_positive": any(
            float((row.get("sv01_builder_summary", {}) or {}).get("hunger_value", 0) or 0) > 0
            or float((row.get("sv01_builder_summary", {}) or {}).get("warmth_value", 0) or 0) > 0
            or float((row.get("sv01_builder_summary", {}) or {}).get("survival_timer_remaining", 0) or 0) > 0
            for row in step8_trace_rows
        ),
        "tb01_builder_construction_positive": any(
            (row.get("tb01_builder_summary", {}) or {}).get("bridge_budget_remaining") is not None
            or (row.get("tb01_builder_summary", {}) or {}).get("step_limit_remaining") is not None
            or int((row.get("tb01_builder_summary", {}) or {}).get("bridge_built_count", 0)) > 0
            or int((row.get("tb01_builder_summary", {}) or {}).get("water_cell_count", 0)) > 0
            or (row.get("tb01_builder_summary", {}) or {}).get("goal_cell") is not None
            for row in step8_trace_rows
        ),
        "first_row_reference_summary": _reference_summary(step8_trace_rows[0]) if step8_trace_rows else {},
        "last_row_reference_summary": _reference_summary(step8_trace_rows[-1]) if step8_trace_rows else {},
        "first_row_builder_summary": _builder_summary(step8_trace_rows[0]) if step8_trace_rows else {},
        "last_row_builder_summary": _builder_summary(step8_trace_rows[-1]) if step8_trace_rows else {},
        "failed_last_5_raw_state_text": [row.get("raw_state_text") for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_frame_summary": [row.get("frame_summary", {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_environment_metadata_summary": [row.get("environment_metadata_summary", {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_ms01_builder_error": [row.get("ms01_builder_error") for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_rs01_builder_error": [row.get("rs01_builder_error") for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_pt01_phase_detector_error": [row.get("pt01_phase_detector_error") for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_sv01_builder_error": [row.get("sv01_builder_error") for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_tb01_builder_error": [row.get("tb01_builder_error") for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_ms01_builder_summary": [dict(row.get("ms01_builder_summary", {}) or {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_rs01_builder_summary": [dict(row.get("rs01_builder_summary", {}) or {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_pt01_phase_detector_summary": [dict(row.get("pt01_phase_detector_summary", {}) or {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_sv01_builder_summary": [dict(row.get("sv01_builder_summary", {}) or {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_tb01_builder_summary": [dict(row.get("tb01_builder_summary", {}) or {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_emitted_hypothesis_ids": [tuple(row.get("emitted_hypothesis_ids", ())) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_emitted_hypothesis_candidate_values": [tuple(row.get("emitted_hypothesis_candidate_values", ())) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_registry_hypothesis_ids_after_update": [tuple(row.get("registry_hypothesis_ids_after_update", ())) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_registry_hypothesis_candidate_values_after_update": [tuple(row.get("registry_hypothesis_candidate_values_after_update", ())) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_registry_hypothesis_confidence_bands_after_update": [tuple(row.get("registry_hypothesis_confidence_bands_after_update", ())) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_hypothesis_update_debug": [dict(row.get("hypothesis_update_debug", {}) or {}) for row in step8_trace_rows[-5:]] if not success else [],
        "failed_last_5_hypothesis_update_debug_error": [row.get("hypothesis_update_debug_error") for row in step8_trace_rows[-5:]] if not success else [],
    }

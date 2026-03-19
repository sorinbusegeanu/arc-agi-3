from __future__ import annotations

from v3_1.contracts.messages import AnalyzedEpisode, RawEpisode
from v3_1.utils.ids import stable_digest


def normalize_events(raw_episode: RawEpisode, analyzed_episode: AnalyzedEpisode, mechanic_graph_snapshot: dict, blackboard_snapshot: dict) -> list[dict]:
    blackboard_snapshot = dict(blackboard_snapshot or {})
    graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    nodes_by_id = dict(graph_state.get("nodes_by_id", {}))
    step_rows = list(dict(analyzed_episode.summary or {}).get("step_rows", []) or [])
    analysis_mode = str(dict(analyzed_episode.summary or {}).get("analysis_mode") or dict(analyzed_episode.metadata or {}).get("analysis_mode") or "probe")
    events: list[dict] = []
    object_to_node = {}
    for node_id, node in nodes_by_id.items():
        object_ref = str(node.get("object_ref") or "")
        if object_ref:
            object_to_node[object_ref] = str(node_id)
    for step in step_rows:
        step_idx = int(step.get("step_idx", 0) or 0)
        episode_id = raw_episode.episode_id
        target_entity_id = str(step.get("target_entity_id") or "")
        target_node_id = object_to_node.get(target_entity_id, target_entity_id)
        action_family = str(step.get("action_family") or "")
        target_entity = dict(blackboard_snapshot.get("entities", {}).get(target_entity_id, {}) or {})
        target_provenance = {str(value) for value in list(target_entity.get("poi_source_provenance", []) or []) if value}
        detector_backed_poi = str(target_entity.get("kind") or "") == "poi" and "detector" in target_provenance
        if detector_backed_poi:
            events.append(_event("poi_visit", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or target_entity_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode))
            if action_family == "interact":
                events.append(_event("verified_trigger_contact", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or target_entity_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode, chain_id=str(step.get("chain_id") or ""), step_id=str(step.get("step_id") or "")))
        if action_family == "interact" and target_node_id:
            events.append(_event("contact", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
        if isinstance(step.get("avatar_cell"), list):
            region_id = f"region:{stable_digest((step.get('area_id'), step.get('avatar_cell')))}"
            events.append(_event("enter_region", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=region_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
        target_node_kind = str(dict(nodes_by_id.get(target_node_id or "", {}) or {}).get("node_kind") or "").lower()
        if int(step.get("changed_cells", 0) or 0) > 0:
            telemetry = dict(step.get("telemetry", {}) or {})
            effect_region = dict(telemetry.get("effect_region", {}) or {})
            effect_node_id = f"effect_region:{stable_digest(effect_region.get('bbox') or step_idx)}"
            events.append(_event("remote_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=effect_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
            if detector_backed_poi:
                events.append(_event("poi_visit_then_remote_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or target_entity_id, other_node_id=effect_node_id, lag_steps=0, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode))
                events.append(_event("verified_remote_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or target_entity_id, other_node_id=effect_node_id, lag_steps=0, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode, chain_id=str(step.get("chain_id") or ""), step_id=str(step.get("step_id") or "")))
            if target_node_id and action_family == "interact":
                events.append(_event("contact_then_remote_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id, other_node_id=effect_node_id, lag_steps=0, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
                events.append(_event("repeated_remote_change_after_specific_trigger", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id, other_node_id=effect_node_id, lag_steps=0, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
            if "gate" in str(target_entity_id).lower() or target_node_kind == "gate":
                events.append(_event("gate_state_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or effect_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
                events.append(_event("gate_state_changed_after_trigger", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or effect_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
                if detector_backed_poi:
                    events.append(_event("poi_visit_then_gate_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or target_entity_id, other_node_id=effect_node_id, lag_steps=0, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode))
                    events.append(_event("verified_gate_match", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or target_entity_id, other_node_id=effect_node_id, lag_steps=0, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode, chain_id=str(step.get("chain_id") or ""), step_id=str(step.get("step_id") or "")))
        for node_id, node in nodes_by_id.items():
            pattern_id = str(node.get("pattern_id") or "")
            if pattern_id:
                events.append(_event("pattern_observed", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier=str(node.get("evidence_tier") or "hypothesized"), provenance="graph", node_id=str(node_id), pattern_id=pattern_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
                if target_node_id and action_family == "interact":
                    events.append(_event("panel_match_observed_after_trigger", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier=str(node.get("evidence_tier") or "hypothesized"), provenance="analysis", node_id=target_node_id, other_node_id=str(node_id), pattern_id=pattern_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
                    if detector_backed_poi:
                        events.append(_event("poi_visit_then_panel_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier=str(node.get("evidence_tier") or "hypothesized"), provenance="analysis", node_id=target_node_id, other_node_id=str(node_id), pattern_id=pattern_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode))
                        events.append(_event("verified_panel_state", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier=str(node.get("evidence_tier") or "hypothesized"), provenance="analysis", node_id=target_node_id, other_node_id=str(node_id), pattern_id=pattern_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode, chain_id=str(step.get("chain_id") or ""), step_id=str(step.get("step_id") or "")))
                elif analysis_mode == "directed_outcome":
                    events.append(_event("expected_match_missing", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="hypothesized", provenance="analysis", node_id=str(node_id), pattern_id=pattern_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
        if bool(raw_episode.won):
            exit_node_id = next((str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "exit"), "exit:unknown")
            events.append(_event("exit_success", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="env_native", node_id=exit_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
        elif action_family in {"move", "interact"}:
            exit_node_id = next((str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "exit"), "")
            if exit_node_id:
                events.append(_event("exit_attempt", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="hypothesized", provenance="analysis", node_id=exit_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
                if target_node_id:
                    events.append(_event("exit_attempt_after_trigger", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="hypothesized", provenance="analysis", node_id=target_node_id, other_node_id=exit_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx]))
                if detector_backed_poi:
                    events.append(_event("poi_visit_then_exit_attempt_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="hypothesized", provenance="analysis", node_id=target_node_id or target_entity_id, other_node_id=exit_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], poi_id=target_entity_id, directed_or_probe=analysis_mode))
                if bool(step.get("position_hold_detected")):
                    events.append(_event("position_hold_after_exit_attempt", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=exit_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], chain_id=str(step.get("chain_id") or ""), step_id=str(step.get("step_id") or "")))
                if bool(step.get("exit_attempt_failed_without_new_support")):
                    events.append(_event("exit_attempt_failed_without_new_support", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=exit_node_id, analysis_mode=analysis_mode, source_step_ids=[step_idx], target_step_ids=[step_idx], chain_id=str(step.get("chain_id") or ""), step_id=str(step.get("step_id") or "")))
    pattern_nodes = [dict(node, node_id=node_id) for node_id, node in nodes_by_id.items() if str(node.get("pattern_id") or "")]
    for idx, left in enumerate(pattern_nodes):
        for right in pattern_nodes[idx + 1:]:
            if str(left.get("pattern_id")) == str(right.get("pattern_id")):
                events.append(_event("pattern_match", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=max(0, len(step_rows) - 1), supporting_row_ids=["pattern_match"], evidence_tier="hypothesized", provenance="graph", node_id=str(left["node_id"]), other_node_id=str(right["node_id"]), pattern_id=str(left.get("pattern_id")), analysis_mode=analysis_mode))
    exit_node_id = next((str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "exit"), "")
    trigger_nodes = [str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "trigger"]
    gate_nodes = [str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "gate"]
    if exit_node_id:
        for trigger_node_id in trigger_nodes:
            events.append(_event("trigger_before_exit", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=0, supporting_row_ids=["trigger_exit"], evidence_tier="hypothesized", provenance="graph", node_id=trigger_node_id, other_node_id=exit_node_id, lag_steps=1, analysis_mode=analysis_mode))
            events.append(_event("exit_attempt_without_trigger", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=0, supporting_row_ids=["exit_without_trigger"], evidence_tier="hypothesized", provenance="graph", node_id=exit_node_id, other_node_id=trigger_node_id, lag_steps=1, analysis_mode=analysis_mode))
        for gate_node_id in gate_nodes:
            for trigger_node_id in trigger_nodes:
                events.append(_event("trigger_before_gate", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=0, supporting_row_ids=["trigger_gate"], evidence_tier="hypothesized", provenance="graph", node_id=trigger_node_id, other_node_id=gate_node_id, lag_steps=1, analysis_mode=analysis_mode))
        if not raw_episode.won:
            events.append(_event("exit_failure", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=max(0, len(step_rows) - 1), supporting_row_ids=["exit_failure"], evidence_tier="observed", provenance="env_native", node_id=exit_node_id, analysis_mode=analysis_mode))
            events.append(_event("counterfactual_failure", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=max(0, len(step_rows) - 1), supporting_row_ids=["counterfactual_failure"], evidence_tier="observed", provenance="env_native", node_id=exit_node_id, analysis_mode=analysis_mode))
    poi_with_followup = {
        str(event.get("poi_id") or "")
        for event in events
        if str(event.get("event_kind") or "").startswith("poi_visit_then_")
    }
    poi_visit_counts: dict[str, int] = {}
    for event in events:
        if str(event.get("event_kind") or "") != "poi_visit":
            continue
        poi_id = str(event.get("poi_id") or "")
        if poi_id:
            poi_visit_counts[poi_id] = poi_visit_counts.get(poi_id, 0) + 1
    for poi_id, count in poi_visit_counts.items():
        if count >= 2 and poi_id not in poi_with_followup:
            events.append(_event("repeated_probe_without_new_effect", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=max(0, len(step_rows) - 1), supporting_row_ids=["repeated_probe_without_new_effect"], evidence_tier="hypothesized", provenance="analysis", node_id=poi_id, analysis_mode=analysis_mode, poi_id=poi_id, directed_or_probe=analysis_mode))
    return events


def _event(event_kind: str, *, round_id: int, episode_id: str, step_index: int, supporting_row_ids: list[str], evidence_tier: str, provenance: str, node_id: str, other_node_id: str | None = None, pattern_id: str | None = None, lag_steps: int | None = None, analysis_mode: str = "probe", source_step_ids: list[int] | None = None, target_step_ids: list[int] | None = None, poi_id: str | None = None, directed_or_probe: str | None = None, chain_id: str | None = None, step_id: str | None = None) -> dict:
    payload = {
        "event_kind": event_kind,
        "step_index": int(step_index),
        "round_id": int(round_id),
        "episode_id": str(episode_id),
        "supporting_row_ids": list(dict.fromkeys(str(value) for value in list(supporting_row_ids or []) if value)),
        "evidence_tier": str(evidence_tier),
        "provenance": str(provenance),
        "node_id": str(node_id),
        "analysis_mode": str(analysis_mode),
        "source_step_ids": list(source_step_ids or ([int(step_index)] if source_step_ids is None else [])),
        "target_step_ids": list(target_step_ids or ([int(step_index)] if target_step_ids is None else [])),
    }
    if other_node_id is not None:
        payload["other_node_id"] = str(other_node_id)
    if pattern_id is not None:
        payload["pattern_id"] = str(pattern_id)
    if lag_steps is not None:
        payload["lag_steps"] = int(lag_steps)
    if poi_id is not None:
        payload["poi_id"] = str(poi_id)
    if directed_or_probe is not None:
        payload["directed_or_probe"] = str(directed_or_probe)
    if chain_id:
        payload["chain_id"] = str(chain_id)
    if step_id:
        payload["step_id"] = str(step_id)
    payload["event_id"] = f"event:{stable_digest(payload)}"
    return payload

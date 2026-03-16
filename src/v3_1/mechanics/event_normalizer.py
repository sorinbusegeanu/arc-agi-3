from __future__ import annotations

from v3_1.contracts.messages import AnalyzedEpisode, RawEpisode
from v3_1.utils.ids import stable_digest


def normalize_events(raw_episode: RawEpisode, analyzed_episode: AnalyzedEpisode, mechanic_graph_snapshot: dict, blackboard_snapshot: dict) -> list[dict]:
    del blackboard_snapshot
    graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    nodes_by_id = dict(graph_state.get("nodes_by_id", {}))
    step_rows = list(dict(analyzed_episode.summary or {}).get("step_rows", []) or [])
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
        if action_family == "interact" and target_node_id:
            events.append(_event("contact", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id))
        if isinstance(step.get("avatar_cell"), list):
            region_id = f"region:{stable_digest((step.get('area_id'), step.get('avatar_cell')))}"
            events.append(_event("enter_region", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=region_id))
        if int(step.get("changed_cells", 0) or 0) > 0:
            telemetry = dict(step.get("telemetry", {}) or {})
            effect_region = dict(telemetry.get("effect_region", {}) or {})
            effect_node_id = f"effect_region:{stable_digest(effect_region.get('bbox') or step_idx)}"
            events.append(_event("remote_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=effect_node_id))
            if "gate" in str(target_entity_id).lower():
                events.append(_event("gate_state_change", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="analysis", node_id=target_node_id or effect_node_id))
        for node_id, node in nodes_by_id.items():
            pattern_id = str(node.get("pattern_id") or "")
            if pattern_id:
                events.append(_event("pattern_observed", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier=str(node.get("evidence_tier") or "hypothesized"), provenance="graph", node_id=str(node_id), pattern_id=pattern_id))
        if bool(raw_episode.won):
            exit_node_id = next((str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "exit"), "exit:unknown")
            events.append(_event("exit_success", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="observed", provenance="env_native", node_id=exit_node_id))
        elif action_family in {"move", "interact"}:
            exit_node_id = next((str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "exit"), "")
            if exit_node_id:
                events.append(_event("exit_attempt", round_id=raw_episode.round_id, episode_id=episode_id, step_index=step_idx, supporting_row_ids=[f"step:{step_idx}"], evidence_tier="hypothesized", provenance="analysis", node_id=exit_node_id))
    pattern_nodes = [dict(node, node_id=node_id) for node_id, node in nodes_by_id.items() if str(node.get("pattern_id") or "")]
    for idx, left in enumerate(pattern_nodes):
        for right in pattern_nodes[idx + 1:]:
            if str(left.get("pattern_id")) == str(right.get("pattern_id")):
                events.append(_event("pattern_match", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=max(0, len(step_rows) - 1), supporting_row_ids=["pattern_match"], evidence_tier="hypothesized", provenance="graph", node_id=str(left["node_id"]), other_node_id=str(right["node_id"]), pattern_id=str(left.get("pattern_id"))))
    exit_node_id = next((str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "exit"), "")
    trigger_nodes = [str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "trigger"]
    gate_nodes = [str(node_id) for node_id, node in nodes_by_id.items() if str(node.get("node_kind") or "") == "gate"]
    if exit_node_id:
        for trigger_node_id in trigger_nodes:
            events.append(_event("trigger_before_exit", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=0, supporting_row_ids=["trigger_exit"], evidence_tier="hypothesized", provenance="graph", node_id=trigger_node_id, other_node_id=exit_node_id, lag_steps=1))
        for gate_node_id in gate_nodes:
            for trigger_node_id in trigger_nodes:
                events.append(_event("trigger_before_gate", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=0, supporting_row_ids=["trigger_gate"], evidence_tier="hypothesized", provenance="graph", node_id=trigger_node_id, other_node_id=gate_node_id, lag_steps=1))
        if not raw_episode.won:
            events.append(_event("exit_failure", round_id=raw_episode.round_id, episode_id=raw_episode.episode_id, step_index=max(0, len(step_rows) - 1), supporting_row_ids=["exit_failure"], evidence_tier="observed", provenance="env_native", node_id=exit_node_id))
    return events


def _event(event_kind: str, *, round_id: int, episode_id: str, step_index: int, supporting_row_ids: list[str], evidence_tier: str, provenance: str, node_id: str, other_node_id: str | None = None, pattern_id: str | None = None, lag_steps: int | None = None) -> dict:
    payload = {
        "event_kind": event_kind,
        "step_index": int(step_index),
        "round_id": int(round_id),
        "episode_id": str(episode_id),
        "supporting_row_ids": list(dict.fromkeys(str(value) for value in list(supporting_row_ids or []) if value)),
        "evidence_tier": str(evidence_tier),
        "provenance": str(provenance),
        "node_id": str(node_id),
    }
    if other_node_id is not None:
        payload["other_node_id"] = str(other_node_id)
    if pattern_id is not None:
        payload["pattern_id"] = str(pattern_id)
    if lag_steps is not None:
        payload["lag_steps"] = int(lag_steps)
    payload["event_id"] = f"event:{stable_digest(payload)}"
    return payload

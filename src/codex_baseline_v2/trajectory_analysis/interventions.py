from __future__ import annotations

from typing import Dict, List, Optional

from codex_baseline_v2.shared.config import ExecutorConfigV2
from codex_baseline_v2.shared.schemas import ExecutorOutcomeV2, InterventionRecordV2, SCHEMA_VERSION, TrajectoryEpisodeV2


def _instruction_payload(episode: TrajectoryEpisodeV2) -> Dict[str, object]:
    metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
    instruction = metadata.get("instruction")
    if isinstance(instruction, dict):
        return instruction
    fallback = {
        "instruction_id": metadata.get("instruction_id"),
        "target_poi_id": metadata.get("target_poi_id"),
        "mode": metadata.get("mode") or metadata.get("collection_mode"),
        "target_trigger_zone_id": metadata.get("target_trigger_zone_id"),
        "intent_class": metadata.get("intent_class"),
        "probe_mode": metadata.get("probe_mode"),
    }
    return {k: v for k, v in fallback.items() if v is not None}


def _route_edge_ids(episode: TrajectoryEpisodeV2) -> List[str]:
    out: List[str] = []
    for step in episode.steps:
        route_ids = step.info.get("route_edge_ids") if isinstance(step.info, dict) else None
        if isinstance(route_ids, list):
            out.extend(str(value) for value in route_ids)
    return sorted(set(out))


def _contact_step_idx(episode: TrajectoryEpisodeV2, outcome: Optional[ExecutorOutcomeV2]) -> Optional[int]:
    for step in episode.steps:
        progress = step.info.get("progress") if isinstance(step.info, dict) else None
        if isinstance(progress, dict) and progress.get("contact") is True:
            return step.step_idx
    if outcome is not None and outcome.contact and episode.steps:
        return episode.steps[-1].step_idx
    return None


def _effect_event_ids(episode: TrajectoryEpisodeV2, contact_step_idx: Optional[int], cfg: ExecutorConfigV2) -> List[str]:
    if contact_step_idx is None:
        return []
    out: List[str] = []
    for step in episode.steps:
        if step.step_idx < contact_step_idx:
            continue
        if step.step_idx > contact_step_idx + cfg.post_contact_observation_steps:
            continue
        out.extend(step.event_ids)
    return sorted(set(out))


def _distance_new(step: object) -> Optional[float]:
    if not isinstance(getattr(step, "info", None), dict):
        return None
    progress = step.info.get("progress")
    if not isinstance(progress, dict):
        return None
    value = progress.get("distance_new")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_intervention_records(
    execution_episodes: List[TrajectoryEpisodeV2],
    executor_outcomes: List[ExecutorOutcomeV2],
    cfg: ExecutorConfigV2,
) -> list[InterventionRecordV2]:
    outcome_by_instruction = {row.instruction_id: row for row in executor_outcomes}
    records: List[InterventionRecordV2] = []
    for episode in execution_episodes:
        instruction = _instruction_payload(episode)
        if not instruction:
            continue
        instruction_id = str(instruction.get("instruction_id", episode.episode_id))
        outcome = outcome_by_instruction.get(instruction_id)
        route_edge_ids = _route_edge_ids(episode)
        contact_step_idx = _contact_step_idx(episode, outcome)
        reached = bool(outcome.reached) if outcome is not None else any(
            distance is not None and distance <= float(cfg.max_local_probe_steps)
            for distance in (_distance_new(step) for step in episode.steps)
        )
        blocked = bool(outcome.blocked) if outcome is not None else all(
            isinstance(step.info.get("progress"), dict) and float(step.info["progress"].get("distance_delta") or 0.0) <= 0.0
            for step in episode.steps
        )
        if outcome is None:
            blocked = all(
                isinstance(step.info.get("progress"), dict) and float(step.info["progress"].get("distance_delta") or 0.0) <= 0.0
                for step in episode.steps[-3:]
            )
        progress_confidence = 0.0
        if outcome is not None and outcome.target_progress:
            progress_confidence = 1.0 if min(outcome.target_progress) <= cfg.target_reach_distance else 0.65
        elif reached:
            progress_confidence = 0.6
        effect_event_ids = _effect_event_ids(episode, contact_step_idx, cfg)
        intended_contact_mode = "unknown"
        mode = str(instruction.get("mode", ""))
        if mode == "poi_interaction":
            intended_contact_mode = "overlap_contact"
        elif mode == "poi_approach":
            intended_contact_mode = "adjacent_contact"
        immediate_event_ids = list(effect_event_ids[:1])
        delayed_event_ids = list(effect_event_ids[1:])
        post_transition_event_ids: List[str] = []
        null_effect = len(effect_event_ids) == 0
        records.append(
            InterventionRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=episode.game_id,
                round_id=int(instruction.get("round_id", 0)),
                instruction_id=instruction_id,
                target_poi_id=instruction.get("target_poi_id"),
                target_area_id=None,
                intended_contact_mode=intended_contact_mode,
                start_episode_id=episode.episode_id,
                start_step_idx=episode.steps[0].step_idx if episode.steps else 0,
                contact_step_idx=contact_step_idx,
                end_step_idx=episode.steps[-1].step_idx if episode.steps else 0,
                route_edge_ids=route_edge_ids,
                reached=reached,
                contact=bool(outcome.contact) if outcome is not None else contact_step_idx is not None,
                blocked=blocked,
                progress_confidence=progress_confidence,
                effect_event_ids=effect_event_ids,
                notes=[mode] if mode else [],
                target_trigger_zone_id=instruction.get("target_trigger_zone_id"),
                intent_class=str(instruction.get("intent_class", "poi_interaction_probe")),
                probe_mode=str(instruction.get("probe_mode", mode or "move_to_visible_poi")) if instruction.get("probe_mode", mode) is not None else None,
                probe_outcome_ids=[],
                immediate_event_ids=immediate_event_ids,
                delayed_event_ids=delayed_event_ids,
                post_transition_event_ids=post_transition_event_ids,
                null_effect=null_effect,
            )
        )
    return records

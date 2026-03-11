from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

from codex_baseline_v2.shared.config import ActionSemanticsConfigV2
from codex_baseline_v2.shared.schemas import (
    ActionContextStatsV2,
    ActionSemanticsStatsV2,
    SCHEMA_VERSION,
    TrajectoryEpisodeV2,
    TrajectoryStepV2,
)
from codex_baseline_v2.shared.utils import BBox, compact_context_key


def _avatar_position(step: TrajectoryStepV2) -> Optional[Tuple[float, float]]:
    return step.actual_avatar_centroid or step.predicted_avatar_centroid


def _occupancy_code(step: TrajectoryStepV2) -> str:
    obs = step.observation
    pos = _avatar_position(step)
    if obs is None or pos is None or not obs or not obs[0]:
        return "unk"
    x = int(round(pos[0]))
    y = int(round(pos[1]))
    height = len(obs)
    width = len(obs[0])
    center = int(obs[y][x]) if 0 <= y < height and 0 <= x < width else -1
    mask = []
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx = x + dx
        ny = y + dy
        if 0 <= ny < height and 0 <= nx < width:
            mask.append("1" if int(obs[ny][nx]) != center else "0")
        else:
            mask.append("x")
    return "".join(mask)


def _target_bucket(step: TrajectoryStepV2) -> str:
    pos = _avatar_position(step)
    geom = step.target_geometry
    if pos is None or geom is None:
        return "none"
    gx, gy = geom.centroid()
    dist = abs(pos[0] - gx) + abs(pos[1] - gy)
    if dist <= 1.5:
        return "near"
    if dist <= 4.0:
        return "mid"
    return "far"


def _step_context_key(step: TrajectoryStepV2) -> str:
    if step.action_context_key:
        return step.action_context_key
    return compact_context_key(step.area_id, _occupancy_code(step), _target_bucket(step), step.action.action_id)


def _motion(prev_step: Optional[TrajectoryStepV2], step: TrajectoryStepV2) -> Tuple[float, float]:
    prev_pos = _avatar_position(prev_step) if prev_step is not None else None
    cur_pos = _avatar_position(step)
    if prev_pos is None or cur_pos is None:
        return (0.0, 0.0)
    return (float(cur_pos[0] - prev_pos[0]), float(cur_pos[1] - prev_pos[1]))


def _change_ratio(step: TrajectoryStepV2) -> float:
    obs = step.observation
    if obs is None or not obs or not obs[0]:
        return 0.0
    change = step.info.get("change_count") if isinstance(step.info, dict) else None
    if change is None:
        return 0.0
    return float(change) / float(max(1, len(obs) * len(obs[0])))


def _transition_like(step: TrajectoryStepV2, change_ratio: float, cfg: ActionSemanticsConfigV2) -> bool:
    if step.done:
        return True
    if step.info.get("transition_like") is True:
        return True
    return change_ratio >= cfg.transition_change_threshold


def _interaction_like(step: TrajectoryStepV2, change_ratio: float, cfg: ActionSemanticsConfigV2) -> bool:
    if step.event_ids:
        return True
    if abs(float(step.reward)) > 0.0:
        return True
    return change_ratio >= cfg.interaction_change_threshold


def _classify_motion(
    sample_count: int,
    blocked_count: int,
    noop_count: int,
    interaction_like_count: int,
    transition_like_count: int,
    mean_dx: float,
    mean_dy: float,
    cfg: ActionSemanticsConfigV2,
) -> str:
    if sample_count <= 0:
        return "ambiguous"
    blocked_rate = blocked_count / float(sample_count)
    noop_rate = noop_count / float(sample_count)
    interaction_rate = interaction_like_count / float(sample_count)
    transition_rate = transition_like_count / float(sample_count)
    mean_motion = abs(mean_dx) + abs(mean_dy)
    if transition_rate >= cfg.transition_change_threshold:
        return "transition_like"
    if interaction_rate >= cfg.interaction_change_threshold and mean_motion <= 1.0:
        return "interaction_like"
    if blocked_rate >= cfg.blocked_motion_threshold:
        return "blocked_like"
    if noop_rate >= 0.6:
        return "noop_like"
    if mean_motion > cfg.noop_motion_threshold:
        return "move_like"
    return "ambiguous"


def _weighted_mean(old_mean: float, old_count: int, new_mean: float, new_count: int) -> float:
    total = old_count + new_count
    if total <= 0:
        return 0.0
    return ((old_mean * old_count) + (new_mean * new_count)) / float(total)


def _weighted_std(
    old_std: float,
    old_mean: float,
    old_count: int,
    new_std: float,
    new_mean: float,
    new_count: int,
) -> float:
    total = old_count + new_count
    if total <= 0:
        return 0.0
    old_var = old_std * old_std
    new_var = new_std * new_std
    merged_mean = _weighted_mean(old_mean, old_count, new_mean, new_count)
    merged_var = (
        old_count * (old_var + (old_mean - merged_mean) ** 2)
        + new_count * (new_var + (new_mean - merged_mean) ** 2)
    ) / float(total)
    return sqrt(max(0.0, merged_var))


def infer_action_semantics_from_episodes(
    episodes: List[TrajectoryEpisodeV2],
    cfg: ActionSemanticsConfigV2,
    existing_table: Optional[List[ActionSemanticsStatsV2]] = None,
    existing_context_table: Optional[List[ActionContextStatsV2]] = None,
) -> tuple[list[ActionSemanticsStatsV2], list[ActionContextStatsV2]]:
    per_action: DefaultDict[int, List[Tuple[float, float, bool, bool, bool, bool]]] = defaultdict(list)
    per_context: DefaultDict[Tuple[int, str], List[Tuple[float, float, bool, bool, bool]]] = defaultdict(list)
    round_id = 0
    game_id = "unknown_game"

    for episode in episodes:
        game_id = episode.game_id
        prev_step: Optional[TrajectoryStepV2] = None
        for step in episode.steps:
            if step.action.action_id is None:
                prev_step = step
                continue
            action_id = int(step.action.action_id)
            dx, dy = _motion(prev_step, step)
            abs_motion = abs(dx) + abs(dy)
            change_ratio = _change_ratio(step)
            transition_like = _transition_like(step, change_ratio, cfg)
            interaction_like = _interaction_like(step, change_ratio, cfg)
            blocked = abs_motion <= cfg.blocked_motion_threshold and not transition_like and not interaction_like
            noop = abs_motion <= cfg.noop_motion_threshold and not transition_like
            success = not blocked
            context_key = _step_context_key(step)
            per_action[action_id].append((dx, dy, success, blocked, noop, interaction_like or transition_like))
            per_context[(action_id, context_key)].append((dx, dy, success, blocked, transition_like))
            prev_step = step

    existing_by_action: Dict[int, ActionSemanticsStatsV2] = {row.action_id: row for row in (existing_table or [])}
    merged_actions: Dict[int, ActionSemanticsStatsV2] = dict(existing_by_action)
    for action_id, rows in per_action.items():
        sample_count = len(rows)
        mean_dx = sum(v[0] for v in rows) / float(max(1, sample_count))
        mean_dy = sum(v[1] for v in rows) / float(max(1, sample_count))
        std_dx = sqrt(sum((v[0] - mean_dx) ** 2 for v in rows) / float(max(1, sample_count)))
        std_dy = sqrt(sum((v[1] - mean_dy) ** 2 for v in rows) / float(max(1, sample_count)))
        success_count = sum(1 for v in rows if v[2])
        blocked_count = sum(1 for v in rows if v[3])
        noop_count = sum(1 for v in rows if v[4])
        effect_like_count = sum(1 for v in rows if v[5])
        transition_like_count = sum(1 for v in rows if v[5] and (abs(v[0]) + abs(v[1]) <= cfg.noop_motion_threshold))
        prior = existing_by_action.get(action_id)
        old_count = prior.sample_count if prior is not None else 0
        total = old_count + sample_count
        merged_mean_dx = _weighted_mean(prior.mean_dx if prior else 0.0, old_count, mean_dx, sample_count)
        merged_mean_dy = _weighted_mean(prior.mean_dy if prior else 0.0, old_count, mean_dy, sample_count)
        merged_std_dx = _weighted_std(prior.std_dx if prior else 0.0, prior.mean_dx if prior else 0.0, old_count, std_dx, mean_dx, sample_count)
        merged_std_dy = _weighted_std(prior.std_dy if prior else 0.0, prior.mean_dy if prior else 0.0, old_count, std_dy, mean_dy, sample_count)
        merged_success = success_count + (prior.success_count if prior else 0)
        merged_blocked = blocked_count + (prior.blocked_count if prior else 0)
        merged_noop = noop_count + (prior.noop_count if prior else 0)
        merged_interaction = effect_like_count + (prior.interaction_like_count if prior else 0)
        merged_transition = transition_like_count + (prior.transition_like_count if prior else 0)
        dominant_motion_class = _classify_motion(
            total,
            merged_blocked,
            merged_noop,
            merged_interaction,
            merged_transition,
            merged_mean_dx,
            merged_mean_dy,
            cfg,
        )
        confidence = min(1.0, total / float(max(1, cfg.min_samples_per_action)))
        merged_actions[action_id] = ActionSemanticsStatsV2(
            schema_version=SCHEMA_VERSION,
            game_id=game_id if prior is None else prior.game_id,
            action_id=action_id,
            sample_count=total,
            success_count=merged_success,
            blocked_count=merged_blocked,
            noop_count=merged_noop,
            interaction_like_count=merged_interaction,
            transition_like_count=merged_transition,
            mean_dx=merged_mean_dx,
            mean_dy=merged_mean_dy,
            std_dx=merged_std_dx,
            std_dy=merged_std_dy,
            dominant_motion_class=dominant_motion_class,
            confidence=confidence,
            last_updated_round=round_id if episodes else (prior.last_updated_round if prior else 0),
        )

    existing_by_context: Dict[Tuple[int, str], ActionContextStatsV2] = {
        (row.action_id, row.context_key): row for row in (existing_context_table or [])
    }
    merged_context: Dict[Tuple[int, str], ActionContextStatsV2] = dict(existing_by_context)
    for key, rows in per_context.items():
        action_id, context_key = key
        sample_count = len(rows)
        mean_dx = sum(v[0] for v in rows) / float(max(1, sample_count))
        mean_dy = sum(v[1] for v in rows) / float(max(1, sample_count))
        success_rate = sum(1 for v in rows if v[2]) / float(max(1, sample_count))
        blocked_rate = sum(1 for v in rows if v[3]) / float(max(1, sample_count))
        transition_rate = sum(1 for v in rows if v[4]) / float(max(1, sample_count))
        prior = existing_by_context.get(key)
        old_count = prior.sample_count if prior is not None else 0
        total = old_count + sample_count
        merged_context[key] = ActionContextStatsV2(
            schema_version=SCHEMA_VERSION,
            game_id=game_id if prior is None else prior.game_id,
            action_id=action_id,
            context_key=context_key,
            sample_count=total,
            mean_dx=_weighted_mean(prior.mean_dx if prior else 0.0, old_count, mean_dx, sample_count),
            mean_dy=_weighted_mean(prior.mean_dy if prior else 0.0, old_count, mean_dy, sample_count),
            success_rate=_weighted_mean(prior.success_rate if prior else 0.0, old_count, success_rate, sample_count),
            blocked_rate=_weighted_mean(prior.blocked_rate if prior else 0.0, old_count, blocked_rate, sample_count),
            transition_rate=_weighted_mean(prior.transition_rate if prior else 0.0, old_count, transition_rate, sample_count),
            confidence=min(1.0, total / float(max(1, cfg.min_samples_per_context))),
        )

    ordered_actions = sorted(merged_actions.values(), key=lambda row: row.action_id)
    ordered_context = sorted(merged_context.values(), key=lambda row: (row.action_id, row.context_key))
    return ordered_actions, ordered_context

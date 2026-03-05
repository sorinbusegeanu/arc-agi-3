from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .lsa_bootstrap_explorer_types import (
    ActionV1,
    AvailableActionsV1,
    BootstrapExplorerReportV1,
    ObsV1,
    ProbeTraceV1,
    StepRecordV1,
    action_tags_for_id,
    canonical_points,
    count_true,
    normalize_action,
    normalize_available_actions,
    normalize_info,
    normalize_obs,
)
from .lsa_env_adapter import EnvAdapter


def run_bootstrap_explorer(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    env_adapter: EnvAdapter,
    probe_steps: int = 4,
    policy_config: Optional[Dict[str, Any]] = None,
    logging_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    policy_config = policy_config or {"mode": "safe_heuristic"}
    logging_config = logging_config or {}
    errors: List[str] = []

    trace_id = str(logging_config.get("trace_id") or _build_trace_id(episode_id, seed, logging_config))
    timestamp_step = int(logging_config.get("timestamp_step") or 0)
    dump_frames = bool(logging_config.get("dump_frames", False))
    frame_dir = logging_config.get("frame_dir")

    steps: List[Dict[str, Any]] = []
    unique_obs_hashes = set()
    max_unique_obs = 4
    invalid_actions = 0

    terminated_early_reason = "probe_steps_reached"
    probe_steps_executed = 0
    ended_with_done = False
    start_obs_shape = (0, 0)

    try:
        raw_obs = env_adapter.reset(episode_id, game_id, seed)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"reset_failed: {exc}")
        trace = _build_trace(
            episode_id=episode_id,
            game_id=game_id,
            seed=seed,
            trace_id=trace_id,
            timestamp_step=timestamp_step,
            probe_steps_requested=probe_steps,
            probe_steps_executed=0,
            terminated_early_reason="env_error",
            steps=[],
        )
        report = _build_report(
            episode_id=episode_id,
            game_id=game_id,
            seed=seed,
            trace_id=trace_id,
            timestamp_step=timestamp_step,
            start_obs_shape=start_obs_shape,
            probe_steps_executed=0,
            ended_with_done=False,
            action_selection_mode=str(policy_config.get("mode", "")),
            errors=errors,
            summary={},
        )
        return trace, report

    for t in range(int(probe_steps)):
        obs = normalize_obs(env_adapter.to_canonical_obs(raw_obs))
        if t == 0:
            start_obs_shape = (obs.h, obs.w)
        available_actions = normalize_available_actions(env_adapter.get_available_actions(raw_obs))
        action = choose_action(obs, available_actions, policy_config)
        action_valid: Optional[bool] = None
        reward: Optional[float] = None
        done = False
        info_passthrough: Optional[Dict[str, Any]] = None
        obs_next: Optional[ObsV1] = None

        unique_obs_hashes.add(_hash_obs(obs))
        if action is None:
            terminated_early_reason = "no_available_actions"
            steps.append(
                StepRecordV1(
                    t=t,
                    obs=obs.to_dict(),
                    available_actions=available_actions.to_dict(),
                    action=None,
                    action_valid=None,
                    reward=None,
                    done=False,
                    info_passthrough=None,
                ).to_dict()
            )
            break

        try:
            raw_obs_next, reward, done, info = env_adapter.step(action.to_dict())
            info_passthrough = normalize_info(info)
            if isinstance(info_passthrough, dict):
                if "action_valid" in info_passthrough:
                    action_valid = bool(info_passthrough.get("action_valid"))
                elif "valid" in info_passthrough:
                    action_valid = bool(info_passthrough.get("valid"))
            obs_next = normalize_obs(env_adapter.to_canonical_obs(raw_obs_next))
            raw_obs = raw_obs_next
        except Exception as exc:  # noqa: BLE001
            errors.append(f"step_failed_t{t}: {exc}")
            terminated_early_reason = "env_error"
            steps.append(
                StepRecordV1(
                    t=t,
                    obs=obs.to_dict(),
                    available_actions=available_actions.to_dict(),
                    action=action.to_dict(),
                    action_valid=action_valid,
                    reward=reward,
                    done=False,
                    info_passthrough=info_passthrough,
                ).to_dict()
            )
            break

        if action_valid is False:
            invalid_actions += 1

        step_record = StepRecordV1(
            t=t,
            obs=obs.to_dict(),
            available_actions=available_actions.to_dict(),
            action=action.to_dict(),
            action_valid=action_valid,
            reward=reward,
            done=bool(done),
            info_passthrough=info_passthrough,
        )
        steps.append(step_record.to_dict())
        probe_steps_executed = t + 1

        if dump_frames and frame_dir is not None:
            _dump_frame(frame_dir, episode_id, trace_id, t, obs)

        if done:
            ended_with_done = True
            terminated_early_reason = "done"
            break

        if len(unique_obs_hashes) >= max_unique_obs:
            terminated_early_reason = "probe_steps_reached"
            break

    if terminated_early_reason not in ("done", "env_error", "no_available_actions"):
        terminated_early_reason = "probe_steps_reached"

    summary: Dict[str, Any] = {}
    if unique_obs_hashes:
        summary["num_unique_obs_hashes"] = len(unique_obs_hashes)
    if invalid_actions:
        summary["num_invalid_actions"] = invalid_actions

    trace = _build_trace(
        episode_id=episode_id,
        game_id=game_id,
        seed=seed,
        trace_id=trace_id,
        timestamp_step=timestamp_step,
        probe_steps_requested=int(probe_steps),
        probe_steps_executed=probe_steps_executed,
        terminated_early_reason=terminated_early_reason,
        steps=steps,
    )
    # Compute cell-level diffs between consecutive unique frames and embed in trace.
    unique_frames: List[List[List[int]]] = []
    seen_frame_keys: set = set()
    for step in steps:
        grid = (step.get("obs") or {}).get("grid")
        if isinstance(grid, list):
            fkey = json.dumps(grid, separators=(",", ":"))
            if fkey not in seen_frame_keys:
                seen_frame_keys.add(fkey)
                unique_frames.append(grid)
                if len(unique_frames) >= 4:
                    break
    trace["frame_diffs"] = _compute_frame_diffs(unique_frames)

    report = _build_report(
        episode_id=episode_id,
        game_id=game_id,
        seed=seed,
        trace_id=trace_id,
        timestamp_step=timestamp_step,
        start_obs_shape=start_obs_shape,
        probe_steps_executed=probe_steps_executed,
        ended_with_done=ended_with_done,
        action_selection_mode=str(policy_config.get("mode", "safe_heuristic")),
        errors=errors,
        summary=summary,
    )
    return trace, report


def choose_action(obs: ObsV1, available_actions: AvailableActionsV1, policy_config: Dict[str, Any]) -> Optional[ActionV1]:
    mode = str(policy_config.get("mode", "safe_heuristic"))
    fixed_actions = policy_config.get("fixed_actions") or []

    if mode == "fixed_action_list":
        for template in fixed_actions:
            action = normalize_action(template)
            if action and _action_available(action, available_actions):
                return action
        mode = "safe_heuristic"

    if mode == "safe_heuristic":
        action = _safe_heuristic_action(available_actions)
        if action is not None:
            return action

    if mode == "coordinate_canonical_points" or available_actions.coord_enabled:
        return _coordinate_action(obs, available_actions)

    return None


def _safe_heuristic_action(available_actions: AvailableActionsV1) -> Optional[ActionV1]:
    if available_actions.discrete_mask:
        for idx, enabled in enumerate(available_actions.discrete_mask):
            if not enabled:
                continue
            tags = action_tags_for_id(available_actions.tags, idx)
            if "safe" in tags or "noop" in tags:
                return ActionV1(type="discrete", id=idx)
        for idx, enabled in enumerate(available_actions.discrete_mask):
            if enabled:
                return ActionV1(type="discrete", id=idx)
    return None


def _coordinate_action(obs: ObsV1, available_actions: AvailableActionsV1) -> Optional[ActionV1]:
    if not available_actions.coord_enabled:
        return None
    point_list = canonical_points(obs.h, obs.w)
    if available_actions.allowed_coords:
        allowed = set(available_actions.allowed_coords)
        point_list = [pt for pt in point_list if pt in allowed]
    if available_actions.coord_bounds:
        x0, y0, x1, y1 = available_actions.coord_bounds
        filtered: List[Tuple[int, int]] = []
        for x, y in point_list:
            if x0 <= x <= x1 and y0 <= y <= y1:
                filtered.append((x, y))
        point_list = filtered
    if not point_list:
        return None
    x, y = point_list[0]
    return ActionV1(type="coord", id=int(available_actions.coord_action_id or 0), x=int(x), y=int(y))


def _action_available(action: ActionV1, available_actions: AvailableActionsV1) -> bool:
    if action.type == "discrete":
        if not available_actions.discrete_mask:
            return False
        if action.id < 0 or action.id >= len(available_actions.discrete_mask):
            return False
        return bool(available_actions.discrete_mask[action.id])
    if action.type == "coord":
        if not available_actions.coord_enabled:
            return False
        if action.x is None or action.y is None:
            return False
        if available_actions.allowed_coords and (action.x, action.y) not in set(available_actions.allowed_coords):
            return False
        if available_actions.coord_bounds:
            x0, y0, x1, y1 = available_actions.coord_bounds
            if not (x0 <= int(action.x) <= x1 and y0 <= int(action.y) <= y1):
                return False
        return True
    return False


def _build_trace(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    trace_id: str,
    timestamp_step: int,
    probe_steps_requested: int,
    probe_steps_executed: int,
    terminated_early_reason: str,
    steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    trace = ProbeTraceV1(
        schema_version="ProbeTraceV1",
        agent_name="lsa_bootstrap_explorer",
        episode_id=episode_id,
        game_id=game_id,
        seed=int(seed),
        trace_id=trace_id,
        timestamp_step=int(timestamp_step),
        probe_steps_requested=int(probe_steps_requested),
        probe_steps_executed=int(probe_steps_executed),
        terminated_early_reason=str(terminated_early_reason),
        steps=steps,
    )
    return trace.to_dict()


def _build_report(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    trace_id: str,
    timestamp_step: int,
    start_obs_shape: Tuple[int, int],
    probe_steps_executed: int,
    ended_with_done: bool,
    action_selection_mode: str,
    errors: List[str],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    report = BootstrapExplorerReportV1(
        schema_version="BootstrapExplorerReportV1",
        agent_name="lsa_bootstrap_explorer",
        episode_id=episode_id,
        game_id=game_id,
        seed=int(seed),
        trace_id=trace_id,
        timestamp_step=int(timestamp_step),
        start_obs_shape=tuple(start_obs_shape),
        probe_steps_executed=int(probe_steps_executed),
        ended_with_done=bool(ended_with_done),
        action_selection_mode=str(action_selection_mode),
        errors=list(errors),
        summary=dict(summary),
    )
    return report.to_dict()


def _hash_obs(obs: ObsV1) -> str:
    payload = json.dumps(obs.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_trace_id(episode_id: str, seed: int, logging_config: Dict[str, Any]) -> str:
    timestamp_step = logging_config.get("timestamp_step", 0)
    base = f"{episode_id}:{seed}:{timestamp_step}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _compute_frame_diffs(frames: List[List[List[int]]]) -> List[Dict[str, Any]]:
    """Return cell-level diffs between each pair of consecutive unique frames."""
    diffs: List[Dict[str, Any]] = []
    for i in range(len(frames) - 1):
        changes: List[Dict[str, Any]] = []
        for y, (row_a, row_b) in enumerate(zip(frames[i], frames[i + 1])):
            for x, (old_val, new_val) in enumerate(zip(row_a, row_b)):
                if old_val != new_val:
                    changes.append({
                        "x": int(x),
                        "y": int(y),
                        "old_value": int(old_val),
                        "new_value": int(new_val),
                    })
        diffs.append({"from_frame": i, "to_frame": i + 1, "changes": changes})
    return diffs


def _dump_frame(frame_dir: str, episode_id: str, trace_id: str, t: int, obs: ObsV1) -> None:
    os.makedirs(frame_dir, exist_ok=True)
    path = f"{frame_dir}/grid_{episode_id}_{trace_id}_{t}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obs.grid, f)

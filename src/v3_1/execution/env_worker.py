from __future__ import annotations

from dataclasses import dataclass, field
import random

from v3_1.contracts.messages import ExecutorRequest, ExecutorOutcome, RawEpisode, RawStep
from v3_1.execution.env_factory import NormalizedEnvAdapter, build_env, normalize_action_lookup
from v3_1.execution.live_avatar_tracker import LiveAvatarTracker
from v3_1.execution.option_execution import choose_directed_action, choose_probe_action
from v3_1.execution.outcomes import summarize_outcome
from v3_1.execution.route_execution import route_instruction


def _avatar_fields_from_belief(belief) -> tuple[list[int] | None, float, str, bool, list[list[int]]]:
    if belief is None:
        return None, 0.0, "unknown", False, []
    return (
        list(belief.cell) if isinstance(getattr(belief, "cell", None), list) else None,
        float(getattr(belief, "confidence", 0.0) or 0.0),
        str(getattr(belief, "source", "unknown") or "unknown"),
        bool(getattr(belief, "ambiguous", False)),
            [list(cell) for cell in list(getattr(belief, "candidate_cells", ()) or []) if isinstance(cell, list)],
    )


def _changed_cells(previous_observation, current_observation) -> int:
    if not isinstance(previous_observation, list) or not isinstance(current_observation, list):
        return 0
    changed = 0
    for prev_row, curr_row in zip(previous_observation, current_observation):
        if not isinstance(prev_row, list) or not isinstance(curr_row, list):
            continue
        for prev_value, curr_value in zip(prev_row, curr_row):
            if int(prev_value) != int(curr_value):
                changed += 1
    return changed


def _effect_region(previous_observation, current_observation) -> dict | None:
    if not isinstance(previous_observation, list) or not isinstance(current_observation, list):
        return None
    xs: list[int] = []
    ys: list[int] = []
    for y, (prev_row, curr_row) in enumerate(zip(previous_observation, current_observation)):
        if not isinstance(prev_row, list) or not isinstance(curr_row, list):
            continue
        for x, (prev_value, curr_value) in enumerate(zip(prev_row, curr_row)):
            if int(prev_value) != int(curr_value):
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return {"bbox": [min(xs), min(ys), max(xs), max(ys)], "changed_cells": len(xs)}


def _movement_direction_from_action(normalized_action: dict) -> str | None:
    action_name = str(normalized_action.get("action_name") or "").lower()
    for direction in ("up", "down", "left", "right"):
        if direction in action_name:
            return direction
    return None


def _boundary_direction(cell: list[int] | None) -> str | None:
    if not isinstance(cell, list) or len(cell) != 2:
        return None
    x = int(cell[0])
    y = int(cell[1])
    if x <= 1:
        return "left"
    if x >= 62:
        return "right"
    if y <= 1:
        return "up"
    if y >= 62:
        return "down"
    return None


def _resolve_area_id(*, info: dict, request_metadata: dict, navigation: dict, objective: dict, fallback: str | None = None) -> str | None:
    return (
        str(info.get("area_id") or info.get("current_area_id") or "")
        or str(request_metadata.get("expected_contact_region_or_boundary") or "")
        or str(navigation.get("local_area") or navigation.get("avatar_area") or "")
        or str(objective.get("target_area_id") or "")
        or (str(fallback) if fallback else "")
        or None
    )


def _step_telemetry(
    *,
    info: dict,
    chosen_action,
    available_actions,
    avatar_before,
    avatar_after,
    target_before,
    target_after,
    boundary_hit: bool,
    invalid_move: bool,
    reward: float,
    done: bool,
    truncated: bool,
    terminal_stop_reason: str | None,
    route_instruction_id: str | None,
    terminal_action_marker: bool,
    effect_region: dict | None,
    effect_changed_cells: int,
    attempted_movement_direction: str | None,
    attempted_interaction_target_cell: list[int] | None,
    attempted_interaction_target_object_id: str | None,
    boundary_contact_observed: bool,
    blocked_movement_observed: bool,
    portal_like_contact_observed: bool,
    terminal_affordance_contact_observed: bool,
    pre_step_area_id: str | None,
    post_step_area_id: str | None,
    attempted_move_into_blocked_boundary: bool,
    repeated_boundary_facing_movement: bool,
    movement_stayed_in_place_after_attempted_boundary_move: bool,
) -> dict:
    payload = dict(info)
    payload.update(
        {
            "chosen_action": chosen_action,
            "available_actions_at_decision_time": list(available_actions),
            "avatar_cell_before": avatar_before,
            "avatar_cell_after": avatar_after,
            "target_cell_before": target_before,
            "target_cell_after": target_after,
            "boundary_hit": bool(boundary_hit),
            "invalid_move": bool(invalid_move),
            "reward_observed": float(reward or 0.0),
            "done_observed": bool(done),
            "truncated_observed": bool(truncated),
            "terminal_stop_reason": terminal_stop_reason,
            "route_instruction_id": route_instruction_id,
            "terminal_action_marker": bool(terminal_action_marker),
            "effect_region": effect_region,
            "effect_changed_cells": int(effect_changed_cells or 0),
            "attempted_movement_direction": attempted_movement_direction,
            "attempted_interaction_target_cell": attempted_interaction_target_cell,
            "attempted_interaction_target_object_id": attempted_interaction_target_object_id,
            "boundary_contact_observed": bool(boundary_contact_observed),
            "blocked_movement_observed": bool(blocked_movement_observed),
            "portal_like_contact_observed": bool(portal_like_contact_observed),
            "terminal_affordance_contact_observed": bool(terminal_affordance_contact_observed),
            "pre_step_avatar_cell": avatar_before,
            "post_step_avatar_cell": avatar_after,
            "pre_step_area_id": pre_step_area_id,
            "post_step_area_id": post_step_area_id,
            "attempted_move_into_blocked_boundary": bool(attempted_move_into_blocked_boundary),
            "repeated_boundary_facing_movement": bool(repeated_boundary_facing_movement),
            "movement_stayed_in_place_after_attempted_boundary_move": bool(movement_stayed_in_place_after_attempted_boundary_move),
        }
    )
    return payload


@dataclass
class EnvWorker:
    worker_id: str
    env: NormalizedEnvAdapter
    reset_counter: int = 0
    last_observation: object | None = None
    last_info: dict = field(default_factory=dict)
    avatar_tracker: LiveAvatarTracker = field(default_factory=LiveAvatarTracker)

    @classmethod
    def from_config(cls, worker_id: str, *, env_factory: str | None, env_id: str | None, env_root: str | None, seed: int | None, render_terminal: bool = False) -> "EnvWorker":
        return cls(worker_id=worker_id, env=build_env(env_factory, env_id=env_id, env_root=env_root, seed=seed, render_terminal=render_terminal))

    def _reset(self, seed: int | None = None) -> tuple[object, dict]:
        observation, info = self.env.reset(seed=seed)
        self.reset_counter += 1
        self.last_observation = observation
        self.last_info = dict(info)
        self.avatar_tracker.reset(observation, info)
        return observation, dict(info)

    def _step(self, action) -> tuple[object, float, bool, bool, dict]:
        observation, reward, done, truncated, info = self.env.step(action)
        self.last_observation = observation
        self.last_info = dict(info)
        return observation, reward, done, truncated, dict(info)

    def _normalize_emitted_action(self, action: object, available_actions: list[dict]) -> dict:
        return normalize_action_lookup(action, available_actions)

    def _run_probe(self, request: ExecutorRequest) -> ExecutorOutcome:
        observation, info = self._reset(seed=request.metadata.get("seed") if isinstance(request.metadata, dict) else None)
        initial_observation = observation
        request_metadata = dict(request.metadata or {})
        steps = []
        rewards: list[float] = []
        action_history: list[object] = []
        routed_history: list[dict] = []
        no_change_aliases: list[str] = []
        rng = random.Random(int(request.metadata.get("seed", 0) or 0))
        for step_idx in range(request.max_steps):
            available_actions = list(info.get("available_actions", self.env.available_actions()))
            action = choose_probe_action(available_actions, action_history, recent_no_change_actions=no_change_aliases[-4:], rng=rng)
            normalized_action = self._normalize_emitted_action(action, available_actions)
            previous_observation = observation
            avatar_before_belief = self.avatar_tracker.current_belief()
            avatar_before, avatar_confidence_before, avatar_source_before, avatar_ambiguous_before, avatar_candidate_cells_before = _avatar_fields_from_belief(avatar_before_belief)
            observation, reward, done, truncated, info = self._step(action)
            avatar_after_belief = self.avatar_tracker.update(previous_observation, observation, action, info=info, step_index=step_idx)
            avatar_after = list(avatar_after_belief.best_cell) if avatar_after_belief.best_cell is not None else None
            pre_area_id = _resolve_area_id(info=self.last_info, request_metadata=request_metadata, navigation=dict(request.navigation or {}), objective=dict(request.objective or {}))
            post_area_id = _resolve_area_id(info=info, request_metadata=request_metadata, navigation=dict(request.navigation or {}), objective=dict(request.objective or {}), fallback=pre_area_id)
            attempted_movement_direction = _movement_direction_from_action(normalized_action)
            attempted_interaction_target_cell = list(request.click_target_coordinates) if isinstance(request.click_target_coordinates, list) else None
            attempted_interaction_target_object_id = str(request_metadata.get("expected_target_id") or request.target_entity_id or "") or None
            blocked_movement_observed = bool(observation == previous_observation and str(normalized_action.get("action_family", "unknown")) == "move")
            boundary_direction = _boundary_direction(avatar_before)
            boundary_contact_observed = bool(blocked_movement_observed and boundary_direction is not None)
            attempted_move_into_blocked_boundary = bool(boundary_contact_observed and attempted_movement_direction is not None and attempted_movement_direction == boundary_direction)
            repeated_boundary_facing_movement = bool(
                attempted_move_into_blocked_boundary
                and no_change_aliases
                and no_change_aliases[-1:] == [str(normalized_action.get("action_name", "")).lower()]
            )
            movement_stayed_in_place_after_attempted_boundary_move = bool(attempted_move_into_blocked_boundary and avatar_before == avatar_after)
            portal_like_contact_observed = bool("portal" in str(request_metadata.get("expected_effect_relation") or "").lower())
            terminal_affordance_contact_observed = bool("terminal" in str(request_metadata.get("expected_effect_relation") or "").lower())
            changed_cells = _changed_cells(previous_observation, observation)
            effect_region = _effect_region(previous_observation, observation)
            rewards.append(float(reward or 0.0))
            action_history.append(action)
            routed_history.append({"mode": "probe", "chosen_action": action})
            if observation == previous_observation:
                no_change_aliases.append(str(normalized_action.get("action_name", "")).lower())
            step_info = _step_telemetry(
                info=info,
                chosen_action=action,
                available_actions=available_actions,
                avatar_before=avatar_before,
                avatar_after=avatar_after,
                target_before=None,
                target_after=None,
                boundary_hit=boundary_contact_observed,
                invalid_move=blocked_movement_observed,
                reward=float(reward or 0.0),
                done=bool(done),
                truncated=bool(truncated),
                terminal_stop_reason="done" if done else ("truncated" if truncated else None),
                route_instruction_id=None,
                terminal_action_marker=False,
                effect_region=effect_region,
                effect_changed_cells=changed_cells,
                attempted_movement_direction=attempted_movement_direction,
                attempted_interaction_target_cell=attempted_interaction_target_cell,
                attempted_interaction_target_object_id=attempted_interaction_target_object_id,
                boundary_contact_observed=boundary_contact_observed,
                blocked_movement_observed=blocked_movement_observed,
                portal_like_contact_observed=portal_like_contact_observed,
                terminal_affordance_contact_observed=terminal_affordance_contact_observed,
                pre_step_area_id=pre_area_id,
                post_step_area_id=post_area_id,
                attempted_move_into_blocked_boundary=attempted_move_into_blocked_boundary,
                repeated_boundary_facing_movement=repeated_boundary_facing_movement,
                movement_stayed_in_place_after_attempted_boundary_move=movement_stayed_in_place_after_attempted_boundary_move,
            )
            step_info.update(
                {
                    "avatar_cell": avatar_after,
                    "avatar_confidence": float(avatar_after_belief.confidence or 0.0),
                    "avatar_source": str(avatar_after_belief.source or "unknown"),
                    "avatar_ambiguous": bool(avatar_after_belief.ambiguous),
                    "avatar_status": str(avatar_after_belief.avatar_status or "unknown"),
                    "avatar_mode_status": str(avatar_after_belief.mode_status or "unknown"),
                    "avatar_candidate_cells": [list(cell) for cell in list(avatar_after_belief.candidate_cells or ())],
                    "avatar_cell_before": avatar_before,
                    "avatar_confidence_before": avatar_confidence_before,
                    "avatar_source_before": avatar_source_before,
                    "avatar_ambiguous_before": avatar_ambiguous_before,
                    "avatar_candidate_cells_before": avatar_candidate_cells_before,
                    "avatar_cell_after": avatar_after,
                    "avatar_confidence_after": float(avatar_after_belief.confidence or 0.0),
                    "avatar_source_after": str(avatar_after_belief.source or "unknown"),
                    "avatar_ambiguous_after": bool(avatar_after_belief.ambiguous),
                }
            )
            steps.append(
                RawStep(
                    session_id=request.session_id,
                    run_id=request.run_id,
                    game_id=request.game_id,
                    episode_id=f"{request.mode}:{request.round_id}:{request.pass_id}",
                    step_idx=step_idx,
                    observation=observation,
                    action=action,
                    action_id=normalized_action.get("action_id"),
                    action_name=normalized_action.get("action_name"),
                    action_family=str(normalized_action.get("action_family", "unknown")),
                    reward=float(reward or 0.0),
                    done=bool(done),
                    truncated=bool(truncated),
                    info=step_info,
                )
            )
            if done or truncated:
                break
        return self._build_outcome(request, steps, rewards, routed_history, initial_observation=initial_observation)

    def _run_directed(self, request: ExecutorRequest) -> ExecutorOutcome:
        observation, info = self._reset(seed=request.metadata.get("seed") if isinstance(request.metadata, dict) else None)
        initial_observation = observation
        request_metadata = dict(request.metadata or {})
        steps = []
        rewards: list[float] = []
        action_history: list[object] = []
        routed_history: list[dict] = []
        avatar_reacquire_attempts = 0
        target_reacquire_attempts = 0
        local_reroute_attempts = 0
        no_progress_steps = 0
        stall_limit = int(request.stop_conditions.get("stall_limit", 3) or 3)

        for step_idx in range(request.max_steps):
            available_actions = list(info.get("available_actions", self.env.available_actions()))
            avatar_belief = self.avatar_tracker.current_belief()
            route_info = dict(info or {})
            avatar_cell, avatar_confidence, avatar_source, avatar_ambiguous, avatar_candidate_cells = _avatar_fields_from_belief(avatar_belief)
            route_info.update(
                {
                    "avatar": avatar_cell,
                    "avatar_confidence": avatar_confidence,
                    "avatar_source": avatar_source,
                    "avatar_ambiguous": avatar_ambiguous,
                    "avatar_candidate_cells": avatar_candidate_cells,
                    "avatar_status": str(getattr(avatar_belief, "avatar_status", "unknown") or "unknown"),
                    "avatar_mode_status": str(getattr(avatar_belief, "mode_status", "unknown") or "unknown"),
                }
            )
            routed = route_instruction(request.action, current_observation=observation, info=route_info)
            if routed is None:
                routed_history.append({"mode": "directed", "failed": True, "failure_reason": "missing_route"})
                break
            if routed.get("failed"):
                failure_reason = str(routed.get("failure_reason") or "execution_failed")
                if failure_reason == "missing_avatar" and avatar_reacquire_attempts < 1 and bool(request.constraints.get("allow_avatar_reacquire_once", False)):
                    avatar_reacquire_attempts += 1
                    patched_info = dict(route_info)
                    routed = route_instruction(request.action, current_observation=observation, info=patched_info)
                elif failure_reason == "missing_target" and target_reacquire_attempts < 1 and bool(request.constraints.get("allow_target_reacquire_once", False)):
                    target_reacquire_attempts += 1
                    patched_action = dict(request.action or {})
                    navigation_target = request.navigation.get("route_target") or request.target_centroid
                    if isinstance(navigation_target, (list, tuple)) and len(navigation_target) == 2:
                        patched_action["centroid"] = [float(navigation_target[0]), float(navigation_target[1])]
                    request = ExecutorRequest(**{**request.__dict__, "action": patched_action})
                    routed = route_instruction(request.action, current_observation=observation, info=info)
                elif failure_reason in {"blocked", "unreachable"} and local_reroute_attempts < int(request.constraints.get("max_local_reroute_attempts", 1) or 1) and bool(request.constraints.get("allow_local_reroute", False)):
                    local_reroute_attempts += 1
                    routed = route_instruction(request.action, current_observation=observation, info={**info, "reroute_attempt": local_reroute_attempts})
                if routed is None or routed.get("failed"):
                    routed_history.append(dict(routed or {}, mode="directed", failed=True))
                    break
            if routed.get("stop"):
                routed_history.append(dict(routed, mode="directed"))
                break
            previous_observation = observation
            avatar_before = avatar_cell
            target_before = list(routed.get("target_centroid", [])) if isinstance(routed.get("target_centroid"), list) else None
            try:
                action = choose_directed_action(request.action, routed, available_actions, action_history)
            except RuntimeError as exc:
                routed_history.append({"mode": "directed", "failed": True, "failure_reason": str(exc)})
                break
            normalized_action = self._normalize_emitted_action(action, available_actions)
            observation, reward, done, truncated, info = self._step(action)
            avatar_after_belief = self.avatar_tracker.update(previous_observation, observation, action, info=info, step_index=step_idx)
            avatar_after = list(avatar_after_belief.best_cell) if avatar_after_belief.best_cell is not None else None
            target_after = list(routed.get("target_centroid", [])) if isinstance(routed.get("target_centroid"), list) else None
            changed_cells = _changed_cells(previous_observation, observation)
            effect_region = _effect_region(previous_observation, observation)
            invalid_move = bool(
                observation == previous_observation
                and str(normalized_action.get("action_family", "unknown")) == "move"
            )
            boundary_hit = bool(
                str(routed.get("failure_reason") or "") == "blocked"
                or (
                    invalid_move
                    and avatar_before is not None
                    and avatar_after is not None
                    and avatar_before == avatar_after
                )
            )
            attempted_movement_direction = _movement_direction_from_action(normalized_action)
            attempted_interaction_target_cell = (
                list(request.click_target_coordinates)
                if isinstance(request.click_target_coordinates, list)
                else list(target_after) if isinstance(target_after, list) else None
            )
            attempted_interaction_target_object_id = str(
                request_metadata.get("expected_target_id")
                or request_metadata.get("expected_trigger_target_id")
                or request_metadata.get("expected_terminal_or_exit_target_id")
                or request.target_entity_id
                or routed.get("target_entity_id")
                or ""
            ) or None
            pre_area_id = _resolve_area_id(info=route_info, request_metadata=request_metadata, navigation=dict(request.navigation or {}), objective=dict(request.objective or {}))
            post_area_id = _resolve_area_id(info=info, request_metadata=request_metadata, navigation=dict(request.navigation or {}), objective=dict(request.objective or {}), fallback=pre_area_id)
            portal_like_contact_observed = bool(
                "portal" in str(request_metadata.get("expected_effect_relation") or "").lower()
                or "portal" in str(request_metadata.get("expected_contact_region_or_boundary") or "").lower()
            )
            terminal_affordance_contact_observed = bool(
                bool(routed.get("terminal"))
                or "terminal" in str(request_metadata.get("expected_effect_relation") or "").lower()
                or "exit" in str(request_metadata.get("expected_effect_relation") or "").lower()
            )
            boundary_direction = _boundary_direction(avatar_before)
            attempted_move_into_blocked_boundary = bool(boundary_hit and attempted_movement_direction is not None and attempted_movement_direction == boundary_direction)
            repeated_boundary_facing_movement = bool(attempted_move_into_blocked_boundary and no_progress_steps > 0)
            movement_stayed_in_place_after_attempted_boundary_move = bool(attempted_move_into_blocked_boundary and avatar_before == avatar_after)
            rewards.append(float(reward or 0.0))
            action_history.append(action)
            routed_history.append(dict(routed or {}, chosen_action=action, mode="directed"))
            step_info = _step_telemetry(
                info=info,
                chosen_action=action,
                available_actions=available_actions,
                avatar_before=avatar_before,
                avatar_after=avatar_after,
                target_before=target_before,
                target_after=target_after,
                boundary_hit=boundary_hit,
                invalid_move=invalid_move,
                reward=float(reward or 0.0),
                done=bool(done),
                truncated=bool(truncated),
                terminal_stop_reason="done" if done else ("truncated" if truncated else None),
                route_instruction_id=f"route:{request.candidate_id}:{step_idx}",
                terminal_action_marker=bool(routed.get("terminal")),
                effect_region=effect_region,
                effect_changed_cells=changed_cells,
                attempted_movement_direction=attempted_movement_direction,
                attempted_interaction_target_cell=attempted_interaction_target_cell,
                attempted_interaction_target_object_id=attempted_interaction_target_object_id,
                boundary_contact_observed=boundary_hit,
                blocked_movement_observed=invalid_move,
                portal_like_contact_observed=portal_like_contact_observed,
                terminal_affordance_contact_observed=terminal_affordance_contact_observed,
                pre_step_area_id=pre_area_id,
                post_step_area_id=post_area_id,
                attempted_move_into_blocked_boundary=attempted_move_into_blocked_boundary,
                repeated_boundary_facing_movement=repeated_boundary_facing_movement,
                movement_stayed_in_place_after_attempted_boundary_move=movement_stayed_in_place_after_attempted_boundary_move,
            )
            step_info.update(
                {
                    "avatar_cell": avatar_after,
                    "avatar_confidence": float(avatar_after_belief.confidence or 0.0),
                    "avatar_source": str(avatar_after_belief.source or "unknown"),
                    "avatar_ambiguous": bool(avatar_after_belief.ambiguous),
                    "avatar_status": str(avatar_after_belief.avatar_status or "unknown"),
                    "avatar_mode_status": str(avatar_after_belief.mode_status or "unknown"),
                    "avatar_candidate_cells": [list(cell) for cell in list(avatar_after_belief.candidate_cells or ())],
                    "avatar_cell_before": avatar_before,
                    "avatar_confidence_before": avatar_confidence,
                    "avatar_source_before": avatar_source,
                    "avatar_ambiguous_before": avatar_ambiguous,
                    "avatar_cell_after": avatar_after,
                    "avatar_confidence_after": float(avatar_after_belief.confidence or 0.0),
                    "avatar_source_after": str(avatar_after_belief.source or "unknown"),
                    "avatar_ambiguous_after": bool(avatar_after_belief.ambiguous),
                }
            )
            steps.append(
                RawStep(
                    session_id=request.session_id,
                    run_id=request.run_id,
                    game_id=request.game_id,
                    episode_id=f"{request.mode}:{request.round_id}:{request.pass_id}",
                    step_idx=step_idx,
                    observation=observation,
                    action=action,
                    action_id=normalized_action.get("action_id"),
                    action_name=normalized_action.get("action_name"),
                    action_family=str(normalized_action.get("action_family", "unknown")),
                    reward=float(reward or 0.0),
                    done=bool(done),
                    truncated=bool(truncated),
                    info=step_info,
                )
            )
            if routed.get("terminal") or done or truncated:
                break
            if observation == previous_observation and routed.get("movement"):
                no_progress_steps += 1
                if no_progress_steps >= stall_limit:
                    routed_history.append({"mode": "directed", "failed": True, "failure_reason": "stalled", "no_progress_steps": no_progress_steps})
                    break
            else:
                no_progress_steps = 0
        return self._build_outcome(request, steps, rewards, routed_history, initial_observation=initial_observation)

    def _build_outcome(self, request: ExecutorRequest, steps: list[RawStep], rewards: list[float], routed_history: list[dict], *, initial_observation: object | None) -> ExecutorOutcome:
        summary = summarize_outcome(steps=steps, request=request, routed_history=routed_history, rewards=rewards)
        episode = RawEpisode(
            session_id=request.session_id,
            run_id=request.run_id,
            game_id=request.game_id,
            round_id=request.round_id,
            pass_id=request.pass_id,
            episode_id=f"{request.mode}:{request.round_id}:{request.pass_id}",
            mode=request.mode,
            worker_id=self.worker_id,
            steps=tuple(steps),
            total_reward=float(sum(rewards)),
            done=bool(steps and steps[-1].done),
            won=bool(steps and steps[-1].done and sum(rewards) > 0),
            metadata={
                "env": self.env.env_metadata(),
                "request_metadata": dict(request.metadata),
                "execution_intent": {
                    "expected_effect_type": request.metadata.get("expected_effect_type"),
                    "expected_effect_relation": request.metadata.get("expected_effect_relation") or request.metadata.get("expected_relation_type"),
                    "expected_target_id": request.metadata.get("expected_target_id"),
                    "expected_trigger_target_id": request.metadata.get("expected_trigger_target_id"),
                    "expected_terminal_or_exit_target_id": request.metadata.get("expected_terminal_or_exit_target_id"),
                    "expected_contact_region_or_boundary": request.metadata.get("expected_contact_region_or_boundary"),
                },
                "objective": dict(request.objective),
                "navigation": dict(request.navigation),
                "terminal_action": dict(request.terminal_action),
                "constraints": dict(request.constraints),
                "stop_conditions": dict(request.stop_conditions),
                "initial_observation": initial_observation,
                "outcome_summary": dict(summary),
            },
        )
        return ExecutorOutcome(
            session_id=request.session_id,
            run_id=request.run_id,
            game_id=request.game_id,
            round_id=request.round_id,
            pass_id=request.pass_id,
            plan_context_id=request.plan_context_id,
            candidate_id=request.candidate_id,
            episode=episode,
            success=bool(summary["execution_success"]),
            termination_reason=str(summary["termination_reason"]),
            reward_delta=float(summary["reward_delta"]),
            outcome=summary,
        )

    def run(self, request: ExecutorRequest) -> ExecutorOutcome:
        if request.mode == "probe":
            return self._run_probe(request)
        return self._run_directed(request)

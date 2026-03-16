from __future__ import annotations

from dataclasses import dataclass, field
import random

from v3_1.contracts.messages import ExecutorRequest, ExecutorOutcome, RawEpisode, RawStep
from v3_1.execution.env_factory import NormalizedEnvAdapter, build_env, normalize_action_lookup
from v3_1.execution.option_execution import choose_directed_action, choose_probe_action
from v3_1.execution.outcomes import summarize_outcome
from v3_1.execution.route_execution import route_instruction


def _avatar_cell(info: dict | None, observation) -> list[int] | None:
    payload = dict(info or {})
    avatar = payload.get("avatar")
    if isinstance(avatar, (list, tuple)) and len(avatar) == 2:
        return [int(float(avatar[0])), int(float(avatar[1]))]
    if not isinstance(observation, list):
        return None
    for y, row in enumerate(observation):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            if int(value) == 1:
                return [x, y]
    return None


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

    @classmethod
    def from_config(cls, worker_id: str, *, env_factory: str | None, env_id: str | None, env_root: str | None, seed: int | None, render_terminal: bool = False) -> "EnvWorker":
        return cls(worker_id=worker_id, env=build_env(env_factory, env_id=env_id, env_root=env_root, seed=seed, render_terminal=render_terminal))

    def _reset(self, seed: int | None = None) -> tuple[object, dict]:
        observation, info = self.env.reset(seed=seed)
        self.reset_counter += 1
        self.last_observation = observation
        self.last_info = dict(info)
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
            avatar_before = _avatar_cell(info, previous_observation)
            observation, reward, done, truncated, info = self._step(action)
            avatar_after = _avatar_cell(info, observation)
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
                boundary_hit=False,
                invalid_move=bool(observation == previous_observation and str(normalized_action.get("action_family", "unknown")) == "move"),
                reward=float(reward or 0.0),
                done=bool(done),
                truncated=bool(truncated),
                terminal_stop_reason="done" if done else ("truncated" if truncated else None),
                route_instruction_id=None,
                terminal_action_marker=False,
                effect_region=effect_region,
                effect_changed_cells=changed_cells,
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
            routed = route_instruction(request.action, current_observation=observation, info=info)
            if routed is None:
                routed_history.append({"mode": "directed", "failed": True, "failure_reason": "missing_route"})
                break
            if routed.get("failed"):
                failure_reason = str(routed.get("failure_reason") or "execution_failed")
                if failure_reason == "missing_avatar" and avatar_reacquire_attempts < 1 and bool(request.constraints.get("allow_avatar_reacquire_once", False)):
                    avatar_reacquire_attempts += 1
                    patched_info = dict(info)
                    patched_info["avatar"] = patched_info.get("avatar")
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
            avatar_before = _avatar_cell(info, previous_observation)
            target_before = list(routed.get("target_centroid", [])) if isinstance(routed.get("target_centroid"), list) else None
            try:
                action = choose_directed_action(request.action, routed, available_actions, action_history)
            except RuntimeError as exc:
                routed_history.append({"mode": "directed", "failed": True, "failure_reason": str(exc)})
                break
            normalized_action = self._normalize_emitted_action(action, available_actions)
            observation, reward, done, truncated, info = self._step(action)
            avatar_after = _avatar_cell(info, observation)
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
                "objective": dict(request.objective),
                "navigation": dict(request.navigation),
                "terminal_action": dict(request.terminal_action),
                "constraints": dict(request.constraints),
                "stop_conditions": dict(request.stop_conditions),
                "initial_observation": initial_observation,
            },
        )
        summary = summarize_outcome(steps=steps, request=request, routed_history=routed_history, rewards=rewards)
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

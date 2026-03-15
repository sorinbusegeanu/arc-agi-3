from __future__ import annotations

from dataclasses import dataclass, field
import random

from v3_1.contracts.messages import ExecutorRequest, ExecutorOutcome, RawEpisode, RawStep
from v3_1.execution.env_factory import NormalizedEnvAdapter, build_env, normalize_action_lookup
from v3_1.execution.option_execution import choose_probe_action
from v3_1.execution.outcomes import summarize_outcome
from v3_1.execution.route_execution import route_instruction


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

    def _select_directed_action(self, *, request: ExecutorRequest, routed: dict, available_actions: list[dict]) -> object:
        desired_action_name = str(routed.get("desired_action_name", "")).lower()
        required_action_family = str(request.required_action_family or request.action_family or "unknown").lower()
        if required_action_family == "click_at":
            target_coordinates = routed.get("click_target_coordinates") or request.click_target_coordinates or request.target_centroid
            for action in available_actions:
                normalized = normalize_action_lookup(action, available_actions)
                if normalized.get("action_family") == "click_at":
                    action_row = dict(action) if isinstance(action, dict) else {"id": normalized.get("action_id"), "name": normalized.get("action_name")}
                    if isinstance(target_coordinates, (list, tuple)) and len(target_coordinates) == 2:
                        action_row["coordinates"] = [float(target_coordinates[0]), float(target_coordinates[1])]
                        action_row["click_target_coordinates"] = [float(target_coordinates[0]), float(target_coordinates[1])]
                    return action_row
            raise RuntimeError("click_at action unavailable for directed execution")
        for action in available_actions:
            normalized = normalize_action_lookup(action, available_actions)
            if desired_action_name and normalized.get("action_name") == desired_action_name:
                return action
            if required_action_family != "unknown" and normalized.get("action_family") == required_action_family and routed.get("terminal"):
                return action
        raise RuntimeError(f"directed action unavailable: family={required_action_family} desired={desired_action_name}")

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
            observation, reward, done, truncated, info = self._step(action)
            rewards.append(float(reward or 0.0))
            action_history.append(action)
            routed_history.append({"mode": "probe", "chosen_action": action})
            if observation == previous_observation:
                no_change_aliases.append(str(action.get("name", "")).lower())
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
                    info=dict(info),
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
        for step_idx in range(request.max_steps):
            available_actions = list(info.get("available_actions", self.env.available_actions()))
            routed = route_instruction(request.action, current_observation=observation, info=info)
            if routed is None:
                routed_history.append({"mode": "directed", "failed": True, "failure_reason": "missing_route"})
                break
            if routed.get("failed"):
                routed_history.append(dict(routed, mode="directed"))
                break
            if routed.get("stop"):
                routed_history.append(dict(routed, mode="directed"))
                break
            previous_observation = observation
            try:
                action = self._select_directed_action(request=request, routed=routed, available_actions=available_actions)
            except RuntimeError as exc:
                routed_history.append({"mode": "directed", "failed": True, "failure_reason": str(exc)})
                break
            normalized_action = self._normalize_emitted_action(action, available_actions)
            observation, reward, done, truncated, info = self._step(action)
            rewards.append(float(reward or 0.0))
            action_history.append(action)
            routed_history.append(dict(routed or {}, chosen_action=action, mode="directed"))
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
                    info=dict(info),
                )
            )
            if routed.get("terminal") or done or truncated:
                break
            if observation == previous_observation and routed.get("movement"):
                routed_history.append({"mode": "directed", "failed": True, "failure_reason": "stalled"})
                break
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
            success=bool(summary["success"]),
            termination_reason=str(summary["termination_reason"]),
            reward_delta=float(summary["reward_delta"]),
            outcome=summary,
        )

    def run(self, request: ExecutorRequest) -> ExecutorOutcome:
        if request.mode == "probe":
            return self._run_probe(request)
        return self._run_directed(request)

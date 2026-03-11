from __future__ import annotations

from dataclasses import dataclass, field

from v3_1.contracts.messages import ExecutorRequest, ExecutorOutcome, RawEpisode, RawStep
from v3_1.execution.env_factory import NormalizedEnvAdapter, build_env
from v3_1.execution.option_execution import choose_directed_action, choose_probe_action
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
    def from_config(cls, worker_id: str, *, env_factory: str | None, env_id: str | None, env_root: str | None, seed: int | None) -> "EnvWorker":
        return cls(worker_id=worker_id, env=build_env(env_factory, env_id=env_id, env_root=env_root, seed=seed))

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

    def _run_probe(self, request: ExecutorRequest) -> ExecutorOutcome:
        observation, info = self._reset(seed=request.metadata.get("seed") if isinstance(request.metadata, dict) else None)
        steps = []
        rewards: list[float] = []
        action_history: list[object] = []
        routed_history: list[dict] = []
        for step_idx in range(request.max_steps):
            available_actions = list(info.get("available_actions", self.env.available_actions()))
            action = choose_probe_action(available_actions, action_history)
            observation, reward, done, truncated, info = self._step(action)
            rewards.append(float(reward or 0.0))
            action_history.append(action)
            routed_history.append({"mode": "probe", "chosen_action": action})
            steps.append(
                RawStep(
                    session_id=request.session_id,
                    run_id=request.run_id,
                    game_id=request.game_id,
                    episode_id=f"{request.mode}:{request.round_id}:{request.pass_id}",
                    step_idx=step_idx,
                    observation=observation,
                    action=action,
                    reward=float(reward or 0.0),
                    done=bool(done),
                    truncated=bool(truncated),
                    info=dict(info),
                )
            )
            if done or truncated:
                break
        return self._build_outcome(request, steps, rewards, routed_history)

    def _run_directed(self, request: ExecutorRequest) -> ExecutorOutcome:
        observation, info = self._reset(seed=request.metadata.get("seed") if isinstance(request.metadata, dict) else None)
        steps = []
        rewards: list[float] = []
        action_history: list[object] = []
        routed_history: list[dict] = []
        for step_idx in range(request.max_steps):
            available_actions = list(info.get("available_actions", self.env.available_actions()))
            routed = route_instruction(request.action, current_observation=observation, info=info)
            action = choose_directed_action(request.action, routed, available_actions, action_history)
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
                    reward=float(reward or 0.0),
                    done=bool(done),
                    truncated=bool(truncated),
                    info=dict(info),
                )
            )
            if done or truncated:
                break
        return self._build_outcome(request, steps, rewards, routed_history)

    def _build_outcome(self, request: ExecutorRequest, steps: list[RawStep], rewards: list[float], routed_history: list[dict]) -> ExecutorOutcome:
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

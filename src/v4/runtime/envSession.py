from __future__ import annotations

from typing import Any, Callable

from arc_agi_agent.envs.loader import make_env

from v4.agentContract.extract import (
    build_v4_step_result,
    build_v4_transition_record,
    extract_v4_environment_metadata,
    extract_v4_observation_from_env_output,
)
from v4.agentContract.types import V4Action, V4Observation, V4StepResult, V4TransitionRecord
from v4.runtime.stopConditions import StopConditionStatusV4, StopReasonV4


class EnvSessionV4:
    def __init__(
        self,
        *,
        env_id: str,
        env_root: str | None = None,
        seed: int = 0,
        op_mode: str = "offline",
        render_mode: str | None = None,
        env_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.env_id = env_id
        self.env_root = env_root
        self.seed = int(seed)
        self.op_mode = op_mode
        self.render_mode = render_mode
        self._env = env_factory() if env_factory is not None else make_env(env_id, env_root=env_root, seed=seed, op_mode=op_mode, render_mode=render_mode)
        self.environment_metadata = extract_v4_environment_metadata(self._env)
        self.current_observation: V4Observation | None = None
        self.last_transition: V4TransitionRecord | None = None
        self.last_step_result: V4StepResult | None = None
        self.step_index = 0
        self.stop_status = StopConditionStatusV4(False, StopReasonV4.CONTINUE)

    @property
    def env(self) -> Any:
        return self._env

    def reset(self) -> V4Observation:
        raw_obs = self._env.reset()
        self.current_observation = extract_v4_observation_from_env_output(raw_obs)
        self.last_transition = None
        self.last_step_result = None
        self.step_index = 0
        self.stop_status = StopConditionStatusV4(False, StopReasonV4.CONTINUE)
        return self.current_observation

    def step(self, action: V4Action) -> tuple[V4TransitionRecord, V4StepResult]:
        if self.current_observation is None:
            raise ValueError("session must be reset before stepping")
        from arcengine import GameAction

        pre_observation = self.current_observation
        game_action = GameAction.from_id(action.action_id)
        payload = dict(action.payload or {})
        payload.setdefault("game_id", pre_observation.game_id)
        raw_post = self._env.step(game_action, data=payload, reasoning=action.reasoning)
        post_observation = extract_v4_observation_from_env_output(raw_post)
        transition = build_v4_transition_record(
            pre_observation,
            action,
            post_observation,
            step_index=self.step_index,
        )
        step_result = build_v4_step_result(transition)
        self.current_observation = post_observation
        self.last_transition = transition
        self.last_step_result = step_result
        self.step_index += 1
        return transition, step_result

    def close(self) -> None:
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass

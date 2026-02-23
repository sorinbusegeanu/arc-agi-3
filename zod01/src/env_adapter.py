from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arc_agi import Arcade, EnvironmentWrapper
from arcengine import FrameDataRaw, GameAction

from .actions import to_game_action
from .types import NormalizedAction


@dataclass
class StepResult:
    observation: FrameDataRaw
    valid_action: bool


class EnvAdapter:
    """Thin deterministic wrapper around arc-agi environment interaction."""

    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        scorecard_id: str | None = None,
        render_mode: str | None = None,
        arcade: Arcade | None = None,
    ) -> None:
        self.game_id = game_id
        self.seed = seed
        self.arcade = arcade or Arcade()
        env = self.arcade.make(game_id, seed=seed, scorecard_id=scorecard_id, render_mode=render_mode)
        if env is None:
            raise RuntimeError(f"Failed to create environment for {game_id}")
        self.env: EnvironmentWrapper = env

    @property
    def observation(self) -> FrameDataRaw:
        obs = self.env.observation_space
        if obs is None:
            raise RuntimeError("Environment has no observation. Ensure reset() has been called.")
        return obs

    @property
    def available_action_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.env.action_space)

    def reset(self) -> FrameDataRaw:
        obs = self.env.reset()
        if obs is None:
            raise RuntimeError("Environment reset failed")
        return obs

    def is_action_valid(self, action: NormalizedAction) -> bool:
        return action.name in self.available_action_names

    def step(self, action: NormalizedAction) -> StepResult:
        if not self.is_action_valid(action):
            return StepResult(observation=self.observation, valid_action=False)
        ga: GameAction = to_game_action(action, game_id=self.game_id)
        data = ga.action_data.model_dump() if hasattr(ga, "action_data") else {}
        reasoning: dict[str, Any] | None = None
        if isinstance(action.reasoning, dict):
            reasoning = action.reasoning
        obs = self.env.step(ga, data=data, reasoning=reasoning)
        if obs is None:
            raise RuntimeError(f"Environment step failed for {action.name}")
        return StepResult(observation=obs, valid_action=True)

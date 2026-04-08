from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


ACTION_IDS = (1, 2, 3, 4, 5, 6, 7)
ACTION_MASK_SIZE = 8


@dataclass(frozen=True)
class V1Action:
    action_id: int
    x: int | None = None
    y: int | None = None


@dataclass
class ObservationPackage:
    current_frame: torch.Tensor
    previous_frame_1: torch.Tensor
    previous_frame_2: torch.Tensor
    valid_action_mask: torch.Tensor
    action6_clickable: bool | None
    raw_metadata: dict[str, Any]
    terminal: bool
    reward: float
    valid_pixel_mask: torch.Tensor
    available_action_ids: tuple[int, ...]
    game_id: str = ""
    game_id_index: int = 0
    current_level_index: int = 0
    level_completed: bool = False
    game_won: bool = False
    deepest_level_index: int = 0
    step_count: int = 0
    changed_cell_mask: torch.Tensor | None = None
    raw_response: Any = None

    def stacked_frames(self) -> torch.Tensor:
        return torch.cat(
            [self.current_frame, self.previous_frame_1, self.previous_frame_2],
            dim=0,
        )


@dataclass
class PolicyOutput:
    action_logits: torch.Tensor
    action_logprob: torch.Tensor
    value: torch.Tensor
    click_logits: torch.Tensor | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutTimestep:
    observation: ObservationPackage
    previous_action: V1Action
    previous_reward: float
    previous_done: bool
    chosen_action: V1Action
    action_logprob: float
    value_estimate: float
    reward: float
    done: bool
    next_observation: ObservationPackage
    step_count: int = 0
    changed_cell_mask: torch.Tensor | None = None
    hidden_state: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutSequence:
    sequence_id: str
    episode_id: str
    env_instance_id: str = ""
    game_id: str = ""
    episode_counter: int = 0
    timesteps: list[RolloutTimestep] = field(default_factory=list)
    sequence_start: bool = True
    truncated: bool = False
    chunk_end_reason: str = ""
    bootstrap_value: float = 0.0
    initial_hidden_state: torch.Tensor | None = None


def action_mask_from_available(available_action_ids: list[int] | tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros(ACTION_MASK_SIZE, dtype=torch.bool)
    for action_id in available_action_ids:
        if 0 <= int(action_id) < ACTION_MASK_SIZE:
            mask[int(action_id)] = True
    return mask

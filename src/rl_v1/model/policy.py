from __future__ import annotations

import torch

from rl_v1.model.action_masking import build_valid_action_mask


def apply_valid_action_mask(logits: torch.Tensor, observation_or_mask, *, evaluation: bool = False) -> tuple[torch.Tensor, dict]:
    if isinstance(observation_or_mask, torch.Tensor):
        mask = observation_or_mask.to(dtype=torch.bool, device=logits.device)
        mask_empty = bool(mask.sum(dim=-1).eq(0).any().item()) if mask.dim() > 1 else bool(mask.sum().item() == 0)
    else:
        built = build_valid_action_mask(observation_or_mask)
        built_mask = built["valid_action_mask"]
        mask_empty = bool(built["valid_action_mask_empty"])
        mask = built_mask.to(dtype=torch.bool, device=logits.device) if isinstance(built_mask, torch.Tensor) else torch.tensor(built_mask, dtype=torch.bool, device=logits.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    empty_rows = mask.sum(dim=-1) == 0
    if empty_rows.any():
        mode = "evaluation" if evaluation else "training"
        raise ValueError(f"empty valid-action mask encountered during {mode} action masking")
    return logits.masked_fill(~mask, -1e9), {"valid_action_mask_empty": bool(mask_empty)}


def sample_masked_action(masked_logits: torch.Tensor, deterministic: bool) -> tuple[int, torch.Tensor, torch.Tensor]:
    distribution = torch.distributions.Categorical(logits=masked_logits)
    action = masked_logits.argmax(dim=-1) if deterministic else distribution.sample()
    return int(action.item()), distribution.log_prob(action), distribution.entropy()


def masked_logprob(masked_logits: torch.Tensor, action_id: int) -> torch.Tensor:
    distribution = torch.distributions.Categorical(logits=masked_logits)
    return distribution.log_prob(torch.tensor([action_id], device=masked_logits.device))

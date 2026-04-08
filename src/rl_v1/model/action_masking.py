from __future__ import annotations

import torch

from rl_v1.data.contracts import ACTION_MASK_SIZE, action_mask_from_available


def build_valid_action_mask(observation):
    existing = getattr(observation, "valid_action_mask", None)
    if existing is not None:
        if isinstance(existing, torch.Tensor):
            if existing.numel() != ACTION_MASK_SIZE:
                raise ValueError(f"valid action mask length must be {ACTION_MASK_SIZE}")
            mask = existing.to(dtype=torch.bool)
            return {"valid_action_mask": mask, "valid_action_mask_empty": bool(mask.sum().item() == 0)}
        existing_list = list(existing)
        if len(existing_list) != ACTION_MASK_SIZE:
            raise ValueError(f"valid action mask length must be {ACTION_MASK_SIZE}")
        mask_list = [bool(item) for item in existing_list]
        return {"valid_action_mask": mask_list, "valid_action_mask_empty": not any(mask_list)}
    available = getattr(observation, "available_action_ids", None)
    if available is None:
        raise ValueError("observation must expose either valid_action_mask or available_action_ids")
    mask = action_mask_from_available(available)
    return {"valid_action_mask": mask, "valid_action_mask_empty": bool(mask.sum().item() == 0)}

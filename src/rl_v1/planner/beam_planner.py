from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_v1.data.contracts import V1Action
from rl_v1.model.policy import apply_valid_action_mask


@dataclass
class PlannerOutput:
    selected_first_action: V1Action
    selected_branch_action_sequence: list[int]
    planner_score: float
    diagnostics: dict[str, Any]


class LatentBeamPlanner:
    def __init__(
        self,
        *,
        beam_width: int,
        search_depth: int,
        action_topk: int,
        discount: float,
        allow_click_action_in_planner: bool = False,
    ) -> None:
        self.beam_width = int(beam_width)
        self.search_depth = int(search_depth)
        self.action_topk = int(action_topk)
        self.discount = float(discount)
        self.allow_click_action_in_planner = bool(allow_click_action_in_planner)

    def plan(self, current_latent, valid_action_mask, policy_logits, model_interfaces):
        beam: list[tuple[torch.Tensor, list[int], float, float, bool]] = [(current_latent, [], 0.0, 1.0, False)]
        branches: list[tuple[list[int], float]] = []
        for depth in range(self.search_depth):
            expanded: list[tuple[torch.Tensor, list[int], float, float, bool]] = []
            for latent, path, reward_sum, discount, done in beam:
                if done:
                    branches.append((path, reward_sum))
                    continue
                if depth == 0 and not path:
                    prior_logits = policy_logits
                else:
                    prior_fn = model_interfaces.get("policy_prior")
                    prior_logits = prior_fn(latent) if prior_fn is not None else policy_logits
                # Root uses real valid action mask. Imagined future states have no exact mask in V1.
                if depth == 0 and not path:
                    masked_logits, _ = apply_valid_action_mask(prior_logits, valid_action_mask, evaluation=True)
                    candidate_ids = torch.nonzero(valid_action_mask[0], as_tuple=False).squeeze(-1).tolist()
                else:
                    masked_logits = prior_logits
                    candidate_ids = list(range(masked_logits.shape[-1]))
                if not self.allow_click_action_in_planner:
                    candidate_ids = [idx for idx in candidate_ids if idx != 6]
                if not candidate_ids:
                    continue
                prior = torch.softmax(masked_logits[0], dim=-1)
                candidates = sorted(
                    candidate_ids,
                    key=lambda idx: float(prior[idx].detach().item()),
                    reverse=True,
                )[: self.action_topk]
                for action_id in candidates:
                    next_latent, predicted_reward, done_logit = model_interfaces["dynamics"](latent, torch.tensor([action_id], device=latent.device))
                    predicted_done = bool(torch.sigmoid(done_logit).item() > 0.5)
                    expanded.append((next_latent, path + [int(action_id)], reward_sum + discount * float(predicted_reward.item()), discount * self.discount, predicted_done))
            expanded.sort(key=lambda item: item[2], reverse=True)
            beam = expanded[: self.beam_width]
            if not beam:
                break
        for latent, path, reward_sum, discount, done in beam:
            if done:
                score = reward_sum
            else:
                score = reward_sum + discount * float(model_interfaces["value_head"](latent).item())
            branches.append((path, score))
        if not branches:
            fallback_logits, _ = apply_valid_action_mask(policy_logits, valid_action_mask, evaluation=True)
            if not self.allow_click_action_in_planner:
                fallback_logits = fallback_logits.clone()
                fallback_logits[:, 6] = -1e9
            fallback = int(fallback_logits.argmax(dim=-1).item())
            branches = [([fallback], 0.0)]
        branches.sort(key=lambda item: item[1], reverse=True)
        path, score = branches[0]
        return PlannerOutput(
            selected_first_action=V1Action(action_id=int(path[0])),
            selected_branch_action_sequence=[int(action) for action in path],
            planner_score=float(score),
            diagnostics={
                "chosen_first_action": int(path[0]),
                "branch_length": len(path),
                "planner_score": float(score),
            },
        )

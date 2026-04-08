from __future__ import annotations

import torch

from rl_v1.data.contracts import PolicyOutput, V1Action
from rl_v1.model.action_masking import build_valid_action_mask
from rl_v1.model.click_head import select_click_coordinates
from rl_v1.model.policy import apply_valid_action_mask, masked_logprob, sample_masked_action
from rl_v1.planner.beam_planner import LatentBeamPlanner


def resolve_acting_mode(cfg, *, evaluation: bool) -> str:
    if cfg.ablations.disable_planner:
        return "policy_only"
    if cfg.acting.mode == "planner_act":
        return "planner_act"
    if cfg.acting.mode == "planner_eval_only":
        return "planner_act" if evaluation else "policy_only"
    return "policy_only"


class ActionSelector:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.planner = None
        if cfg.planner.enabled and not cfg.ablations.disable_planner:
            self.planner = LatentBeamPlanner(
                beam_width=cfg.planner.beam_width,
                search_depth=cfg.planner.search_depth,
                action_topk=cfg.planner.action_topk,
                discount=cfg.planner.discount,
                allow_click_action_in_planner=cfg.planner.allow_click_action_in_planner,
            )

    def select_action(self, model, observation, prev_action_id, prev_reward, prev_done, hidden, *, deterministic: bool, evaluation: bool):
        step = model.encode_observation(observation, prev_action_id, prev_reward, prev_done, hidden)
        mask_info = build_valid_action_mask(observation)
        if bool(mask_info["valid_action_mask_empty"]):
            raw = getattr(observation, "raw_metadata", {}) or {}
            episode_id = raw.get("episode_id")
            env_id = raw.get("env_instance_id")
            game_id = raw.get("game_id")
            raise ValueError(
                "empty valid-action mask in action selection: "
                f"episode_id={episode_id} env_id={env_id} game_id={game_id}"
            )
        masked_logits, mask_diag = apply_valid_action_mask(step.policy_logits, observation, evaluation=evaluation)
        acting_mode = resolve_acting_mode(self.cfg, evaluation=evaluation)
        planner_diag = {}
        if acting_mode == "planner_act" and self.planner is not None and getattr(model, "dynamics", None) is not None:
            planned = self.planner.plan(
                step.latent.detach(),
                observation.valid_action_mask.unsqueeze(0).to(step.latent.device),
                masked_logits.detach(),
                {
                    "dynamics": model.dynamics,
                    "value_head": model.value_head,
                    "policy_prior": lambda latent: model.policy_head(latent, torch.ones((latent.shape[0], masked_logits.shape[-1]), dtype=torch.bool, device=latent.device)),
                },
            )
            action = planned.selected_first_action
            planner_diag = planned.diagnostics
        else:
            action_id, _logprob, _entropy = sample_masked_action(masked_logits, deterministic=deterministic)
            action = V1Action(action_id=action_id)
        click_logits = None
        action_logprob = masked_logprob(masked_logits, action.action_id)
        entropy = torch.distributions.Categorical(logits=masked_logits).entropy()
        if (
            action.action_id == 6
            and bool(observation.valid_action_mask[6].item())
            and getattr(model, "click_head", None) is not None
            and not self.cfg.ablations.disable_click_head
        ):
            click = select_click_coordinates(model.click_head, step.latent, step.spatial, observation.valid_pixel_mask.unsqueeze(0).to(step.latent.device))
            action = V1Action(action_id=6, x=int(click["x"]), y=int(click["y"]))
            click_logits = click["logits"]
            action_logprob = action_logprob + click["logprob"]
        diagnostics = dict(step.diagnostics)
        diagnostics["acting_mode"] = acting_mode
        diagnostics["planner"] = planner_diag
        diagnostics["valid_action_mask_empty"] = bool(mask_diag["valid_action_mask_empty"])
        return action, PolicyOutput(action_logits=masked_logits, action_logprob=action_logprob, value=step.value, click_logits=click_logits, diagnostics=diagnostics), step.hidden, diagnostics, entropy

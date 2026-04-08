from __future__ import annotations

import torch
from torch.nn import functional as F


def compute_gae(
    rewards,
    values,
    dones,
    gamma: float,
    gae_lambda: float,
    *,
    bootstrap_value: float = 0.0,
    terminal_chunk: bool = True,
):
    advantages = []
    gae = 0.0
    next_value = 0.0 if terminal_chunk else float(bootstrap_value)
    for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
        nonterminal = 0.0 if done else 1.0
        delta = reward + gamma * next_value * nonterminal - value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages.append(gae)
        next_value = value
    advantages.reverse()
    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t = advantages_t + torch.tensor(values, dtype=torch.float32)
    return advantages_t, returns_t


def compute_losses(
    *,
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    value_pred: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    transition_pred: torch.Tensor | None,
    transition_target: torch.Tensor | None,
    reward_pred: torch.Tensor | None,
    reward_target: torch.Tensor | None,
    done_logit: torch.Tensor | None,
    done_target: torch.Tensor | None,
    optimization_cfg,
    loss_weights_cfg,
):
    if new_logprob.ndim != 1 or old_logprob.ndim != 1:
        raise ValueError(
            f"new_logprob/old_logprob must be 1D tensors; got {tuple(new_logprob.shape)} and {tuple(old_logprob.shape)}"
        )
    if value_pred.ndim != 1 or returns.ndim != 1 or advantages.ndim != 1 or entropy.ndim != 1:
        raise ValueError(
            "value_pred/returns/advantages/entropy must be 1D tensors; got "
            f"{tuple(value_pred.shape)}, {tuple(returns.shape)}, {tuple(advantages.shape)}, {tuple(entropy.shape)}"
        )
    expected = new_logprob.shape[0]
    if not (
        old_logprob.shape[0] == expected
        and value_pred.shape[0] == expected
        and returns.shape[0] == expected
        and advantages.shape[0] == expected
        and entropy.shape[0] == expected
    ):
        raise ValueError(
            "batched loss tensors must have the same leading dimension N; got "
            f"N(new)={new_logprob.shape[0]}, N(old)={old_logprob.shape[0]}, "
            f"N(value)={value_pred.shape[0]}, N(returns)={returns.shape[0]}, "
            f"N(advantages)={advantages.shape[0]}, N(entropy)={entropy.shape[0]}"
        )
    advantages = advantages.to(new_logprob.device)
    returns = returns.to(new_logprob.device)
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * normalized_advantages
    clipped = torch.clamp(ratio, 1.0 - optimization_cfg.clip_eps, 1.0 + optimization_cfg.clip_eps) * normalized_advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(value_pred, returns)
    entropy_bonus = entropy.mean()
    total_loss = policy_loss + optimization_cfg.value_coef * value_loss - optimization_cfg.entropy_coef * entropy_bonus
    latent_transition_loss = torch.tensor(0.0, device=new_logprob.device)
    reward_prediction_loss = torch.tensor(0.0, device=new_logprob.device)
    done_prediction_loss = torch.tensor(0.0, device=new_logprob.device)
    if transition_pred is not None and transition_target is not None:
        if transition_pred.shape != transition_target.shape or transition_pred.numel() == 0 or transition_target.numel() == 0:
            raise ValueError(
                "invalid transition tensors for mse loss: "
                f"transition_pred.shape={tuple(transition_pred.shape)}, "
                f"transition_target.shape={tuple(transition_target.shape)}"
            )
        latent_transition_loss = F.mse_loss(transition_pred, transition_target)
        total_loss = total_loss + loss_weights_cfg.transition_coef * latent_transition_loss
    if reward_pred is not None and reward_target is not None:
        reward_prediction_loss = F.mse_loss(reward_pred, reward_target)
        total_loss = total_loss + loss_weights_cfg.reward_coef * reward_prediction_loss
    if done_logit is not None and done_target is not None:
        done_prediction_loss = F.binary_cross_entropy_with_logits(done_logit, done_target)
        total_loss = total_loss + loss_weights_cfg.done_coef * done_prediction_loss
    return total_loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy_bonus": float(entropy_bonus.detach().cpu()),
        "latent_transition_loss": float(latent_transition_loss.detach().cpu()),
        "reward_prediction_loss": float(reward_prediction_loss.detach().cpu()),
        "done_prediction_loss": float(done_prediction_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }

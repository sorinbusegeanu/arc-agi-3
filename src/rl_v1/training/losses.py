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


def compute_world_model_losses(
    *,
    transition_pred: torch.Tensor | None,
    transition_target: torch.Tensor | None,
    reward_pred: torch.Tensor | None,
    reward_target: torch.Tensor | None,
    done_logit: torch.Tensor | None,
    done_target: torch.Tensor | None,
    change_mask_logits: torch.Tensor | None = None,
    change_mask_target: torch.Tensor | None = None,
    next_frame_logits: torch.Tensor | None = None,
    next_frame_target: torch.Tensor | None = None,
    loss_weights_cfg=None,
):
    device = None
    for tensor in (transition_pred, transition_target, reward_pred, reward_target, done_logit, done_target):
        if tensor is not None:
            device = tensor.device
            break
    if device is None:
        device = torch.device("cpu")
    total_loss = torch.tensor(0.0, device=device)
    parts: dict[str, float] = {}
    if transition_pred is not None and transition_target is not None:
        if transition_pred.shape != transition_target.shape or transition_pred.numel() == 0 or transition_target.numel() == 0:
            raise ValueError(
                "invalid transition tensors for mse loss: "
                f"transition_pred.shape={tuple(transition_pred.shape)}, "
                f"transition_target.shape={tuple(transition_target.shape)}"
            )
        latent_transition_loss = F.mse_loss(transition_pred, transition_target)
        parts["transition_loss"] = float(latent_transition_loss.detach().cpu())
        coef = 1.0 if loss_weights_cfg is None else float(loss_weights_cfg.transition_coef)
        total_loss = total_loss + coef * latent_transition_loss
    if change_mask_logits is not None and change_mask_target is not None:
        change_mask_loss = F.binary_cross_entropy_with_logits(change_mask_logits, change_mask_target)
        parts["change_mask_loss"] = float(change_mask_loss.detach().cpu())
        probs = torch.sigmoid(change_mask_logits)
        pred = probs >= 0.5
        tgt = change_mask_target >= 0.5
        tp = (pred & tgt).sum().float()
        fp = (pred & ~tgt).sum().float()
        fn = (~pred & tgt).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)
        parts["change_mask_precision"] = float(precision.detach().cpu())
        parts["change_mask_recall"] = float(recall.detach().cpu())
        parts["change_mask_f1"] = float(f1.detach().cpu())
        coef = 1.0 if loss_weights_cfg is None else float(getattr(loss_weights_cfg, "change_mask_coef", 1.0))
        total_loss = total_loss + coef * change_mask_loss
    if next_frame_logits is not None and next_frame_target is not None:
        next_frame_loss = F.mse_loss(next_frame_logits, next_frame_target)
        parts["next_frame_loss"] = float(next_frame_loss.detach().cpu())
        coef = 1.0 if loss_weights_cfg is None else float(getattr(loss_weights_cfg, "next_frame_coef", 1.0))
        total_loss = total_loss + coef * next_frame_loss
    if reward_pred is not None and reward_target is not None:
        reward_prediction_loss = F.mse_loss(reward_pred, reward_target)
        parts["reward_prediction_loss"] = float(reward_prediction_loss.detach().cpu())
        parts["reward_prediction_mae"] = float((reward_pred - reward_target).abs().mean().detach().cpu())
        coef = 1.0 if loss_weights_cfg is None else float(loss_weights_cfg.reward_coef)
        total_loss = total_loss + coef * reward_prediction_loss
    if done_logit is not None and done_target is not None:
        done_prediction_loss = F.binary_cross_entropy_with_logits(done_logit, done_target)
        parts["done_prediction_loss"] = float(done_prediction_loss.detach().cpu())
        done_pred = (torch.sigmoid(done_logit) >= 0.5).to(done_target.dtype)
        parts["done_accuracy"] = float((done_pred == done_target).float().mean().detach().cpu())
        coef = 1.0 if loss_weights_cfg is None else float(loss_weights_cfg.done_coef)
        total_loss = total_loss + coef * done_prediction_loss
    parts["world_total_loss"] = float(total_loss.detach().cpu())
    return total_loss, parts


def compute_rl_losses(
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
    add_world_aux: bool = True,
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
    aux_total = torch.tensor(0.0, device=new_logprob.device)
    aux_parts = {
        "transition_loss": 0.0,
        "reward_prediction_loss": 0.0,
        "done_prediction_loss": 0.0,
    }
    if add_world_aux:
        aux_total, world_parts = compute_world_model_losses(
            transition_pred=transition_pred,
            transition_target=transition_target,
            reward_pred=reward_pred,
            reward_target=reward_target,
            done_logit=done_logit,
            done_target=done_target,
            loss_weights_cfg=loss_weights_cfg,
        )
        total_loss = total_loss + aux_total
        aux_parts["transition_loss"] = world_parts.get("transition_loss", 0.0)
        aux_parts["reward_prediction_loss"] = world_parts.get("reward_prediction_loss", 0.0)
        aux_parts["done_prediction_loss"] = world_parts.get("done_prediction_loss", 0.0)
    return total_loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy_bonus": float(entropy_bonus.detach().cpu()),
        "transition_loss": aux_parts["transition_loss"],
        "reward_prediction_loss": aux_parts["reward_prediction_loss"],
        "done_prediction_loss": aux_parts["done_prediction_loss"],
        "total_loss": float(total_loss.detach().cpu()),
    }


def compute_losses(
    *,
    mode: str = "train_rl",
    **kwargs,
):
    if mode == "world_pretrain":
        return compute_world_model_losses(
            transition_pred=kwargs.get("transition_pred"),
            transition_target=kwargs.get("transition_target"),
            reward_pred=kwargs.get("reward_pred"),
            reward_target=kwargs.get("reward_target"),
            done_logit=kwargs.get("done_logit"),
            done_target=kwargs.get("done_target"),
            change_mask_logits=kwargs.get("change_mask_logits"),
            change_mask_target=kwargs.get("change_mask_target"),
            next_frame_logits=kwargs.get("next_frame_logits"),
            next_frame_target=kwargs.get("next_frame_target"),
            loss_weights_cfg=kwargs.get("loss_weights_cfg"),
        )
    return compute_rl_losses(
        new_logprob=kwargs["new_logprob"],
        old_logprob=kwargs["old_logprob"],
        value_pred=kwargs["value_pred"],
        returns=kwargs["returns"],
        advantages=kwargs["advantages"],
        entropy=kwargs["entropy"],
        transition_pred=kwargs.get("transition_pred"),
        transition_target=kwargs.get("transition_target"),
        reward_pred=kwargs.get("reward_pred"),
        reward_target=kwargs.get("reward_target"),
        done_logit=kwargs.get("done_logit"),
        done_target=kwargs.get("done_target"),
        optimization_cfg=kwargs["optimization_cfg"],
        loss_weights_cfg=kwargs["loss_weights_cfg"],
    )

from __future__ import annotations

from typing import Any, Dict

import torch


def build_optimizer(modules: Dict[str, Any], cfg: Dict[str, Any]):
    ppo_cfg = dict(cfg.get("ppo", {}))
    algo = str(cfg.get("algo", "a2c")).lower()
    lr = float(ppo_cfg.get("lr", cfg.get("lr", 3e-4))) if algo == "ppo" else float(cfg.get("lr", 3e-4))
    beta1 = float(ppo_cfg.get("adam_beta1", 0.9))
    beta2 = float(ppo_cfg.get("adam_beta2", 0.999))
    adam_eps = float(ppo_cfg.get("adam_eps", 1e-8))
    weight_decay = float(ppo_cfg.get("weight_decay", 0.0))
    mode = str(cfg.get("optimizers", {}).get("mode", "single")).lower()

    if mode == "param_groups":
        controller_lr_mult = float(cfg.get("optimizers", {}).get("controller_lr_mult", 1.0))
        value_lr_mult = float(cfg.get("optimizers", {}).get("value_lr_mult", 1.0))
        groups = [
            {
                "params": list(modules["encoder"].parameters()) + list(modules["memory"].parameters()) + list(modules["actor"].parameters()),
                "lr": lr,
            },
            {
                "params": list(modules["controller"].parameters()),
                "lr": lr * controller_lr_mult,
            },
            {
                "params": list(modules["value"].parameters()),
                "lr": lr * value_lr_mult,
            },
        ]
        return torch.optim.Adam(
            groups,
            betas=(beta1, beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
        )

    params = (
        list(modules["encoder"].parameters())
        + list(modules["memory"].parameters())
        + list(modules["controller"].parameters())
        + list(modules["actor"].parameters())
        + list(modules["value"].parameters())
    )
    return torch.optim.Adam(
        params,
        lr=lr,
        betas=(beta1, beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
    )

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_cfg() -> Dict[str, Any]:
    return {
        "hidden_dim": 256,
        "controller_hidden": 128,
        "num_modes": 3,
        "sample_mode_train": True,
    }


class HierarchicalController(nn.Module):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = {**_default_cfg(), **(cfg or {})}
        hd = int(self.cfg["hidden_dim"])
        ch = int(self.cfg["controller_hidden"])
        nm = int(self.cfg["num_modes"])
        self.net = nn.Sequential(
            nn.Linear(hd, ch),
            nn.ReLU(),
            nn.Linear(ch, nm),
        )

    def forward(
        self,
        h_t: torch.Tensor,
        aux: Optional[Dict[str, Any]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = {**self.cfg, **(cfg or {})}
        logits = self.net(h_t)
        probs = F.softmax(logits, dim=1)

        is_train = bool((ctx or {}).get("is_train", False))
        sample_train = bool(cfg_eff.get("sample_mode_train", True))
        if is_train and sample_train:
            mode_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            mode_id = torch.argmax(probs, dim=1)

        return {
            "schema_version": "CONTROLLER_OUT_V1",
            "mode_logits": logits,
            "mode_probs": probs,
            "mode_id": mode_id,
        }

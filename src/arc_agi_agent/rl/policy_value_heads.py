from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_key_normalize_v1 import sorted_action_ids


def _default_cfg() -> Dict[str, Any]:
    return {
        "coord_mode": "proposal",
        "coord_topK": 16,
        "policy_mlp_hidden": [256, 128],
        "value_mlp_hidden": [128],
    }


def _build_mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Module:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = {**_default_cfg(), **(cfg or {})}
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.discrete_head = _build_mlp(hidden_dim, cfg["policy_mlp_hidden"], 1)
        self.coord_head = _build_mlp(hidden_dim + 3, cfg["policy_mlp_hidden"], 1)

    def forward(
        self,
        h_t: torch.Tensor,
        available_actions: List[str],
        coord_candidates: Optional[List[Dict[str, Any]]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = {**self.cfg, **(cfg or {})}
        action_schema = ctx.get("action_schema") if ctx else None
        action_ids = sorted_action_ids(action_schema)
        if not action_ids:
            action_ids = sorted(list(set(available_actions)))
        if not action_ids:
            action_ids = ["ACTION1"]

        mask = [1.0 if a in available_actions else 0.0 for a in action_ids]
        mask_t = torch.tensor(mask, dtype=torch.float32, device=h_t.device).unsqueeze(0)

        logits = []
        for _ in action_ids:
            logits.append(self.discrete_head(h_t))
        pi_discrete = torch.cat(logits, dim=1)
        # mask invalid
        masked = pi_discrete + (mask_t - 1.0) * 1e9
        if mask_t.sum() <= 0:
            masked = pi_discrete
            mask_t = torch.ones_like(mask_t)

        pi_coord = None
        coord_logits_topk = None
        if coord_candidates:
            logits_c = []
            for cand in coord_candidates:
                x = float(cand.get("x", 0.0))
                y = float(cand.get("y", 0.0))
                tag = cand.get("tag", "")
                tag_code = float(abs(hash(tag)) % 1000) / 1000.0
                feat = torch.tensor([[x, y, tag_code]], dtype=torch.float32, device=h_t.device)
                inp = torch.cat([h_t, feat], dim=1)
                logits_c.append(self.coord_head(inp))
            pi_coord = torch.cat(logits_c, dim=1)
            k = min(5, pi_coord.shape[1])
            topv, topi = torch.topk(pi_coord, k)
            coord_logits_topk = {"indices": topi.squeeze(0).tolist(), "logits": topv.squeeze(0).tolist()}

        return {
            "schema_version": "POLICY_OUT_V1",
            "pi_discrete": masked,
            "pi_coord": pi_coord,
            "action_mask": mask_t,
            "action_ids": action_ids,
            "coord_logits_topk": coord_logits_topk,
        }


class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = {**_default_cfg(), **(cfg or {})}
        self.cfg = cfg
        self.value_head = _build_mlp(hidden_dim, cfg["value_mlp_hidden"], 1)

    def forward(self, h_t: torch.Tensor, cfg: Optional[Dict[str, Any]] = None, ctx: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        return self.value_head(h_t).squeeze(1)

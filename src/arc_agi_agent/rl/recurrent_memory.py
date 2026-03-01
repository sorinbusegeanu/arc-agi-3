from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .action_key_normalize_v1 import action_key_to_index


def _default_cfg() -> Dict[str, Any]:
    return {
        "cell": "gru",
        "hidden_dim": 256,
        "action_emb_dim": 32,
        "max_actions": 64,
        "reward_clip": 1.0,
    }


class RecurrentMemory(nn.Module):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = {**_default_cfg(), **(cfg or {})}
        self.cell = str(self.cfg["cell"]).lower()
        self.hidden_dim = int(self.cfg["hidden_dim"])
        self.max_actions = int(self.cfg["max_actions"])
        self.action_emb = nn.Embedding(self.max_actions, int(self.cfg["action_emb_dim"]))

        self._input_dim: Optional[int] = None
        self._core: Optional[nn.Module] = None

    def _build_core(self, input_dim: int, device: torch.device) -> None:
        self._input_dim = int(input_dim)
        if self.cell == "lstm":
            self._core = nn.LSTMCell(self._input_dim, self.hidden_dim).to(device)
        else:
            self._core = nn.GRUCell(self._input_dim, self.hidden_dim).to(device)

    def _zeros(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_dim), dtype=torch.float32, device=device)

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Any:
        dev = device or next(self.parameters()).device
        if self.cell == "lstm":
            z = self._zeros(batch_size, dev)
            return (z, z.clone())
        return self._zeros(batch_size, dev)

    def _action_feat(self, action: Optional[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        idx = action_key_to_index(action or {}, self.max_actions)
        emb = self.action_emb(torch.tensor([idx], dtype=torch.long, device=device))
        coords = torch.tensor(
            [[float((action or {}).get("x", 0.0)), float((action or {}).get("y", 0.0))]],
            dtype=torch.float32,
            device=device,
        )
        return torch.cat([emb, coords], dim=1)

    def step(
        self,
        z_t: torch.Tensor,
        prev_action: Optional[Dict[str, Any]],
        prev_reward: float,
        prev_done: bool,
        h_prev: Optional[Any],
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = {**self.cfg, **(cfg or {})}
        reward_clip = float(cfg_eff.get("reward_clip", 1.0))
        reward = float(prev_reward)
        if reward_clip > 0:
            reward = max(-reward_clip, min(reward_clip, reward))

        device = z_t.device
        done_t = torch.tensor([[1.0 if prev_done else 0.0]], dtype=torch.float32, device=device)
        rew_t = torch.tensor([[reward]], dtype=torch.float32, device=device)
        act_t = self._action_feat(prev_action, device)
        inp = torch.cat([z_t, act_t, rew_t, done_t], dim=1)

        if self._core is None or self._input_dim != int(inp.shape[1]):
            self._build_core(int(inp.shape[1]), device)

        if h_prev is None or prev_done:
            h_prev = self.reset(batch_size=int(z_t.shape[0]), device=device)

        if self.cell == "lstm":
            if isinstance(h_prev, tuple) and len(h_prev) == 2:
                h_prev_use = h_prev
            else:
                z = self._zeros(int(z_t.shape[0]), device)
                h_prev_use = (z, z.clone())
            h_t, c_t = self._core(inp, h_prev_use)  # type: ignore[arg-type]
            return {"schema_version": "REC_STATE_V1", "h_t": (h_t, c_t)}

        if isinstance(h_prev, tuple):
            h_prev_use = h_prev[0]
        else:
            h_prev_use = h_prev
        h_t = self._core(inp, h_prev_use)  # type: ignore[arg-type]
        return {"schema_version": "REC_STATE_V1", "h_t": h_t}

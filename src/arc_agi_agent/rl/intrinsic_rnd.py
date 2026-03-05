from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class RNDNormState:
    mean: float = 0.0
    var: float = 1.0
    count: int = 0

    def update(self, x: float) -> None:
        self.count += 1
        if self.count == 1:
            self.mean = float(x)
            self.var = 0.0
            return
        delta = float(x) - self.mean
        self.mean += delta / float(self.count)
        delta2 = float(x) - self.mean
        self.var += delta * delta2

    def std(self) -> float:
        if self.count <= 1:
            return 1.0
        return math.sqrt(self.var / float(self.count - 1))


def compute_phi(err_scalar: float, norm_state: Optional[RNDNormState], phi_clip: float) -> float:
    if norm_state is None:
        norm_state = RNDNormState()
    norm_state.update(float(err_scalar))
    denom = max(1e-8, float(norm_state.std()))
    z = (float(err_scalar) - float(norm_state.mean)) / denom
    if z < 0.0:
        z = 0.0
    if phi_clip > 0.0:
        z = max(-phi_clip, min(phi_clip, z))
    return float(z)


class IntrinsicRND(nn.Module):
    def __init__(self, grid_embed_dim: int, rnd_hidden: int, rnd_out: int) -> None:
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(grid_embed_dim, rnd_hidden),
            nn.ReLU(),
            nn.Linear(rnd_hidden, rnd_out),
        )
        self.predictor = nn.Sequential(
            nn.Linear(grid_embed_dim, rnd_hidden),
            nn.ReLU(),
            nn.Linear(rnd_hidden, rnd_out),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, grid_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.predictor(grid_embed)
        with torch.no_grad():
            tgt = self.target(grid_embed)
        err_vec = pred - tgt
        err_scalar = (err_vec.pow(2).mean(dim=1) if err_vec.dim() > 1 else err_vec.pow(2).mean()).view(-1)
        return pred, tgt, err_vec, err_scalar

    @staticmethod
    def compute_phi(err_scalar: float, norm_state: Optional[RNDNormState], phi_clip: float) -> float:
        return compute_phi(err_scalar, norm_state, phi_clip)

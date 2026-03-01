from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_grid import canonical_grid
from .obs_norm_v1 import normalize_obs_v1


def _default_cfg() -> Dict[str, Any]:
    return {
        "embed_dim": 256,
        "use_multigrid": False,
        "cnn_channels": [32, 64, 128],
        "meta_hidden": 128,
        "meta_max_dim": 64,
        "frame_stack": 4,
    }


class ObservationEncoder(nn.Module):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = {**_default_cfg(), **(cfg or {})}
        chs = list(self.cfg["cnn_channels"])
        self.frame_stack = max(1, int(self.cfg.get("frame_stack", 4)))

        layers: List[nn.Module] = []
        in_ch = self.frame_stack
        for ch in chs:
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_ch = ch
        self.grid_backbone = nn.Sequential(*layers)

        self.meta_max_dim = int(self.cfg["meta_max_dim"])
        meta_hidden = int(self.cfg["meta_hidden"])
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_max_dim, meta_hidden),
            nn.ReLU(),
        )

        self.proj = nn.Linear(chs[-1] + meta_hidden, int(self.cfg["embed_dim"]))

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _grid_tensor(
        self,
        grids: List[Dict[str, Any]],
        grid_stack: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        device = self._device()
        if grid_stack is not None and len(grid_stack) > 0:
            stack_np = [np.asarray(g, dtype=np.int64) for g in grid_stack]
            target_h, target_w = int(stack_np[-1].shape[0]), int(stack_np[-1].shape[1])
            padded: List[np.ndarray] = []
            for g in stack_np:
                arr = np.asarray(g, dtype=np.int64)
                if arr.shape != (target_h, target_w):
                    canvas = np.zeros((target_h, target_w), dtype=np.int64)
                    h = min(target_h, int(arr.shape[0]))
                    w = min(target_w, int(arr.shape[1]))
                    canvas[:h, :w] = arr[:h, :w]
                    arr = canvas
                padded.append(arr)
            while len(padded) < self.frame_stack:
                padded.insert(0, padded[0].copy())
            if len(padded) > self.frame_stack:
                padded = padded[-self.frame_stack :]
            arr = torch.tensor(np.stack(padded, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
            vmax = max(1.0, float(arr.max().item()))
            return arr / vmax
        if not grids:
            return torch.zeros((1, self.frame_stack, 64, 64), dtype=torch.float32, device=device)

        if not bool(self.cfg.get("use_multigrid", False)):
            g = grids[0]
            grid = np.asarray(g["grid"], dtype=np.int64)
            stack = [grid.copy() for _ in range(self.frame_stack)]
            arr = torch.tensor(np.stack(stack, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
            vmax = max(1.0, float(arr.max().item()))
            return arr / vmax

        max_h = max(int(g.get("height", 0)) for g in grids)
        max_w = max(int(g.get("width", 0)) for g in grids)
        planes: List[torch.Tensor] = []
        for g in grids:
            arr = torch.tensor(g["grid"], dtype=torch.float32, device=device)
            h, w = int(arr.shape[0]), int(arr.shape[1])
            if h != max_h or w != max_w:
                canvas = torch.zeros((max_h, max_w), dtype=torch.float32, device=device)
                canvas[:h, :w] = arr
                arr = canvas
            planes.append(arr)
        while len(planes) < self.frame_stack:
            planes.append(planes[-1].clone())
        if len(planes) > self.frame_stack:
            planes = planes[: self.frame_stack]
        stacked = torch.stack(planes, dim=0)
        vmax = max(1.0, float(stacked.max().item()))
        return (stacked / vmax).unsqueeze(0)

    def _meta_tensor(self, meta_vector: List[float]) -> torch.Tensor:
        device = self._device()
        vec = list(meta_vector[: self.meta_max_dim])
        if len(vec) < self.meta_max_dim:
            vec.extend([0.0] * (self.meta_max_dim - len(vec)))
        return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)

    def encode(
        self,
        observation: Any,
        fp_report: Optional[Any] = None,
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = {**self.cfg, **(cfg or {})}
        ctx_eff = ctx or {}
        obs_norm = ctx_eff.get("obs_norm") if isinstance(ctx_eff.get("obs_norm"), dict) else normalize_obs_v1(observation, fp_report=fp_report)
        grid_stack = ctx_eff.get("grid_stack")
        if not isinstance(grid_stack, list):
            grid_stack = [canonical_grid(obs_norm)]

        grid_t = self._grid_tensor(
            obs_norm.get("grids", []),
            grid_stack=grid_stack,
        )
        grid_feat = self.grid_backbone(grid_t)
        grid_embed = F.adaptive_avg_pool2d(grid_feat, output_size=1).flatten(1)

        meta_t = self._meta_tensor(obs_norm.get("meta_vector", []))
        meta_embed = self.meta_mlp(meta_t)

        z_t = self.proj(torch.cat([grid_embed, meta_embed], dim=1))

        return {
            "schema_version": "ENCODER_OUT_V1",
            "z_t": z_t,
            "grid_embed": grid_embed,
            "meta_embed": meta_embed,
            "obs_norm": obs_norm,
            "debug": {
                "grid_tensor_shape": list(grid_t.shape),
                "meta_dim": int(meta_t.shape[1]),
                "meta_keys": obs_norm.get("meta_keys", []),
            },
        }

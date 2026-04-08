from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class VisualEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, encoder_dim: int = 256, channels: tuple[int, ...] = (32, 64, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            current = out_channels
        layers.append(nn.Conv2d(current, encoder_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.encoder_dim = encoder_dim

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.net(frames)
        batch, channels, height, width = spatial.shape
        tokens = spatial.flatten(2).transpose(1, 2)
        device = spatial.device
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([yy, xx], dim=-1).reshape(1, height * width, 2)
        pos_embed = _sinusoidal_expand(pos, channels)
        tokens = tokens + pos_embed
        return spatial, tokens


def _sinusoidal_expand(pos: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 4
    freqs = torch.exp(torch.linspace(0.0, 1.0, half, device=pos.device) * -4.0)
    y = pos[..., 0:1] * freqs
    x = pos[..., 1:2] * freqs
    enc = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)
    if enc.shape[-1] < dim:
        enc = F.pad(enc, (0, dim - enc.shape[-1]))
    return enc[..., :dim]


class SlotAttention(nn.Module):
    def __init__(self, slot_count: int, slot_dim: int, iters: int = 3) -> None:
        super().__init__()
        self.slot_count = slot_count
        self.slot_dim = slot_dim
        self.iters = iters
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.1)
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        batch = tokens.shape[0]
        tokens = self.norm_inputs(tokens)
        mu = self.slot_mu.expand(batch, self.slot_count, -1)
        sigma = self.slot_logsigma.exp().expand(batch, self.slot_count, -1)
        slots = mu + sigma * torch.randn_like(mu)
        k = self.project_k(tokens)
        v = self.project_v(tokens)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots))
            attn_logits = torch.einsum("bsd,btd->bst", q, k) / (self.slot_dim**0.5)
            attn = attn_logits.softmax(dim=1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
            updates = torch.einsum("bst,btd->bsd", attn, v)
            slots = self.gru(
                updates.reshape(batch * self.slot_count, self.slot_dim),
                slots_prev.reshape(batch * self.slot_count, self.slot_dim),
            ).reshape(batch, self.slot_count, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        diagnostics = {
            "slot_norm_mean": float(slots.norm(dim=-1).mean().detach().cpu()),
            "slot_norm_std": float(slots.norm(dim=-1).std().detach().cpu()),
            "slot_similarity_mean": float(_pairwise_similarity(slots).mean().detach().cpu()),
            "slot_collapse_fraction": float((slots.std(dim=1).mean(dim=-1) < 1e-3).float().mean().detach().cpu()),
        }
        return slots, diagnostics


def _pairwise_similarity(slots: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(slots, dim=-1)
    return torch.matmul(normalized, normalized.transpose(1, 2))


class SlotRelationBlock(nn.Module):
    def __init__(self, slot_dim: int, heads: int, layers: int) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim,
            nhead=heads,
            dim_feedforward=slot_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.encoder(slots)


class SlotPooler(nn.Module):
    def __init__(self, dim: int, mode: str = "attention") -> None:
        super().__init__()
        self.mode = mode
        self.attn = nn.Linear(dim, 1)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.mode == "mean":
            pooled = slots.mean(dim=1)
            return pooled, {"pooling_mode": "mean"}
        weights = torch.softmax(self.attn(slots).squeeze(-1), dim=-1)
        pooled = torch.einsum("bs,bsd->bd", weights, slots)
        return pooled, {
            "pooling_mode": "attention",
            "pooling_weight_entropy": float((-(weights * (weights.clamp_min(1e-8)).log()).sum(dim=-1).mean()).detach().cpu()),
        }


class RecurrentCore(nn.Module):
    def __init__(
        self,
        scene_dim: int,
        hidden_dim: int,
        action_vocab: int,
        max_game_ids: int,
        game_embed_dim: int,
        max_level_index: int,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_vocab, hidden_dim)
        self.game_embedding = nn.Embedding(max_game_ids, game_embed_dim)
        self.gru = nn.GRUCell(scene_dim + hidden_dim + game_embed_dim + 3, hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.hidden_dim = hidden_dim
        self.max_level_index = int(max_level_index)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        scene: torch.Tensor,
        prev_action: torch.Tensor,
        game_id_index: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        current_level_index: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_action = self.action_embedding(prev_action)
        embedded_game = self.game_embedding(game_id_index.long())
        normalized_level = current_level_index.float() / float(max(1, self.max_level_index - 1))
        x = torch.cat(
            [
                scene,
                embedded_action,
                embedded_game,
                prev_reward.unsqueeze(-1),
                prev_done.unsqueeze(-1),
                normalized_level.unsqueeze(-1),
            ],
            dim=-1,
        )
        next_hidden = self.gru(x, hidden)
        latent = self.proj(next_hidden)
        return next_hidden, latent


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, action_vocab: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_vocab))

    def forward(self, latent: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(latent)
        action_mask = action_mask.clone()
        empty_rows = action_mask.sum(dim=-1) == 0
        if empty_rows.any():
            action_mask[empty_rows, 1] = True
        invalid = ~action_mask
        return logits.masked_fill(invalid, -1e9)


class ClickPolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, spatial_dim: int) -> None:
        super().__init__()
        self.lat_proj = nn.Linear(hidden_dim, spatial_dim)
        self.conv = nn.Conv2d(spatial_dim, 1, kernel_size=1)

    def forward(self, latent: torch.Tensor, spatial: torch.Tensor, valid_pixel_mask: torch.Tensor) -> torch.Tensor:
        gating = self.lat_proj(latent).unsqueeze(-1).unsqueeze(-1)
        logits = self.conv(torch.tanh(spatial + gating))
        mask = F.interpolate(valid_pixel_mask.float(), size=logits.shape[-2:], mode="nearest")
        logits = logits.masked_fill(mask <= 0, -1e9)
        return logits


class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


class DynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int, action_vocab: int) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_vocab, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_latent = nn.Linear(hidden_dim, hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor, action_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.action_embedding(action_ids)
        trunk = self.trunk(torch.cat([latent, embedded], dim=-1))
        return self.next_latent(trunk), self.reward_head(trunk).squeeze(-1), self.done_head(trunk).squeeze(-1)

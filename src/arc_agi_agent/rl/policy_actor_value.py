from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_key_normalize_v1 import action_id_to_index


def _default_cfg() -> Dict[str, Any]:
    return {
        "hidden_dim": 256,
        "action_emb_dim": 32,
        "max_actions": 128,
        "actor_hidden": [256, 128],
        "value_hidden": [128],
        "num_modes": 3,
        "mode_action_bias": {},
        "mode_action_allow": {},
        "mode_coord_bias": {},
        "coord_features": {
            "patch_radius": 1,
            "use_fp_objects": True,
        },
    }


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.ReLU())
        last = int(h)
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def _safe_temperature(value: Any) -> float:
    try:
        t = float(value)
    except Exception:
        t = 1.0
    if not math.isfinite(t) or t <= 0.0:
        t = 1.0
    return t


def apply_action_mask_and_bias(
    logits: torch.Tensor,
    action_mask: Optional[torch.Tensor],
    mode_bias: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    out = logits
    if mode_bias is not None:
        out = out + mode_bias
    if action_mask is not None:
        out = out + (action_mask.float() - 1.0) * 1e9
    temp = _safe_temperature(temperature)
    if temp != 1.0:
        out = out / temp
    return out


def apply_coord_mask_and_bias(
    coord_logits: torch.Tensor,
    coord_mask: Optional[torch.Tensor],
    coord_bias: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    out = coord_logits
    if coord_bias is not None:
        out = out + coord_bias
    if coord_mask is not None:
        out = out + (coord_mask.float() - 1.0) * 1e9
    temp = _safe_temperature(temperature)
    if temp != 1.0:
        out = out / temp
    return out


class PolicyActor(nn.Module):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = {**_default_cfg(), **(cfg or {})}
        h = int(self.cfg["hidden_dim"])
        aemb = int(self.cfg["action_emb_dim"])
        self.max_actions = int(self.cfg["max_actions"])
        self.action_emb = nn.Embedding(self.max_actions, aemb)
        self.discrete_head = _mlp(h + aemb, list(self.cfg["actor_hidden"]), 1)
        self.coord_head = _mlp(h + 8, list(self.cfg["actor_hidden"]), 1)

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if is_dataclass(value):
            return asdict(value)
        return {}

    @staticmethod
    def _stable_tag_id(tag: Any) -> float:
        raw = str(tag or "")
        digest = hashlib.sha1(raw.encode("utf-8")).digest()
        intval = int.from_bytes(digest[:4], byteorder="big", signed=False)
        return float(intval % 1024) / 1024.0

    def _extract_fp_centroids(self, fp_report: Any) -> List[tuple[float, float]]:
        out: List[tuple[float, float]] = []
        fp = self._as_dict(fp_report)
        feats = fp.get("features_v1", {}) if isinstance(fp, dict) else {}
        obj_idx = feats.get("object_index") if isinstance(feats, dict) else None
        if isinstance(obj_idx, list):
            for obj in obj_idx:
                c = obj.get("centroid") if isinstance(obj, dict) else None
                if isinstance(c, (list, tuple)) and len(c) == 2:
                    try:
                        out.append((float(c[1]), float(c[0])))  # (x, y)
                    except Exception:
                        continue
        return out

    def _coord_feat(
        self,
        cand: Dict[str, Any],
        grid: Optional[np.ndarray],
        centroids: List[tuple[float, float]],
        cfg_eff: Dict[str, Any],
    ) -> List[float]:
        x = float(cand.get("x", 0.0))
        y = float(cand.get("y", 0.0))
        if grid is not None and grid.ndim == 2 and grid.shape[0] > 0 and grid.shape[1] > 0:
            h = int(grid.shape[0])
            w = int(grid.shape[1])
        else:
            h = max(1, int(cand.get("grid_h", 64) or 64))
            w = max(1, int(cand.get("grid_w", 64) or 64))

        x_n = min(1.0, max(0.0, x / max(1.0, float(w))))
        y_n = min(1.0, max(0.0, y / max(1.0, float(h))))
        tag_n = self._stable_tag_id(cand.get("tag"))

        patch_radius = int(cfg_eff.get("coord_features", {}).get("patch_radius", 1))
        patch_radius = max(0, patch_radius)
        num_non_bg = 0.0
        num_unique = 0.0
        center_color_n = 0.0
        if grid is not None and grid.ndim == 2 and h > 0 and w > 0:
            xi = max(0, min(w - 1, int(round(x))))
            yi = max(0, min(h - 1, int(round(y))))
            y0 = max(0, yi - patch_radius)
            y1 = min(h, yi + patch_radius + 1)
            x0 = max(0, xi - patch_radius)
            x1 = min(w, xi + patch_radius + 1)
            patch = grid[y0:y1, x0:x1]
            if patch.size > 0:
                num_non_bg = float((patch != 0).sum()) / float(patch.size)
                num_unique = float(np.unique(patch).size) / 10.0
                center_color_n = float(grid[yi, xi]) / 10.0

        edge_dist = min(
            max(0.0, x),
            max(0.0, y),
            max(0.0, float(w - 1) - x),
            max(0.0, float(h - 1) - y),
        )
        edge_dist_n = min(1.0, edge_dist / max(1.0, float(max(h, w))))

        nearest_centroid_n = 1.0
        if centroids:
            d = min(abs(x - cx) + abs(y - cy) for cx, cy in centroids)
            nearest_centroid_n = min(1.0, float(d) / max(1.0, float(h + w)))

        return [
            x_n,
            y_n,
            tag_n,
            num_non_bg,
            num_unique,
            center_color_n,
            nearest_centroid_n,
            edge_dist_n,
        ]

    def _mode_idx(self, mode_id: Any, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(mode_id, torch.Tensor):
            out = mode_id.to(device).view(-1)
            if out.numel() == 1 and batch > 1:
                out = out.repeat(batch)
            return out.long()
        return torch.full((batch,), int(mode_id or 0), dtype=torch.long, device=device)

    def _apply_mode_priors(
        self,
        logits: torch.Tensor,
        action_ids: List[str],
        mode_idx: int,
        cfg_eff: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        allow_cfg = cfg_eff.get("mode_action_allow", {}).get(str(mode_idx), None)
        if isinstance(allow_cfg, list) and allow_cfg:
            allow = set(str(a) for a in allow_cfg)
            mask = torch.tensor([1.0 if aid in allow else 0.0 for aid in action_ids], dtype=torch.float32, device=device).unsqueeze(0)
            logits = logits + (mask - 1.0) * 1e9

        bias_cfg = cfg_eff.get("mode_action_bias", {}).get(str(mode_idx), {})
        if isinstance(bias_cfg, dict) and bias_cfg:
            bias = torch.tensor([float(bias_cfg.get(aid, 0.0)) for aid in action_ids], dtype=torch.float32, device=device).unsqueeze(0)
            logits = logits + bias
        return logits

    def forward(
        self,
        h_t: torch.Tensor,
        mode_id: Any,
        available_actions: List[str],
        coord_candidates: Optional[List[Dict[str, Any]]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = {**self.cfg, **(cfg or {})}
        device = h_t.device
        batch = int(h_t.shape[0])
        mode = self._mode_idx(mode_id, batch=batch, device=device)
        mode_scalar = int(mode[0].item()) if mode.numel() > 0 else 0
        ctx_eff = ctx or {}

        action_ids = [str(a) for a in available_actions] if available_actions else ["ACTION1"]
        discrete_logits: List[torch.Tensor] = []
        for aid in action_ids:
            idx = action_id_to_index(aid, self.max_actions)
            emb = self.action_emb(torch.tensor([idx], dtype=torch.long, device=device)).repeat(batch, 1)
            discrete_logits.append(self.discrete_head(torch.cat([h_t, emb], dim=1)))
        pi_discrete = torch.cat(discrete_logits, dim=1)
        pi_discrete = self._apply_mode_priors(pi_discrete, action_ids, mode_scalar, cfg_eff, device)

        pi_coord = None
        coord_logits_topk = None
        coord_feature_vectors = None
        if coord_candidates:
            coord_logits: List[torch.Tensor] = []
            coord_feats_out: List[List[float]] = []
            mode_coord_bias = float(cfg_eff.get("mode_coord_bias", {}).get(str(mode_scalar), 0.0))
            grid_raw = ctx_eff.get("grid")
            grid = np.asarray(grid_raw) if grid_raw is not None else None
            if grid is not None and (grid.ndim != 2):
                grid = None
            use_fp_objects = bool(cfg_eff.get("coord_features", {}).get("use_fp_objects", True))
            centroids = self._extract_fp_centroids(ctx_eff.get("fp_report")) if use_fp_objects else []
            for cand in coord_candidates:
                feat_vec_raw = cand.get("feat_vec")
                if isinstance(feat_vec_raw, list) and len(feat_vec_raw) == 8:
                    feat_vec = [float(v) for v in feat_vec_raw]
                else:
                    feat_vec = self._coord_feat(cand, grid=grid, centroids=centroids, cfg_eff=cfg_eff)
                coord_feats_out.append(feat_vec)
                feat = torch.tensor([feat_vec], dtype=torch.float32, device=device).repeat(batch, 1)
                coord_logits.append(self.coord_head(torch.cat([h_t, feat], dim=1)) + mode_coord_bias)
            pi_coord = torch.cat(coord_logits, dim=1)
            coord_feature_vectors = coord_feats_out
            k = min(5, int(pi_coord.shape[1]))
            topv, topi = torch.topk(pi_coord, k=k, dim=1)
            coord_logits_topk = {
                "indices": topi[0].detach().cpu().tolist(),
                "logits": topv[0].detach().cpu().tolist(),
            }

        return {
            "schema_version": "POLICY_OUT_V1",
            "pi_discrete": pi_discrete,
            "pi_coord": pi_coord,
            "action_ids": action_ids,
            "mode_id": mode,
            "coord_logits_topk": coord_logits_topk,
            "coord_feature_vectors": coord_feature_vectors,
        }

    def compute_logp_components(
        self,
        h_t: torch.Tensor,
        batch_step: Dict[str, Any],
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg_eff = {**self.cfg, **(cfg or {})}
        _ = ctx
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        if h_t.dim() != 2:
            h_t = h_t.reshape(h_t.shape[0], -1)
        device = h_t.device
        batch = int(h_t.shape[0])

        mode_id = int(batch_step.get("mode_id", 0))
        mode_index = torch.full((batch,), mode_id, dtype=torch.long, device=device)
        mode_logits_raw = batch_step.get("mode_logits")
        if not torch.is_tensor(mode_logits_raw):
            raise RuntimeError("compute_logp_components requires batch_step['mode_logits'] tensor")
        mode_logits = mode_logits_raw.to(device=device, dtype=torch.float32)
        controller_cfg = cfg_eff.get("controller", {}) if isinstance(cfg_eff.get("controller"), dict) else {}
        mode_temperature = _safe_temperature(controller_cfg.get("temperature", 1.0))
        mode_logits_eff = mode_logits / mode_temperature if mode_temperature != 1.0 else mode_logits
        logp_mode, _ = self.mode_logp_entropy(mode_logits_eff, mode_index)

        action_ids = [str(a) for a in (batch_step.get("action_ids") or ["ACTION1"])]
        nd = len(action_ids)
        action_indices = torch.tensor([action_id_to_index(a, self.max_actions) for a in action_ids], dtype=torch.long, device=device)
        aemb = self.action_emb(action_indices)
        h_exp = h_t.unsqueeze(1).expand(-1, nd, -1)
        emb_exp = aemb.unsqueeze(0).expand(batch, -1, -1)
        discrete_in = torch.cat([h_exp, emb_exp], dim=2).contiguous()
        raw_action_logits = self.discrete_head(discrete_in.view(batch * nd, -1)).view(batch, nd)

        action_mask_raw = batch_step.get("available_actions_mask")
        if isinstance(action_mask_raw, list) and action_mask_raw:
            base_action_mask = torch.tensor(
                [bool(action_mask_raw[i]) if i < len(action_mask_raw) else False for i in range(nd)],
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0).expand(batch, -1)
        else:
            base_action_mask = torch.ones((batch, nd), dtype=torch.bool, device=device)

        allow_cfg = cfg_eff.get("mode_action_allow", {}).get(str(mode_id), None)
        allow_mask = torch.ones((batch, nd), dtype=torch.bool, device=device)
        if isinstance(allow_cfg, list) and allow_cfg:
            allow = set(str(a) for a in allow_cfg)
            allow_mask = torch.tensor([a in allow for a in action_ids], dtype=torch.bool, device=device).unsqueeze(0).expand(batch, -1)
        final_action_mask = base_action_mask & allow_mask

        bias_cfg = cfg_eff.get("mode_action_bias", {}).get(str(mode_id), {})
        mode_bias = None
        if isinstance(bias_cfg, dict) and bias_cfg:
            mode_bias = torch.tensor([float(bias_cfg.get(a, 0.0)) for a in action_ids], dtype=torch.float32, device=device).unsqueeze(0).expand(batch, -1)

        actor_temperature = _safe_temperature(cfg_eff.get("temperature", 1.0))
        action_logits = apply_action_mask_and_bias(raw_action_logits, final_action_mask, mode_bias, actor_temperature)
        action_index = int(batch_step.get("action_index", 0))
        action_index_t = torch.full((batch,), max(0, min(action_index, nd - 1)), dtype=torch.long, device=device)
        logp_discrete, _ = self.action_logp_entropy(action_logits, action_index_t, action_mask=None)

        coords = batch_step.get("coord_candidates") or []
        k = int(cfg_eff.get("coord_topK", max(1, len(coords) if isinstance(coords, list) else 1)))
        k = max(1, k)
        coord_mask = torch.zeros((batch, k), dtype=torch.bool, device=device)
        coord_feats = torch.zeros((batch, k, 8), dtype=torch.float32, device=device)
        if isinstance(coords, list):
            lim = min(k, len(coords))
            for j in range(lim):
                c = coords[j]
                fv = c.get("feat_vec") if isinstance(c, dict) else None
                if isinstance(fv, list) and len(fv) == 8:
                    coord_feats[:, j, :] = torch.tensor([float(x) for x in fv], dtype=torch.float32, device=device).unsqueeze(0).expand(batch, -1)
                    coord_mask[:, j] = True
        coord_mask_raw = batch_step.get("coord_mask")
        if isinstance(coord_mask_raw, list) and coord_mask_raw:
            lim_m = min(k, len(coord_mask_raw))
            coord_mask[:, :lim_m] = torch.tensor([bool(x) for x in coord_mask_raw[:lim_m]], dtype=torch.bool, device=device).unsqueeze(0).expand(batch, -1)

        h_coord = h_t.unsqueeze(1).expand(-1, k, -1)
        coord_in = torch.cat([h_coord, coord_feats], dim=2).contiguous()
        raw_coord_logits = self.coord_head(coord_in.view(batch * k, -1)).view(batch, k)
        coord_bias_v = float(cfg_eff.get("mode_coord_bias", {}).get(str(mode_id), 0.0))
        coord_bias = torch.full((batch, 1), coord_bias_v, dtype=torch.float32, device=device)
        coord_logits = apply_coord_mask_and_bias(raw_coord_logits, coord_mask, coord_bias, actor_temperature)

        chosen_coord_index = int(batch_step.get("chosen_coord_index", -1))
        coord_index_safe = torch.full((batch,), max(0, min(chosen_coord_index, k - 1)), dtype=torch.long, device=device)
        logp_coord_all, _ = self.coord_logp_entropy(coord_logits, coord_index_safe, coord_mask=None)

        has_coord = bool(batch_step.get("has_coord", int(chosen_coord_index >= 0)))
        action_id = action_ids[action_index_t[0].item()] if action_ids else "ACTION1"
        coord_use = bool(action_id.upper() == "ACTION6" and has_coord and chosen_coord_index >= 0)
        logp_coord = logp_coord_all if coord_use else torch.zeros_like(logp_coord_all)

        logp_total = logp_mode + logp_discrete + logp_coord
        return logp_mode, logp_discrete, logp_coord, logp_total

    @staticmethod
    def mode_logp_entropy(mode_logits: torch.Tensor, mode_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mode_index.dim() == 0:
            mode_index = mode_index.unsqueeze(0)
        mode_index = mode_index.long().view(-1, 1)
        logp_all = F.log_softmax(mode_logits, dim=1)
        probs = F.softmax(mode_logits, dim=1)
        logp = logp_all.gather(1, mode_index).squeeze(1)
        entropy = -(probs * logp_all).sum(dim=1)
        return logp, entropy

    @staticmethod
    def action_logp_entropy(
        action_logits: torch.Tensor,
        action_index: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_logits = action_logits
        if action_mask is not None:
            masked_logits = action_logits + (action_mask.float() - 1.0) * 1e9
        if action_index.dim() == 0:
            action_index = action_index.unsqueeze(0)
        action_index = action_index.long().view(-1, 1)
        logp_all = F.log_softmax(masked_logits, dim=1)
        probs = F.softmax(masked_logits, dim=1)
        logp = logp_all.gather(1, action_index).squeeze(1)
        entropy = -(probs * logp_all).sum(dim=1)
        return logp, entropy

    @staticmethod
    def coord_logp_entropy(
        coord_logits: torch.Tensor,
        coord_index: torch.Tensor,
        coord_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_logits = coord_logits
        if coord_mask is not None:
            masked_logits = coord_logits + (coord_mask.float() - 1.0) * 1e9
        if coord_index.dim() == 0:
            coord_index = coord_index.unsqueeze(0)
        coord_index = coord_index.long().view(-1, 1)
        logp_all = F.log_softmax(masked_logits, dim=1)
        probs = F.softmax(masked_logits, dim=1)
        logp = logp_all.gather(1, coord_index).squeeze(1)
        entropy = -(probs * logp_all).sum(dim=1)
        return logp, entropy


class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg_eff = {**_default_cfg(), **(cfg or {})}
        self.net = _mlp(int(hidden_dim), list(cfg_eff["value_hidden"]), 1)

    def forward(self, h_t: torch.Tensor, cfg: Optional[Dict[str, Any]] = None, ctx: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        return self.net(h_t).squeeze(1)

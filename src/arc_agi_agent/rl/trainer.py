from __future__ import annotations

import logging
import hashlib
from array import array
import json
import math
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .policy_actor_value import apply_action_mask_and_bias, apply_coord_mask_and_bias

logger = logging.getLogger(__name__)


def _default_cfg() -> Dict[str, Any]:
    return {
        "algo": "a2c",
        "gamma": 0.99,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "controller_coef": 1.0,
        "actor_coef": 1.0,
        "aux_mode_ce_coef": 0.2,
        "max_grad_norm": 1.0,
        "log": {
            "heartbeat_enabled": True,
            "heartbeat_every_minibatches": 10,
        },
        "ppo": {
            "clip_eps": 0.2,
            "clip_eps_coord": 0.2,
            "epochs": 4,
            "minibatches": 4,
            "target_kl": 0.02,
            "early_stop_kl": True,
            "kl_metric": "action",
            "recompute_post_update_metrics": False,
            "recompute_values": True,
            "allow_bias_cache_max": 256,
            "preupdate_eval_mode": False,
            "use_gae": True,
            "gae_lambda": 0.95,
            "adv_norm": True,
            "vf_clip": True,
            "vf_clip_eps": 0.2,
            "mode_entropy_coef": 0.01,
            "coord_coef": 1.0,
        },
    }


def _merge_cfg(defaults: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(defaults)
    if not isinstance(override, dict):
        return out
    for k, v in override.items():
        if k in {"ppo", "log"} and isinstance(v, dict):
            base = dict(out.get(k, {})) if isinstance(out.get(k, {}), dict) else {}
            base.update(v)
            out[k] = base
        else:
            out[k] = v
    return out


def normalize_ppo_cfg(ppo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(ppo_cfg or {})
    legacy_keys = {
        "clip_range_actor",
        "clip_range_controller",
        "vf_clip_range",
        "num_epochs_per_iter",
        "minibatches_per_epoch",
        "advantage_normalization",
    }
    used_legacy = sorted(k for k in cfg.keys() if k in legacy_keys)
    if used_legacy:
        raise RuntimeError(f"Legacy PPO config keys are not supported: {used_legacy}")
    clip_eps_shared = float(cfg.get("clip_eps", 0.2))

    if "clip_eps_actor" not in cfg:
        cfg["clip_eps_actor"] = clip_eps_shared
    if "clip_eps_controller" not in cfg:
        cfg["clip_eps_controller"] = clip_eps_shared
    if "clip_eps_coord" not in cfg:
        cfg["clip_eps_coord"] = float(cfg.get("clip_eps_actor", clip_eps_shared))
    if "vf_clip_eps" not in cfg:
        cfg["vf_clip_eps"] = clip_eps_shared

    allowed = {
        "clip_eps",
        "clip_eps_actor",
        "clip_eps_controller",
        "clip_eps_coord",
        "epochs",
        "minibatches",
        "target_kl",
        "target_kl_early",
        "target_kl_late",
        "preupdate_kl_max",
        "early_stop_kl",
        "kl_metric",
        "recompute_post_update_metrics",
        "recompute_values",
        "allow_bias_cache_max",
        "preupdate_eval_mode",
        "use_gae",
        "gae_lambda",
        "adv_norm",
        "vf_clip",
        "vf_clip_eps",
        "mode_entropy_coef",
        "entropy_coef_controller",
        "entropy_coef_actor_phase1",
        "entropy_coef_actor_phase2",
        "aux_ce_coef",
        "gamma",
        "value_coef",
        "coord_coef",
        "max_grad_norm",
        "lr",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "weight_decay",
        "bptt_chunk_len",
        "lr_adapt_enabled",
        "lr_adapt_target_kl",
        "lr_adapt_min_lr_mult",
        "lr_adapt_max_lr_mult",
        "lr_adapt_downscale",
        "lr_adapt_upscale",
        "lr_adapt_upscale_patience_iters",
        "lr_adapt_use_abs_kl",
        "value_recompute_chunk",
        "max_steps_per_train_step",
    }
    unknown = sorted(set(cfg.keys()) - allowed)
    if unknown:
        raise RuntimeError(f"Unknown PPO config keys: {unknown}")
    return cfg


def _build_discrete_pi(
    pi_discrete_raw: torch.Tensor,
    mode_index: torch.Tensor,
    action_ids_union: List[str],
    action_available_by_id_rows: List[Dict[str, bool]],
    cfg_eff: Dict[str, Any],
    raw_mask_type: Optional[List[str]] = None,
    raw_mask_preview: Optional[List[List[str]]] = None,
    step_indices: Optional[List[int]] = None,
    apply_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pi_discrete_raw.device
    action_mask_rows: List[List[bool]] = []
    for row_i, avail_map in enumerate(action_available_by_id_rows):
        row = [bool(avail_map.get(a, False)) for a in action_ids_union]
        if sum(1 for v in row if v) == 0 and len(row) > 0:
            step_idx = step_indices[row_i] if step_indices and row_i < len(step_indices) else row_i
            typ = raw_mask_type[row_i] if raw_mask_type and row_i < len(raw_mask_type) else "unknown"
            preview = raw_mask_preview[row_i] if raw_mask_preview and row_i < len(raw_mask_preview) else []
            raise RuntimeError(
                f"available_actions_mask has zero valid actions step={step_idx} typ={typ} preview={preview}"
            )
        action_mask_rows.append(row)
    action_mask = torch.tensor(action_mask_rows, dtype=torch.bool, device=device).contiguous()

    allow_cfg_all = cfg_eff.get("mode_action_allow", {})
    bias_cfg_all = cfg_eff.get("mode_action_bias", {}) if apply_bias else {}
    allow_mask = torch.ones_like(action_mask, dtype=torch.bool, device=device)
    mode_bias = torch.zeros_like(action_mask, dtype=torch.float32, device=device)
    for mode_val in torch.unique(mode_index).detach().cpu().tolist():
        row_sel = mode_index == int(mode_val)
        if not bool(row_sel.any()):
            continue
        allow_cfg = allow_cfg_all.get(str(mode_val), None) if isinstance(allow_cfg_all, dict) else None
        if isinstance(allow_cfg, list) and allow_cfg:
            allow_set = set(str(a) for a in allow_cfg)
            allow_row = torch.tensor([a in allow_set for a in action_ids_union], dtype=torch.bool, device=device).unsqueeze(0)
            allow_mask[row_sel] = allow_row
        if apply_bias:
            bias_cfg = bias_cfg_all.get(str(mode_val), {}) if isinstance(bias_cfg_all, dict) else {}
            if isinstance(bias_cfg, dict) and bias_cfg:
                bias_row = torch.tensor([float(bias_cfg.get(a, 0.0)) for a in action_ids_union], dtype=torch.float32, device=device).unsqueeze(0)
                mode_bias[row_sel] = bias_row

    action_temperature = float(cfg_eff.get("temperature", 1.0))
    final_action_mask = action_mask & allow_mask
    pi_discrete = apply_action_mask_and_bias(
        pi_discrete_raw,
        final_action_mask,
        mode_bias,
        action_temperature,
    )
    return pi_discrete, final_action_mask, action_mask


def _build_coord_pi(
    pi_coord_raw: torch.Tensor,
    coord_mask: torch.Tensor,
    mode_index: torch.Tensor,
    cfg_eff: Dict[str, Any],
    action_temperature: float,
    apply_bias: bool = True,
) -> torch.Tensor:
    device = pi_coord_raw.device
    bias_coord_all = cfg_eff.get("mode_coord_bias", {}) if apply_bias else {}
    mode_coord_bias = torch.zeros((pi_coord_raw.shape[0], 1), dtype=torch.float32, device=device)
    if apply_bias and isinstance(bias_coord_all, dict) and bias_coord_all:
        for mode_val in torch.unique(mode_index).detach().cpu().tolist():
            sel_mode = mode_index == int(mode_val)
            if not bool(sel_mode.any()):
                continue
            mode_coord_bias[sel_mode] = float(bias_coord_all.get(str(mode_val), 0.0))
    pi_coord = apply_coord_mask_and_bias(
        pi_coord_raw,
        coord_mask,
        mode_coord_bias,
        action_temperature,
    )
    return pi_coord


def _coord_feat_dim(cfg_eff: Dict[str, Any], actor: Any) -> int:
    if hasattr(actor, "coord_feat_dim"):
        return int(getattr(actor, "coord_feat_dim"))
    if "coord_feat_dim" in cfg_eff:
        return int(cfg_eff.get("coord_feat_dim"))
    raise RuntimeError("coord_feat_dim must be provided by actor or cfg")


def _approx_kl_from_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    return (torch.exp(log_ratio) - 1.0) - log_ratio


def _returns(rewards: List[float], dones: List[bool], gamma: float) -> List[float]:
    out = [0.0] * len(rewards)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if dones[i]:
            g = 0.0
        g = rewards[i] + gamma * g
        out[i] = g
    return out


def _compute_gae(
    rewards: List[float],
    dones: List[bool],
    values: List[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[List[float], List[float]]:
    n = len(rewards)
    adv = [0.0] * n
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        v_t = values[t]
        v_next = 0.0 if t == n - 1 or dones[t] else values[t + 1]
        delta = rewards[t] + gamma * v_next - v_t
        if dones[t]:
            last_gae = delta
        else:
            last_gae = delta + gamma * gae_lambda * last_gae
        adv[t] = last_gae
    ret = [adv[i] + values[i] for i in range(n)]
    return ret, adv


def _zero_grad(optim: Any) -> None:
    if isinstance(optim, dict):
        for opt in optim.values():
            opt.zero_grad()
    else:
        optim.zero_grad()


def _step_optim(optim: Any) -> None:
    if isinstance(optim, dict):
        for opt in optim.values():
            opt.step()
    else:
        optim.step()


def _get_optimizer_lr(optim: Any) -> float:
    if isinstance(optim, dict):
        for opt in optim.values():
            if hasattr(opt, "param_groups") and opt.param_groups:
                return float(opt.param_groups[0].get("lr", 0.0))
        return 0.0
    if hasattr(optim, "param_groups") and optim.param_groups:
        return float(optim.param_groups[0].get("lr", 0.0))
    return 0.0


def _set_optimizer_lr(optim: Any, lr: float) -> None:
    if isinstance(optim, dict):
        for opt in optim.values():
            if hasattr(opt, "param_groups"):
                for g in opt.param_groups:
                    g["lr"] = float(lr)
        return
    if hasattr(optim, "param_groups"):
        for g in optim.param_groups:
            g["lr"] = float(lr)


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32, device=device)


def _classify_raw_mask(raw: Any, nd: int) -> tuple[str, int, List[str]]:
    if raw is None:
        return "none", -1, []
    if not isinstance(raw, list):
        return "other", -1, [str(type(raw).__name__)]
    n = len(raw)
    preview = [str(x) for x in raw[:8]]
    if n == 0:
        return "empty", 0, preview
    if n == nd and all(isinstance(x, bool) for x in raw):
        return "bool_nd", n, preview
    if n == nd and all(isinstance(x, int) and int(x) in (0, 1) for x in raw):
        return "int01_nd", n, preview
    if all(isinstance(x, int) for x in raw) and ((n != nd) or any(int(x) >= 2 for x in raw)):
        return "index_list", n, preview
    if all(isinstance(x, str) for x in raw):
        return "str_list", n, preview
    return "other", n, preview


def _normalize_available_actions_mask(
    raw: Any, nd: int, action_ids: List[str], step_idx: Optional[int] = None
) -> tuple[List[bool], str, int, List[str]]:
    typ, raw_len, preview = _classify_raw_mask(raw, nd)
    aid_to_idx = {str(a): i for i, a in enumerate(action_ids)} if typ == "str_list" else {}
    if nd <= 0:
        return [], typ, raw_len, preview
    if typ in {"none", "empty"}:
        raise RuntimeError(
            f"available_actions_mask missing/invalid step={step_idx} typ={typ} nd={nd} raw_len={raw_len} preview={preview} action_ids={action_ids[:8]}"
        )
    elif typ == "bool_nd":
        pre = [bool(x) for x in raw[:nd]]
    elif typ == "int01_nd":
        pre = [bool(int(x)) for x in raw[:nd]]
    elif typ == "index_list":
        pre = [False] * nd
        for idx in raw:
            ii = int(idx)
            if 0 <= ii < nd:
                pre[ii] = True
    elif typ == "str_list":
        pre = [False] * nd
        for aid in raw:
            ii = aid_to_idx.get(str(aid))
            if ii is not None and 0 <= ii < nd:
                pre[ii] = True
    else:
        raise RuntimeError(
            f"available_actions_mask missing/invalid step={step_idx} typ={typ} nd={nd} raw_len={raw_len} preview={preview} action_ids={action_ids[:8]}"
        )
    if sum(1 for v in pre if v) == 0:
        raise RuntimeError(
            f"available_actions_mask has zero valid actions step={step_idx} typ={typ} nd={nd} raw_len={raw_len} preview={preview} action_ids={action_ids[:8]}"
        )
    return list(pre), typ, raw_len, preview


def _mode_logp_entropy_from_logits(logits: torch.Tensor, mode_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logp_all = F.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * logp_all).sum(dim=1)
    idx = mode_index.view(-1, 1)
    logp = logp_all.gather(1, idx).squeeze(1)
    return logp, entropy


def _snapshot_param_tensors(modules: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        for p_name, p in module.named_parameters():
            snap[f"{name}.{p_name}"] = p.detach().clone()
    return snap


def _param_delta_norm(modules: Dict[str, Any], snap: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    with torch.no_grad():
        for name, module in modules.items():
            for p_name, p in module.named_parameters():
                key = f"{name}.{p_name}"
                prev = snap.get(key)
                if prev is None:
                    continue
                diff = (p.detach() - prev.to(device=p.device, dtype=p.dtype)).float()
                total += float((diff * diff).sum().detach().cpu().item())
    return float(math.sqrt(max(0.0, total)))


def _rollout_cfg_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    controller_cfg = cfg.get("controller", {}) if isinstance(cfg.get("controller"), dict) else {}
    allow_raw = cfg.get("mode_action_allow", {}) if isinstance(cfg.get("mode_action_allow"), dict) else {}
    bias_raw = cfg.get("mode_action_bias", {}) if isinstance(cfg.get("mode_action_bias"), dict) else {}
    coord_bias_raw = cfg.get("mode_coord_bias", {}) if isinstance(cfg.get("mode_coord_bias"), dict) else {}

    allow_norm: Dict[str, List[str]] = {}
    for k, v in allow_raw.items():
        if isinstance(v, (list, tuple, set)):
            allow_norm[str(k)] = [str(a) for a in list(v)]
        else:
            allow_norm[str(k)] = []

    bias_norm: Dict[str, Dict[str, float]] = {}
    for k, v in bias_raw.items():
        if isinstance(v, dict):
            bias_norm[str(k)] = {str(a): float(b) for a, b in v.items()}
        else:
            bias_norm[str(k)] = {}

    coord_bias_norm: Dict[str, float] = {}
    for k, v in coord_bias_raw.items():
        try:
            coord_bias_norm[str(k)] = float(v)
        except Exception:
            coord_bias_norm[str(k)] = 0.0

    payload = {
        "coord_topK": int(cfg.get("coord_topK", 16)),
        "mode_action_allow": allow_norm,
        "mode_action_bias": bias_norm,
        "mode_coord_bias": coord_bias_norm,
        "controller_temperature": float(controller_cfg.get("temperature", 1.0)),
        "actor_temperature": float(cfg.get("temperature", 1.0)),
    }
    return payload


def _rollout_cfg_hash(cfg: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    payload = _rollout_cfg_payload(cfg)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest(), payload


def _empty_ppo_report(batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
    episodes = list(batch.get("episodes", [])) if isinstance(batch, dict) else []
    total_steps = 0
    win_eps = 0
    done_eps = 0
    win_present = False
    total_reward = 0.0
    episode_returns: List[float] = []
    for ep in episodes:
        steps = list(ep.get("steps", [])) if isinstance(ep, dict) else []
        total_steps += len(steps)
        if any(bool(s.get("done", False)) for s in steps):
            done_eps += 1
        ep_win_present = any(isinstance(s, dict) and ("win" in s) for s in steps)
        win_present = win_present or ep_win_present
        if ep_win_present and any(bool(s.get("win", False)) for s in steps):
            win_eps += 1
        if steps:
            episode_returns.append(float(sum(float(s.get("reward", 0.0)) for s in steps)))
        for s in steps:
            total_reward += float(s.get("reward", 0.0))
    mean_return = float(sum(episode_returns) / max(1, len(episode_returns))) if episode_returns else 0.0
    avg_reward = mean_return
    mean_episode_len = float(total_steps / max(1, len(episodes)))
    if not win_present:
        win_eps = 0
    win_rate = float(win_eps / max(1, len(episodes)))
    done_rate = float(done_eps / max(1, len(episodes)))
    return {
        "losses": {
            "total": 0.0,
            "actor_policy": 0.0,
            "controller_policy": 0.0,
            "value": 0.0,
            "entropy_actor": 0.0,
            "entropy_controller": 0.0,
            "aux_mode_ce": 0.0,
            "grad_norm_total": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "clipfrac_mode": 0.0,
            "clipfrac_action": 0.0,
            "clipfrac_coord": 0.0,
            "ppo_epochs_ran": 0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "train_stage": stage,
            "skipped_update": 1.0,
        },
        "mean_return": mean_return,
        "avg_reward": avg_reward,
        "mean_episode_len": mean_episode_len,
        "win_rate": win_rate,
        "done_rate": done_rate,
    }


class Trainer:
    def __init__(self) -> None:
        self.phase_state: Dict[str, Any] = {
            "stage": "exploration",
            "consecutive_eval_win_hits": 0,
        }
        self.lr_adapt_enabled = False
        self.lr_adapt_target_kl = 0.02
        self.lr_adapt_min_lr_mult = 0.1
        self.lr_adapt_max_lr_mult = 3.0
        self.lr_adapt_downscale = 0.5
        self.lr_adapt_upscale = 1.05
        self.lr_adapt_upscale_patience_iters = 5
        self.lr_adapt_use_abs_kl = True
        self._lr_base = 0.0
        self._lr_current = 0.0
        self._lr_low_kl_counter = 0

    def update_phase_from_eval(self, eval_metrics: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_eff = _merge_cfg(_default_cfg(), cfg)
        eval_cfg = cfg_eff.get("eval", {}) if isinstance(cfg_eff.get("eval"), dict) else {}
        threshold = float(eval_cfg.get("win_switch_threshold", 0.05))
        k = int(eval_cfg.get("win_switch_consecutive_k", 3))
        win_rate = float(eval_metrics.get("WinRate", 0.0))
        if win_rate >= threshold:
            self.phase_state["consecutive_eval_win_hits"] = int(self.phase_state.get("consecutive_eval_win_hits", 0)) + 1
        else:
            self.phase_state["consecutive_eval_win_hits"] = 0
        if self.phase_state.get("stage") == "exploration" and int(self.phase_state["consecutive_eval_win_hits"]) >= k:
            self.phase_state["stage"] = "win"
        return dict(self.phase_state)

    def train_step(
        self,
        batch: Dict[str, Any],
        modules: Dict[str, Any],
        optim: Any,
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = _merge_cfg(_default_cfg(), cfg)
        algo = str(cfg_eff.get("algo", "a2c")).lower()
        if algo == "ppo":
            return self._train_step_ppo(batch, modules, optim, cfg_eff, ctx)
        iter_idx = int((ctx or {}).get("iter_idx", -1))
        ep_count = len(batch.get("episodes", []))
        step_count = sum(len(ep.get("steps", [])) for ep in batch.get("episodes", []))
        logger.info(
            "train_step_start iter=%s algo=a2c episodes=%s steps=%s",
            iter_idx,
            ep_count,
            step_count,
        )
        t0 = time.time()
        out = self._train_step_a2c(batch, modules, optim, cfg_eff, ctx)
        logger.info(
            "train_step_end iter=%s algo=a2c elapsed_s=%.1f",
            iter_idx,
            time.time() - t0,
        )
        return out

    def _train_step_a2c(
        self,
        batch: Dict[str, Any],
        modules: Dict[str, Any],
        optim: Any,
        cfg_eff: Dict[str, Any],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        required = {"actor", "value"}
        missing = [k for k in required if k not in modules]
        if missing:
            raise RuntimeError(f"trainer missing modules keys: missing={missing} available={sorted(modules.keys())}")
        gamma = float(cfg_eff["gamma"])
        entropy_coef = float(cfg_eff["entropy_coef"])
        value_coef = float(cfg_eff["value_coef"])
        controller_coef = float(cfg_eff.get("controller_coef", 1.0))
        actor_coef = float(cfg_eff.get("actor_coef", 1.0))
        aux_mode_ce_coef = float(cfg_eff.get("aux_mode_ce_coef", 0.2))
        controller_cfg = cfg_eff.get("controller", {}) if isinstance(cfg_eff.get("controller"), dict) else {}
        controller_temp = float(controller_cfg.get("temperature", 1.0))
        if controller_temp <= 0:
            controller_temp = 1.0

        device = next(modules["actor"].parameters()).device

        actor_policy_losses: List[torch.Tensor] = []
        controller_policy_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        actor_entropies: List[torch.Tensor] = []
        controller_entropies: List[torch.Tensor] = []
        aux_losses: List[torch.Tensor] = []
        episode_returns: List[float] = []
        total_steps = 0
        win_eps = 0
        done_eps = 0
        win_present = False

        iter_idx = int((ctx or {}).get("iter_idx", -1))
        total_eps = len(batch.get("episodes", []))
        t_start = time.time()
        last_emit = t_start

        for ep_idx, ep in enumerate(batch.get("episodes", [])):
            steps = ep.get("steps", [])
            if not steps:
                continue
            rewards = [float(s.get("reward", 0.0)) for s in steps]
            if rewards:
                episode_returns.append(float(sum(rewards)))
            dones = [bool(s.get("done", False)) for s in steps]
            rets = _returns(rewards, dones, gamma)

            for i, step in enumerate(steps):
                h_raw = _as_float_tensor(step.get("h_t"), device=device)
                if h_raw.dim() == 1:
                    h_t = h_raw.unsqueeze(0)
                elif h_raw.dim() == 2:
                    h_t = h_raw
                else:
                    h_t = h_raw.reshape(1, -1)

                mode_id = int(step.get("mode_id", 0))
                action_ids = [str(a) for a in (step.get("action_ids") or ["ACTION1"])]
                action_index = int(step.get("action_index", 0))
                if action_index < 0 or action_index >= len(action_ids):
                    raise RuntimeError(
                        f"A2C invalid action_index ep={ep_idx} step={i} action_index={action_index} action_ids={action_ids}"
                    )
                coord_candidates = step.get("coord_candidates") or []
                chosen_coord_index = step.get("chosen_coord_index")

                if modules["controller"] is not None:
                    ctrl_out = modules["controller"].forward(h_t, ctx={"is_train": True})
                    ctrl_logits = ctrl_out["mode_logits"] / float(controller_temp)
                else:
                    _num_modes = max(1, int(cfg_eff.get("num_modes", 4)))
                    ctrl_logits = torch.zeros((1, _num_modes), dtype=torch.float32, device=device)
                actor_out = modules["actor"].forward(h_t, mode_id, action_ids, coord_candidates, cfg=cfg_eff)
                value = modules["value"].forward(h_t).view(1)
                ret = torch.tensor([rets[i]], dtype=torch.float32, device=device)
                adv = ret - value

                mode_logp, mode_entropy = _mode_logp_entropy_from_logits(
                    ctrl_logits,
                    torch.tensor([mode_id], dtype=torch.long, device=device),
                )
                mask_obj = step.get("available_actions_mask", None)
                try:
                    final_row, raw_typ, raw_len, raw_preview = _normalize_available_actions_mask(
                        mask_obj,
                        len(action_ids),
                        action_ids,
                        step_idx=i,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"A2C requires available_actions_mask ep={ep_idx} step={i}: {exc}"
                    ) from exc
                allow_cfg_all = cfg_eff.get("mode_action_allow", {})
                bias_cfg_all = cfg_eff.get("mode_action_bias", {})
                allow_mask = torch.ones((1, len(action_ids)), dtype=torch.bool, device=device)
                mode_bias = torch.zeros((1, len(action_ids)), dtype=torch.float32, device=device)
                allow_cfg = allow_cfg_all.get(str(mode_id), None) if isinstance(allow_cfg_all, dict) else None
                if isinstance(allow_cfg, list) and allow_cfg:
                    allow_set = set(str(a) for a in allow_cfg)
                    allow_row = torch.tensor([a in allow_set for a in action_ids], dtype=torch.bool, device=device).unsqueeze(0)
                    allow_mask[:] = allow_row
                bias_cfg = bias_cfg_all.get(str(mode_id), {}) if isinstance(bias_cfg_all, dict) else {}
                if isinstance(bias_cfg, dict) and bias_cfg:
                    bias_row = torch.tensor([float(bias_cfg.get(a, 0.0)) for a in action_ids], dtype=torch.float32, device=device).unsqueeze(0)
                    mode_bias[:] = bias_row
                action_temperature = float(cfg_eff.get("temperature", 1.0))
                avail_mask = torch.tensor([final_row], dtype=torch.bool, device=device)
                final_action_mask = avail_mask & allow_mask
                pi_discrete = apply_action_mask_and_bias(
                    actor_out["pi_discrete"],
                    final_action_mask,
                    mode_bias,
                    action_temperature,
                )
                act_logp, action_entropy = modules["actor"].action_logp_entropy(
                    pi_discrete,
                    torch.tensor([action_index], dtype=torch.long, device=device),
                    action_mask=None,
                )

                action_id = str(action_ids[action_index])
                if action_id.upper() == "ACTION6" and chosen_coord_index is not None and actor_out.get("pi_coord") is not None:
                    k = min(int(cfg_eff.get("coord_topK", 16)), len(coord_candidates))
                    if k > 0:
                        if int(chosen_coord_index) < 0 or int(chosen_coord_index) >= k:
                            raise RuntimeError(
                                f"A2C invalid chosen_coord_index ep={ep_idx} step={i} chosen_coord_index={chosen_coord_index} k={k}"
                            )
                        coord_mask = torch.ones((1, k), dtype=torch.bool, device=device)
                        bias_coord_all = cfg_eff.get("mode_coord_bias", {})
                        mode_coord_bias = float(bias_coord_all.get(str(mode_id), 0.0)) if isinstance(bias_coord_all, dict) else 0.0
                        mode_coord_bias_t = torch.tensor([[mode_coord_bias]], dtype=torch.float32, device=device)
                        pi_coord_final = apply_coord_mask_and_bias(
                            actor_out["pi_coord"],
                            coord_mask,
                            mode_coord_bias_t,
                            action_temperature,
                        )
                        cidx = int(chosen_coord_index)
                        c_logp, c_entropy = modules["actor"].coord_logp_entropy(
                            pi_coord_final,
                            torch.tensor([cidx], dtype=torch.long, device=device),
                            coord_mask=None,
                        )
                        act_logp = act_logp + c_logp
                        action_entropy = action_entropy + c_entropy

                actor_policy_losses.append(-(act_logp.squeeze(0) * adv.detach().squeeze(0)))
                controller_policy_losses.append(-(mode_logp.squeeze(0) * adv.detach().squeeze(0)))
                value_losses.append(F.mse_loss(value, ret))
                actor_entropies.append(action_entropy.squeeze(0))
                controller_entropies.append(mode_entropy.squeeze(0))

                aux = step.get("reward_aux", {}) if isinstance(step.get("reward_aux"), dict) else {}
                if "mode_target" in aux:
                    target = torch.tensor([int(aux["mode_target"])], dtype=torch.long, device=device)
                    w = float(aux.get("mode_weight", 1.0))
                    aux_losses.append(F.cross_entropy(ctrl_out["mode_logits"], target) * w)

                total_steps += 1
                now = time.time()
                if now - last_emit >= 5.0:
                    logger.debug(
                        "train_step_a2c_progress iter=%s ep=%s/%s steps=%s elapsed_s=%.1f",
                        iter_idx,
                        ep_idx + 1,
                        total_eps,
                        total_steps,
                        now - t_start,
                    )
                    last_emit = now

            if any(bool(s.get("done", False)) for s in steps):
                done_eps += 1
            ep_win_present = any(isinstance(s, dict) and ("win" in s) for s in steps)
            win_present = win_present or ep_win_present
            if ep_win_present and any(bool(s.get("win", False)) for s in steps):
                win_eps += 1

        if not actor_policy_losses:
            return {
                "losses": {
                    "total": 0.0,
                    "actor_policy": 0.0,
                    "controller_policy": 0.0,
                    "value": 0.0,
                    "entropy_actor": 0.0,
                    "entropy_controller": 0.0,
                    "aux_mode_ce": 0.0,
                    "grad_norm_total": 0.0,
                    "approx_kl": 0.0,
                    "clip_frac": 0.0,
                },
                "mean_return": 0.0,
                "mean_episode_len": 0.0,
                "win_rate": 0.0,
                "done_rate": 0.0,
            }

        actor_policy_loss = torch.stack(actor_policy_losses).mean()
        controller_policy_loss = torch.stack(controller_policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_actor = torch.stack(actor_entropies).mean()
        entropy_controller = torch.stack(controller_entropies).mean()
        aux_mode_ce = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=device)

        total_loss = (
            actor_coef * actor_policy_loss
            + controller_coef * controller_policy_loss
            + value_coef * value_loss
            + aux_mode_ce_coef * aux_mode_ce
            - entropy_coef * (entropy_actor + entropy_controller)
        )

        logger.debug("train_step_a2c_backward_start iter=%s", iter_idx)
        t_backward = time.time()
        _zero_grad(optim)
        total_loss.backward()
        logger.debug("train_step_a2c_backward_done iter=%s elapsed_s=%.3f", iter_idx, time.time() - t_backward)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (list(modules["controller"].parameters()) if modules["controller"] is not None else [])
            + list(modules["actor"].parameters())
            + list(modules["value"].parameters()),
            float(cfg_eff["max_grad_norm"]),
        )
        t_step = time.time()
        _step_optim(optim)
        logger.debug("train_step_a2c_optim_done iter=%s elapsed_s=%.3f", iter_idx, time.time() - t_step)

        mean_return = float(sum(episode_returns) / max(1, len(episode_returns))) if episode_returns else 0.0
        avg_reward = mean_return
        mean_episode_len = float(total_steps / max(1, len(batch.get("episodes", []))))
        if not win_present:
            win_eps = 0
        win_rate = float(win_eps / max(1, len(batch.get("episodes", []))))
        done_rate = float(done_eps / max(1, len(batch.get("episodes", []))))

        return {
            "losses": {
                "total": float(total_loss.item()),
                "actor_policy": float(actor_policy_loss.item()),
                "controller_policy": float(controller_policy_loss.item()),
                "value": float(value_loss.item()),
                "entropy_actor": float(entropy_actor.item()),
                "entropy_controller": float(entropy_controller.item()),
                "aux_mode_ce": float(aux_mode_ce.item()),
                "grad_norm_total": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                "approx_kl": 0.0,
                "clip_frac": 0.0,
            },
            "mean_return": mean_return,
            "avg_reward": avg_reward,
            "mean_episode_len": mean_episode_len,
            "win_rate": win_rate,
            "done_rate": done_rate,
        }

    def _train_step_ppo(
        self,
        batch: Dict[str, Any],
        modules: Dict[str, Any],
        optim: Any,
        cfg_eff: Dict[str, Any],
        ctx: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        required = {"actor", "value"}
        missing = [k for k in required if k not in modules]
        if missing:
            raise RuntimeError(f"trainer missing modules keys: missing={missing} available={sorted(modules.keys())}")
        ppo_cfg = normalize_ppo_cfg(dict(cfg_eff.get("ppo", {})))
        kl_metric = str(ppo_cfg.get("kl_metric", "action")).lower()
        if kl_metric not in {"mode", "action", "coord"}:
            raise RuntimeError(f"Invalid ppo.kl_metric={kl_metric} (expected 'mode', 'action', or 'coord')")
        gamma = float(ppo_cfg.get("gamma", cfg_eff.get("gamma", 0.99)))
        value_coef = float(ppo_cfg.get("value_coef", cfg_eff.get("value_coef", 0.5)))
        controller_coef = float(cfg_eff.get("controller_coef", 1.0))
        actor_coef = float(cfg_eff.get("actor_coef", 1.0))
        aux_mode_ce_coef = float(ppo_cfg.get("aux_ce_coef", cfg_eff.get("aux_mode_ce_coef", 0.2)))
        controller_cfg = cfg_eff.get("controller", {}) if isinstance(cfg_eff.get("controller"), dict) else {}
        controller_temp = float(controller_cfg.get("temperature", 1.0))
        if controller_temp <= 0:
            controller_temp = 1.0

        clip_eps_actor = float(ppo_cfg.get("clip_eps_actor", 0.2))
        clip_eps_controller = float(ppo_cfg.get("clip_eps_controller", 0.2))
        clip_eps_coord = float(ppo_cfg.get("clip_eps_coord", clip_eps_actor))
        ppo_epochs = int(ppo_cfg.get("epochs", 4))
        ppo_minibatches = max(1, int(ppo_cfg.get("minibatches", 4)))
        stage = str(self.phase_state.get("stage", "exploration"))
        pipeline_mode = str((cfg_eff.get("pipeline", {}) or {}).get("mode", "")).lower()
        rl_cfg = cfg_eff.get("rl", {}) if isinstance(cfg_eff.get("rl"), dict) else {}
        debug_logp_mismatch_dump = bool(rl_cfg.get("debug_logp_mismatch_dump", pipeline_mode == "rl_only"))
        in_early_phase = stage != "win"
        target_kl = float(ppo_cfg.get("target_kl_early" if in_early_phase else "target_kl_late", ppo_cfg.get("target_kl", 0.02)))
        preupdate_kl_max = float(ppo_cfg.get("preupdate_kl_max", ppo_cfg.get("target_kl_early", target_kl)))
        early_stop_kl = bool(ppo_cfg.get("early_stop_kl", True))
        use_gae = bool(ppo_cfg.get("use_gae", True))
        gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
        adv_norm = bool(ppo_cfg.get("adv_norm", True))
        vf_clip = bool(ppo_cfg.get("vf_clip", True))
        vf_clip_eps = float(ppo_cfg.get("vf_clip_eps", 0.2))
        entropy_coef = float(
            ppo_cfg.get(
                "entropy_coef_actor_phase1" if in_early_phase else "entropy_coef_actor_phase2",
                cfg_eff.get("entropy_coef", 0.01),
            )
        )
        mode_entropy_coef = float(ppo_cfg.get("entropy_coef_controller", ppo_cfg.get("mode_entropy_coef", entropy_coef)))
        coord_coef = float(ppo_cfg.get("coord_coef", 1.0))
        recompute_post_update_metrics = bool(ppo_cfg.get("recompute_post_update_metrics", False))
        preupdate_eval_mode = bool(ppo_cfg.get("preupdate_eval_mode", False))
        max_grad_norm = float(ppo_cfg.get("max_grad_norm", cfg_eff.get("max_grad_norm", 1.0)))
        self.lr_adapt_enabled = bool(ppo_cfg.get("lr_adapt_enabled", True))
        self.lr_adapt_target_kl = float(ppo_cfg.get("lr_adapt_target_kl", 0.02))
        self.lr_adapt_min_lr_mult = float(ppo_cfg.get("lr_adapt_min_lr_mult", 0.1))
        self.lr_adapt_max_lr_mult = float(ppo_cfg.get("lr_adapt_max_lr_mult", 3.0))
        self.lr_adapt_downscale = float(ppo_cfg.get("lr_adapt_downscale", 0.5))
        self.lr_adapt_upscale = float(ppo_cfg.get("lr_adapt_upscale", 1.05))
        self.lr_adapt_upscale_patience_iters = int(ppo_cfg.get("lr_adapt_upscale_patience_iters", 5))
        self.lr_adapt_use_abs_kl = bool(ppo_cfg.get("lr_adapt_use_abs_kl", True))
        current_optimizer_lr = _get_optimizer_lr(optim)
        if self._lr_base <= 0.0:
            self._lr_base = float(current_optimizer_lr)
        if self._lr_current <= 0.0:
            self._lr_current = float(current_optimizer_lr)
        if self._lr_low_kl_counter < 0:
            self._lr_low_kl_counter = 0
        log_cfg = cfg_eff.get("log", {}) if isinstance(cfg_eff.get("log"), dict) else {}
        heartbeat_enabled = bool(log_cfg.get("heartbeat_enabled", True))
        heartbeat_every_minibatches = max(1, int(log_cfg.get("heartbeat_every_minibatches", 10)))
        debug_perf_checks = bool((ctx or {}).get("debug_perf_checks", False))
        num_modes = int(cfg_eff.get("num_modes", (getattr(modules.get("controller"), "cfg", {}) or {}).get("num_modes", 4)))

        device = next(modules["actor"].parameters()).device
        pre_train_param_snapshot = _snapshot_param_tensors(modules)
        rollout_cfg_hash = (ctx or {}).get("rollout_cfg_hash")
        trainer_cfg_hash, trainer_payload = _rollout_cfg_hash(cfg_eff)
        if rollout_cfg_hash is not None and str(rollout_cfg_hash) != str(trainer_cfg_hash):
            logger.error("rollout cfg mismatch trainer_payload=%s", json.dumps(trainer_payload, sort_keys=True, separators=(",", ":")))
            rollout_payload = (ctx or {}).get("rollout_cfg_payload")
            if isinstance(rollout_payload, dict):
                logger.error(
                    "rollout cfg mismatch rollout_payload=%s",
                    json.dumps(rollout_payload, sort_keys=True, separators=(",", ":")),
                )
            else:
                logger.error("rollout cfg mismatch rollout_payload=missing")
            logger.error(
                "rollout cfg mismatch rollout_cfg_hash=%s trainer_cfg_hash=%s trainer_iter=%s; skipping update",
                rollout_cfg_hash,
                trainer_cfg_hash,
                int((ctx or {}).get("iter_idx", -1)),
            )
            return _empty_ppo_report(batch, stage=stage)

        flat: List[Dict[str, Any]] = []
        episode_returns: List[float] = []
        total_steps = 0
        win_eps = 0
        done_eps = 0
        win_present = False
        recompute_values_cfg = bool(ppo_cfg.get("recompute_values", True))
        value_recompute_chunk = max(1, int(ppo_cfg.get("value_recompute_chunk", 8192)))
        max_steps_per_train_step = int(ppo_cfg.get("max_steps_per_train_step", 200_000))
        iter_idx = int((ctx or {}).get("iter_idx", -1))
        total_eps = len(batch.get("episodes", []))
        t_pre = time.time()
        t_recompute_values = 0.0
        t_flatten_records = 0.0
        last_emit = t_pre
        logger.info(
            "train_step_preprocess_start iter=%s episodes=%s recompute_values=%s",
            iter_idx,
            total_eps,
            recompute_values_cfg,
        )
        episodes = list(batch.get("episodes", []))
        ep_slices: List[tuple[int, int]] = []
        values_all: List[Optional[float]] = []
        missing_indices: List[int] = []
        total_steps_counted: Optional[int] = None

        if recompute_values_cfg:
            h_rows_all: List[torch.Tensor] = []
            step_offset = 0
            for ep in episodes:
                steps = ep.get("steps", [])
                start = step_offset
                if steps:
                    for s in steps:
                        h_raw = _as_float_tensor(s.get("h_t"), device=device)
                        if h_raw.dim() == 1:
                            h_t = h_raw.unsqueeze(0)
                        elif h_raw.dim() == 2:
                            h_t = h_raw
                        else:
                            h_t = h_raw.reshape(1, -1)
                        h_rows_all.append(h_t)
                        old_val = s.get("old_value", None)
                        if old_val is None:
                            missing_indices.append(step_offset)
                            values_all.append(None)
                        else:
                            values_all.append(float(old_val))
                        step_offset += 1
                ep_slices.append((start, step_offset))
            total_steps_counted = step_offset

            if max_steps_per_train_step > 0 and total_steps_counted > max_steps_per_train_step:
                logger.error(
                    "train_step_skipped_too_many_steps iter=%s episodes=%s steps=%s workers=%s games=%s max_steps_per_train_step=%s",
                    iter_idx,
                    total_eps,
                    total_steps_counted,
                    (ctx or {}).get("workers", "unknown"),
                    (ctx or {}).get("games", "unknown"),
                    max_steps_per_train_step,
                )
                return _empty_ppo_report(batch, stage=stage)

            if missing_indices:
                value_prev = modules["value"].training
                modules["value"].eval()
                h_all = torch.cat(h_rows_all, dim=0).contiguous() if h_rows_all else torch.zeros((0, 1), device=device)
                with torch.no_grad():
                    t_rv0 = time.time()
                    for start in range(0, len(missing_indices), value_recompute_chunk):
                        chunk_idx = missing_indices[start : start + value_recompute_chunk]
                        idx_t = torch.tensor(chunk_idx, dtype=torch.long, device=device)
                        h_chunk = h_all.index_select(0, idx_t).contiguous()
                        vals_chunk = modules["value"].forward(h_chunk).view(-1).detach().cpu().tolist()
                        for j, v in enumerate(vals_chunk):
                            values_all[chunk_idx[j]] = float(v)
                    t_recompute_values += time.time() - t_rv0
                modules["value"].train(value_prev)

        if total_steps_counted is None:
            total_steps_counted = sum(len(ep.get("steps", []) or []) for ep in episodes)
            if max_steps_per_train_step > 0 and total_steps_counted > max_steps_per_train_step:
                logger.error(
                    "train_step_skipped_too_many_steps iter=%s episodes=%s steps=%s workers=%s games=%s max_steps_per_train_step=%s",
                    iter_idx,
                    total_eps,
                    total_steps_counted,
                    (ctx or {}).get("workers", "unknown"),
                    (ctx or {}).get("games", "unknown"),
                    max_steps_per_train_step,
                )
                return _empty_ppo_report(batch, stage=stage)

        for ep_idx, ep in enumerate(episodes):
            steps = ep.get("steps", [])
            if not steps:
                continue
            rewards = [float(s.get("reward", 0.0)) for s in steps]
            dones = [bool(s.get("done", False)) for s in steps]
            if recompute_values_cfg:
                start, end = ep_slices[ep_idx]
                values = [float(v) if v is not None else 0.0 for v in values_all[start:end]]
            else:
                values = [float(s.get("old_value", 0.0)) for s in steps]
            if use_gae:
                rets, advs = _compute_gae(rewards, dones, values, gamma=gamma, gae_lambda=gae_lambda)
            else:
                rets = _returns(rewards, dones, gamma)
                advs = [rets[i] - values[i] for i in range(len(rets))]
            episode_returns.append(float(sum(rewards)))

            t_flat0 = time.time()
            for i, s in enumerate(steps):
                if "old_logp_mode" not in s:
                    raise RuntimeError("PPO batch missing old_logp_mode")
                action_ids = list(s.get("action_ids") or ["ACTION1"])
                action_index = int(s.get("action_index", 0))
                if action_index < 0 or action_index >= len(action_ids):
                    raise RuntimeError(
                        f"PPO invalid action_index ep={ep_idx} step={i} action_index={action_index} action_ids={action_ids}"
                    )
                action_id = str(action_ids[action_index])
                coord_candidates = list(s.get("coord_candidates") or [])
                has_coord_flag = int(s.get("mask_has_coord", 0)) == 1
                coord_use = bool(
                    str(action_id).upper() == "ACTION6"
                    and s.get("chosen_coord_index") is not None
                    and len(coord_candidates) > 0
                    and has_coord_flag
                )
                if coord_use:
                    k = min(int(cfg_eff.get("coord_topK", 16)), len(coord_candidates))
                    chosen_coord_index = int(s.get("chosen_coord_index"))
                    if chosen_coord_index < 0 or chosen_coord_index >= k:
                        raise RuntimeError(
                            f"PPO invalid chosen_coord_index ep={ep_idx} step={i} chosen_coord_index={chosen_coord_index} k={k}"
                        )
                rec = {
                    "h_t": s.get("h_t"),
                    "mode_id": int(s.get("mode_id", 0)),
                    "action_ids": action_ids,
                    "action_index": action_index,
                    "ep_idx": int(ep_idx),
                    "step_idx": int(i),
                    "grid_embed": s.get("grid_embed"),
                    "available_actions_mask": s.get("available_actions_mask", None),
                    "coord_candidates": coord_candidates,
                    "chosen_coord_index": -1 if s.get("chosen_coord_index") is None else int(s.get("chosen_coord_index")),
                    "mask_has_coord": int(s.get("mask_has_coord", 0)),
                    "mask_valid_step": int(s.get("mask_valid_step", 1)),
                    "old_logp_mode": float(s.get("old_logp_mode", 0.0)),
                    "old_logp_action": float(s.get("old_logp_action_discrete", s.get("old_logp_action", 0.0))),
                    "old_logp_coord": float(s.get("old_logp_coord", 0.0)),
                    "old_logp_total": float(
                        float(s.get("old_logp_mode", 0.0))
                        + float(s.get("old_logp_action_discrete", s.get("old_logp_action", 0.0)))
                        + (float(s.get("old_logp_coord", 0.0)) if coord_use else 0.0)
                    ),
                    "old_value": float(s.get("old_value", 0.0)) if s.get("old_value") is not None else 0.0,
                    "old_value_present": s.get("old_value") is not None,
                    "ret": float(rets[i]),
                    "adv": float(advs[i]),
                    "aux_mode_target": (s.get("reward_aux", {}) or {}).get("mode_target"),
                    "aux_mode_weight": float((s.get("reward_aux", {}) or {}).get("mode_weight", 1.0)),
                    "done": bool(s.get("done", False)),
                    "win": bool(s.get("win", False)),
                    "coord_use": bool(coord_use),
                    "intrinsic_terms": s.get("intrinsic_terms", {}),
                    "flash_event": bool((s.get("reward_terms", {}) or {}).get("flash_event", False)),
                }
                flat.append(rec)
                total_steps += 1
            t_flatten_records += time.time() - t_flat0
            if any(bool(s.get("done", False)) for s in steps):
                done_eps += 1
            ep_win_present = any(isinstance(s, dict) and ("win" in s) for s in steps)
            win_present = win_present or ep_win_present
            if ep_win_present and any(bool(s.get("win", False)) for s in steps):
                win_eps += 1
            now = time.time()
            if now - last_emit >= 5.0:
                logger.debug(
                    "train_step_preprocess_progress iter=%s ep=%s/%s steps=%s elapsed_s=%.1f t_recompute=%.1f t_flatten=%.1f",
                    iter_idx,
                    ep_idx + 1,
                    total_eps,
                    total_steps,
                    now - t_pre,
                    t_recompute_values,
                    t_flatten_records,
                )
                last_emit = now

        logger.info(
            "train_step_preprocess_done iter=%s episodes=%s steps=%s elapsed_s=%.1f t_recompute=%.1f t_flatten=%.1f",
            iter_idx,
            total_eps,
            total_steps,
            time.time() - t_pre,
            t_recompute_values,
            t_flatten_records,
        )

        if not flat:
            return {
                "losses": {
                    "total": 0.0,
                    "actor_policy": 0.0,
                    "controller_policy": 0.0,
                    "value": 0.0,
                    "entropy_actor": 0.0,
                    "entropy_controller": 0.0,
                    "aux_mode_ce": 0.0,
                    "grad_norm_total": 0.0,
                    "approx_kl": 0.0,
                    "clip_frac": 0.0,
                    "clipfrac_mode": 0.0,
                    "clipfrac_action": 0.0,
                    "clipfrac_coord": 0.0,
                    "ppo_epochs_ran": 0,
                    "adv_mean": 0.0,
                    "adv_std": 0.0,
                },
                "mean_return": 0.0,
                "mean_episode_len": 0.0,
                "win_rate": 0.0,
                "done_rate": 0.0,
            }

        t_flatten = 0.0
        t_mask = 0.0
        t_pre_eval = 0.0
        t_mb_eval = 0.0
        t_mb_step = 0.0

        # Stage scalar and hidden-state tensors on device once per train step.
        # Each record stores h_t as either [H] or [1,H]; normalize to [H].
        t_flatten_start = time.time()
        h_rows: List[torch.Tensor] = []
        for r in flat:
            h_row = torch.as_tensor(r["h_t"], dtype=torch.float32).to(device=device)
            if h_row.dim() == 2 and h_row.shape[0] == 1:
                h_row = h_row[0]
            elif h_row.dim() > 1:
                h_row = h_row.reshape(-1)
            h_rows.append(h_row)
        h_t_all = torch.stack(h_rows, dim=0).contiguous()
        mode_id_all = torch.tensor([int(r["mode_id"]) for r in flat], dtype=torch.long, device=device)
        if mode_id_all.numel() > 0:
            min_mode = int(mode_id_all.min().detach().cpu().item())
            max_mode = int(mode_id_all.max().detach().cpu().item())
            if min_mode < 0 or max_mode >= num_modes:
                bad_mask = (mode_id_all < 0) | (mode_id_all >= num_modes)
                bad_idx = bad_mask.nonzero(as_tuple=False).view(-1)[:8].detach().cpu().tolist()
                raise RuntimeError(
                    f"PPO invalid mode_id_all min={min_mode} max={max_mode} num_modes={num_modes} bad_indices={bad_idx}"
                )
        chosen_coord_index_all = torch.tensor([int(r["chosen_coord_index"]) for r in flat], dtype=torch.long, device=device)
        mask_valid_step_all = torch.tensor([int(r["mask_valid_step"]) == 1 for r in flat], dtype=torch.bool, device=device).contiguous()
        old_logp_mode_all = torch.tensor([float(r["old_logp_mode"]) for r in flat], dtype=torch.float32, device=device).contiguous()
        old_logp_action_all = torch.tensor([float(r["old_logp_action"]) for r in flat], dtype=torch.float32, device=device).contiguous()
        old_logp_coord_all = torch.tensor([float(r["old_logp_coord"]) for r in flat], dtype=torch.float32, device=device).contiguous()
        old_value_all = torch.tensor([float(r["old_value"]) for r in flat], dtype=torch.float32, device=device).contiguous()
        old_value_present_all = torch.tensor([bool(r.get("old_value_present", False)) for r in flat], dtype=torch.bool, device=device)
        ret_all = torch.tensor([float(r["ret"]) for r in flat], dtype=torch.float32, device=device).contiguous()
        adv_t = torch.tensor([r["adv"] for r in flat], dtype=torch.float32, device=device).contiguous()
        coord_use_all = torch.tensor([bool(r.get("coord_use", False)) for r in flat], dtype=torch.bool, device=device).contiguous()
        action_ids_all = [list(r["action_ids"]) for r in flat]
        action_ids_sig_all = [
            hashlib.sha1("|".join(map(str, ids)).encode("utf-8")).hexdigest() for ids in action_ids_all
        ]
        action_index_cpu = [int(r["action_index"]) for r in flat]
        coord_candidates_all = [list(r["coord_candidates"]) for r in flat]
        any_coord_use_flat = any(bool(r.get("coord_use", False)) for r in flat)
        coord_feat_dim_flat = _coord_feat_dim(cfg_eff, modules["actor"]) if any_coord_use_flat else None
        coord_sig_all: List[Optional[str]] = []
        for r in flat:
            if not bool(r.get("coord_use", False)):
                coord_sig_all.append(None)
                continue
            coords = list(r.get("coord_candidates") or [])
            k = min(int(cfg_eff.get("coord_topK", 16)), len(coords))
            if k <= 0:
                coord_sig_all.append(None)
                continue
            if coord_feat_dim_flat is None:
                raise RuntimeError("coord_feat_dim must be provided by actor or cfg")
            sig_ints: List[int] = [int(k), int(coord_feat_dim_flat)]
            for cj, cand in enumerate(coords[:k]):
                fv = cand.get("feat_vec") if isinstance(cand, dict) else None
                if not isinstance(fv, list) or len(fv) != coord_feat_dim_flat:
                    raise RuntimeError(
                        f"Inconsistent coord feature dim at ep={r.get('ep_idx', '?')} step={r.get('step_idx', '?')} cand={cj}: got {len(fv) if isinstance(fv, list) else None} expected {coord_feat_dim_flat}"
                    )
                for x in fv:
                    sig_ints.append(int(round(float(x) * 10000.0)))
            raw = array("i", sig_ints).tobytes()
            coord_sig_all.append(hashlib.sha1(raw).hexdigest())
        aux_mode_target_all = [r.get("aux_mode_target") for r in flat]
        aux_mode_weight_all = torch.tensor([float(r.get("aux_mode_weight", 1.0)) for r in flat], dtype=torch.float32, device=device)
        aux_present_all = torch.tensor([t is not None for t in aux_mode_target_all], dtype=torch.bool, device=device)
        aux_target_t_all = torch.tensor([int(t) if t is not None else 0 for t in aux_mode_target_all], dtype=torch.long, device=device)
        intrinsic_cfg = cfg_eff.get("intrinsic", {}) if isinstance(cfg_eff.get("intrinsic", {}), dict) else {}
        intrinsic_enabled = bool(intrinsic_cfg.get("enabled", False)) and str(intrinsic_cfg.get("method", "")) == "rnd_grid_embed"
        intrinsic_enabled = intrinsic_enabled and ("intrinsic_rnd" in modules)
        grid_embed_all: Optional[torch.Tensor] = None
        intrinsic_valid_mask: Optional[torch.Tensor] = None
        rnd_phi_mean = 0.0
        rnd_err_raw_mean = 0.0
        if intrinsic_enabled:
            grid_embed_dim = int(getattr(modules["intrinsic_rnd"].predictor[0], "in_features", 0))
            grid_rows: List[torch.Tensor] = []
            present_mask: List[bool] = []
            flash_mask: List[bool] = []
            phi_vals: List[float] = []
            err_vals: List[float] = []
            for r in flat:
                ge = r.get("grid_embed")
                if isinstance(ge, list) and ge and (grid_embed_dim == 0 or len(ge) == grid_embed_dim):
                    grid_rows.append(torch.tensor(ge, dtype=torch.float32))
                    present_mask.append(True)
                else:
                    grid_rows.append(torch.zeros((max(1, grid_embed_dim),), dtype=torch.float32))
                    present_mask.append(False)
                terms = r.get("intrinsic_terms", {}) if isinstance(r.get("intrinsic_terms", {}), dict) else {}
                phi_vals.append(float(terms.get("rnd_phi", 0.0)) if present_mask[-1] else 0.0)
                err_vals.append(float(terms.get("rnd_err_raw", 0.0)) if present_mask[-1] else 0.0)
                flash_mask.append(bool(r.get("flash_event", False)))
            if grid_rows:
                grid_embed_all_cpu = torch.stack(grid_rows, dim=0).contiguous()
                grid_embed_all = grid_embed_all_cpu.to(device=device, non_blocking=True)
                present_mask_t = torch.tensor(present_mask, dtype=torch.bool, device=device)
                flash_mask_t = torch.tensor(flash_mask, dtype=torch.bool, device=device)
                intrinsic_valid_mask = mask_valid_step_all & present_mask_t & (~flash_mask_t)
                denom = max(1, sum(1 for v in present_mask if v))
                rnd_phi_mean = float(sum(phi_vals) / float(denom))
                rnd_err_raw_mean = float(sum(err_vals) / float(denom))
        action_env_mask_local_all: List[torch.Tensor] = []
        action_mask_raw_type_all: List[str] = []
        action_mask_raw_len_all: List[int] = []
        action_mask_raw_preview_all: List[List[str]] = []
        legacy_mask_rows = 0
        legacy_mask_type_counts = {
            "none": 0,
            "empty": 0,
            "bool_nd": 0,
            "int01_nd": 0,
            "index_list": 0,
            "str_list": 0,
            "other": 0,
        }
        t_flatten += time.time() - t_flatten_start
        t_mask_start = time.time()
        for idx_r, r in enumerate(flat):
            mask_obj = r.get("available_actions_mask", None)
            sample_action_ids = list(r.get("action_ids") or [])
            final_row, raw_typ, raw_len, raw_preview = _normalize_available_actions_mask(
                mask_obj,
                len(sample_action_ids),
                sample_action_ids,
                step_idx=idx_r,
            )
            action_env_mask_local_all.append(torch.tensor(final_row, dtype=torch.bool).contiguous())
            action_mask_raw_type_all.append(str(raw_typ))
            action_mask_raw_len_all.append(int(raw_len))
            action_mask_raw_preview_all.append(list(raw_preview))
            if raw_typ in legacy_mask_type_counts:
                legacy_mask_type_counts[raw_typ] += 1
            else:
                legacy_mask_type_counts["other"] += 1
            if raw_typ != "bool_nd":
                legacy_mask_rows += 1
        t_mask += time.time() - t_mask_start
        if legacy_mask_rows > 0:
            logger.error(
                "noncanonical_available_actions_mask rows=%s total=%s types none=%s empty=%s bool_nd=%s int01_nd=%s index_list=%s str_list=%s other=%s",
                legacy_mask_rows,
                len(flat),
                legacy_mask_type_counts["none"],
                legacy_mask_type_counts["empty"],
                legacy_mask_type_counts["bool_nd"],
                legacy_mask_type_counts["int01_nd"],
                legacy_mask_type_counts["index_list"],
                legacy_mask_type_counts["str_list"],
                legacy_mask_type_counts["other"],
            )

        valid_adv = adv_t[mask_valid_step_all]
        if valid_adv.numel() == 0:
            return _empty_ppo_report(batch, stage=stage)
        adv_mean_t = valid_adv.mean()
        adv_std_t = valid_adv.std(unbiased=False)
        adv_mean = float(adv_mean_t.detach().cpu().item())
        adv_std = float(adv_std_t.detach().cpu().item())
        if adv_norm and valid_adv.numel() > 1:
            adv_t = adv_t.clone()
            valid_idx_t = mask_valid_step_all.nonzero(as_tuple=False).view(-1)
            adv_t_valid = (valid_adv - adv_mean_t) / (adv_std_t + 1e-8)
            adv_t.index_copy_(0, valid_idx_t, adv_t_valid)
        adv_t = adv_t.contiguous()

        mask_valid_step_cpu = [int(r.get("mask_valid_step", 1)) == 1 for r in flat]

        if debug_perf_checks:
            logger.info(
                "trainer_perf_tensors iter=%s h_t=%s/%s adv=%s/%s ret=%s/%s old_logp_mode=%s/%s valid_mask=%s/%s",
                int((ctx or {}).get("iter_idx", -1)),
                tuple(h_t_all.shape),
                str(h_t_all.device),
                tuple(adv_t.shape),
                str(adv_t.device),
                tuple(ret_all.shape),
                str(ret_all.device),
                tuple(old_logp_mode_all.shape),
                str(old_logp_mode_all.device),
                tuple(mask_valid_step_all.shape),
                str(mask_valid_step_all.device),
            )
            if device.type == "cuda":
                assert h_t_all.is_cuda
                assert adv_t.is_cuda
                assert ret_all.is_cuda
                assert old_logp_mode_all.is_cuda
                assert old_logp_action_all.is_cuda
                assert old_logp_coord_all.is_cuda
                assert mask_valid_step_all.is_cuda

        rng_seed = int((ctx or {}).get("global_seed", 0)) * 1000003 + int((ctx or {}).get("iter_idx", 0))
        iter_idx = int((ctx or {}).get("iter_idx", -1))
        n = len(flat)
        mb_size = max(1, math.ceil(n / ppo_minibatches))
        total_mb_per_epoch = max(1, math.ceil(n / mb_size))
        train_t0 = time.time()
        mb_done = 0
        logger.info(
            "train_step_start iter=%s algo=ppo samples=%s epochs=%s minibatches_per_epoch=%s mb_size=%s",
            iter_idx,
            n,
            ppo_epochs,
            total_mb_per_epoch,
            mb_size,
        )
        rollout_policy_version = (ctx or {}).get("rollout_policy_version", None)

        def _set_eval_for_recompute() -> tuple[bool, bool, bool]:
            ctrl_prev = modules["controller"].training if modules["controller"] is not None else False
            actor_prev = modules["actor"].training
            value_prev = modules["value"].training
            if modules["controller"] is not None:
                modules["controller"].eval()
            modules["actor"].eval()
            modules["value"].eval()
            return ctrl_prev, actor_prev, value_prev

        def _restore_train_mode(prev: tuple[bool, bool, bool]) -> None:
            if modules["controller"] is not None:
                modules["controller"].train(prev[0])
            modules["actor"].train(prev[1])
            modules["value"].train(prev[2])

        allow_bias_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        allow_bias_cache_keys: List[tuple] = []

        allow_bias_cache_max = int(ppo_cfg.get("allow_bias_cache_max", 256))

        def _vectorized_logp_eval(valid_indices: torch.Tensor, diagnostics: bool = False) -> Dict[str, Any]:
            if isinstance(valid_indices, torch.Tensor):
                if valid_indices.numel() == 0:
                    zf = torch.zeros((0,), dtype=torch.float32, device=device)
                    zb = torch.zeros((0,), dtype=torch.bool, device=device)
                    return {
                        "old_mode": zf,
                        "old_action": zf,
                        "old_coord": zf,
                        "old_total": zf,
                        "new_mode": zf,
                        "new_action": zf,
                        "new_coord": zf,
                        "new_total": zf,
                        "action_mask_union": torch.zeros((0, 0), dtype=torch.bool, device=device),
                        "coord_mask": torch.zeros((0, 0), dtype=torch.bool, device=device),
                        "coord_use": zb,
                        "coord_use_count": 0,
                        "mode_index": torch.zeros((0,), dtype=torch.long, device=device),
                        "chosen_coord_index": torch.zeros((0,), dtype=torch.long, device=device),
                    }
                idx_t = valid_indices.to(device=device)
                idx_t_cpu = valid_indices.detach().cpu()
            else:
                if not valid_indices:
                    zf = torch.zeros((0,), dtype=torch.float32, device=device)
                    zb = torch.zeros((0,), dtype=torch.bool, device=device)
                    return {
                        "old_mode": zf,
                        "old_action": zf,
                        "old_coord": zf,
                        "old_total": zf,
                        "new_mode": zf,
                        "new_action": zf,
                        "new_coord": zf,
                        "new_total": zf,
                        "action_mask_union": torch.zeros((0, 0), dtype=torch.bool, device=device),
                        "coord_mask": torch.zeros((0, 0), dtype=torch.bool, device=device),
                        "coord_use": zb,
                        "coord_use_count": 0,
                        "mode_index": torch.zeros((0,), dtype=torch.long, device=device),
                        "chosen_coord_index": torch.zeros((0,), dtype=torch.long, device=device),
                    }
                idx_t_cpu = torch.tensor(list(valid_indices), dtype=torch.long)
                idx_t = idx_t_cpu.to(device=device)

            if idx_t.numel() == 0:
                zf = torch.zeros((0,), dtype=torch.float32, device=device)
                zb = torch.zeros((0,), dtype=torch.bool, device=device)
                return {
                    "old_mode": zf,
                    "old_action": zf,
                    "old_coord": zf,
                    "old_total": zf,
                    "new_mode": zf,
                    "new_action": zf,
                    "new_coord": zf,
                    "new_total": zf,
                    "action_mask_union": torch.zeros((0, 0), dtype=torch.bool, device=device),
                    "coord_mask": torch.zeros((0, 0), dtype=torch.bool, device=device),
                    "coord_use": zb,
                    "coord_use_count": 0,
                    "mode_index": torch.zeros((0,), dtype=torch.long, device=device),
                    "chosen_coord_index": torch.zeros((0,), dtype=torch.long, device=device),
                }

            m = int(idx_t.numel())
            h_t = h_t_all.index_select(0, idx_t).contiguous()
            mode_index = mode_id_all.index_select(0, idx_t)
            mode_index_cpu = mode_index.detach().to("cpu")
            chosen_coord_index = chosen_coord_index_all.index_select(0, idx_t)
            old_logp_mode = old_logp_mode_all.index_select(0, idx_t)
            old_logp_action = old_logp_action_all.index_select(0, idx_t)
            old_logp_coord = old_logp_coord_all.index_select(0, idx_t)
            coord_use_stored = coord_use_all.index_select(0, idx_t)
            coord_use_cpu = coord_use_stored.detach().to("cpu")

            if modules["controller"] is not None:
                ctrl_out = modules["controller"].forward(h_t, ctx={"is_train": False})
                ctrl_logits = ctrl_out["mode_logits"] / float(controller_temp)
            else:
                _num_modes = max(1, int(cfg_eff.get("num_modes", 4)))
                ctrl_logits = torch.zeros((m, _num_modes), dtype=torch.float32, device=device)
                ctrl_out = {"mode_logits": ctrl_logits}
            mode_logp_new, mode_entropy_new = _mode_logp_entropy_from_logits(ctrl_logits, mode_index)

            action_logp_new = torch.zeros((m,), dtype=torch.float32, device=device)
            action_entropy_new = torch.zeros((m,), dtype=torch.float32, device=device)
            coord_logp_new = torch.zeros((m,), dtype=torch.float32, device=device)
            coord_entropy_all = torch.zeros((m,), dtype=torch.float32, device=device)
            coord_use = coord_use_stored
            any_coord_use = bool(coord_use.any())
            action_temperature = float(cfg_eff.get("temperature", 1.0))

            uniform_action_ids = True
            if m > 0:
                first_sig = action_ids_sig_all[int(idx_t_cpu[0].item())]
                for row_j in range(1, m):
                    sample_i = int(idx_t_cpu[row_j].item())
                    if action_ids_sig_all[sample_i] != first_sig:
                        uniform_action_ids = False
                        break
            if uniform_action_ids and m > 0:
                first_mode = int(mode_index_cpu[0].item())
                uniform_mode = True
                for row_j in range(1, m):
                    if int(mode_index_cpu[row_j].item()) != first_mode:
                        uniform_mode = False
                        break
            else:
                uniform_mode = False

            groups: Dict[tuple, List[int]] = {}
            if uniform_action_ids and uniform_mode:
                groups[(int(mode_index_cpu[0].item()), first_sig)] = list(range(m))
            else:
                for row_j in range(m):
                    sample_i = int(idx_t_cpu[row_j].item())
                    mode_id = int(mode_index_cpu[row_j])
                    sig = (mode_id, action_ids_sig_all[sample_i])
                    groups.setdefault(sig, []).append(row_j)

            for sig, rows in groups.items():
                mode_id, _action_sig = sig
                idx_rows = torch.tensor(rows, dtype=torch.long, device=device)
                h_group = h_t.index_select(0, idx_rows).contiguous()
                sample_indices = [int(idx_t_cpu[r].item()) for r in rows]
                action_ids_row = action_ids_all[sample_indices[0]]
                env_masks = [action_env_mask_local_all[sample_i] for sample_i in sample_indices]
                env_mask_batch = torch.stack(env_masks, dim=0).to(device=device, non_blocking=True)
                cache_key = (mode_id, action_ids_sig_all[sample_indices[0]])
                if cache_key in allow_bias_cache:
                    allow_row_t, bias_row_t = allow_bias_cache[cache_key]
                else:
                    allow_cfg = (cfg_eff.get("mode_action_allow", {}) or {}).get(str(mode_id), None)
                    allow_row = [True for _ in action_ids_row]
                    if isinstance(allow_cfg, list) and allow_cfg:
                        allow_set = set(str(a) for a in allow_cfg)
                        allow_row = [a in allow_set for a in action_ids_row]
                    allow_row_t = torch.tensor([allow_row], dtype=torch.bool, device=device)
                    bias_cfg = (cfg_eff.get("mode_action_bias", {}) or {}).get(str(mode_id), {})
                    bias_row = [float(bias_cfg.get(a, 0.0)) for a in action_ids_row] if isinstance(bias_cfg, dict) else [0.0 for _ in action_ids_row]
                    bias_row_t = torch.tensor([bias_row], dtype=torch.float32, device=device)
                    if cache_key not in allow_bias_cache:
                        allow_bias_cache[cache_key] = (allow_row_t, bias_row_t)
                        allow_bias_cache_keys.append(cache_key)
                        if len(allow_bias_cache_keys) > allow_bias_cache_max:
                            evict_key = allow_bias_cache_keys.pop(0)
                            allow_bias_cache.pop(evict_key, None)
                allow_mask = allow_row_t.expand(env_mask_batch.shape[0], -1)
                mode_bias = bias_row_t.expand(env_mask_batch.shape[0], -1)
                final_mask = env_mask_batch & allow_mask
                valid_counts = final_mask.sum(dim=1)
                bad = (valid_counts <= 0).nonzero(as_tuple=False)
                if bad.numel() > 0:
                    bad_idx = int(bad[0].item())
                    bad_sample_i = sample_indices[bad_idx]
                    raise RuntimeError(
                        f"PPO final_action_mask has zero valid actions step={bad_sample_i} mode_id={mode_id} action_ids={action_ids_row[:8]}"
                    )
                actor_out = modules["actor"].forward(
                    h_group,
                    mode_id,
                    action_ids_row,
                    [],
                    cfg=cfg_eff,
                )
                pi_discrete = apply_action_mask_and_bias(
                    actor_out["pi_discrete"],
                    final_mask,
                    mode_bias,
                    action_temperature,
                )
                local_indices = [action_index_cpu[sample_i] for sample_i in sample_indices]
                if any(idx < 0 or idx >= len(action_ids_row) for idx in local_indices):
                    raise RuntimeError(
                        f"PPO invalid action_index step={sample_indices} action_index={local_indices} action_ids={action_ids_row}"
                    )
                local_idx_t = torch.tensor(local_indices, dtype=torch.long, device=device)
                act_logp, act_ent = modules["actor"].action_logp_entropy(
                    pi_discrete,
                    local_idx_t,
                    action_mask=None,
                )
                action_logp_new.index_copy_(0, idx_rows, act_logp.view(-1))
                action_entropy_new.index_copy_(0, idx_rows, act_ent.view(-1))

            if not any_coord_use:
                coord_mask = torch.zeros((m, 0), dtype=torch.bool, device=device)
                old_total = old_logp_mode + old_logp_action
                new_total = mode_logp_new + action_logp_new
                return {
                    "old_mode": old_logp_mode,
                    "old_action": old_logp_action,
                    "old_coord": old_logp_coord,
                    "old_total": old_total,
                    "new_mode": mode_logp_new,
                    "new_action": action_logp_new,
                    "new_coord": coord_logp_new,
                    "new_total": new_total,
                    "mode_entropy": mode_entropy_new,
                    "action_entropy": action_entropy_new,
                    "coord_entropy": coord_entropy_all,
                    "mode_logits": ctrl_out["mode_logits"],
                    "action_mask_union": torch.zeros((m, 0), dtype=torch.bool, device=device),
                    "coord_mask": coord_mask,
                    "coord_use": coord_use,
                    "coord_use_count": int(coord_use_cpu.sum().item()) if coord_use_cpu.numel() > 0 else 0,
                    "mode_index": mode_index,
                    "chosen_coord_index": chosen_coord_index,
                }

            coord_mask_list: List[List[bool]] = [[] for _ in range(m)] if diagnostics else []
            valid_coords: List[int] = [0 for _ in range(m)] if diagnostics else []
            coord_feat_dim = _coord_feat_dim(cfg_eff, modules["actor"])
            groups = {}
            for row_j in range(m):
                sample_i = int(idx_t_cpu[row_j].item())
                mode_id = int(mode_index_cpu[row_j])
                if bool(coord_use_cpu[row_j]):
                    k_coord = min(int(cfg_eff.get("coord_topK", 16)), len(coord_candidates_all[sample_i]))
                    coord_sig = coord_sig_all[sample_i]
                else:
                    k_coord = 0
                    coord_sig = None
                sig = (mode_id, action_ids_sig_all[sample_i], int(k_coord), coord_sig)
                groups.setdefault(sig, []).append(row_j)

            for sig, rows in groups.items():
                mode_id, _action_sig, k_coord, _coord_sig = sig
                sample_indices = [int(idx_t_cpu[r].item()) for r in rows]
                action_ids_row = action_ids_all[sample_indices[0]]
                if k_coord > 0:
                    idx_rows = torch.tensor(rows, dtype=torch.long, device=device)
                    h_group = h_t.index_select(0, idx_rows).contiguous()
                    coords_batch = [coord_candidates_all[sample_i][:k_coord] for sample_i in sample_indices]
                    batched_ok = True
                    coords_feat = torch.zeros((len(rows), k_coord, coord_feat_dim), dtype=torch.float32, device=device)
                    coords_mask = torch.zeros((len(rows), k_coord), dtype=torch.bool, device=device)
                    for b, coords in enumerate(coords_batch):
                        if len(coords) < k_coord:
                            batched_ok = False
                            break
                        for cj, cand in enumerate(coords[:k_coord]):
                            fv = cand.get("feat_vec") if isinstance(cand, dict) else None
                            if not isinstance(fv, list) or len(fv) != coord_feat_dim:
                                batched_ok = False
                                break
                            coords_feat[b, cj, :] = torch.tensor(fv, dtype=torch.float32, device=device)
                            coords_mask[b, cj] = True
                        if not batched_ok:
                            break
                    if batched_ok:
                        try:
                            actor_out_coord = modules["actor"].forward(
                                h_group,
                                mode_id,
                                action_ids_row,
                                coords_feat,
                                cfg=cfg_eff,
                            )
                            pi_raw = actor_out_coord["pi_coord"]
                            if pi_raw.dim() != 3 or pi_raw.shape[0] != len(rows):
                                batched_ok = False
                            else:
                                pi_coord = _build_coord_pi(
                                    pi_raw[:, :k_coord],
                                    coords_mask,
                                    torch.full((len(rows),), int(mode_id), dtype=torch.long, device=device),
                                    cfg_eff,
                                    action_temperature,
                                    apply_bias=True,
                                )
                                cidx_t = chosen_coord_index.index_select(0, idx_rows)
                                if bool((cidx_t < 0).any()) or bool((cidx_t >= k_coord).any()):
                                    cidx_list = [int(v) for v in cidx_t.detach().cpu().tolist()]
                                    raise RuntimeError(
                                        f"PPO invalid chosen_coord_index step={sample_indices} chosen_coord_index={cidx_list} k={k_coord}"
                                    )
                                c_logp, c_ent = modules["actor"].coord_logp_entropy(
                                    pi_coord,
                                    cidx_t,
                                    coord_mask=None,
                                )
                                coord_logp_new.index_copy_(0, idx_rows, c_logp.view(-1))
                                coord_entropy_all.index_copy_(0, idx_rows, c_ent.view(-1))
                                if diagnostics:
                                    for row_j in rows:
                                        coord_mask_list[row_j] = [True for _ in range(k_coord)]
                                        valid_coords[row_j] = int(k_coord)
                        except Exception:
                            batched_ok = False
                    if not batched_ok:
                        for row_j in rows:
                            sample_i = valid_indices_list[row_j]
                            coords = coord_candidates_all[sample_i][:k_coord]
                            h_row = h_t[row_j].unsqueeze(0)
                            coord_mask = torch.ones((1, k_coord), dtype=torch.bool, device=device)
                            actor_out_coord = modules["actor"].forward(
                                h_row,
                                mode_id,
                                action_ids_row,
                                coords,
                                cfg=cfg_eff,
                            )
                            pi_coord = _build_coord_pi(
                                actor_out_coord["pi_coord"][:, :k_coord],
                                coord_mask,
                                torch.full((1,), int(mode_id), dtype=torch.long, device=device),
                                cfg_eff,
                                action_temperature,
                                apply_bias=True,
                            )
                            coord_rows_t = torch.tensor([row_j], dtype=torch.long, device=device)
                            cidx_t = chosen_coord_index.index_select(0, coord_rows_t)
                            if bool((cidx_t < 0).any()) or bool((cidx_t >= k_coord).any()):
                                coord_sample_indices = [sample_i]
                                cidx_list = [int(cidx_t[0].detach().cpu().item())]
                                raise RuntimeError(
                                    f"PPO invalid chosen_coord_index step={coord_sample_indices} chosen_coord_index={cidx_list} k={k_coord}"
                                )
                            c_logp, c_ent = modules["actor"].coord_logp_entropy(
                                pi_coord,
                                cidx_t,
                                coord_mask=None,
                            )
                            coord_logp_new.index_copy_(0, coord_rows_t, c_logp.view(-1))
                            coord_entropy_all.index_copy_(0, coord_rows_t, c_ent.view(-1))
                            if diagnostics:
                                coord_mask_list[row_j] = [True for _ in range(k_coord)]
                                valid_coords[row_j] = int(k_coord)

            if diagnostics:
                max_k = max([len(r) for r in coord_mask_list], default=0)
                if max_k > 0:
                    coord_mask = torch.zeros((m, max_k), dtype=torch.bool, device=device)
                    for rj, row in enumerate(coord_mask_list):
                        for cj in range(len(row)):
                            coord_mask[rj, cj] = True
                else:
                    coord_mask = torch.zeros((m, 0), dtype=torch.bool, device=device)
            else:
                coord_mask = torch.zeros((m, 0), dtype=torch.bool, device=device)
            old_total = old_logp_mode + old_logp_action + torch.where(coord_use, old_logp_coord, torch.zeros_like(old_logp_coord))
            new_total = mode_logp_new + action_logp_new + coord_logp_new

            return {
                "old_mode": old_logp_mode,
                "old_action": old_logp_action,
                "old_coord": old_logp_coord,
                "old_total": old_total,
                "new_mode": mode_logp_new,
                "new_action": action_logp_new,
                "new_coord": coord_logp_new,
                "new_total": new_total,
                "mode_entropy": mode_entropy_new,
                "action_entropy": action_entropy_new,
                "coord_entropy": coord_entropy_all,
                "mode_logits": ctrl_out["mode_logits"],
                "action_mask_union": torch.zeros((m, 0), dtype=torch.bool, device=device),
                "coord_mask": coord_mask,
                "coord_use": coord_use,
                "coord_use_count": int(coord_use_cpu.sum().item()) if coord_use_cpu.numel() > 0 else 0,
                "mode_index": mode_index,
                "chosen_coord_index": chosen_coord_index,
            }

        # Pre-update sanity check: recompute logp on incoming batch before any optimizer step.
        pre_valid_idx_t = mask_valid_step_all.nonzero(as_tuple=False).view(-1)
        with torch.no_grad():
            t_pre_eval_start = time.time()
            if preupdate_eval_mode:
                prev_states = _set_eval_for_recompute()
                pre_eval = _vectorized_logp_eval(pre_valid_idx_t, diagnostics=True)
                _restore_train_mode(prev_states)
            else:
                pre_eval = _vectorized_logp_eval(pre_valid_idx_t, diagnostics=True)
            t_pre_eval += time.time() - t_pre_eval_start
            pre_old_mode = pre_eval["old_mode"]
            pre_old_action = pre_eval["old_action"]
            pre_old_coord = pre_eval["old_coord"]
            pre_old_total = pre_eval["old_total"]
            pre_new_mode = pre_eval["new_mode"]
            pre_new_action = pre_eval["new_action"]
            pre_new_coord = pre_eval["new_coord"]
            pre_new_total = pre_eval["new_total"]
            pre_coord_use = pre_eval.get("coord_use")
            pre_log_ratio_mode = pre_new_mode - pre_old_mode
            pre_log_ratio_action = pre_new_action - pre_old_action
            if isinstance(pre_coord_use, torch.Tensor) and bool(pre_coord_use.any()):
                pre_log_ratio_coord = pre_new_coord[pre_coord_use] - pre_old_coord[pre_coord_use]
                approx_kl_coord_pre = float(_approx_kl_from_log_ratio(pre_log_ratio_coord).mean().detach().cpu().item())
                max_abs_coord = float((pre_old_coord[pre_coord_use] - pre_new_coord[pre_coord_use]).abs().max().detach().cpu().item())
            else:
                approx_kl_coord_pre = 0.0
                max_abs_coord = 0.0
            approx_kl_mode_pre = float(_approx_kl_from_log_ratio(pre_log_ratio_mode).mean().detach().cpu().item()) if pre_log_ratio_mode.numel() > 0 else 0.0
            approx_kl_action_pre = float(_approx_kl_from_log_ratio(pre_log_ratio_action).mean().detach().cpu().item()) if pre_log_ratio_action.numel() > 0 else 0.0
            if kl_metric == "mode":
                approx_kl_pre_update = approx_kl_mode_pre
            elif kl_metric == "coord":
                approx_kl_pre_update = approx_kl_coord_pre
            else:
                approx_kl_pre_update = approx_kl_action_pre
            max_abs_logp_diff_pre_update = 0.0
            if pre_old_mode.numel() > 0:
                max_abs_logp_diff_pre_update = max(max_abs_logp_diff_pre_update, float((pre_old_mode - pre_new_mode).abs().max().detach().cpu().item()))
            if pre_old_action.numel() > 0:
                max_abs_logp_diff_pre_update = max(max_abs_logp_diff_pre_update, float((pre_old_action - pre_new_action).abs().max().detach().cpu().item()))
            max_abs_logp_diff_pre_update = max(max_abs_logp_diff_pre_update, max_abs_coord)
        coord_use_count = int(pre_eval.get("coord_use_count", 0)) if isinstance(pre_eval, dict) else 0
        logger.info(
            "approx_kl_pre_update=%.6f max_abs_logp_diff_pre_update=%.6f coord_use_count=%s",
            approx_kl_pre_update,
            max_abs_logp_diff_pre_update,
            coord_use_count,
        )
        if approx_kl_pre_update > preupdate_kl_max:
            if kl_metric == "mode":
                diff = (pre_old_mode - pre_new_mode).abs()
                max_i = int(diff.argmax().detach().cpu().item()) if diff.numel() > 0 else -1
            elif kl_metric == "coord":
                if isinstance(pre_coord_use, torch.Tensor) and bool(pre_coord_use.any()):
                    diff = (pre_old_coord - pre_new_coord).abs()
                    diff = torch.where(pre_coord_use, diff, torch.full_like(diff, -1.0))
                    max_i = int(diff.argmax().detach().cpu().item()) if diff.numel() > 0 and float(diff.max().detach().cpu().item()) >= 0.0 else -1
                else:
                    max_i = -1
            else:
                diff = (pre_old_action - pre_new_action).abs()
                max_i = int(diff.argmax().detach().cpu().item()) if diff.numel() > 0 else -1
            if max_i >= 0 and max_i < int(pre_valid_idx_t.numel()):
                global_i = int(pre_valid_idx_t[max_i].detach().cpu().item())
            else:
                global_i = -1
            local_action_ids = action_ids_all[global_i] if global_i >= 0 else []
            local_action_index = action_index_cpu[global_i] if global_i >= 0 else -1
            action_id_dbg = local_action_ids[local_action_index] if 0 <= local_action_index < len(local_action_ids) else "ACTION1"
            sample_old = float(pre_old_total[max_i].detach().cpu().item()) if max_i >= 0 else 0.0
            sample_new = float(pre_new_total[max_i].detach().cpu().item()) if max_i >= 0 else 0.0
            mode_id_dbg = (
                int(pre_eval["mode_index"][max_i].detach().cpu().item())
                if (isinstance(pre_eval, dict) and max_i >= 0 and pre_eval.get("mode_index") is not None and pre_eval["mode_index"].numel() > 0)
                else None
            )
            allow_dbg = (cfg_eff.get("mode_action_allow", {}) or {}).get(str(mode_id_dbg), None) if mode_id_dbg is not None else None
            bias_dbg = (cfg_eff.get("mode_action_bias", {}) or {}).get(str(mode_id_dbg), {}) if mode_id_dbg is not None else {}
            coord_bias_dbg = float((cfg_eff.get("mode_coord_bias", {}) or {}).get(str(mode_id_dbg), 0.0)) if mode_id_dbg is not None else 0.0
            if global_i >= 0 and mode_id_dbg is not None:
                env_mask_dbg = action_env_mask_local_all[global_i]
                allow_cfg_dbg = (cfg_eff.get("mode_action_allow", {}) or {}).get(str(mode_id_dbg), None)
                allow_row_dbg = [True for _ in local_action_ids]
                if isinstance(allow_cfg_dbg, list) and allow_cfg_dbg:
                    allow_set_dbg = set(str(a) for a in allow_cfg_dbg)
                    allow_row_dbg = [a in allow_set_dbg for a in local_action_ids]
                allow_mask_dbg = torch.tensor(allow_row_dbg, dtype=torch.bool, device=env_mask_dbg.device)
                final_mask_dbg = env_mask_dbg & allow_mask_dbg
                valid_actions_dbg = int(final_mask_dbg.sum().detach().cpu().item())
            else:
                valid_actions_dbg = None
            actor_temp_dbg = float(cfg_eff.get("temperature", 1.0))
            controller_cfg = cfg_eff.get("controller", {}) if isinstance(cfg_eff.get("controller"), dict) else {}
            controller_temp_dbg = float(controller_cfg.get("temperature", 1.0))
            logger.error(
                "rollout policy mismatch approx_kl_pre_update=%.6f preupdate_kl_max=%.6f max_abs_logp_diff_pre_update=%.6f coord_use_count=%s subset_i=%s global_i=%s action_id=%s action_index=%s old_logp=%.6f new_logp=%.6f valid_actions=%s allow=%s bias=%s coord_bias=%.6f actor_temp=%.6f controller_temp=%.6f actor_forward=%s rollout_policy_version=%s trainer_iter=%s",
                approx_kl_pre_update,
                preupdate_kl_max,
                max_abs_logp_diff_pre_update,
                coord_use_count,
                max_i,
                global_i,
                action_id_dbg,
                local_action_index,
                sample_old,
                sample_new,
                valid_actions_dbg,
                allow_dbg,
                bias_dbg,
                coord_bias_dbg,
                actor_temp_dbg,
                controller_temp_dbg,
                "canonical",
                rollout_policy_version,
                iter_idx,
            )
            if debug_logp_mismatch_dump and pre_old_total.numel() > 0:
                controller_cfg = cfg_eff.get("controller", {}) if isinstance(cfg_eff.get("controller"), dict) else {}
                raw_mask_type_dbg = action_mask_raw_type_all[global_i] if global_i >= 0 else None
                raw_mask_len_dbg = action_mask_raw_len_all[global_i] if global_i >= 0 else None
                raw_mask_preview_dbg = action_mask_raw_preview_all[global_i] if global_i >= 0 else None
                valid_coords_dbg = min(int(cfg_eff.get("coord_topK", 16)), len(coord_candidates_all[global_i])) if global_i >= 0 else None
                logger.error(
                    "logp_mismatch_dump sample=%s stored(mode=%.6f action=%.6f coord=%.6f total=%.6f) "
                    "recomputed(mode=%.6f action=%.6f coord=%.6f total=%.6f) mode_id=%s action_id=%s action_index=%s coord_idx=%s "
                    "has_coord=%s valid_actions=%s valid_coords=%s raw_mask_type=%s raw_mask_len=%s raw_mask_preview=%s controller_temp=%.6f actor_temp=%.6f",
                    max_i,
                    float(pre_old_mode[max_i].detach().cpu().item()),
                    float(pre_old_action[max_i].detach().cpu().item()),
                    float(pre_old_coord[max_i].detach().cpu().item()),
                    float(pre_old_total[max_i].detach().cpu().item()),
                    float(pre_new_mode[max_i].detach().cpu().item()),
                    float(pre_new_action[max_i].detach().cpu().item()),
                    float(pre_new_coord[max_i].detach().cpu().item()),
                    float(pre_new_total[max_i].detach().cpu().item()),
                    int(pre_eval["mode_index"][max_i].detach().cpu().item()) if pre_eval["mode_index"].numel() > 0 else None,
                    action_id_dbg,
                    local_action_index,
                    int(pre_eval["chosen_coord_index"][max_i].detach().cpu().item()) if pre_eval["chosen_coord_index"].numel() > 0 else None,
                    bool(pre_eval["coord_use"][max_i].detach().cpu().item()) if pre_eval["coord_use"].numel() > 0 else None,
                    valid_actions_dbg,
                    valid_coords_dbg,
                    raw_mask_type_dbg,
                    raw_mask_len_dbg,
                    raw_mask_preview_dbg,
                    float(controller_cfg.get("temperature", 1.0)),
                    float(cfg_eff.get("temperature", 1.0)),
                )
            return _empty_ppo_report(batch, stage=stage)

        last_total = torch.tensor(0.0, device=device)
        last_actor = torch.tensor(0.0, device=device)
        last_ctrl = torch.tensor(0.0, device=device)
        last_value = torch.tensor(0.0, device=device)
        last_ent_a = torch.tensor(0.0, device=device)
        last_ent_m = torch.tensor(0.0, device=device)
        last_aux = torch.tensor(0.0, device=device)

        approx_kl_vals: List[torch.Tensor] = []
        approx_kl_post_update_vals: List[torch.Tensor] = []
        approx_kl_mode_vals: List[torch.Tensor] = []
        approx_kl_action_vals: List[torch.Tensor] = []
        approx_kl_coord_vals: List[torch.Tensor] = []
        clip_mode_vals: List[torch.Tensor] = []
        clip_action_vals: List[torch.Tensor] = []
        clip_coord_vals: List[torch.Tensor] = []
        grad_norm_vals: List[torch.Tensor] = []
        epochs_ran = 0
        entropy_action_discrete_vals: List[torch.Tensor] = []
        entropy_action_coord_vals: List[torch.Tensor] = []
        entropy_mode_sum_t = torch.zeros((num_modes,), dtype=torch.float32, device=device)
        entropy_mode_count_t = torch.zeros((num_modes,), dtype=torch.float32, device=device)
        rnd_loss_vals: List[torch.Tensor] = []
        rnd_err_raw_vals: List[torch.Tensor] = []
        rnd_updates_done = 0
        rnd_updates_limit = int(intrinsic_cfg.get("rnd_updates_per_iter", 0)) if intrinsic_enabled else 0

        stop_early = False
        device_check_done = False
        for epoch in range(ppo_epochs):
            if stop_early:
                break
            epochs_ran += 1
            logger.info(
                "train_step_epoch_start iter=%s epoch=%s/%s elapsed_sec=%.1f",
                iter_idx,
                epoch + 1,
                ppo_epochs,
                time.time() - train_t0,
            )
            gen = torch.Generator(device="cpu")
            gen.manual_seed(rng_seed + epoch)
            perm = torch.randperm(n, generator=gen, device="cpu")

            for mb_start in range(0, n, mb_size):
                mb_idx_t_cpu = perm[mb_start : mb_start + mb_size]
                if mb_idx_t_cpu.numel() == 0:
                    continue
                mb_done += 1

                mode_losses: List[torch.Tensor] = []
                action_losses: List[torch.Tensor] = []
                coord_losses: List[torch.Tensor] = []
                value_losses: List[torch.Tensor] = []
                ent_action: List[torch.Tensor] = []
                ent_mode: List[torch.Tensor] = []
                aux_losses: List[torch.Tensor] = []
                approx_kl_mode_mb: List[torch.Tensor] = []
                approx_kl_action_mb: List[torch.Tensor] = []
                approx_kl_coord_mb: List[torch.Tensor] = []
                clip_mode_mb: List[torch.Tensor] = []
                clip_action_mb: List[torch.Tensor] = []
                clip_coord_mb: List[torch.Tensor] = []
                ent_action_discrete: List[torch.Tensor] = []
                ent_action_coord: List[torch.Tensor] = []

                mb_idx_t = mb_idx_t_cpu.to(device=device)
                valid_sel = mask_valid_step_all.index_select(0, mb_idx_t)
                valid_idx_t = mb_idx_t[valid_sel]
                if valid_idx_t.numel() == 0:
                    continue
                idx_t = valid_idx_t
                m = int(idx_t.numel())
                h_t = h_t_all.index_select(0, idx_t).contiguous()
                mode_index = mode_id_all.index_select(0, idx_t)
                if mode_index.numel() > 0:
                    min_mode = int(mode_index.min().detach().cpu().item())
                    max_mode = int(mode_index.max().detach().cpu().item())
                    if min_mode < 0 or max_mode >= num_modes:
                        bad_mask = (mode_index < 0) | (mode_index >= num_modes)
                        bad_vals = mode_index[bad_mask].detach().cpu().tolist()[:8]
                        raise RuntimeError(
                            f"PPO invalid mode_index iter={iter_idx} epoch={epoch + 1} minibatch={mb_done} min={min_mode} max={max_mode} num_modes={num_modes} bad_values={bad_vals}"
                        )
                chosen_coord_index = chosen_coord_index_all.index_select(0, idx_t)
                old_logp_mode = old_logp_mode_all.index_select(0, idx_t)
                old_logp_action = old_logp_action_all.index_select(0, idx_t)
                old_logp_coord = old_logp_coord_all.index_select(0, idx_t)
                adv = adv_t.index_select(0, idx_t)
                ret = ret_all.index_select(0, idx_t)
                old_value = old_value_all.index_select(0, idx_t)

                t_mb_eval_start = time.time()
                if preupdate_eval_mode:
                    prev_states = _set_eval_for_recompute()
                    eval_out = _vectorized_logp_eval(valid_idx_t)
                    _restore_train_mode(prev_states)
                else:
                    eval_out = _vectorized_logp_eval(valid_idx_t)
                t_mb_eval += time.time() - t_mb_eval_start
                value_new = modules["value"].forward(h_t).view(-1)

                mode_logp_new = eval_out["new_mode"]
                action_logp_new = eval_out["new_action"]
                coord_logp_new = eval_out["new_coord"]
                mode_entropy_new = eval_out["mode_entropy"]
                action_entropy_new = eval_out["action_entropy"]
                coord_entropy_all = eval_out["coord_entropy"]
                coord_use = eval_out["coord_use"]
                coord_mask = eval_out["coord_mask"]
                ctrl_logits = eval_out["mode_logits"] / float(controller_temp)

                ratio_mode = torch.exp(mode_logp_new - old_logp_mode)
                ratio_mode_clip = torch.clamp(ratio_mode, 1.0 - clip_eps_controller, 1.0 + clip_eps_controller)
                loss_mode = -torch.min(ratio_mode * adv, ratio_mode_clip * adv)
                mode_losses.append(loss_mode.mean())
                clip_mode_mb.append((ratio_mode.detach() - ratio_mode_clip.detach()).abs().gt(1e-8).float().mean())

                ratio_action = torch.exp(action_logp_new - old_logp_action)
                ratio_action_clip = torch.clamp(ratio_action, 1.0 - clip_eps_actor, 1.0 + clip_eps_actor)
                loss_action = -torch.min(ratio_action * adv, ratio_action_clip * adv)
                action_losses.append(loss_action.mean())
                clip_action_mb.append((ratio_action.detach() - ratio_action_clip.detach()).abs().gt(1e-8).float().mean())
                ent_action_discrete.append(action_entropy_new.mean())
                ent_action.append(action_entropy_new.mean())

                if bool(coord_use.any()):
                    cu = coord_use
                    ratio_coord = torch.exp(coord_logp_new[cu] - old_logp_coord[cu])
                    ratio_coord_clip = torch.clamp(ratio_coord, 1.0 - clip_eps_coord, 1.0 + clip_eps_coord)
                    loss_coord = -torch.min(ratio_coord * adv[cu], ratio_coord_clip * adv[cu])
                    coord_losses.append(loss_coord.mean())
                    clip_coord_mb.append((ratio_coord.detach() - ratio_coord_clip.detach()).abs().gt(1e-8).float().mean())
                    log_ratio_coord = coord_logp_new[cu] - old_logp_coord[cu]
                    approx_kl_coord_mb.append(_approx_kl_from_log_ratio(log_ratio_coord).mean())
                    ent_action_coord.append(coord_entropy_all[cu].mean())

                log_ratio_mode = mode_logp_new - old_logp_mode
                log_ratio_action = action_logp_new - old_logp_action
                approx_kl_mode_mb.append(_approx_kl_from_log_ratio(log_ratio_mode).mean())
                approx_kl_action_mb.append(_approx_kl_from_log_ratio(log_ratio_action).mean())

                v_loss_unclipped = (value_new - ret).pow(2)
                if vf_clip:
                    old_value_present = old_value_present_all.index_select(0, idx_t)
                    v_clip = old_value + (value_new - old_value).clamp(-vf_clip_eps, vf_clip_eps)
                    v_loss_clipped = torch.max(v_loss_unclipped, (v_clip - ret).pow(2))
                    v_loss = torch.where(old_value_present, v_loss_clipped, v_loss_unclipped).mean()
                else:
                    v_loss = v_loss_unclipped.mean()
                value_losses.append(v_loss)

                ent_mode.append(mode_entropy_new.mean())
                entropy_mode_sum_t.scatter_add_(0, mode_index, mode_entropy_new)
                entropy_mode_count_t.scatter_add_(0, mode_index, torch.ones_like(mode_entropy_new))

                aux_present = aux_present_all.index_select(0, idx_t)
                if bool(aux_present.any()):
                    aux_targets_t = aux_target_t_all.index_select(0, idx_t)
                    aux_w = aux_mode_weight_all.index_select(0, idx_t)
                    aux_loss_all = F.cross_entropy(ctrl_logits, aux_targets_t, reduction="none")
                    aux_loss_masked = aux_loss_all.mul(aux_w)
                    aux_losses.append(aux_loss_masked[aux_present].mean())

                if not device_check_done:
                    if h_t.device != device:
                        raise RuntimeError(f"PPO update expected h_t on {device}, got {h_t.device}")
                    if ctrl_logits.device != device:
                        raise RuntimeError(
                            f"PPO update expected controller logits on {device}, got {ctrl_logits.device}"
                        )
                    if value_new.device != device:
                        raise RuntimeError(f"PPO update expected value head output on {device}, got {value_new.device}")
                    if device.type == "cuda":
                        assert h_t.is_cuda
                        assert adv.is_cuda
                        assert ret.is_cuda
                        assert old_logp_mode.is_cuda
                        assert old_logp_action.is_cuda
                        assert old_logp_coord.is_cuda
                        assert coord_mask.is_cuda
                    device_check_done = True

                if not action_losses or not mode_losses or not value_losses:
                    continue

                loss_mode_mb = torch.stack(mode_losses).mean()
                loss_action_mb = torch.stack(action_losses).mean()
                loss_coord_mb = torch.stack(coord_losses).mean() if coord_losses else torch.tensor(0.0, device=device)
                loss_value_mb = torch.stack(value_losses).mean()
                coord_entropy_used = coord_entropy_all[coord_use].mean() if bool(coord_use.any()) else torch.tensor(0.0, device=device)
                coord_frac = coord_use.float().mean() if coord_use.numel() > 0 else torch.tensor(0.0, device=device)
                ent_a_mb = (torch.stack(ent_action).mean() if ent_action else torch.tensor(0.0, device=device)) + (
                    coord_entropy_used * coord_frac
                )
                ent_m_mb = torch.stack(ent_mode).mean() if ent_mode else torch.tensor(0.0, device=device)
                aux_mb = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=device)

                loss_pi = actor_coef * loss_action_mb + controller_coef * loss_mode_mb + coord_coef * loss_coord_mb
                total_loss = loss_pi + value_coef * loss_value_mb + aux_mode_ce_coef * aux_mb - entropy_coef * ent_a_mb - mode_entropy_coef * ent_m_mb

                t_mb_step_start = time.time()
                _zero_grad(optim)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (list(modules["controller"].parameters()) if modules["controller"] is not None else [])
                    + list(modules["actor"].parameters())
                    + list(modules["value"].parameters()),
                    max_grad_norm,
                )
                _step_optim(optim)
                t_mb_step += time.time() - t_mb_step_start

                if intrinsic_enabled and grid_embed_all is not None and intrinsic_valid_mask is not None:
                    if rnd_updates_limit <= 0 or rnd_updates_done < rnd_updates_limit:
                        rnd_sel = intrinsic_valid_mask.index_select(0, idx_t)
                        rnd_idx_t = idx_t[rnd_sel]
                        if rnd_idx_t.numel() > 0:
                            grid_embed_mb = grid_embed_all.index_select(0, rnd_idx_t)
                            pred, tgt, _err_vec, err_scalar = modules["intrinsic_rnd"](grid_embed_mb)
                            rnd_loss = F.mse_loss(pred, tgt)
                            _zero_grad(optim)
                            rnd_loss.backward()
                            _step_optim(optim)
                            rnd_loss_vals.append(rnd_loss.detach())
                            rnd_err_raw_vals.append(err_scalar.mean().detach())
                            rnd_updates_done += 1

                last_total = total_loss.detach()
                last_actor = loss_action_mb.detach()
                last_ctrl = loss_mode_mb.detach()
                last_value = loss_value_mb.detach()
                last_ent_a = ent_a_mb.detach()
                last_ent_m = ent_m_mb.detach()
                last_aux = aux_mb.detach()

                grad_norm_vals.append(grad_norm if torch.is_tensor(grad_norm) else torch.tensor(float(grad_norm), dtype=torch.float32, device=device))
                if recompute_post_update_metrics:
                    with torch.no_grad():
                        t_mb_eval_start = time.time()
                        if preupdate_eval_mode:
                            prev_states = _set_eval_for_recompute()
                            post_eval = _vectorized_logp_eval(valid_idx_t)
                            _restore_train_mode(prev_states)
                        else:
                            post_eval = _vectorized_logp_eval(valid_idx_t)
                        t_mb_eval += time.time() - t_mb_eval_start
                        post_log_ratio_mode = post_eval["new_mode"] - post_eval["old_mode"]
                        post_log_ratio_action = post_eval["new_action"] - post_eval["old_action"]
                        post_coord_use = post_eval["coord_use"]
                        if bool(post_coord_use.any()):
                            post_log_ratio_coord = post_eval["new_coord"][post_coord_use] - post_eval["old_coord"][post_coord_use]
                            approx_kl_coord_t = _approx_kl_from_log_ratio(post_log_ratio_coord).mean()
                        else:
                            approx_kl_coord_t = torch.tensor(0.0, dtype=torch.float32, device=device)
                        approx_kl_mode_t = _approx_kl_from_log_ratio(post_log_ratio_mode).mean()
                        approx_kl_action_t = _approx_kl_from_log_ratio(post_log_ratio_action).mean()
                else:
                    approx_kl_mode_t = (
                        torch.stack(approx_kl_mode_mb).mean()
                        if approx_kl_mode_mb
                        else torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                    approx_kl_action_t = (
                        torch.stack(approx_kl_action_mb).mean()
                        if approx_kl_action_mb
                        else torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                    approx_kl_coord_t = (
                        torch.stack(approx_kl_coord_mb).mean()
                        if approx_kl_coord_mb
                        else torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                if kl_metric == "mode":
                    approx_kl_now_t = approx_kl_mode_t
                elif kl_metric == "coord":
                    approx_kl_now_t = approx_kl_coord_t
                else:
                    approx_kl_now_t = approx_kl_action_t
                approx_kl_vals.append(approx_kl_now_t)
                approx_kl_post_update_vals.append(approx_kl_now_t)
                approx_kl_mode_vals.append(approx_kl_mode_t)
                approx_kl_action_vals.append(approx_kl_action_t)
                approx_kl_coord_vals.append(approx_kl_coord_t)
                if clip_mode_mb:
                    clip_mode_vals.append(torch.stack(clip_mode_mb).mean())
                if clip_action_mb:
                    clip_action_vals.append(torch.stack(clip_action_mb).mean())
                if clip_coord_mb:
                    clip_coord_vals.append(torch.stack(clip_coord_mb).mean())
                if ent_action_discrete:
                    entropy_action_discrete_vals.append(torch.stack(ent_action_discrete).mean())
                if ent_action_coord:
                    entropy_action_coord_vals.append(torch.stack(ent_action_coord).mean())

                if heartbeat_enabled and (mb_done % heartbeat_every_minibatches == 0):
                    approx_kl_now = float(approx_kl_now_t.detach().cpu().item())
                    logger.info(
                        "train_step_heartbeat iter=%s epoch=%s/%s minibatch=%s/%s elapsed_sec=%.1f approx_kl_pre_update=%.6f approx_kl_post_update=%.6f",
                        iter_idx,
                        epoch + 1,
                        ppo_epochs,
                        mb_done,
                        ppo_epochs * total_mb_per_epoch,
                        time.time() - train_t0,
                        float(approx_kl_pre_update),
                        approx_kl_now,
                    )

                approx_kl_mode_mb_t = torch.stack(approx_kl_mode_mb).mean() if approx_kl_mode_mb else torch.tensor(0.0, device=device)
                approx_kl_action_mb_t = torch.stack(approx_kl_action_mb).mean() if approx_kl_action_mb else torch.tensor(0.0, device=device)
                approx_kl_coord_mb_t = torch.stack(approx_kl_coord_mb).mean() if approx_kl_coord_mb else torch.tensor(0.0, device=device)
                if kl_metric == "mode":
                    approx_kl_for_stop = approx_kl_mode_mb_t
                elif kl_metric == "coord":
                    approx_kl_for_stop = approx_kl_coord_mb_t
                else:
                    approx_kl_for_stop = approx_kl_action_mb_t
                if early_stop_kl and float(approx_kl_for_stop.detach().cpu().item()) > target_kl:
                    logger.info(
                        "train_step_early_stop iter=%s epoch=%s/%s minibatch=%s approx_kl_post_update=%.6f target_kl=%.6f elapsed_sec=%.1f",
                        iter_idx,
                        epoch + 1,
                        ppo_epochs,
                        mb_done,
                        float(approx_kl_for_stop.detach().cpu().item()),
                        float(target_kl),
                        time.time() - train_t0,
                    )
                    stop_early = True
                    break

        param_delta_norm = _param_delta_norm(modules, pre_train_param_snapshot)
        approx_kl_post_update_iter = (
            float(torch.stack(approx_kl_post_update_vals).mean().detach().cpu().item())
            if approx_kl_post_update_vals
            else 0.0
        )
        approx_kl_mode_iter = (
            float(torch.stack(approx_kl_mode_vals).mean().detach().cpu().item()) if approx_kl_mode_vals else 0.0
        )
        approx_kl_action_iter = (
            float(torch.stack(approx_kl_action_vals).mean().detach().cpu().item()) if approx_kl_action_vals else 0.0
        )
        approx_kl_coord_iter = (
            float(torch.stack(approx_kl_coord_vals).mean().detach().cpu().item()) if approx_kl_coord_vals else 0.0
        )
        logger.info(
            "train_step_kl_summary iter=%s approx_kl_pre_update=%.6f approx_kl_post_update=%.6f param_delta_norm=%.6e lr_current=%.8f",
            iter_idx,
            float(approx_kl_pre_update),
            float(approx_kl_post_update_iter),
            float(param_delta_norm),
            float(self._lr_current),
        )
        if kl_metric == "mode":
            base_kl = approx_kl_mode_iter
        elif kl_metric == "coord":
            base_kl = approx_kl_coord_iter
        else:
            base_kl = approx_kl_action_iter
        kl_value = float(abs(base_kl)) if self.lr_adapt_use_abs_kl else float(base_kl)
        if self.lr_adapt_enabled:
            if kl_value > float(self.lr_adapt_target_kl):
                self._lr_current = max(
                    float(self._lr_base) * float(self.lr_adapt_min_lr_mult),
                    float(self._lr_current) * float(self.lr_adapt_downscale),
                )
                self._lr_low_kl_counter = 0
            elif kl_value < 0.5 * float(self.lr_adapt_target_kl):
                self._lr_low_kl_counter += 1
                if int(self._lr_low_kl_counter) >= int(self.lr_adapt_upscale_patience_iters):
                    self._lr_current = min(
                        float(self._lr_base) * float(self.lr_adapt_max_lr_mult),
                        float(self._lr_current) * float(self.lr_adapt_upscale),
                    )
                    self._lr_low_kl_counter = 0
            else:
                self._lr_low_kl_counter = 0
            _set_optimizer_lr(optim, float(self._lr_current))

        logger.info(
            "train_step_end iter=%s algo=ppo epochs_ran=%s minibatches_done=%s elapsed_sec=%.1f",
            iter_idx,
            epochs_ran,
            mb_done,
            time.time() - train_t0,
        )
        logger.info(
            "train_step_perf iter=%s t_flatten=%.3f t_mask=%.3f t_pre_eval=%.3f t_mb_eval=%.3f t_mb_step=%.3f",
            iter_idx,
            t_flatten,
            t_mask,
            t_pre_eval,
            t_mb_eval,
            t_mb_step,
        )
        if intrinsic_enabled and grid_embed_all is not None:
            logger.info(
                "train_step_rnd_summary iter=%s grid_embed_rows=%s grid_embed_dim=%s rnd_updates=%s",
                iter_idx,
                int(grid_embed_all.shape[0]),
                int(grid_embed_all.shape[1]) if grid_embed_all.dim() > 1 else 0,
                int(rnd_updates_done),
            )

        n_episodes = len(batch.get("episodes", []))
        mean_return = float(sum(episode_returns) / max(1, len(episode_returns))) if episode_returns else 0.0
        avg_reward = mean_return
        mean_episode_len = float(total_steps / max(1, n_episodes))
        win_rate = float(win_eps / max(1, n_episodes))
        done_rate = float(done_eps) / float(max(1, n_episodes))

        return {
            "losses": {
                "total": float(last_total.item()),
                "actor_policy": float(last_actor.item()),
                "controller_policy": float(last_ctrl.item()),
                "value": float(last_value.item()),
                "entropy_actor": float(last_ent_a.item()),
                "entropy_controller": float(last_ent_m.item()),
                "aux_mode_ce": float(last_aux.item()),
                "grad_norm_total": float(torch.stack(grad_norm_vals).mean().detach().cpu().item()) if grad_norm_vals else 0.0,
                "approx_kl": float(torch.stack(approx_kl_vals).mean().detach().cpu().item()) if approx_kl_vals else 0.0,
                "approx_kl_mode": float(approx_kl_mode_iter),
                "approx_kl_action": float(approx_kl_action_iter),
                "approx_kl_coord": float(approx_kl_coord_iter),
                "approx_kl_pre_update": float(approx_kl_pre_update),
                "approx_kl_post_update": float(approx_kl_post_update_iter),
                "param_delta_norm": float(param_delta_norm),
                "train/clip_eps_actor_effective": float(clip_eps_actor),
                "train/clip_eps_controller_effective": float(clip_eps_controller),
                "train/clip_eps_coord_effective": float(clip_eps_coord),
                "train/vf_clip_eps_effective": float(vf_clip_eps),
                "train/ppo_epochs_effective": int(ppo_epochs),
                "train/ppo_minibatches_effective": int(ppo_minibatches),
                "train/rnd_loss": float(torch.stack(rnd_loss_vals).mean().detach().cpu().item()) if rnd_loss_vals else 0.0,
                "train/rnd_err_raw_mean": float(torch.stack(rnd_err_raw_vals).mean().detach().cpu().item()) if rnd_err_raw_vals else 0.0,
                "train/rnd_phi_mean": float(rnd_phi_mean),
                "clip_frac": float(
                    torch.stack([v for v in [torch.stack(clip_mode_vals).mean() if clip_mode_vals else None,
                                             torch.stack(clip_action_vals).mean() if clip_action_vals else None,
                                             torch.stack(clip_coord_vals).mean() if clip_coord_vals else None] if v is not None]).mean().detach().cpu().item()
                    if any([clip_mode_vals, clip_action_vals, clip_coord_vals]) else 0.0
                ),
                "clipfrac_mode": float(torch.stack(clip_mode_vals).mean().detach().cpu().item()) if clip_mode_vals else 0.0,
                "clipfrac_action": float(torch.stack(clip_action_vals).mean().detach().cpu().item()) if clip_action_vals else 0.0,
                "clipfrac_coord": float(torch.stack(clip_coord_vals).mean().detach().cpu().item()) if clip_coord_vals else 0.0,
                "ClipFrac_action": float(torch.stack(clip_action_vals).mean().detach().cpu().item()) if clip_action_vals else 0.0,
                "ClipFrac_coord": float(torch.stack(clip_coord_vals).mean().detach().cpu().item()) if clip_coord_vals else 0.0,
                "ClipFrac_controller": float(torch.stack(clip_mode_vals).mean().detach().cpu().item()) if clip_mode_vals else 0.0,
                "entropy_action_discrete": float(torch.stack(entropy_action_discrete_vals).mean().detach().cpu().item()) if entropy_action_discrete_vals else 0.0,
                "entropy_action_coord": float(torch.stack(entropy_action_coord_vals).mean().detach().cpu().item()) if entropy_action_coord_vals else 0.0,
                "entropy_by_mode": {
                    str(k): float(
                        entropy_mode_sum_t[k].detach().cpu().item()
                        / max(1.0, entropy_mode_count_t[k].detach().cpu().item())
                    )
                    for k in range(num_modes)
                    if float(entropy_mode_count_t[k].detach().cpu().item()) > 0.0
                },
                "ppo_epochs_ran": int(epochs_ran),
                "adv_mean": float(adv_mean),
                "adv_std": float(adv_std),
                "train_stage": stage,
            },
            "mean_return": mean_return,
            "avg_reward": avg_reward,
            "mean_episode_len": mean_episode_len,
            "win_rate": win_rate,
            "done_rate": done_rate,
        }

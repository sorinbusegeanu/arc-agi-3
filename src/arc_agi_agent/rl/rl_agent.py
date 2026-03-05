from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from ..action_schema import build_action_schema_from_env
from ..config import RLConfig
from .action_key_normalize_v1 import action_key_to_index
from .coord_proposer import CoordProposer
from .hierarchical_controller import HierarchicalController
from .intrinsic_rnd import IntrinsicRND
from .module_control import module_enabled
from .observation_encoder import ObservationEncoder
from .policy_actor_value import PolicyActor, ValueHead
from .recurrent_memory import RecurrentMemory
from .reward_shaper import RewardShaper
from .rollout_collector import RolloutCollector
from .trainer import Trainer
from .optim import build_optimizer

logger = logging.getLogger(__name__)


def _cfg_to_dict(cfg: RLConfig) -> Dict[str, Any]:
    return {
        "pipeline": dict(cfg.pipeline),
        "modules": dict(cfg.modules),
        "action_source": "rl_agent",
        "embed_dim": cfg.embed_dim,
        "hidden_dim": cfg.hidden_dim,
        "action_emb_dim": cfg.action_emb_dim,
        "coord_topK": cfg.coord_topK,
        "rollout_batch_episodes": cfg.rollout_batch_episodes,
        "rollout_max_steps": cfg.rollout_max_steps,
        "frame_stack": cfg.frame_stack,
        "hud_probe_enabled": cfg.hud_probe_enabled,
        "hud_probe_steps": cfg.hud_probe_steps,
        "hud_cache_dir": cfg.hud_cache_dir,
        "hud_detect_window": cfg.hud_detect_window,
        "hud_change_rate_threshold": cfg.hud_change_rate_threshold,
        "hud_min_changed_cells_per_step": cfg.hud_min_changed_cells_per_step,
        "hud_edge_margin": cfg.hud_edge_margin,
        "hud_min_component_area": cfg.hud_min_component_area,
        "hud_dilate_px": cfg.hud_dilate_px,
        "episodes_per_iter": cfg.episodes_per_iter,
        "max_steps_per_episode": cfg.max_steps_per_episode,
        "stochastic_actions_train": cfg.stochastic_actions_train,
        "deterministic_eval": cfg.deterministic_eval,
        "save_trajectory_batches": cfg.save_trajectory_batches,
        "reward": dict(cfg.reward),
        "algo": cfg.algo,
        "lr": cfg.lr,
        "gamma": cfg.gamma,
        "entropy_coef": cfg.entropy_coef,
        "value_coef": cfg.value_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "updates_per_iter": cfg.updates_per_iter,
        "optimizers": dict(cfg.optimizers),
        "ppo": dict(cfg.ppo),
        "ckpt": dict(cfg.ckpt),
        "log": dict(cfg.log),
        "train": dict(cfg.train),
        "eval": dict(cfg.eval),
        "controller": dict(cfg.controller),
        "num_modes": cfg.num_modes,
        "modes": list(cfg.modes),
        "mode_action_allow": dict(cfg.mode_action_allow),
        "mode_action_bias": dict(cfg.mode_action_bias),
        "mode_coord_bias": dict(cfg.mode_coord_bias),
        "aux": dict(cfg.aux),
    }


def _load_json_defaults() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "rl_config_defaults.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


class RLAgent:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        base = _cfg_to_dict(RLConfig())
        json_defaults = _load_json_defaults()
        self.cfg = {**base, **json_defaults, **(cfg or {})}

        module_enabled(self.cfg, "rl_encoder", required=True)
        _encoder_sub = self.cfg.get("encoder", {}) if isinstance(self.cfg.get("encoder"), dict) else {}
        self.encoder = ObservationEncoder(
            {
                "embed_dim": self.cfg["embed_dim"],
                "frame_stack": int(self.cfg.get("frame_stack", 4)),
                **_encoder_sub,
            }
        )
        module_enabled(self.cfg, "rl_memory", required=True)
        self.memory = RecurrentMemory(
            {
                "hidden_dim": self.cfg["hidden_dim"],
                "action_emb_dim": self.cfg["action_emb_dim"],
            }
        )
        _ctrl_enabled = bool(module_enabled(self.cfg, "rl_controller", required=False))
        self.controller = HierarchicalController(
            {
                "hidden_dim": self.cfg["hidden_dim"],
                "num_modes": max(1, len(self.cfg.get("modes", [0, 1, 2]))),
                "sample_mode_train": bool(self.cfg.get("controller", {}).get("sample_mode_train", True)),
            }
        ) if _ctrl_enabled else None
        module_enabled(self.cfg, "rl_actor", required=True)
        self.actor = PolicyActor(
            {
                "hidden_dim": self.cfg["hidden_dim"],
                "action_emb_dim": self.cfg["action_emb_dim"],
                "num_modes": max(1, len(self.cfg.get("modes", [0, 1, 2]))),
                "mode_action_allow": self.cfg.get("mode_action_allow", {}),
                "mode_action_bias": self.cfg.get("mode_action_bias", {}),
                "mode_coord_bias": self.cfg.get("mode_coord_bias", {}),
            }
        )
        module_enabled(self.cfg, "rl_value", required=True)
        self.value = ValueHead(self.cfg["hidden_dim"])  # type: ignore[arg-type]

        self.intrinsic_rnd = None
        intrinsic_cfg = self.cfg.get("intrinsic", {}) if isinstance(self.cfg.get("intrinsic", {}), dict) else {}
        if bool(intrinsic_cfg.get("enabled", False)) and str(intrinsic_cfg.get("method", "")) == "rnd_grid_embed":
            grid_embed_dim = int(self.encoder.cfg.get("cnn_channels", [128])[-1])
            rnd_hidden = int(intrinsic_cfg.get("rnd_hidden", 256))
            rnd_out = int(intrinsic_cfg.get("rnd_out", 128))
            self.intrinsic_rnd = IntrinsicRND(grid_embed_dim, rnd_hidden, rnd_out)

        if module_enabled(self.cfg, "rl_coord_proposer", required=True):
            self.coord_proposer = CoordProposer()
        if module_enabled(self.cfg, "rl_reward_shaper", required=True):
            self.reward_shaper = RewardShaper()
        if module_enabled(self.cfg, "rl_rollout_collector", required=True):
            self.collector = RolloutCollector(cfg=self.cfg)
        if module_enabled(self.cfg, "rl_trainer", required=True):
            self.trainer = Trainer()

        requested_device = str(self.cfg.get("device", "cuda:0")).lower()
        if requested_device == "cpu":
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for RLAgent when device is not CPU.")
            self.device = torch.device(requested_device if requested_device != "cuda" else "cuda:0")
        self.encoder.to(self.device)
        self.memory.to(self.device)
        if self.controller is not None:
            self.controller.to(self.device)
        self.actor.to(self.device)
        self.value.to(self.device)
        if self.intrinsic_rnd is not None:
            self.intrinsic_rnd.to(self.device)
        self.policy_version = -1

    def _build_modules(self) -> Dict[str, Any]:
        modules = {
            "encoder": self.encoder,
            "memory": self.memory,
            "controller": self.controller,
            "actor": self.actor,
            "value": self.value,
        }
        if self.intrinsic_rnd is not None:
            modules["intrinsic_rnd"] = self.intrinsic_rnd
        return modules

    @staticmethod
    def _normalize_available_actions_mask(mask_obj: Any, nd: int, action_ids: Optional[List[str]] = None) -> tuple[List[bool], bool]:
        if nd <= 0:
            return [], False
        pre_zero = False
        aid_to_idx = {str(a): i for i, a in enumerate(action_ids or [])}
        if mask_obj is None:
            out = [True] * nd
        else:
            mask_raw = list(mask_obj) if isinstance(mask_obj, list) else []
            m = len(mask_raw)
            if m == 0:
                out = [True] * nd
            elif m == nd and all(isinstance(x, bool) for x in mask_raw):
                out = [bool(mask_raw[i]) for i in range(nd)]
            elif m == nd and all(isinstance(x, int) and int(x) in (0, 1) for x in mask_raw):
                out = [bool(int(mask_raw[i])) for i in range(nd)]
            elif all(isinstance(x, int) for x in mask_raw) and ((m != nd) or any(int(x) >= 2 for x in mask_raw)):
                out = [False] * nd
                for idx in mask_raw:
                    ii = int(idx)
                    if 0 <= ii < nd:
                        out[ii] = True
            elif all(isinstance(x, str) for x in mask_raw):
                out = [False] * nd
                for aid in mask_raw:
                    ai = aid_to_idx.get(str(aid))
                    if ai is not None and 0 <= ai < nd:
                        out[ai] = True
            elif m >= nd:
                out = [bool(mask_raw[i]) for i in range(nd)]
            else:
                out = [bool(mask_raw[i]) if i < m else False for i in range(nd)]
        if sum(1 for x in out if x) == 0:
            pre_zero = True
            out = [True] * nd
        return out, pre_zero

    def apply_policy_snapshot(self, state_dict: Dict[str, Any], policy_version: int) -> None:
        actor_sd = state_dict.get("actor", {})
        if not actor_sd:
            raise ValueError(f"apply_policy_snapshot: actor state_dict is empty (policy_version={policy_version})")
        modules = self._build_modules()
        modules["encoder"].load_state_dict(state_dict["encoder"])
        modules["memory"].load_state_dict(state_dict["memory"])
        if modules["controller"] is not None and "controller" in state_dict:
            modules["controller"].load_state_dict(state_dict["controller"])
        modules["actor"].load_state_dict(actor_sd)
        modules["value"].load_state_dict(state_dict["value"])
        modules["encoder"].to(self.device)
        modules["memory"].to(self.device)
        if modules["controller"] is not None:
            modules["controller"].to(self.device)
        modules["actor"].to(self.device)
        modules["value"].to(self.device)
        self.policy_version = int(policy_version)

    def pack_rollout_batch_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        episodes = batch.get("episodes", []) if isinstance(batch, dict) else []
        if not episodes:
            batch["tensor_pack"] = {}
            return batch

        bsz = len(episodes)
        max_t = max(len(ep.get("steps", [])) for ep in episodes)
        h_dim = 0
        max_actions = 0
        for ep in episodes:
            for s in ep.get("steps", []):
                h_raw = s.get("h_t")
                if isinstance(h_raw, list):
                    if h_raw and isinstance(h_raw[0], list):
                        h_dim = max(h_dim, len(h_raw[0]))
                    else:
                        h_dim = max(h_dim, len(h_raw))
                action_ids = s.get("action_ids") or []
                if isinstance(action_ids, list):
                    max_actions = max(max_actions, len(action_ids))
                mask = s.get("available_actions_mask", None)
                if isinstance(mask, list):
                    max_actions = max(max_actions, len(mask))

        z_t = torch.zeros((bsz, max_t, h_dim), dtype=torch.float32)
        prev_action_idx = torch.zeros((bsz, max_t), dtype=torch.int64)
        prev_action_coord = torch.zeros((bsz, max_t, 2), dtype=torch.float32)
        prev_action_has_coord = torch.zeros((bsz, max_t), dtype=torch.bool)
        prev_reward = torch.zeros((bsz, max_t), dtype=torch.float32)
        prev_done = torch.zeros((bsz, max_t), dtype=torch.bool)
        old_logp_mode = torch.zeros((bsz, max_t), dtype=torch.float32)
        old_logp_action_discrete = torch.zeros((bsz, max_t), dtype=torch.float32)
        old_logp_coord = torch.zeros((bsz, max_t), dtype=torch.float32)
        value_pred = torch.zeros((bsz, max_t), dtype=torch.float32)
        action_index = torch.full((bsz, max_t), -1, dtype=torch.int64)
        chosen_coord_index = torch.full((bsz, max_t), -1, dtype=torch.int64)
        available_actions_mask = torch.zeros((bsz, max_t, max_actions), dtype=torch.bool)
        reward = torch.zeros((bsz, max_t), dtype=torch.float32)
        done = torch.zeros((bsz, max_t), dtype=torch.bool)
        valid_step = torch.zeros((bsz, max_t), dtype=torch.bool)

        for b, ep in enumerate(episodes):
            zero_mask_warned = False
            steps: List[Dict[str, Any]] = ep.get("steps", [])
            for t, s in enumerate(steps):
                h_raw = s.get("h_t")
                if isinstance(h_raw, list):
                    if h_raw and isinstance(h_raw[0], list):
                        h_vec = h_raw[0]
                    else:
                        h_vec = h_raw
                    if h_dim > 0:
                        h_ten = torch.tensor(h_vec[:h_dim], dtype=torch.float32)
                        z_t[b, t, : int(h_ten.numel())] = h_ten

                ak = s.get("action_key", {}) if isinstance(s.get("action_key"), dict) else {}
                prev_action_idx[b, t] = int(action_key_to_index(ak, int(getattr(self.memory, "max_actions", 64))))
                if "x" in ak and "y" in ak:
                    prev_action_coord[b, t, 0] = float(ak.get("x", 0.0))
                    prev_action_coord[b, t, 1] = float(ak.get("y", 0.0))
                    prev_action_has_coord[b, t] = True
                prev_reward[b, t] = float(s.get("reward", 0.0))
                prev_done[b, t] = bool(s.get("done", False))
                old_logp_mode[b, t] = float(s.get("old_logp_mode", 0.0))
                old_logp_action_discrete[b, t] = float(s.get("old_logp_action_discrete", s.get("old_logp_action", 0.0)))
                old_logp_coord[b, t] = float(s.get("old_logp_coord", 0.0))
                value_pred[b, t] = float(s.get("value_pred", s.get("old_value", 0.0)))
                action_index[b, t] = int(s.get("action_index", -1))
                chosen_coord_index[b, t] = int(-1 if s.get("chosen_coord_index") is None else s.get("chosen_coord_index"))
                reward[b, t] = float(s.get("reward", 0.0))
                done[b, t] = bool(s.get("done", False))
                valid_step[b, t] = bool(int(s.get("mask_valid_step", 1)) == 1)
                if max_actions > 0:
                    mask_norm, pre_zero = self._normalize_available_actions_mask(
                        s.get("available_actions_mask", None),
                        max_actions,
                        action_ids=list(s.get("action_ids") or []),
                    )
                    if pre_zero and not zero_mask_warned:
                        logger.error(
                            "rollout_mask_zero_valid_fallback episode=%s step=%s nd=%s",
                            b,
                            t,
                            max_actions,
                        )
                        zero_mask_warned = True
                    m = torch.tensor(mask_norm, dtype=torch.bool)
                    available_actions_mask[b, t, : int(m.numel())] = m

        batch["tensor_pack"] = {
            "z_t": z_t,
            "prev_action_idx": prev_action_idx,
            "prev_action_coord": prev_action_coord,
            "prev_action_has_coord": prev_action_has_coord,
            "prev_reward": prev_reward,
            "prev_done": prev_done,
            "old_logp_mode": old_logp_mode,
            "old_logp_action_discrete": old_logp_action_discrete,
            "old_logp_coord": old_logp_coord,
            "value_pred": value_pred,
            "action_index": action_index,
            "chosen_coord_index": chosen_coord_index,
            "available_actions_mask": available_actions_mask,
            "reward": reward,
            "done": done,
            "valid_step": valid_step,
        }
        return batch

    def _save_checkpoint(self, path: str, modules: Dict[str, Any], optim: Any, iter_idx: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "iter": int(iter_idx),
            "cfg": self.cfg,
            "encoder": modules["encoder"].state_dict(),
            "memory": modules["memory"].state_dict(),
            "controller": modules["controller"].state_dict() if modules["controller"] is not None else None,
            "actor": modules["actor"].state_dict(),
            "value": modules["value"].state_dict(),
            "optim": optim.state_dict() if optim is not None else None,
        }
        torch.save(payload, path)

    def _load_checkpoint(self, path: str, modules: Dict[str, Any], optim: Optional[Any]) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        modules["encoder"].load_state_dict(payload.get("encoder", {}))
        modules["memory"].load_state_dict(payload.get("memory", {}))
        if modules["controller"] is not None and payload.get("controller") is not None:
            modules["controller"].load_state_dict(payload["controller"])
        modules["actor"].load_state_dict(payload.get("actor", {}))
        modules["value"].load_state_dict(payload.get("value", {}))
        if optim is not None and payload.get("optim") is not None:
            optim.load_state_dict(payload["optim"])
        return payload

    def infer_action(
        self,
        observation: Any,
        fp_report: Optional[Any],
        action_schema: Dict[str, Any],
        h_prev: Optional[Any],
        prev_action: Optional[Dict[str, Any]],
        prev_reward: float,
        prev_done: bool,
        stochastic: bool,
    ) -> Dict[str, Any]:
        modules = self._build_modules()
        with torch.no_grad():
            enc = modules["encoder"].encode(observation, fp_report=fp_report, ctx={"action_schema": action_schema})
            z_t = enc["z_t"]
            mem = modules["memory"].step(z_t, prev_action, prev_reward, prev_done, h_prev)
            h_t = mem["h_t"]
            h_core = h_t[0] if isinstance(h_t, tuple) else h_t
            if modules["controller"] is not None:
                ctrl = modules["controller"].forward(h_core, ctx={"is_train": stochastic})
                mode_id = int(ctrl["mode_id"].view(-1)[0].item())
                _ctrl_mode_logits = ctrl["mode_logits"]
            else:
                mode_id = 0
                _num_modes = max(1, int(self.cfg.get("num_modes", 4)))
                _ctrl_mode_logits = torch.zeros((1, _num_modes), dtype=torch.float32, device=h_core.device)
            action_ids = [str(a.get("action_id")) for a in action_schema.get("actions", []) if a.get("action_id")]
            actor = modules["actor"].forward(h_core, mode_id=mode_id, available_actions=action_ids, coord_candidates=[])
            logits = actor["pi_discrete"]
            if stochastic:
                probs = torch.softmax(logits, dim=1)
                idx = int(torch.multinomial(probs, 1).item())
            else:
                idx = int(torch.argmax(logits, dim=1).item())
            action_id = actor["action_ids"][idx]
            mode_logp_t, _ = modules["actor"].mode_logp_entropy(
                _ctrl_mode_logits,
                torch.tensor([mode_id], dtype=torch.long, device=logits.device),
            )
            mask_norm, _ = self._normalize_available_actions_mask(None, len(actor["action_ids"]), action_ids=list(actor["action_ids"]))
            _, action_logp_t, coord_logp_t_eff, _ = modules["actor"].compute_logp_components(
                h_core,
                {
                    "mode_id": mode_id,
                    "mode_logits": _ctrl_mode_logits,
                    "action_ids": actor["action_ids"],
                    "action_index": idx,
                    "available_actions_mask": [1 if x else 0 for x in mask_norm],
                    "coord_candidates": [],
                    "chosen_coord_index": -1,
                    "has_coord": False,
                },
                cfg=self.cfg,
                ctx=None,
            )
            logp_mode = float(mode_logp_t.detach().cpu().item()) if modules["controller"] is not None else 0.0
            logp_action_discrete = float(action_logp_t.detach().cpu().item())
            logp_coord = float(coord_logp_t_eff.detach().cpu().item()) if coord_logp_t_eff is not None else None
            logp_total = float(logp_mode + logp_action_discrete + (logp_coord or 0.0))
        return {
            "action": {"type": "simple", "action_id": action_id},
            "mode_id": mode_id,
            "h_t": h_t,
            "logp": {
                "logp_mode": logp_mode,
                "logp_action_discrete": logp_action_discrete,
                "logp_coord": logp_coord,
                "logp_total": logp_total,
            },
        }

    def run(self, env_factory: Any, args: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg_eff = {**self.cfg, **(cfg or {})}
        episodes_per_iter_resolved = int(getattr(args, "episodes_per_iter_resolved", cfg_eff.get("episodes_per_iter", 64)))
        max_steps_resolved = int(getattr(args, "max_steps_resolved", getattr(args, "max_actions", cfg_eff.get("max_steps_per_episode", 100))))
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        modules = self._build_modules()

        optim = build_optimizer(modules, cfg_eff)

        if getattr(args, "checkpoint", None):
            self._load_checkpoint(args.checkpoint, modules, optim if args.mode == "train" else None)

        if args.mode == "collect":
            batch = self.collector.collect(
                env_factory,
                modules,
                cfg={
                    "episodes_per_batch": args.episodes or episodes_per_iter_resolved,
                    "max_steps_per_episode": max_steps_resolved,
                    "stochastic_actions_train": bool(cfg_eff["stochastic_actions_train"]),
                    "coord_topK": int(cfg_eff.get("coord_topK", 16)),
                    "reward": cfg_eff.get("reward", {}),
                },
            )
            batch["policy_version"] = int(getattr(self, "policy_version", -1))
            for ep in batch.get("episodes", []):
                if isinstance(ep, dict):
                    ep["policy_version"] = int(getattr(self, "policy_version", -1))
            return {"batch": batch}

        if args.mode == "eval":
            batch = self.collector.collect(
                env_factory,
                modules,
                cfg={
                    "episodes_per_batch": args.episodes or cfg_eff["eval"]["episodes"],
                    "max_steps_per_episode": max_steps_resolved,
                    "stochastic_actions_train": False,
                    "coord_topK": int(cfg_eff.get("coord_topK", 16)),
                    "reward": cfg_eff.get("reward", {}),
                },
            )
            batch["policy_version"] = int(getattr(self, "policy_version", -1))
            for ep in batch.get("episodes", []):
                if isinstance(ep, dict):
                    ep["policy_version"] = int(getattr(self, "policy_version", -1))
            return {"batch": batch}

        num_iters = int(args.iters or cfg_eff["train"]["num_iters"])
        best_loss = float("inf")
        best_winrate = -1.0
        best_return = float("-inf")

        for iter_idx in range(num_iters):
            batch = self.collector.collect(
                env_factory,
                modules,
                cfg={
                    "episodes_per_batch": args.episodes or episodes_per_iter_resolved,
                    "max_steps_per_episode": max_steps_resolved,
                    "stochastic_actions_train": bool(cfg_eff["stochastic_actions_train"]),
                    "coord_topK": int(cfg_eff.get("coord_topK", 16)),
                    "reward": cfg_eff.get("reward", {}),
                },
            )
            report = self.trainer.train_step(batch, modules, optim, cfg=cfg_eff, ctx={"global_seed": int(getattr(args, "seed", 0)), "iter_idx": iter_idx})

            metrics_path = os.path.join(outdir, "metrics", f"train_iter_{iter_idx:06d}.json")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            last_ckpt = os.path.join(outdir, "checkpoints", "last.pt")
            self._save_checkpoint(last_ckpt, modules, optim, iter_idx)
            loss = float(report.get("losses", {}).get("total", float("inf")))
            if loss < best_loss:
                best_loss = loss
                self._save_checkpoint(os.path.join(outdir, "checkpoints", "best.pt"), modules, optim, iter_idx)
            win_rate = float(report.get("win_rate", 0.0))
            if win_rate > best_winrate:
                best_winrate = win_rate
                self._save_checkpoint(os.path.join(outdir, "checkpoints", "best_winrate.pt"), modules, optim, iter_idx)
            mean_return = float(report.get("mean_return", float("-inf")))
            if mean_return > best_return:
                best_return = mean_return
                self._save_checkpoint(os.path.join(outdir, "checkpoints", "best_return.pt"), modules, optim, iter_idx)

            if int(cfg_eff["eval"].get("every_iters", 0)) > 0 and (iter_idx + 1) % int(cfg_eff["eval"]["every_iters"]) == 0:
                _ = self.collector.collect(
                    env_factory,
                    modules,
                    cfg={
                        "episodes_per_batch": int(cfg_eff["eval"]["episodes"]),
                        "max_steps_per_episode": max_steps_resolved,
                        "stochastic_actions_train": False,
                        "coord_topK": int(cfg_eff.get("coord_topK", 16)),
                        "reward": cfg_eff.get("reward", {}),
                    },
                )

        return {"ok": True}

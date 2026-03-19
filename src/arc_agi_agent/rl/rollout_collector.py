from __future__ import annotations

import csv
import io
import logging
import os
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..action_schema import build_action_schema_from_env
from ..fp_analyst import FPAnalyst
from ..normalize import normalize_observation
from ..transition_event_compiler import compile_transition_event, to_json as event_to_json
from ..transition_event_compiler_config import TransitionEventCompilerConfig
from .coord_proposer import CoordProposer
from .module_control import ModuleDisabledError, module_enabled
from .obs_norm_v1 import normalize_obs_v1
from .canonical_grid import canonical_grid
from .reward_shaper import RewardShaper
from .intrinsic_rnd import RNDNormState

logger = logging.getLogger(__name__)

_REWARD_LOG_FIELDS = [
    "game_id", "ep", "step",
    "r_win", "r_effect", "r_revert", "r_potential", "r_step", "r_total",
    "m_noop", "flash", "effect_flag", "revert_flag", "cells_changed",
]


def _write_reward_log(path: str, game_id: str, ep: int, step: int, terms: Dict[str, Any], r_total: float) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # Truncate and write header only on the very first step of a run.
        first_step = ep == 0 and step == 0
        row = [
            game_id, ep, step,
            round(float(terms.get("r_win", 0.0)), 4),
            round(float(terms.get("r_effect", 0.0)), 4),
            round(float(terms.get("r_revert", 0.0)), 4),
            round(float(terms.get("r_potential", 0.0)), 4),
            round(float(terms.get("r_step", 0.0)), 4),
            round(r_total, 4),
            int(terms.get("m_noop", 1)),
            int(bool(terms.get("flash_event", False))),
            int(bool(terms.get("effect_flag", False))),
            int(bool(terms.get("revert_flag", False))),
            int(terms.get("cells_changed", 0)),
        ]
        buf = io.StringIO()
        w = csv.writer(buf)
        if first_step:
            w.writerow(_REWARD_LOG_FIELDS)
        w.writerow(row)
        mode = "w" if first_step else "a"
        with open(path, mode, encoding="utf-8", newline="") as f:
            f.write(buf.getvalue())
    except Exception:
        pass


def _default_cfg() -> Dict[str, Any]:
    return {
        "episodes_per_batch": 8,
        "max_steps_per_episode": 40,
        "stochastic_actions_train": True,
        "coord_topK": 16,
        "frame_stack": 4,
        "mode": "train",
        "rl": {"fast_collect": True, "save_full_batch": False},
    }


def _action_mask(meta: Dict[str, Any], action_ids: List[str]) -> List[int]:
    avail = meta.get("available_actions_sorted") or meta.get("available_actions") or []
    if not isinstance(avail, list):
        return [1] * len(action_ids)
    avail_set: set = set()
    for a in avail:
        s = str(a)
        avail_set.add(s)
        # env provides integer indices (e.g. [1,2,3,4]) — also add "ACTION{n}" form
        if s.isdigit():
            avail_set.add(f"ACTION{s}")
    return [1 if a in avail_set else 0 for a in action_ids]


def normalize_available_actions_mask(raw: Any, nd: int, action_id_to_idx: Dict[str, int]) -> List[bool]:
    if nd <= 0:
        return []
    if raw is None:
        return [True] * nd
    if isinstance(raw, list) and len(raw) == 0:
        return [True] * nd
    if isinstance(raw, list):
        if len(raw) == nd and all(isinstance(x, bool) for x in raw):
            out = [bool(x) for x in raw]
        elif len(raw) == nd and all(isinstance(x, int) and (x in (0, 1)) for x in raw):
            out = [bool(int(x)) for x in raw]
        elif all(isinstance(x, int) for x in raw) and ((len(raw) != nd) or any(int(x) >= 2 for x in raw)):
            out = [False] * nd
            for idx in raw:
                ii = int(idx)
                if 0 <= ii < nd:
                    out[ii] = True
        elif all(isinstance(x, str) for x in raw):
            out = [False] * nd
            for aid in raw:
                ai = action_id_to_idx.get(str(aid))
                if ai is not None and 0 <= ai < nd:
                    out[ai] = True
        else:
            logger.error("available_actions_mask_unrecognized type=%s len=%s; using all-valid", type(raw).__name__, len(raw))
            out = [True] * nd
    else:
        logger.error("available_actions_mask_unrecognized type=%s; using all-valid", type(raw).__name__)
        out = [True] * nd
    if sum(1 for x in out if x) == 0:
        out = [True] * nd
    return out


def _pick_discrete(logits: torch.Tensor, stochastic: bool) -> int:
    if stochastic:
        probs = torch.softmax(logits, dim=1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(torch.argmax(logits, dim=1).item())


def _entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=1)
    logp = torch.log_softmax(logits, dim=1)
    return float((-(probs * logp).sum(dim=1).mean()).item())


def _extract_done(meta: Dict[str, Any]) -> bool:
    if isinstance(meta.get("terminal"), bool):
        return bool(meta["terminal"])
    if isinstance(meta.get("done"), bool):
        return bool(meta["done"])
    state = str(meta.get("state", "")).upper()
    if state in {"WIN", "WON", "GAME_OVER", "LOST"}:
        return True
    # Handle enum-style values like "GAMESTATE.GAME_OVER"
    if state.endswith(".WIN") or state.endswith(".WON") or state.endswith(".GAME_OVER") or state.endswith(".LOST"):
        return True
    return False


def fp_report_minimal(fp_report: Any) -> Dict[str, Any]:
    if isinstance(fp_report, dict):
        report = fp_report
    else:
        report = {}
        state = getattr(fp_report, "state_summary", None)
        if state is not None:
            grid_summaries = []
            for gs in getattr(state, "grid_summaries", []) or []:
                grid_summaries.append(
                    {
                        "name": str(getattr(gs, "name", "")),
                        "height": int(getattr(gs, "height", 0)),
                        "width": int(getattr(gs, "width", 0)),
                    }
                )
            report["state_summary"] = {
                "step_idx": int(getattr(state, "step_idx", 0)),
                "grid_summaries": grid_summaries,
            }
        feats = getattr(fp_report, "features_v1", None)
        if feats is not None:
            report["features_v1"] = {
                "grid_index": getattr(feats, "grid_index", {}) or {},
                "object_index": getattr(feats, "object_index", []) or [],
                "interaction_points": getattr(feats, "interaction_points", []) or [],
                "meta_features": getattr(feats, "meta_features", {}) or {},
            }
        diff = getattr(fp_report, "diff_summary", None)
        if diff is not None:
            report["diff_summary"] = {
                "changed_cells_count": int(getattr(diff, "changed_cells_count", 0)),
                "changed_bbox": getattr(diff, "changed_bbox", None),
                "event_signatures": [
                    {"kind": str(getattr(es, "kind", "")), "confidence": float(getattr(es, "confidence", 0.0))}
                    for es in (getattr(diff, "event_signatures", []) or [])
                ],
                "per_object_deltas": [
                    {
                        "object_id": str(getattr(d, "object_id", "")),
                        "event": str(getattr(d, "event", "")),
                        "dy": float(getattr(d, "dy", 0.0)),
                        "dx": float(getattr(d, "dx", 0.0)),
                    }
                    for d in (getattr(diff, "per_object_deltas", []) or [])
                ],
            }
        debug = getattr(fp_report, "debug", None)
        if debug is not None:
            report["debug"] = {"grid_hash": str(getattr(debug, "grid_hash", ""))}

    # Strip viz payloads in RL hot path.
    report.pop("viz_artifacts", None)
    return report


def _fp_grid_dims(fp_report: Dict[str, Any], fallback_h: int = 64, fallback_w: int = 64) -> Tuple[int, int]:
    feats = fp_report.get("features_v1", {})
    if isinstance(feats, dict):
        gi = feats.get("grid_index", {})
        if isinstance(gi, dict):
            grids = gi.get("grids")
            if isinstance(grids, list) and grids:
                g0 = grids[0]
                return int(g0.get("height", fallback_h)), int(g0.get("width", fallback_w))
    state = fp_report.get("state_summary", {})
    if isinstance(state, dict):
        gs = state.get("grid_summaries")
        if isinstance(gs, list) and gs:
            g0 = gs[0]
            return int(g0.get("height", fallback_h)), int(g0.get("width", fallback_w))
    return fallback_h, fallback_w


class RolloutCollector:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        self.analyst = FPAnalyst() if module_enabled(self.cfg, "fp_analyst") else None
        self.coord_proposer = CoordProposer() if module_enabled(self.cfg, "rl_coord_proposer") else None
        self.reward_shaper = RewardShaper() if module_enabled(self.cfg, "rl_reward_shaper") else None
        self.event_cfg = TransitionEventCompilerConfig() if module_enabled(self.cfg, "transition_event") else None

    def collect(
        self,
        env_factory: Any,
        modules: Dict[str, Any],
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = {**_default_cfg(), **(cfg or {})}
        cfg_eff = {
            **cfg_eff,
            "modules": self.cfg.get("modules", {}),
            "pipeline": self.cfg.get("pipeline", {}),
            "log": self.cfg.get("log", {}),
            "rl": {**_default_cfg()["rl"], **(self.cfg.get("rl", {}) if isinstance(self.cfg.get("rl"), dict) else {}), **(cfg_eff.get("rl", {}) if isinstance(cfg_eff.get("rl"), dict) else {})},
        }
        if "rl" not in cfg_eff:
            cfg_eff["rl"] = {}
        rl_only_mode = str(self.cfg.get("pipeline", {}).get("mode", "")).lower() == "rl_only"
        if rl_only_mode:
            cfg_eff["rl"]["fast_collect"] = bool(cfg_eff["rl"].get("fast_collect", True))
        else:
            cfg_eff["rl"]["fast_collect"] = bool(cfg_eff["rl"].get("fast_collect", False))
        episodes = int(cfg_eff["episodes_per_batch"])
        max_steps = int(cfg_eff["max_steps_per_episode"])
        stochastic = bool(cfg_eff.get("stochastic_actions_train", True))
        fast_collect = bool(cfg_eff.get("rl", {}).get("fast_collect", False))
        trace_enabled = bool(cfg_eff.get("log", {}).get("write_trace", False))
        save_full_batch = bool(cfg_eff.get("rl", {}).get("save_full_batch", False))
        frame_stack_len = max(1, int(cfg_eff.get("frame_stack", 4)))
        minimal_batch_mode = bool(rl_only_mode and (not trace_enabled) and (not save_full_batch))

        batch = {"schema_version": "TRAJECTORY_BATCH_V1", "episodes": [], "available_actions_mask_format": "bool_nd"}

        for ep_idx in range(episodes):
            env, game_id, seed = env_factory(ep_idx)
            obs = env.reset()
            fp_prev = None
            if self.analyst is None:
                raise ModuleDisabledError("fp_analyst")
            if fast_collect:
                    fp_curr = fp_report_minimal(
                        self.analyst.analyze_fast(
                            obs,
                            cfg={"pipeline": self.cfg.get("pipeline", {}), "fp_analyst": self.cfg.get("fp_analyst", {})},
                            ctx={"pipeline": self.cfg.get("pipeline", {})},
                        )
                    )
            else:
                fp_curr = fp_report_minimal(
                    self.analyst.analyze(
                        obs,
                        prev_observation=None,
                        action_taken=None,
                        ctx={"pipeline": self.cfg.get("pipeline", {}), "fp_analyst": self.cfg.get("fp_analyst", {})},
                    )
                )

            h_t = None
            # _h_norm = h_t.norm().item() if isinstance(h_t, torch.Tensor) else 0.0
            # logger.info("episode_start game_id=%s ep_idx=%s h_t_norm=%.6f", game_id, ep_idx, _h_norm)
            prev_action = None
            prev_reward = 0.0
            prev_done = False
            done = False
            visit_counts: Dict[str, int] = {}
            state_hash_t_minus_2: Optional[str] = None
            state_hash_prev: Optional[str] = None
            rnd_norm_state: Optional[RNDNormState] = None
            prev_rnd_phi = 0.0
            episode_ctx: Dict[str, Any] = {"episode_intrinsic_sum": 0.0}
            steps: List[Dict[str, Any]] = []
            obs_norm_curr = normalize_obs_v1(obs, fp_report=fp_curr)
            grid_curr = canonical_grid(obs_norm_curr)
            grid_prev_prev: Optional[np.ndarray] = None
            frame_buffer: Deque[Any] = deque(maxlen=frame_stack_len)
            frame_buffer.append(grid_curr.copy())

            height, width = _fp_grid_dims(fp_curr)
            action_schema = build_action_schema_from_env(env.action_space, width=width, height=height)
            action_ids = [str(a.action_id) for a in action_schema.actions]
            logger.debug("game_id,step,action,reward,changed_pixels_masked_count,changed_cells_raw,win")
            intrinsic_cfg = cfg_eff.get("intrinsic", {}) if isinstance(cfg_eff.get("intrinsic", {}), dict) else {}
            intrinsic_enabled = bool(intrinsic_cfg.get("enabled", False)) and str(intrinsic_cfg.get("method", "")) == "rnd_grid_embed"
            intrinsic_enabled = intrinsic_enabled and ("intrinsic_rnd" in modules)
            if intrinsic_enabled and rnd_norm_state is None:
                rnd_norm_state = RNDNormState()

            for step_idx in range(max_steps):
                grid_embed_vec: Optional[list[float]] = None
                rnd_err_raw = 0.0
                rnd_phi = 0.0
                intrinsic_terms: Dict[str, Any] = {}
                grid_stack_list = list(frame_buffer)
                while len(grid_stack_list) < frame_stack_len:
                    grid_stack_list.insert(0, grid_stack_list[0].copy())
                with torch.no_grad():
                    enc = modules["encoder"].encode(
                        obs,
                        fp_report=fp_curr,
                        ctx={
                            "action_schema": {"actions": [{"action_id": a} for a in action_ids]},
                            "obs_norm": obs_norm_curr,
                            "grid_stack": grid_stack_list,
                        },
                    )
                    z_t = enc["z_t"]
                    mem = modules["memory"].step(z_t, prev_action, prev_reward, prev_done, h_t)
                    h_t = mem["h_t"]
                    h_core = h_t[0] if isinstance(h_t, tuple) else h_t

                    if modules["controller"] is not None:
                        ctrl = modules["controller"].forward(h_core, ctx={"is_train": stochastic})
                        mode_id = int(ctrl["mode_id"].view(-1)[0].item())
                        mode_logits = ctrl["mode_logits"]
                    else:
                        mode_id = 0
                        _num_modes = max(1, int(cfg_eff.get("num_modes", 4)))
                        mode_logits = torch.zeros((1, _num_modes), dtype=torch.float32, device=h_core.device)

                    if intrinsic_enabled and isinstance(enc.get("grid_embed"), torch.Tensor):
                        grid_embed_t = enc["grid_embed"].detach()
                        grid_embed_vec = grid_embed_t.view(-1).to(dtype=torch.float16).cpu().tolist()
                        with torch.no_grad():
                            _, _, _, err_scalar = modules["intrinsic_rnd"](grid_embed_t)
                        rnd_err_raw = float(err_scalar.view(-1)[0].item())
                        rnd_phi = float(
                            modules["intrinsic_rnd"].compute_phi(
                                rnd_err_raw,
                                rnd_norm_state,
                                float(intrinsic_cfg.get("rnd_phi_clip", 5.0)),
                            )
                        )
                        intrinsic_terms = {"rnd_err_raw": rnd_err_raw, "rnd_phi": rnd_phi}

                    if self.coord_proposer is None:
                        raise ModuleDisabledError("rl_coord_proposer")
                    cand_out = self.coord_proposer.propose(fp_curr, fp_prev, cfg={"coord_topK": int(cfg_eff.get("coord_topK", 16))})
                    coords = cand_out.get("coords", [])

                    grid0 = None
                    if isinstance(enc.get("obs_norm"), dict):
                        grids = enc["obs_norm"].get("grids", [])
                        if isinstance(grids, list) and grids:
                            grid0 = grids[0].get("grid") if isinstance(grids[0], dict) else None
                    actor = modules["actor"].forward(
                        h_core,
                        mode_id,
                        available_actions=action_ids,
                        coord_candidates=coords,
                        ctx={"grid": grid0, "fp_report": fp_curr},
                    )
                    feat_vecs = actor.get("coord_feature_vectors") if isinstance(actor, dict) else None
                    if isinstance(feat_vecs, list) and len(feat_vecs) == len(coords):
                        coords = [{**c, "feat_vec": feat_vecs[i]} for i, c in enumerate(coords)]

                    # Apply env available-actions mask to logits before selection
                    mask_raw = _action_mask(enc["obs_norm"]["meta"], actor["action_ids"])
                    aid_to_idx = {str(aid): i for i, aid in enumerate(actor["action_ids"])}
                    available_actions_mask = normalize_available_actions_mask(mask_raw, len(actor["action_ids"]), aid_to_idx)
                    env_mask_t = torch.tensor(available_actions_mask, dtype=torch.bool, device=h_core.device).unsqueeze(0)
                    pi_discrete_masked = actor["pi_discrete"] + (env_mask_t.float() - 1.0) * 1e9

                    action_idx = _pick_discrete(pi_discrete_masked, stochastic=stochastic)
                    action_id = actor["action_ids"][action_idx]
                    policy_entropy = _entropy_from_logits(pi_discrete_masked)
                    mode_entropy = _entropy_from_logits(mode_logits)
                    old_value = float(modules["value"].forward(h_core).view(-1)[0].item())

                    chosen_coord_index = None
                    chosen_coord_tag = None
                    old_logp_coord = 0.0
                    mask_has_coord = 0
                    action = {"type": "simple", "action_id": action_id}
                    if action_id.upper() == "ACTION6" and coords and actor.get("pi_coord") is not None:
                        chosen_coord_index = _pick_discrete(actor["pi_coord"], stochastic=stochastic)
                        c = coords[chosen_coord_index]
                        chosen_coord_tag = c.get("tag")
                        action = {"type": "coord", "action_id": "ACTION6", "x": int(c["x"]), "y": int(c["y"]) }
                        mask_has_coord = 1
                    old_logp_mode_t, old_logp_action_t, old_logp_coord_t, old_logp_total_t = modules["actor"].compute_logp_components(
                        h_core,
                        {
                            "mode_id": mode_id,
                            "mode_logits": mode_logits,
                            "action_ids": actor["action_ids"],
                            "action_index": action_idx,
                            "available_actions_mask": available_actions_mask,
                            "coord_candidates": coords,
                            "chosen_coord_index": int(chosen_coord_index) if chosen_coord_index is not None else -1,
                            "has_coord": bool(mask_has_coord == 1),
                        },
                        cfg=cfg_eff,
                        ctx={"grid": grid0, "fp_report": fp_curr},
                    )
                    old_logp_coord = float(old_logp_coord_t.detach().cpu().item())
                    old_mode_entropy_t = torch.tensor([mode_entropy], dtype=torch.float32, device=mode_logits.device)
                    old_action_entropy_t = torch.tensor([policy_entropy], dtype=torch.float32, device=mode_logits.device)
                from arcengine import GameAction

                action_obj = GameAction.from_name(action["action_id"])
                if action.get("type") == "coord":
                    obs_next = env.step(action_obj, data={"x": action.get("x"), "y": action.get("y")})
                else:
                    obs_next = env.step(action_obj)

                if fast_collect:
                    fp_next = fp_report_minimal(
                        self.analyst.analyze_fast(
                            obs_next,
                            cfg={"pipeline": self.cfg.get("pipeline", {}), "fp_analyst": self.cfg.get("fp_analyst", {})},
                            ctx={"pipeline": self.cfg.get("pipeline", {})},
                        )
                    )
                else:
                    fp_next = fp_report_minimal(
                        self.analyst.analyze(
                            obs_next,
                            prev_observation=obs,
                            action_taken=action,
                            ctx={"pipeline": self.cfg.get("pipeline", {}), "fp_analyst": self.cfg.get("fp_analyst", {})},
                        )
                    )
                if self.event_cfg is None:
                    raise ModuleDisabledError("transition_event")
                norm_next = normalize_observation(obs_next, schema_warnings=[])
                obs_norm_next = normalize_obs_v1(obs_next, fp_report=fp_next)
                grid_next = canonical_grid(obs_norm_next)
                effect_transition: Dict[str, float] = {"effect_changed_cells_masked": 0.0, "changed_cells_masked_count": 0.0, "changed_cells_raw": 0.0, "H": 0.0, "W": 0.0, "den": 1.0}
                if self.reward_shaper is not None:
                    try:
                        effect_transition = self.reward_shaper.effect_from_transition(
                            game_id,
                            np.asarray(grid_curr, dtype=np.int64),
                            np.asarray(grid_next, dtype=np.int64),
                            {"use_hud_mask": False},
                        )
                    except Exception:
                        effect_transition = {"effect_changed_cells_masked": 0.0, "changed_cells_masked_count": 0.0, "changed_cells_raw": 0.0, "H": 0.0, "W": 0.0, "den": 1.0}
                compiler_ctx = {
                    "game_id": game_id,
                    "seed": seed,
                    "step_idx": step_idx,
                    "pipeline": self.cfg.get("pipeline", {}),
                    "transition_event": self.cfg.get("transition_event", {}),
                    "done": _extract_done(norm_next.meta or {}),
                    "win": str((norm_next.meta or {}).get("state", "")).upper() in {"WIN", "WON", "SUCCESS"},
                }
                compile_kwargs: Dict[str, Any] = {}
                if rl_only_mode:
                    compile_kwargs = {
                        "prev_grid_norm": grid0,
                        "next_grid_norm": norm_next.grids[0] if getattr(norm_next, "grids", None) else None,
                        "prev_meta_norm": enc["obs_norm"].get("meta", {}) if isinstance(enc.get("obs_norm"), dict) else {},
                        "next_meta_norm": norm_next.meta if isinstance(getattr(norm_next, "meta", None), dict) else {},
                    }
                event = compile_transition_event(
                    obs,
                    obs_next,
                    action,
                    fp_prev_report=fp_curr,
                    fp_curr_report=fp_next,
                    ctx=compiler_ctx,
                    cfg=self.event_cfg,
                    **compile_kwargs,
                )
                event_json = event_to_json(event)
                done = bool(compiler_ctx["done"])
                win = bool(compiler_ctx["win"])
                state_label = str((norm_next.meta or {}).get("state", "")).upper()

                if self.reward_shaper is None:
                    raise ModuleDisabledError("rl_reward_shaper")
                reward = self.reward_shaper.compute(
                    event_json,
                    done,
                    win,
                    t=step_idx,
                    visit_counts=visit_counts,
                    state_hash_t_minus_2=state_hash_t_minus_2,
                    cfg=cfg_eff.get("reward"),
                    ctx={
                        "grid_prev_prev": grid_prev_prev,
                        "grid_prev": grid_curr,
                        "grid_curr": grid_next,
                        "game_id": game_id,
                        "effect_transition": effect_transition,
                    },
                )
                prev_rnd_phi = rnd_phi
                reward_total = float(reward["r_total"])
                reward_terms = reward.get("terms", {}) if isinstance(reward.get("terms"), dict) else {}
                raw_env_reward = None
                if isinstance(event_json, dict):
                    raw_env_reward = event_json.get("reward")
                    if raw_env_reward is None:
                        raw_env_reward = event_json.get("env_reward")
                if reward_total < -1.5 or reward_total > 2.0:
                    logger.error(
                        "reward_out_of_range game_id=%s step=%s action_id=%s flash_event=%s env_reward=%s r_total=%.6f terms=%s",
                        game_id,
                        step_idx,
                        action_id,
                        bool(reward_terms.get("flash_event", False)),
                        raw_env_reward,
                        reward_total,
                        reward_terms,
                    )
                    raise RuntimeError("reward_total out of expected range")
                logger.info(
                    "reward_step game_id=%s step=%s action_id=%s r_total=%.4f r_win=%.4f r_effect=%.4f r_match_poi=%.4f r_revert=%.4f r_potential=%.4f m_noop=%s flash=%s delta_c=%s",
                    game_id,
                    step_idx,
                    action_id,
                    reward_total,
                    float(reward_terms.get("r_win", 0.0)),
                    float(reward_terms.get("r_effect", 0.0)),
                    float(reward_terms.get("r_match_poi", 0.0)),
                    float(reward_terms.get("r_revert", 0.0)),
                    float(reward_terms.get("r_potential", 0.0)),
                    int(reward_terms.get("m_noop", 1)),
                    int(bool(reward_terms.get("flash_event", False))),
                    int(reward_terms.get("delta_c", 0)),
                )
                debug_reward_log = cfg_eff.get("debug_reward_log")
                if debug_reward_log:
                    _write_reward_log(str(debug_reward_log), game_id, ep_idx, step_idx, reward_terms, reward_total)
                action_log = (
                    f"ACTION6({int(action.get('x')) if action.get('x') is not None else 0},{int(action.get('y')) if action.get('y') is not None else 0})"
                    if action.get("type") == "coord" and chosen_coord_index is not None
                    else str(action_id)
                )
                logger.debug(
                    "%s,%s,%s,%s,%s,%s,%s",
                    game_id,
                    step_idx,
                    action_log,
                    reward_total,
                    int(effect_transition.get("changed_cells_masked_count", 0.0)),
                    int(effect_transition.get("changed_cells_raw", 0.0)),
                    int(bool(win)),
                )
                if state_label in {"GAME_OVER", "WIN"}:
                    logger.debug("game_end,%s,%s,%s", game_id, step_idx, state_label)

                current_hash = reward_terms.get("state_hash") or ""
                state_hash_t_minus_2 = state_hash_prev
                state_hash_prev = current_hash or None

                # if not stochastic and step_idx < 10:
                #     _probs = torch.softmax(pi_discrete_masked.view(-1), dim=0).detach().cpu()
                #     _top5_vals, _top5_idx = torch.topk(_probs, min(5, len(_probs)))
                #     _top5 = [(actor["action_ids"][int(i)], round(float(v), 4)) for i, v in zip(_top5_idx.tolist(), _top5_vals.tolist())]
                #     _mask_size = int(sum(1 for m in available_actions_mask if m))
                #     _sh_before = event_json.get("state_hash_before_filtered") or event_json.get("state_hash_before", "?")
                #     _sh_after = event_json.get("state_hash_after_filtered") or event_json.get("state_hash_after", "?")
                #     logger.info(
                #         "eval_step_diag game_id=%s ep=%s step=%s chosen=%s entropy=%.4f mask_size=%s/%s "
                #         "top5=%s hash_before=%s hash_after=%s",
                #         game_id, ep_idx, step_idx, action_id,
                #         policy_entropy, _mask_size, len(available_actions_mask),
                #         _top5,
                #         str(_sh_before)[-8:] if _sh_before and _sh_before != "?" else "?",
                #         str(_sh_after)[-8:] if _sh_after and _sh_after != "?" else "?",
                #     )

                step = {
                    "step_idx": step_idx,
                    # Keep rollout payload fully serializable so multiprocessing
                    # does not pass tensor storages via file descriptors.
                    "h_t": h_core.detach().cpu().tolist(),
                    "mode_id": mode_id,
                    "action_ids": actor["action_ids"],
                    "action_index": action_idx,
                    "action_key": action,
                    "coord_candidates": [{"x": int(c.get("x", 0)), "y": int(c.get("y", 0)), "tag": c.get("tag"), "feat_vec": c.get("feat_vec")} for c in coords],
                    "chosen_coord_index": chosen_coord_index,
                    "coord_tag": chosen_coord_tag,
                    "old_logp_mode": float(old_logp_mode_t.detach().cpu().item()),
                    "old_logp_action_discrete": float(old_logp_action_t.detach().cpu().item()),
                    "old_logp_action": float(old_logp_action_t.detach().cpu().item()),
                    "old_logp_coord": float(old_logp_coord),
                    "old_logp_total": float(old_logp_total_t.detach().cpu().item()),
                    "value_pred": float(old_value),
                    "old_value": float(old_value),
                    "mask_valid_step": 1,
                    "mask_has_coord": int(mask_has_coord),
                    "has_coord": bool(mask_has_coord == 1),
                    "reward_total": reward_total,
                    "reward": reward_total,
                    "reward_terms": reward.get("terms", {}),
                    "reward_aux": reward.get("aux", {}),
                    "intrinsic_terms": intrinsic_terms,
                    "done": done,
                    "win": bool(win),
                    "available_actions_mask": available_actions_mask,
                    "grid_stack_t": [g.tolist() for g in grid_stack_list],
                    "state_hash_before_filtered": event_json.get("state_hash_before_filtered") or event_json.get("state_hash_before"),
                    "state_hash_after_filtered": event_json.get("state_hash_after_filtered") or event_json.get("state_hash_after"),
                    "state_hash_before": event_json.get("state_hash_before_filtered") or event_json.get("state_hash_before"),
                    "state_hash_after": event_json.get("state_hash_after_filtered") or event_json.get("state_hash_after"),
                    "transition_event": {
                        "state_hash_before": event_json.get("state_hash_before"),
                        "state_hash_after": event_json.get("state_hash_after"),
                        "state_hash_before_filtered": event_json.get("state_hash_before_filtered"),
                        "state_hash_after_filtered": event_json.get("state_hash_after_filtered"),
                        "grid_delta": event_json.get("grid_delta"),
                        "frame_policy": event_json.get("frame_policy"),
                    },
                }
                if grid_embed_vec is not None:
                    step["grid_embed"] = grid_embed_vec
                terms = step.get("reward_terms", {}) if isinstance(step.get("reward_terms"), dict) else {}
                step["effect_flag_filtered"] = bool(terms.get("effect_flag_filtered", terms.get("effect_flag_raw", False)))
                step["novel_flag_filtered"] = bool(terms.get("novel_flag_filtered", False))
                step["repeat_flag_filtered"] = bool(terms.get("repeat_flag_filtered", False))
                if not minimal_batch_mode:
                    step["z_t"] = z_t.detach().cpu().tolist()
                    step["mode_logits"] = mode_logits.detach().cpu().tolist()
                    step["mode_entropy"] = mode_entropy
                    step["pi_discrete_logits"] = actor["pi_discrete"].detach().cpu().tolist()
                    step["policy_entropy"] = policy_entropy
                    step["chosen_coord_tag"] = chosen_coord_tag
                    step["old_mode_entropy"] = float(old_mode_entropy_t.detach().cpu().item())
                    step["old_action_entropy"] = float(old_action_entropy_t.detach().cpu().item())
                    step["state_hash_before"] = event_json.get("state_hash_before")
                    step["state_hash_after"] = event_json.get("state_hash_after")
                steps.append(step)

                obs = obs_next
                obs_norm_curr = obs_norm_next
                fp_prev = fp_curr
                fp_curr = fp_next
                grid_prev_prev = np.asarray(grid_curr, dtype=np.int64).copy()
                grid_curr = grid_next
                frame_buffer.append(grid_next.copy())
                prev_action = action
                prev_reward = reward_total
                prev_done = done

                if done:
                    break

            if steps:
                r_win_sum = 0.0
                r_effect_sum = 0.0
                r_match_poi_sum = 0.0
                r_revert_sum = 0.0
                r_potential_sum = 0.0
                r_total_sum = 0.0
                for s in steps:
                    terms = s.get("reward_terms", {}) if isinstance(s.get("reward_terms"), dict) else {}
                    r_win_sum += float(terms.get("r_win", 0.0))
                    r_effect_sum += float(terms.get("r_effect", 0.0))
                    r_match_poi_sum += float(terms.get("r_match_poi", 0.0))
                    r_revert_sum += float(terms.get("r_revert", 0.0))
                    r_potential_sum += float(terms.get("r_potential", 0.0))
                    r_total_sum += float(s.get("reward", 0.0))
                logger.info(
                    "reward_breakdown game_id=%s seed=%s steps=%s r_total=%.4f r_win=%.4f r_effect=%.4f r_match_poi=%.4f r_revert=%.4f r_potential=%.4f",
                    game_id,
                    seed,
                    len(steps),
                    r_total_sum,
                    r_win_sum,
                    r_effect_sum,
                    r_match_poi_sum,
                    r_revert_sum,
                    r_potential_sum,
                )
            batch["episodes"].append(
                {
                    "game_id": game_id,
                    "seed": seed,
                    "steps": steps,
                    "done": bool(steps and steps[-1].get("done", False)),
                    "win": bool(steps and any(s.get("win", False) for s in steps)),
                    "num_steps": len(steps),
                }
            )

        return batch

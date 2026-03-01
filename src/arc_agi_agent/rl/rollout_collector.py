from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
    avail_set = set(str(a) for a in avail) if isinstance(avail, list) else set()
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
        collect_mode = str(cfg_eff.get("mode", "train")).lower()
        minimal_batch_mode = bool(rl_only_mode and (not trace_enabled) and (not save_full_batch))
        hud_probe_acc = (ctx or {}).get("hud_probe_accumulator")

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
            prev_action = None
            prev_reward = 0.0
            prev_done = False
            done = False
            seen_hashes: set[str] = set()
            recent_hashes: List[str] = []
            steps: List[Dict[str, Any]] = []
            obs_norm_curr = normalize_obs_v1(obs, fp_report=fp_curr)
            grid_curr = canonical_grid(obs_norm_curr)
            frame_buffer: Deque[Any] = deque(maxlen=frame_stack_len)
            frame_buffer.append(grid_curr.copy())

            height, width = _fp_grid_dims(fp_curr)
            action_schema = build_action_schema_from_env(env.action_space, width=width, height=height)
            action_ids = [str(a.action_id) for a in action_schema.actions]
            logger.debug("game_id,step,action,reward,changed_pixels_masked_count,changed_cells_raw,win")

            for step_idx in range(max_steps):
                grid_stack_list = list(frame_buffer)
                while len(grid_stack_list) < frame_stack_len:
                    grid_stack_list.insert(0, grid_stack_list[0].copy())
                if collect_mode == "probe":
                    mode_id = 0
                    mode_logits = torch.zeros((1, max(1, len(cfg_eff.get("modes", [])))), dtype=torch.float32)
                    coords = []
                    grid0 = grid_stack_list[-1].tolist() if grid_stack_list else None
                    avail = obs_norm_curr.get("meta", {}).get("available_actions_sorted", [])
                    cand_ids: List[str] = []
                    if isinstance(avail, list) and avail:
                        for a in avail:
                            if isinstance(a, int):
                                if 0 <= int(a) < len(action_ids):
                                    cand_ids.append(str(action_ids[int(a)]))
                                continue
                            s = str(a)
                            if s in action_ids:
                                cand_ids.append(s)
                                continue
                            if s.isdigit():
                                ii = int(s)
                                if 0 <= ii < len(action_ids):
                                    cand_ids.append(str(action_ids[ii]))
                    if not cand_ids:
                        cand_ids = list(action_ids)
                    non_coord = [a for a in cand_ids if str(a).upper() != "ACTION6"]
                    action_id = random.choice(non_coord if non_coord else cand_ids)
                    action_idx = action_ids.index(action_id) if action_id in action_ids else 0
                    policy_entropy = 0.0
                    mode_entropy = 0.0
                    old_value = 0.0
                    chosen_coord_index = None
                    chosen_coord_tag = None
                    old_logp_coord = 0.0
                    mask_has_coord = 0
                    action = {"type": "simple", "action_id": action_id}
                    available_actions_mask = [True] * len(action_ids)
                    old_logp_mode_t = torch.tensor([0.0], dtype=torch.float32)
                    old_logp_action_t = torch.tensor([0.0], dtype=torch.float32)
                    old_logp_total_t = torch.tensor([0.0], dtype=torch.float32)
                    old_mode_entropy_t = torch.tensor([0.0], dtype=torch.float32)
                    old_action_entropy_t = torch.tensor([0.0], dtype=torch.float32)
                    h_core = torch.zeros((1, 1), dtype=torch.float32)
                    z_t = h_core
                    enc = {"obs_norm": obs_norm_curr}
                else:
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

                        ctrl = modules["controller"].forward(h_core, ctx={"is_train": stochastic})
                        mode_id_t = ctrl["mode_id"]
                        mode_id = int(mode_id_t.view(-1)[0].item())
                        mode_logits = ctrl["mode_logits"]

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
                        action_idx = _pick_discrete(actor["pi_discrete"], stochastic=stochastic)
                        action_id = actor["action_ids"][action_idx]
                        policy_entropy = _entropy_from_logits(actor["pi_discrete"])
                        mode_entropy = _entropy_from_logits(ctrl["mode_logits"])
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

                        mask_raw = _action_mask(enc["obs_norm"]["meta"], actor["action_ids"])
                        aid_to_idx = {str(aid): i for i, aid in enumerate(actor["action_ids"])}
                        available_actions_mask = normalize_available_actions_mask(mask_raw, len(actor["action_ids"]), aid_to_idx)
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
                            {
                                "hud_specs": cfg_eff.get("hud_specs", {}),
                                "use_hud_mask": bool(collect_mode != "probe"),
                            },
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
                    seen_hashes,
                    recent_hashes,
                    cfg=cfg_eff.get("reward"),
                    ctx={
                        "grid_prev": grid_curr,
                        "grid_curr": grid_next,
                        "game_id": game_id,
                        "hud_specs": cfg_eff.get("hud_specs", {}),
                        "effect_transition": effect_transition,
                    },
                )
                if collect_mode == "probe" and hud_probe_acc is not None:
                    try:
                        hud_probe_acc.observe(game_id, grid_curr, grid_next)
                    except Exception:
                        pass
                reward_total = float(reward["r_total"])
                reward_terms = reward.get("terms", {}) if isinstance(reward.get("terms"), dict) else {}
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

                hash_for_visit = reward_terms.get("state_hash")
                if not hash_for_visit:
                    after_hash = event_json.get("state_hash_after")
                    after_hash_filtered = event_json.get("state_hash_after_filtered")
                    hash_for_visit = after_hash_filtered or after_hash
                if hash_for_visit:
                    seen_hashes.add(hash_for_visit)
                    recent_hashes.append(hash_for_visit)
                    loop_n = int(cfg_eff.get("reward", {}).get("loop_window_N", 25)) if isinstance(cfg_eff.get("reward"), dict) else 25
                    if len(recent_hashes) > loop_n:
                        recent_hashes.pop(0)

                if collect_mode == "probe":
                    obs = obs_next
                    obs_norm_curr = obs_norm_next
                    fp_prev = fp_curr
                    fp_curr = fp_next
                    grid_curr = grid_next
                    frame_buffer.append(grid_next.copy())
                    prev_action = action
                    prev_reward = reward_total
                    prev_done = done
                    if done:
                        break
                    continue

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
                terms = step.get("reward_terms", {}) if isinstance(step.get("reward_terms"), dict) else {}
                step["effect_flag_filtered"] = bool(terms.get("effect_flag_filtered", terms.get("effect_flag_raw", False)))
                step["novel_flag_filtered"] = bool(terms.get("novel_flag_filtered", False))
                step["repeat_flag_filtered"] = bool(terms.get("repeat_flag_filtered", False))
                if not minimal_batch_mode:
                    step["z_t"] = z_t.detach().cpu().tolist()
                    step["mode_logits"] = ctrl["mode_logits"].detach().cpu().tolist()
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
                grid_curr = grid_next
                frame_buffer.append(grid_next.copy())
                prev_action = action
                prev_reward = reward_total
                prev_done = done

                if done:
                    break

            batch["episodes"].append(
                {
                    "game_id": game_id,
                    "seed": seed,
                    "steps": steps,
                    "done": bool(done if collect_mode == "probe" else (steps and steps[-1].get("done", False))),
                    "win": bool(steps and any(s.get("win", False) for s in steps)),
                    "num_steps": len(steps),
                }
            )

        return batch

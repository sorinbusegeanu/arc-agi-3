from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

import torch
from ..config import RLConfig
from ..hud_probe_accumulator import HudProbeAccumulator
from .coverage_ledger import CoverageLedgerV1
from .module_control import apply_rl_only_mode, assert_no_non_rl_trace_entry, assert_rl_only_guards, configure_rl_only_logging
from .optim import build_optimizer
from .trainer import _rollout_cfg_hash

logger = logging.getLogger(__name__)


def _prepare_paths() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    arc_agi_path = os.path.join(base_dir, "other_repos", "arc-agi")
    arcengine_path = os.path.join(base_dir, "other_repos", "ARCEngine")
    arcengine_pkg = os.path.join(arcengine_path, "arcengine")
    for p in (arc_agi_path, arcengine_path, arcengine_pkg):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "ENVIRONMENTS_DIR" not in os.environ:
        os.environ["ENVIRONMENTS_DIR"] = os.path.join(base_dir, "environment_files")


def _quiet_arcade_logger() -> logging.Logger:
    lg = logging.getLogger("arc_agi_agent.rl.arcade_quiet")
    lg.setLevel(logging.ERROR)
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    return lg


def _setup_debug_log(path: str) -> None:
    log_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    root = logging.getLogger()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(fh)
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)


def _setup_stdout_log_copy(log_path: str) -> None:
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    root = logging.getLogger()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(fh)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _configure_torch_mp_sharing() -> None:
    # Avoid FD exhaustion ("Too many open files") when tensors are passed
    # through multiprocessing queues.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception as exc:
        logger.warning("torch_mp_sharing_strategy_not_set error=%s", exc)


def _resolve_games(selector: str, op_mode: str) -> List[str]:
    if os.path.isfile(selector):
        with open(selector, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if "," in selector:
        return [g.strip() for g in selector.split(",") if g.strip()]
    if selector.strip().lower() == "all":
        from arc_agi import Arcade, OperationMode

        arcade = Arcade(operation_mode=OperationMode(op_mode), logger=_quiet_arcade_logger())
        game_ids: List[str] = []
        seen = set()
        for env_info in arcade.get_environments():
            base_id = env_info.game_id.split("-", 1)[0]
            if base_id not in seen:
                seen.add(base_id)
                game_ids.append(base_id)
        return game_ids
    return [selector]


def _build_env_factory(games: List[str], seed_base: int, op_mode: str, render_terminal: bool = False):
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(operation_mode=OperationMode(op_mode), logger=_quiet_arcade_logger())

    def factory(ep_idx: int):
        game_id = games[ep_idx % len(games)]
        env_seed = int(seed_base) + int(ep_idx)
        env = arcade.make(game_id, seed=env_seed, render_mode=("terminal" if render_terminal else None))
        return env, game_id, env_seed

    return factory


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _state_dict_cpu(sd: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out



def _split_counts(total: int, workers: int) -> List[int]:
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _sample_noncanonical_masks(batch: Dict[str, Any], sample_rows: int = 32) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = 0
    for ep in batch.get("episodes", []):
        if not isinstance(ep, dict):
            continue
        for s in ep.get("steps", []):
            if not isinstance(s, dict):
                continue
            action_ids = s.get("action_ids") or []
            nd = len(action_ids) if isinstance(action_ids, list) else 0
            raw = s.get("available_actions_mask", None)
            ok = isinstance(raw, list) and len(raw) == nd and all(isinstance(x, bool) for x in raw)
            if not ok:
                out.append(
                    {
                        "nd": int(nd),
                        "raw_type": type(raw).__name__,
                        "raw_len": (len(raw) if isinstance(raw, list) else -1),
                        "preview": [str(x) for x in (raw[:8] if isinstance(raw, list) else [])],
                    }
                )
            seen += 1
            if seen >= sample_rows:
                return out
    return out


def _collect_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    _prepare_paths()
    from .rl_agent import RLAgent

    # Keep rollout workers CPU-only and prevent CPU thread oversubscription.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    worker_torch_threads = max(1, int(payload.get("torch_num_threads", 1)))
    try:
        torch.set_num_threads(worker_torch_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    games = payload["games"]
    op_mode = payload["op_mode"]
    seed_base = int(payload["seed_base"])
    episodes = int(payload["episodes"])
    max_actions = int(payload["max_actions"])
    stochastic = bool(payload["stochastic"])
    render_terminal = bool(payload.get("render_terminal", False))
    collect_mode = str(payload.get("collect_mode", "train"))
    cfg = payload["cfg"]
    cfg = apply_rl_only_mode(cfg, True)
    cfg["device"] = "cpu"
    assert_rl_only_guards(cfg)
    configure_rl_only_logging(cfg)
    policy_state_dict = payload["policy_state_dict"]
    policy_version = int(payload.get("policy_version", 0))
    rollout_cfg_hash = str(payload.get("rollout_cfg_hash", ""))
    rollout_cfg_payload = payload.get("rollout_cfg_payload")
    worker_id = str(payload.get("worker_id", "unknown"))
    iter_idx = int(payload.get("iter_idx", -1))
    logger.info(
        "collect_worker_start iter=%s worker=%s seed_base=%s episodes=%s max_actions=%s stochastic=%s device=%s torch_threads=%s",
        iter_idx,
        worker_id,
        seed_base,
        episodes,
        max_actions,
        stochastic,
        "cpu",
        worker_torch_threads,
    )

    agent = RLAgent(cfg=cfg)
    if int(policy_version) != int(getattr(agent, "policy_version", -1)):
        agent.apply_policy_snapshot(policy_state_dict, policy_version=policy_version)
    modules = agent._build_modules()
    logger.info("worker_policy_version=%s", int(getattr(agent, "policy_version", -1)))

    env_factory = _build_env_factory(games=games, seed_base=seed_base, op_mode=op_mode, render_terminal=render_terminal)
    batch = agent.collector.collect(
        env_factory,
        modules,
        cfg={
            "episodes_per_batch": episodes,
            "max_steps_per_episode": max_actions,
            "stochastic_actions_train": stochastic,
            "coord_topK": int(cfg.get("coord_topK", 16)),
            "reward": cfg.get("reward", {}),
            "hud_specs": cfg.get("hud_specs", {}),
            "mode": collect_mode,
            "hud_cache_dir": str(cfg.get("hud_cache_dir", "runs/cache")),
        },
    )
    ep_count = len(batch.get("episodes", []))
    step_count = sum(len(ep.get("steps", [])) for ep in batch.get("episodes", []))
    batch["policy_version"] = int(getattr(agent, "policy_version", policy_version))
    batch["rollout_policy_version"] = int(getattr(agent, "policy_version", policy_version))
    batch["rollout_cfg_hash"] = rollout_cfg_hash
    batch["rollout_cfg_payload"] = dict(rollout_cfg_payload or {})
    for ep in batch.get("episodes", []):
        if isinstance(ep, dict):
            ep["policy_version"] = int(getattr(agent, "policy_version", policy_version))
            ep["rollout_policy_version"] = int(getattr(agent, "policy_version", policy_version))
            ep["rollout_cfg_hash"] = rollout_cfg_hash
            ep["rollout_cfg_payload"] = dict(rollout_cfg_payload or {})
    logger.info(
        "collect_worker_done iter=%s worker=%s episodes=%s steps=%s",
        iter_idx,
        worker_id,
        ep_count,
        step_count,
    )
    return batch


def _merge_batches(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {"schema_version": "TRAJECTORY_BATCH_V1", "episodes": []}
    for p in parts:
        batch["episodes"].extend(p.get("episodes", []))
    return batch


def _collect_with_progress(pool_obj: Any, payloads: List[Dict[str, Any]], iter_idx: Optional[int], total_target_episodes: int) -> List[Dict[str, Any]]:
    async_by_worker: Dict[int, Any] = {}
    for p in payloads:
        wid = int(p.get("worker_id", 0))
        async_by_worker[wid] = pool_obj.apply_async(_collect_worker, (p,))

    done_parts: Dict[int, Dict[str, Any]] = {}
    done_workers = 0
    done_episodes = 0
    done_steps = 0
    total_workers = max(1, len(async_by_worker))
    last_emit = 0.0
    while async_by_worker:
        progressed = False
        for wid, res in list(async_by_worker.items()):
            if not res.ready():
                continue
            part = res.get()
            done_parts[wid] = part
            del async_by_worker[wid]
            progressed = True
            done_workers += 1
            ep_count = len(part.get("episodes", []))
            st_count = sum(len(ep.get("steps", [])) for ep in part.get("episodes", []))
            done_episodes += ep_count
            done_steps += st_count
            logger.debug(
                "collect_batch_progress iter=%s workers_done=%s/%s episodes_done=%s/%s steps_done=%s",
                iter_idx,
                done_workers,
                total_workers,
                done_episodes,
                total_target_episodes,
                done_steps,
            )
        now = time.time()
        if (not progressed) and now - last_emit >= 5.0:
            logger.debug(
                "collect_batch_wait iter=%s workers_done=%s/%s episodes_done=%s/%s",
                iter_idx,
                done_workers,
                total_workers,
                done_episodes,
                total_target_episodes,
            )
            last_emit = now
        if not progressed:
            time.sleep(0.2)
    return [done_parts[k] for k in sorted(done_parts.keys())]


def _collect_batch(
    agent: Any,
    modules: Dict[str, Any],
    games: List[str],
    op_mode: str,
    seed_base: int,
    episodes: int,
    max_actions: int,
    stochastic: bool,
    workers: int,
    pool: Optional[Any] = None,
    iter_idx: Optional[int] = None,
    policy_state_dict: Optional[Dict[str, Any]] = None,
    policy_version: int = 0,
    rollout_cfg_hash: str = "",
    rollout_cfg_payload: Optional[Dict[str, Any]] = None,
    render_terminal: bool = False,
    collect_mode: str = "train",
) -> Dict[str, Any]:
    logger.info(
        "collect_batch_start iter=%s episodes=%s max_actions=%s workers=%s stochastic=%s games=%s",
        iter_idx,
        episodes,
        max_actions,
        workers,
        stochastic,
        len(games),
    )
    if workers <= 1 or episodes <= 1:
        if policy_state_dict is not None:
            agent.apply_policy_snapshot(policy_state_dict, policy_version=policy_version)
        env_factory = _build_env_factory(games, seed_base=seed_base, op_mode=op_mode, render_terminal=render_terminal)
        batch = agent.collector.collect(
            env_factory,
            modules,
            cfg={
                "episodes_per_batch": episodes,
                "max_steps_per_episode": max_actions,
                "stochastic_actions_train": stochastic,
                "coord_topK": int(agent.cfg.get("coord_topK", 16)),
                "reward": agent.cfg.get("reward", {}),
                "hud_specs": agent.cfg.get("hud_specs", {}),
                "mode": str(collect_mode),
                "hud_cache_dir": str(agent.cfg.get("hud_cache_dir", "runs/cache")),
            },
        )
        ep_count = len(batch.get("episodes", []))
        step_count = sum(len(ep.get("steps", [])) for ep in batch.get("episodes", []))
        logger.info(
            "collect_batch_done iter=%s episodes=%s steps=%s workers=1",
            iter_idx,
            ep_count,
            step_count,
        )
        batch["policy_version"] = int(policy_version)
        batch["rollout_policy_version"] = int(policy_version)
        batch["rollout_cfg_hash"] = str(rollout_cfg_hash)
        batch["rollout_cfg_payload"] = dict(rollout_cfg_payload or {})
        for ep in batch.get("episodes", []):
            if isinstance(ep, dict):
                ep["policy_version"] = int(policy_version)
                ep["rollout_policy_version"] = int(policy_version)
                ep["rollout_cfg_hash"] = str(rollout_cfg_hash)
                ep["rollout_cfg_payload"] = dict(rollout_cfg_payload or {})
        return batch

    counts = _split_counts(episodes, workers)
    payloads: List[Dict[str, Any]] = []
    start = 0
    state_dicts = policy_state_dict
    if state_dicts is None:
        state_dicts = {
            "encoder": _state_dict_cpu(modules["encoder"].state_dict()),
            "memory": _state_dict_cpu(modules["memory"].state_dict()),
            "controller": _state_dict_cpu(modules["controller"].state_dict()),
            "actor": _state_dict_cpu(modules["actor"].state_dict()),
            "value": _state_dict_cpu(modules["value"].state_dict()),
        }
    for c in counts:
        if c <= 0:
            continue
        payloads.append(
            {
                "games": games,
                "op_mode": op_mode,
                "seed_base": seed_base + start,
                "episodes": c,
                "max_actions": max_actions,
                "stochastic": stochastic,
                "cfg": agent.cfg,
                "policy_state_dict": state_dicts,
                "policy_version": int(policy_version),
                "rollout_cfg_hash": str(rollout_cfg_hash),
                "rollout_cfg_payload": dict(rollout_cfg_payload or {}),
                "worker_id": len(payloads),
                "iter_idx": int(iter_idx if iter_idx is not None else -1),
                "torch_num_threads": 1,
                "render_terminal": bool(render_terminal),
                "collect_mode": str(collect_mode),
            }
        )
        start += c

    logger.info(
        "collect_batch_dispatch iter=%s payloads=%s episode_splits=%s",
        iter_idx,
        len(payloads),
        [int(p.get("episodes", 0)) for p in payloads],
    )
    if pool is not None:
        parts = _collect_with_progress(pool, payloads, iter_idx=iter_idx, total_target_episodes=episodes)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=min(workers, len(payloads))) as tmp_pool:
            parts = _collect_with_progress(tmp_pool, payloads, iter_idx=iter_idx, total_target_episodes=episodes)
    batch = _merge_batches(parts)
    ep_count = len(batch.get("episodes", []))
    step_count = sum(len(ep.get("steps", [])) for ep in batch.get("episodes", []))
    logger.info(
        "collect_batch_done iter=%s episodes=%s steps=%s workers=%s",
        iter_idx,
        ep_count,
        step_count,
        workers,
    )
    batch["policy_version"] = int(policy_version)
    batch["rollout_policy_version"] = int(policy_version)
    batch["rollout_cfg_hash"] = str(rollout_cfg_hash)
    batch["rollout_cfg_payload"] = dict(rollout_cfg_payload or {})
    for ep in batch.get("episodes", []):
        if isinstance(ep, dict):
            ep["policy_version"] = int(policy_version)
            ep["rollout_policy_version"] = int(policy_version)
            ep["rollout_cfg_hash"] = str(rollout_cfg_hash)
            ep["rollout_cfg_payload"] = dict(rollout_cfg_payload or {})
    return batch


def _entropy_from_logits_list(logits_2d: Any) -> float:
    try:
        t = torch.tensor(logits_2d, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        p = torch.softmax(t, dim=1)
        lp = torch.log_softmax(t, dim=1)
        return float((-(p * lp).sum(dim=1).mean()).item())
    except Exception:
        return 0.0


def _episode_stats(ep: Dict[str, Any], gamma: float, mode_names: List[str], loop_n: int) -> Dict[str, Any]:
    steps = ep.get("steps", [])
    disc_return = 0.0
    total_reward = 0.0
    seen: set[str] = set()
    recent: List[str] = []
    effect = 0
    novelty = 0
    loops = 0

    mode_counts = {m: 0 for m in mode_names}
    mode_entropy_vals: List[float] = []
    mode_aux_correct = 0
    mode_aux_total = 0

    action_counts: Dict[str, int] = {}
    coord_select = 0
    coord_tag_counts: Dict[str, int] = {}
    policy_entropy_vals: List[float] = []

    for i, s in enumerate(steps):
        r = float(s.get("reward", 0.0))
        disc_return += (gamma**i) * r
        total_reward += r

        ev = s.get("transition_event", {}) or {}
        gd = ev.get("grid_delta", {}) or {}
        changed = int(gd.get("changed_cells_count", 0) or 0)
        md = ev.get("meta_delta", {}) or {}
        meta_changed = bool(md)
        if changed > 0 or meta_changed:
            effect += 1

        state_after = s.get("state_hash_after")
        if state_after and state_after not in seen:
            novelty += 1
        if state_after and state_after in recent[-loop_n:]:
            loops += 1
        if state_after:
            seen.add(state_after)
            recent.append(state_after)

        mode_id = int(s.get("mode_id", 0))
        if 0 <= mode_id < len(mode_names):
            mode_counts[mode_names[mode_id]] += 1

        mode_entropy = s.get("mode_entropy")
        if isinstance(mode_entropy, (int, float)):
            mode_entropy_vals.append(float(mode_entropy))
        else:
            mode_entropy_vals.append(_entropy_from_logits_list(s.get("mode_logits", [])))

        reward_aux = s.get("reward_aux", {}) if isinstance(s.get("reward_aux"), dict) else {}
        if "mode_target" in reward_aux:
            mode_aux_total += 1
            pred = int(torch.tensor(s.get("mode_logits", [[0.0]])).argmax().item()) if s.get("mode_logits") else mode_id
            if pred == int(reward_aux["mode_target"]):
                mode_aux_correct += 1

        action = s.get("action_key", {}) if isinstance(s.get("action_key"), dict) else {}
        aid = str(action.get("action_id", "ACTION1"))
        action_counts[aid] = action_counts.get(aid, 0) + 1
        if aid.upper() == "ACTION6":
            coord_select += 1
            tag = str(s.get("chosen_coord_tag") or "unknown")
            coord_tag_counts[tag] = coord_tag_counts.get(tag, 0) + 1

        pe = s.get("policy_entropy")
        if isinstance(pe, (int, float)):
            policy_entropy_vals.append(float(pe))
        else:
            policy_entropy_vals.append(_entropy_from_logits_list(s.get("pi_discrete_logits", [])))

    n_steps = max(1, len(steps))
    steps = ep.get("steps", []) if isinstance(ep.get("steps"), list) else []
    win_flag = bool(ep.get("win", False))
    if not win_flag and steps:
        win_flag = any(bool(s.get("win", False)) for s in steps if isinstance(s, dict))
    return {
        "game_id": ep.get("game_id"),
        "seed": ep.get("seed"),
        "win": win_flag,
        "return": float(disc_return),
        "total_reward": float(total_reward),
        "steps": len(steps),
        "effect_rate": float(effect / n_steps),
        "novelty_rate": float(novelty / n_steps),
        "loop_rate": float(loops / n_steps),
        "mode_usage": {k: float(v / n_steps) for k, v in mode_counts.items()},
        "mode_entropy": float(sum(mode_entropy_vals) / max(1, len(mode_entropy_vals))),
        "mode_aux_acc": float(mode_aux_correct / max(1, mode_aux_total)) if mode_aux_total > 0 else None,
        "action_usage": {k: float(v / n_steps) for k, v in action_counts.items()},
        "coord_select_rate": float(coord_select / n_steps),
        "coord_tag_usage": {k: float(v / max(1, coord_select)) for k, v in coord_tag_counts.items()},
        "policy_entropy": float(sum(policy_entropy_vals) / max(1, len(policy_entropy_vals))),
    }


def _aggregate_metrics(batch: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    algo = str(cfg.get("algo", "a2c")).lower()
    ppo_cfg = cfg.get("ppo", {}) if isinstance(cfg.get("ppo"), dict) else {}
    if algo == "ppo" and "gamma" in ppo_cfg:
        gamma = float(ppo_cfg.get("gamma", 0.99))
    elif "gamma" in cfg:
        gamma = float(cfg.get("gamma", 0.99))
    else:
        gamma = 0.99
    loop_n = int(cfg.get("reward", {}).get("loop_window_N", 25)) if isinstance(cfg.get("reward"), dict) else 25
    mode_names = list(cfg.get("modes", ["probe", "exploit", "escape_loop", "focus_click"]))

    episodes = batch.get("episodes", [])
    per_ep = [_episode_stats(ep, gamma=gamma, mode_names=mode_names, loop_n=loop_n) for ep in episodes]
    if not per_ep:
        return {
            "WinRate": 0.0,
            "MeanReturn": 0.0,
            "AvgReward": 0.0,
            "MedianReturn": 0.0,
            "MeanEpisodeLen": 0.0,
            "EffectRate": 0.0,
            "NoveltyRate": 0.0,
            "LoopRate": 0.0,
            "ModeEntropy": 0.0,
            "ModeAuxAcc": None,
            "PolicyEntropy": 0.0,
            "CoordSelectRate": 0.0,
            "ModeUsage": {m: 0.0 for m in mode_names},
            "ActionUsage": {},
            "CoordTagUsage": {},
            "total_episodes": 0,
            "total_steps": 0,
            "wins": 0,
        }, []

    returns = [e["return"] for e in per_ep]
    total_rewards = [float(e.get("total_reward", 0.0)) for e in per_ep]
    steps = [int(e["steps"]) for e in per_ep]
    wins = [1 if e["win"] else 0 for e in per_ep]
    effect_rates = [float(e["effect_rate"]) for e in per_ep]
    novelty_rates = [float(e["novelty_rate"]) for e in per_ep]
    loop_rates = [float(e["loop_rate"]) for e in per_ep]
    mode_entropy = [float(e["mode_entropy"]) for e in per_ep]
    policy_entropy = [float(e["policy_entropy"]) for e in per_ep]
    coord_rates = [float(e["coord_select_rate"]) for e in per_ep]

    aux_vals = [float(e["mode_aux_acc"]) for e in per_ep if e["mode_aux_acc"] is not None]

    mode_usage: Dict[str, float] = {m: 0.0 for m in mode_names}
    action_usage: Dict[str, float] = {}
    coord_tag_usage: Dict[str, float] = {}

    for e in per_ep:
        for k, v in e["mode_usage"].items():
            mode_usage[k] = mode_usage.get(k, 0.0) + float(v)
        for k, v in e["action_usage"].items():
            action_usage[k] = action_usage.get(k, 0.0) + float(v)
        for k, v in e["coord_tag_usage"].items():
            coord_tag_usage[k] = coord_tag_usage.get(k, 0.0) + float(v)

    n = float(len(per_ep))
    mode_usage = {k: v / n for k, v in mode_usage.items()}
    action_usage = {k: v / n for k, v in action_usage.items()}
    coord_tag_usage = {k: v / n for k, v in coord_tag_usage.items()}

    agg = {
        "WinRate": float(sum(wins) / max(1, len(wins))),
        "MeanReturn": float(sum(returns) / max(1, len(returns))),
        "MedianReturn": float(statistics.median(returns) if returns else 0.0),
        "AvgReward": float(sum(total_rewards) / max(1, len(total_rewards))),
        "MeanEpisodeLen": float(sum(steps) / max(1, len(steps))),
        "EffectRate": float(sum(effect_rates) / max(1, len(effect_rates))),
        "NoveltyRate": float(sum(novelty_rates) / max(1, len(novelty_rates))),
        "LoopRate": float(sum(loop_rates) / max(1, len(loop_rates))),
        "ModeEntropy": float(sum(mode_entropy) / max(1, len(mode_entropy))),
        "ModeAuxAcc": float(sum(aux_vals) / len(aux_vals)) if aux_vals else None,
        "PolicyEntropy": float(sum(policy_entropy) / max(1, len(policy_entropy))),
        "CoordSelectRate": float(sum(coord_rates) / max(1, len(coord_rates))),
        "ModeUsage": mode_usage,
        "ActionUsage": action_usage,
        "CoordTagUsage": coord_tag_usage,
        "total_episodes": int(len(per_ep)),
        "total_steps": int(sum(steps)),
        "wins": int(sum(wins)),
    }
    return agg, per_ep


def _update_coverage_ledger(ledger: CoverageLedgerV1, batch: Dict[str, Any]) -> None:
    for ep in batch.get("episodes", []):
        for step in ep.get("steps", []):
            trans = step.get("transition_event", {}) if isinstance(step.get("transition_event"), dict) else {}
            terms = step.get("reward_terms", {}) if isinstance(step.get("reward_terms"), dict) else {}
            ledger.update(
                {
                    "state_hash_before_filtered": step.get("state_hash_before_filtered")
                    or trans.get("state_hash_before_filtered")
                    or step.get("state_hash_before")
                    or trans.get("state_hash_before"),
                    "state_hash_after_filtered": step.get("state_hash_after_filtered")
                    or trans.get("state_hash_after_filtered")
                    or step.get("state_hash_after")
                    or trans.get("state_hash_after"),
                    "action_key": step.get("action_key"),
                    "coord_tag": step.get("chosen_coord_tag"),
                    "effect_flag_filtered": terms.get("effect_flag_filtered"),
                    "effect_flag": terms.get("effect_flag_raw"),
                }
            )


def _prefixed(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        out[f"{prefix}{k}"] = v
    return out


def _exploration_better(new: Dict[str, Any], old: Optional[Dict[str, Any]]) -> bool:
    if old is None:
        return True
    key = [
        float(new.get("EvalMeanReturn", -math.inf)),
        float(new.get("EvalEffectRate", -math.inf)),
        -float(new.get("EvalLoopRate", math.inf)),
        float(new.get("EvalNoveltyRate", -math.inf)),
        -int(new.get("iter", 1_000_000_000)),
    ]
    old_key = [
        float(old.get("EvalMeanReturn", -math.inf)),
        float(old.get("EvalEffectRate", -math.inf)),
        -float(old.get("EvalLoopRate", math.inf)),
        float(old.get("EvalNoveltyRate", -math.inf)),
        -int(old.get("iter", 1_000_000_000)),
    ]
    return tuple(key) > tuple(old_key)


def _win_better(new: Dict[str, Any], old: Optional[Dict[str, Any]]) -> bool:
    if old is None:
        return True
    key = [
        float(new.get("EvalWinRate", -math.inf)),
        float(new.get("EvalMeanReturn", -math.inf)),
        -float(new.get("EvalMeanEpisodeLen", math.inf)),
        -float(new.get("EvalLoopRate", math.inf)),
        -int(new.get("iter", 1_000_000_000)),
    ]
    old_key = [
        float(old.get("EvalWinRate", -math.inf)),
        float(old.get("EvalMeanReturn", -math.inf)),
        -float(old.get("EvalMeanEpisodeLen", math.inf)),
        -float(old.get("EvalLoopRate", math.inf)),
        -int(old.get("iter", 1_000_000_000)),
    ]
    return tuple(key) > tuple(old_key)


def _load_config(path: Optional[str], base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not path:
        return base_cfg
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    out = dict(base_cfg)
    out.update(user_cfg)
    if "controller_temperature" in out:
        controller_cfg = dict(out.get("controller", {})) if isinstance(out.get("controller"), dict) else {}
        controller_cfg["temperature"] = float(out.get("controller_temperature"))
        out["controller"] = controller_cfg
    return out


def _default_cfg_dict() -> Dict[str, Any]:
    defaults_path = os.path.join(os.path.dirname(__file__), "rl_config_defaults.json")
    json_defaults: Dict[str, Any] = {}
    if os.path.isfile(defaults_path):
        try:
            with open(defaults_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                json_defaults = loaded
        except Exception:
            json_defaults = {}
    cfg = RLConfig()
    base = {
        "pipeline": dict(cfg.pipeline),
        "modules": dict(cfg.modules),
        "action_source": "rl_agent",
        "embed_dim": cfg.embed_dim,
        "hidden_dim": cfg.hidden_dim,
        "action_emb_dim": cfg.action_emb_dim,
        "controller_hidden": cfg.controller_hidden,
        "num_modes": cfg.num_modes,
        "modes": list(cfg.modes),
        "controller": dict(cfg.controller),
        "mode_action_allow": dict(cfg.mode_action_allow),
        "mode_action_bias": dict(cfg.mode_action_bias),
        "mode_coord_bias": dict(cfg.mode_coord_bias),
        "coord_mode": cfg.coord_mode,
        "coord_topK": cfg.coord_topK,
        "rollout_batch_episodes": cfg.rollout_batch_episodes,
        "rollout_max_steps": cfg.rollout_max_steps,
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
        "controller_coef": cfg.controller_coef,
        "actor_coef": cfg.actor_coef,
        "aux_mode_ce_coef": cfg.aux_mode_ce_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "updates_per_iter": cfg.updates_per_iter,
        "optimizers": dict(cfg.optimizers),
        "ppo": dict(cfg.ppo),
        "ckpt": dict(cfg.ckpt),
        "log": dict(cfg.log),
        "rl": {"debug_perf_checks": False},
        "hud_probe_enabled": False,
        "hud_probe_steps": 30,
        "hud_cache_dir": "runs/cache",
        "hud_detect_window": 30,
        "hud_change_rate_threshold": 0.8,
        "hud_min_changed_cells_per_step": 1,
        "hud_edge_margin": 4,
        "hud_min_component_area": 20,
        "hud_dilate_px": 1,
        "hud_max_area_frac": 0.35,
        "hud_min_elongation": 1.5,
        "hud_min_confidence": 0.85,
        "aux": dict(cfg.aux),
        "train": dict(cfg.train),
        "eval": dict(cfg.eval),
    }
    base.update(json_defaults)
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="RL trainer/evaluator")
    parser.add_argument("--mode", choices=["collect", "train", "eval"], required=True)
    parser.add_argument("--games", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--op-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rl-only", default="true")
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--eval-holdout", default=None)
    parser.add_argument("--eval-easy", default=None)
    parser.add_argument("--render-terminal", action="store_true", help="Render game in terminal during env steps")
    parser.add_argument("--debug", default=None, help="Write debug logs to this file path")
    parser.add_argument("--config", default=None)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    # Ensure only selected INFO logs are emitted to stdout.
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stdout, sys.stderr):
            root_logger.removeHandler(h)

    class _StdoutFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            if record.name == "arc_agi_agent.rl.trainer":
                return True
            msg = record.getMessage()
            return msg.startswith("collect_batch_start") or msg.startswith("iter_collect_done") or msg.startswith("iter_train_done")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler.addFilter(_StdoutFilter())
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    if args.debug:
        _setup_debug_log(str(args.debug))

    _prepare_paths()
    _configure_torch_mp_sharing()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for RL training/inference; CPU fallback is disabled.")

    from .rl_agent import RLAgent

    games_train = _resolve_games(args.games, args.op_mode)
    if not games_train:
        raise SystemExit("No games resolved")

    eval_holdout_selector = args.eval_holdout or args.games
    games_eval_holdout = _resolve_games(eval_holdout_selector, args.op_mode)
    games_eval_easy = _resolve_games(args.eval_easy, args.op_mode) if args.eval_easy else []

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, "rl", run_id)
    os.makedirs(run_dir, exist_ok=True)
    _setup_stdout_log_copy(os.path.join(run_dir, "log.log"))

    cfg = _default_cfg_dict()
    cfg = _load_config(args.config, cfg)
    cfg = apply_rl_only_mode(cfg, args.rl_only)
    assert_rl_only_guards(cfg)
    configure_rl_only_logging(cfg)
    logger.info(
        "run_start mode=%s games=%s workers=%s iters=%s op_mode=%s outdir=%s",
        args.mode,
        args.games,
        args.workers,
        args.iters,
        args.op_mode,
        args.outdir,
    )

    algo = str(cfg.get("algo", "a2c")).lower()

    def _resolve_train_episodes_per_iter() -> int:
        if args.episodes is not None and int(args.episodes) > 0:
            return int(args.episodes)
        if algo == "ppo":
            if cfg.get("rollout_batch_episodes") is not None:
                return int(cfg["rollout_batch_episodes"])
            ppo_cfg = cfg.get("ppo", {}) if isinstance(cfg.get("ppo"), dict) else {}
            if ppo_cfg.get("episodes_per_iter") is not None:
                return int(ppo_cfg["episodes_per_iter"])
            if cfg.get("episodes_per_iter") is not None:
                return int(cfg["episodes_per_iter"])
            return 64
        return int(cfg.get("episodes_per_iter", 64))

    def _resolve_max_steps() -> int:
        if args.max_actions is not None and int(args.max_actions) > 0:
            return int(args.max_actions)
        if cfg.get("rollout_max_steps") is not None:
            return int(cfg["rollout_max_steps"])
        ppo_cfg = cfg.get("ppo", {}) if isinstance(cfg.get("ppo"), dict) else {}
        if ppo_cfg.get("max_steps_per_episode") is not None:
            return int(ppo_cfg["max_steps_per_episode"])
        if cfg.get("max_steps_per_episode") is not None:
            return int(cfg["max_steps_per_episode"])
        return 100

    train_episodes_per_iter = _resolve_train_episodes_per_iter()
    resolved_max_steps = _resolve_max_steps()
    resolved_eval_episodes = int(args.episodes) if (args.episodes is not None and int(args.episodes) > 0) else int(cfg["eval"]["episodes"])

    agent = RLAgent(cfg=cfg)
    modules = agent._build_modules()

    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise SystemExit(f"wandb is required by default but is not available: {exc}") from exc

    ignore_globs = (
        "*.log,**/*.log,debug.log,output.log,**/debug.log,**/output.log,"
        "episodes.jsonl,**/episodes.jsonl,*episodes.jsonl,**/*episodes.jsonl,"
        "seeds.jsonl,**/seeds.jsonl,traces/*.jsonl,**/traces/*.jsonl"
    )
    if os.environ.get("WANDB_IGNORE_GLOBS"):
        os.environ["WANDB_IGNORE_GLOBS"] = f"{os.environ['WANDB_IGNORE_GLOBS']},{ignore_globs}"
    else:
        os.environ["WANDB_IGNORE_GLOBS"] = ignore_globs

    wandb_run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "arc-agi-rl"),
        name=args.run_name,
        mode=os.environ.get("WANDB_MODE", "online"),
        config={
            "mode": args.mode,
            "games": args.games,
            "seed": args.seed,
            "max_actions": resolved_max_steps,
            "episodes": args.episodes,
            "iters": args.iters,
            "workers": args.workers,
            "phase": args.phase,
            "op_mode": args.op_mode,
        },
        dir=run_dir,
    )
    logger.info("wandb_initialized run_name=%s run_dir=%s", args.run_name, run_dir)

    optim: Any = build_optimizer(modules, cfg)

    if args.checkpoint:
        agent._load_checkpoint(args.checkpoint, modules, optim if args.mode == "train" else None)
        logger.info("checkpoint_loaded path=%s", args.checkpoint)
        print(f"[checkpoint_loaded] path={args.checkpoint}", flush=True)

    _write_json(os.path.join(run_dir, "configs", "resolved_config.json"), cfg)

    if args.mode in ("collect", "eval"):
        episodes = int(resolved_eval_episodes)
        eval_iter = 0
        seed_base = int(args.seed + eval_iter * 100000)
        target_games = games_eval_holdout if args.mode == "eval" else games_train
        rollout_cfg_hash, rollout_cfg_payload = _rollout_cfg_hash(cfg)
        batch = _collect_batch(
                    agent,
                    modules,
                    games=target_games,
                    op_mode=args.op_mode,
                    seed_base=seed_base,
                    episodes=episodes,
                    max_actions=resolved_max_steps,
                    stochastic=args.mode == "collect" and bool(cfg.get("stochastic_actions_train", True)),
                    workers=max(1, int(args.workers)),
                    pool=None,
                    iter_idx=0,
                    rollout_cfg_hash=str(rollout_cfg_hash),
                    rollout_cfg_payload=rollout_cfg_payload,
                    render_terminal=bool(args.render_terminal),
                )
        _write_json(os.path.join(run_dir, "trajectories", "batch.json"), batch)
        logger.info(
            "collect_eval_done mode=%s episodes=%s steps=%s",
            args.mode,
            len(batch.get("episodes", [])),
            sum(len(ep.get("steps", [])) for ep in batch.get("episodes", [])),
        )
        metrics, per_ep = _aggregate_metrics(batch, cfg)
        prefix = "Eval" if args.mode == "eval" else "Train"
        m = _prefixed(metrics, prefix)
        _write_json(os.path.join(run_dir, "metrics", "metrics_eval.json" if args.mode == "eval" else "metrics_collect.json"), m)
        rows = [{"split": args.mode, **r} for r in per_ep]
        _append_jsonl(os.path.join(run_dir, "seeds.jsonl"), [{"split": args.mode, "game_id": r["game_id"], "seed": r["seed"]} for r in per_ep])
        _append_jsonl(os.path.join(run_dir, "metrics", "episodes.jsonl"), rows)
        if args.mode == "eval":
            wandb_run.log(
                {
                    "eval/mean_return": float(metrics.get("MeanReturn", 0.0)),
                    "eval/avg_reward": float(metrics.get("AvgReward", 0.0)),
                    "eval/mean_episode_len": float(metrics.get("MeanEpisodeLen", 0.0)),
                    "eval/win_rate": float(metrics.get("WinRate", 0.0)),
                }
            )
        else:
            wandb_run.log(
                {
                    "train/mean_return": float(metrics.get("MeanReturn", 0.0)),
                    "train/avg_reward": float(metrics.get("AvgReward", 0.0)),
                    "train/mean_episode_len": float(metrics.get("MeanEpisodeLen", 0.0)),
                    "train/win_rate": float(metrics.get("WinRate", 0.0)),
                }
            )
        logger.info("collect_eval_metrics_logged mode=%s keys=%s", args.mode, sorted(metrics.keys()))
        wandb_run.finish()
        logger.info("run_finish mode=%s", args.mode)
        return 0

    num_iters = int(args.iters or cfg["train"]["num_iters"])
    eval_every = int(cfg["eval"].get("every_iters", 5))
    trace_eval_episodes = int(cfg["eval"].get("trace_eval_episodes", 10))
    save_every = int(cfg.get("ckpt", {}).get("save_every_iters", 1))
    switch_threshold = float(cfg.get("eval", {}).get("win_switch_threshold", 0.05))
    switch_k = int(cfg.get("eval", {}).get("win_switch_consecutive_k", 3))

    summary: Dict[str, Any] = {
        "stage": "exploration",
        "consecutive_eval_win_hits": 0,
        "best_total_reward": None,
    }
    _write_json(os.path.join(run_dir, "metrics", "summary.json"), summary)

    if bool(cfg.get("hud_probe_enabled", False)):
        probe_steps = int(cfg.get("hud_probe_steps", 30))
        probe_eps = max(1, int(cfg.get("hud_probe_episodes", len(games_train))))
        hud_cache_dir = str(cfg.get("hud_cache_dir", "runs/cache"))
        probe_cfg = {
            "hud_detect_window": int(cfg.get("hud_detect_window", 30)),
            # Minimum region activity fraction per candidate HUD bbox.
            "hud_change_rate_threshold": float(cfg.get("hud_change_rate_threshold", 0.8)),
            "hud_min_changed_cells_per_step": int(cfg.get("hud_min_changed_cells_per_step", 1)),
            "hud_edge_margin": int(cfg.get("hud_edge_margin", 4)),
            "hud_min_component_area": int(cfg.get("hud_min_component_area", 20)),
            "hud_dilate_px": int(cfg.get("hud_dilate_px", 1)),
            "hud_max_area_frac": float(cfg.get("hud_max_area_frac", 0.35)),
            "hud_min_elongation": float(cfg.get("hud_min_elongation", 1.5)),
            "hud_min_confidence": float(cfg.get("hud_min_confidence", 0.85)),
            "hud_probe_min_steps": int(cfg.get("hud_detect_window", 30)),
        }
        logger.info("hud_probe_start enabled=True episodes=%s steps=%s cache_dir=%s", probe_eps, probe_steps, hud_cache_dir)
        probe_acc = HudProbeAccumulator(cfg=probe_cfg)
        probe_env_factory = _build_env_factory(games=games_train, seed_base=int(args.seed + 777000000), op_mode=args.op_mode, render_terminal=bool(args.render_terminal))
        _ = agent.collector.collect(
            probe_env_factory,
            modules,
            cfg={
                "episodes_per_batch": probe_eps,
                "max_steps_per_episode": probe_steps,
                "stochastic_actions_train": True,
                "coord_topK": int(cfg.get("coord_topK", 16)),
                "reward": cfg.get("reward", {}),
                "mode": "probe",
                "hud_cache_dir": hud_cache_dir,
            },
            ctx={"hud_probe_accumulator": probe_acc},
        )
        specs = probe_acc.finalize_all()
        cfg["hud_specs"] = specs
        agent.cfg["hud_specs"] = specs
        for gid in sorted(set(str(g) for g in games_train)):
            if gid in specs:
                spec = specs[gid]
                logger.info(
                    "hud_probe_spec game_id=%s hud_bbox=%s hud_area=%s hud_area_frac=%.6f confidence=%.6f steps_seen=%s source=in_memory",
                    gid,
                    spec.get("hud_bbox"),
                    spec.get("hud_area"),
                    float(spec.get("hud_area_frac", 0.0)),
                    float(spec.get("confidence", 0.0)),
                    spec.get("steps_seen"),
                )
            else:
                logger.info("hud_probe_spec game_id=%s hud_bbox=None status=not_detected source=in_memory", gid)
        try:
            if hasattr(agent, "collector") and getattr(agent.collector, "reward_shaper", None) is not None:
                agent.collector.reward_shaper.reset_hud_cache()
        except Exception:
            pass
        logger.info("hud_probe_done specs_saved=%s source=in_memory cache_dir=%s", len(specs), hud_cache_dir)

    shared_pool: Optional[Any] = None
    if int(args.workers) > 1:
        mp_ctx = get_context("spawn")
        shared_pool = mp_ctx.Pool(processes=int(args.workers))
        logger.info("shared_pool_created workers=%s", args.workers)

    run_completed = False
    policy_version = 0
    try:
        for iter_idx in range(num_iters):
            logger.info("iter_start iter=%s/%s stage=%s", iter_idx, num_iters, summary.get("stage"))
            coverage_ledger = CoverageLedgerV1()
            if args.phase >= 2 and isinstance(cfg.get("reward"), dict):
                anneal_iters = int(cfg["reward"].get("anneal_iters", 0) or 0)
                if anneal_iters > 0:
                    start = float(cfg["reward"].get("alpha_novel_start", cfg["reward"].get("alpha_novel", 0.0)))
                    end = float(cfg["reward"].get("alpha_novel_end", 0.0))
                    t = min(1.0, float(iter_idx) / float(anneal_iters))
                    cfg["reward"]["alpha_novel"] = start + (end - start) * t

            policy_state_dict = {
                "encoder": _state_dict_cpu(modules["encoder"].state_dict()),
                "memory": _state_dict_cpu(modules["memory"].state_dict()),
                "controller": _state_dict_cpu(modules["controller"].state_dict()),
                "actor": _state_dict_cpu(modules["actor"].state_dict()),
                "value": _state_dict_cpu(modules["value"].state_dict()),
            }
            rollout_cfg_hash, rollout_cfg_payload = _rollout_cfg_hash(cfg)
            logger.info("dispatch_policy_version=%s", policy_version)
            train_batch = _collect_batch(
                agent,
                modules,
                games=games_train,
                op_mode=args.op_mode,
                seed_base=int(args.seed + iter_idx * 100000),
                episodes=int(train_episodes_per_iter),
                max_actions=resolved_max_steps,
                stochastic=bool(cfg.get("stochastic_actions_train", True)),
                workers=max(1, int(args.workers)),
                pool=shared_pool,
                iter_idx=iter_idx,
                policy_state_dict=policy_state_dict,
                policy_version=int(policy_version),
                rollout_cfg_hash=str(rollout_cfg_hash),
                rollout_cfg_payload=rollout_cfg_payload,
                render_terminal=bool(args.render_terminal),
            )
            logger.info(
                "iter_collect_done iter=%s episodes=%s steps=%s",
                iter_idx,
                len(train_batch.get("episodes", [])),
                sum(len(ep.get("steps", [])) for ep in train_batch.get("episodes", [])),
            )
            _update_coverage_ledger(coverage_ledger, train_batch)

            _append_jsonl(
                os.path.join(run_dir, "seeds.jsonl"),
                [{"split": "train", "iter": iter_idx, "game_id": ep.get("game_id"), "seed": ep.get("seed")} for ep in train_batch.get("episodes", [])],
            )

            noncanonical_masks = _sample_noncanonical_masks(train_batch, sample_rows=32)
            if noncanonical_masks:
                pipeline_mode = str((cfg.get("pipeline", {}) or {}).get("mode", "")).lower()
                first = noncanonical_masks[0]
                if pipeline_mode == "rl_only":
                    logger.error(
                        "noncanonical_available_actions_mask_in_batch iter=%s rows=%s first_nd=%s first_raw_type=%s first_raw_len=%s first_preview=%s; skipping_train_step",
                        iter_idx,
                        len(noncanonical_masks),
                        first.get("nd"),
                        first.get("raw_type"),
                        first.get("raw_len"),
                        first.get("preview"),
                    )
                    continue
                logger.warning(
                    "noncanonical_available_actions_mask_in_batch iter=%s rows=%s first_nd=%s first_raw_type=%s first_raw_len=%s first_preview=%s",
                    iter_idx,
                    len(noncanonical_masks),
                    first.get("nd"),
                    first.get("raw_type"),
                    first.get("raw_len"),
                    first.get("preview"),
                )

            train_loss = agent.trainer.train_step(
                train_batch,
                modules,
                optim,
                cfg=cfg,
                ctx={
                    "global_seed": int(args.seed),
                    "iter_idx": int(iter_idx),
                    "debug_perf_checks": bool((cfg.get("rl", {}) or {}).get("debug_perf_checks", False)),
                    "rollout_policy_version": train_batch.get("rollout_policy_version", train_batch.get("policy_version")),
                    "rollout_cfg_hash": train_batch.get("rollout_cfg_hash"),
                    "rollout_cfg_payload": train_batch.get("rollout_cfg_payload"),
                },
            )
            policy_version += 1
            train_metrics_agg, _ = _aggregate_metrics(train_batch, cfg)
            coverage_summary = coverage_ledger.summary()
            avg_reward = float(train_metrics_agg.get("AvgReward", 0.0))
            logger.info(
                "iter_train_done iter=%s loss_total=%.6f win_rate=%.6f mean_return=%.6f avg_reward=%.3f",
                iter_idx,
                float(train_loss.get("losses", {}).get("total", 0.0)),
                float(train_metrics_agg.get("WinRate", 0.0)),
                float(train_metrics_agg.get("MeanReturn", 0.0)),
                avg_reward,
            )
            if args.phase != 0:
                best_total = summary.get("best_total_reward")
                best_val = float(best_total.get("avg_reward", float("-inf"))) if isinstance(best_total, dict) else float("-inf")
                if avg_reward > best_val:
                    summary["best_total_reward"] = {
                        "iter": int(iter_idx),
                        "avg_reward": float(avg_reward),
                        "mean_return": float(train_metrics_agg.get("MeanReturn", 0.0)),
                        "win_rate": float(train_metrics_agg.get("WinRate", 0.0)),
                    }
                    agent._save_checkpoint(
                        os.path.join(run_dir, "checkpoints", "best_total_reward.ckpt"),
                        modules,
                        optim,
                        iter_idx,
                    )
                    logger.info(
                        "checkpoint_saved iter=%s kind=best_total_reward avg_reward=%.3f path=%s",
                        iter_idx,
                        avg_reward,
                        os.path.join(run_dir, "checkpoints", "best_total_reward.ckpt"),
                    )
            train_report = {"iter": iter_idx, **train_loss, **_prefixed(train_metrics_agg, "Train"), "coverage": coverage_summary}
            _write_json(os.path.join(run_dir, "metrics", f"train_iter_{iter_idx:06d}.json"), train_report)
            logger.info("iter_metrics_written iter=%s file=train_iter_%06d.json", iter_idx, iter_idx)
            train_log_payload = {
                "train/loss_total": float(train_loss.get("losses", {}).get("total", 0.0)),
                "train/loss_actor_policy": float(train_loss.get("losses", {}).get("actor_policy", 0.0)),
                "train/loss_controller_policy": float(train_loss.get("losses", {}).get("controller_policy", 0.0)),
                "train/loss_value": float(train_loss.get("losses", {}).get("value", 0.0)),
                "train/loss_entropy_actor": float(train_loss.get("losses", {}).get("entropy_actor", 0.0)),
                "train/loss_entropy_controller": float(train_loss.get("losses", {}).get("entropy_controller", 0.0)),
                "train/loss_aux_mode_ce": float(train_loss.get("losses", {}).get("aux_mode_ce", 0.0)),
                "train/grad_norm_total": float(train_loss.get("losses", {}).get("grad_norm_total", 0.0)),
                "train/mean_return": float(train_metrics_agg.get("MeanReturn", 0.0)),
                "train/avg_reward": float(avg_reward),
                "train/mean_episode_len": float(train_metrics_agg.get("MeanEpisodeLen", 0.0)),
                "train/win_rate": float(train_metrics_agg.get("WinRate", 0.0)),
            }
            if algo == "ppo":
                train_log_payload.update(
                    {
                        "train/approx_kl": float(train_loss.get("losses", {}).get("approx_kl", 0.0)),
                        "train/clipfrac_mode": float(train_loss.get("losses", {}).get("clipfrac_mode", 0.0)),
                        "train/clipfrac_action": float(train_loss.get("losses", {}).get("clipfrac_action", 0.0)),
                        "train/clipfrac_coord": float(train_loss.get("losses", {}).get("clipfrac_coord", 0.0)),
                        "train/ppo_epochs_ran": float(train_loss.get("losses", {}).get("ppo_epochs_ran", 0.0)),
                        "train/adv_mean": float(train_loss.get("losses", {}).get("adv_mean", 0.0)),
                        "train/adv_std": float(train_loss.get("losses", {}).get("adv_std", 0.0)),
                    }
                )
            wandb_run.log(train_log_payload, step=iter_idx)

            if save_every > 0 and iter_idx % save_every == 0:
                agent._save_checkpoint(
                    os.path.join(run_dir, "checkpoints", f"latest_{iter_idx:06d}.ckpt"),
                    modules,
                    optim,
                    iter_idx,
                )
                logger.info("checkpoint_saved iter=%s kind=latest path=%s", iter_idx, os.path.join(run_dir, "checkpoints", f"latest_{iter_idx:06d}.ckpt"))

            if eval_every > 0 and (iter_idx + 1) % eval_every == 0:
                logger.info("eval_start iter=%s episodes=%s", iter_idx, int(resolved_eval_episodes))
                print(f"[eval_start] iter={iter_idx} episodes={int(resolved_eval_episodes)}", flush=True)
                eval_seed_base = int(args.seed + (iter_idx + 1) * 1_000_000)
                eval_batch = _collect_batch(
                    agent,
                    modules,
                    games=games_eval_holdout,
                    op_mode=args.op_mode,
                    seed_base=eval_seed_base,
                    episodes=int(resolved_eval_episodes),
                    max_actions=resolved_max_steps,
                    stochastic=False,
                    workers=max(1, int(args.workers)),
                    pool=shared_pool,
                    iter_idx=iter_idx,
                    rollout_cfg_hash=str(rollout_cfg_hash),
                    rollout_cfg_payload=rollout_cfg_payload,
                    render_terminal=bool(args.render_terminal),
                )

                _append_jsonl(
                    os.path.join(run_dir, "seeds.jsonl"),
                    [{"split": "eval_holdout", "iter": iter_idx, "game_id": ep.get("game_id"), "seed": ep.get("seed")} for ep in eval_batch.get("episodes", [])],
                )

                eval_metrics_agg, eval_per_ep = _aggregate_metrics(eval_batch, cfg)
                trainer_phase = agent.trainer.update_phase_from_eval(eval_metrics_agg, cfg=cfg)
                logger.info(
                    "eval_done iter=%s win_rate=%.6f mean_return=%.6f trainer_stage=%s",
                    iter_idx,
                    float(eval_metrics_agg.get("WinRate", 0.0)),
                    float(eval_metrics_agg.get("MeanReturn", 0.0)),
                    trainer_phase.get("stage"),
                )
                eval_report = {"iter": iter_idx, **_prefixed(eval_metrics_agg, "Eval")}
                _write_json(os.path.join(run_dir, "metrics", f"eval_iter_{iter_idx:06d}.json"), eval_report)
                print(
                    (
                        f"[eval] iter={iter_idx} "
                        f"win_rate={float(eval_metrics_agg.get('WinRate', 0.0)):.6f} "
                        f"mean_return={float(eval_metrics_agg.get('MeanReturn', 0.0)):.6f} "
                        f"median_return={float(eval_metrics_agg.get('MedianReturn', 0.0)):.6f} "
                        f"effect_rate={float(eval_metrics_agg.get('EffectRate', 0.0)):.6f} "
                        f"novelty_rate={float(eval_metrics_agg.get('NoveltyRate', 0.0)):.6f} "
                        f"loop_rate={float(eval_metrics_agg.get('LoopRate', 0.0)):.6f}"
                    ),
                    flush=True,
                )
                wandb_run.log(
                    {
                        "eval/mean_return": float(eval_metrics_agg.get("MeanReturn", 0.0)),
                        "eval/avg_reward": float(eval_metrics_agg.get("AvgReward", 0.0)),
                        "eval/mean_episode_len": float(eval_metrics_agg.get("MeanEpisodeLen", 0.0)),
                        "eval/win_rate": float(eval_metrics_agg.get("WinRate", 0.0)),
                    },
                    step=iter_idx,
                )
                _append_jsonl(
                    os.path.join(run_dir, "metrics", f"eval_iter_{iter_idx:06d}_episodes.jsonl"),
                    eval_per_ep,
                )

                for t_idx, ep in enumerate(eval_batch.get("episodes", [])[: max(0, trace_eval_episodes)]):
                    _append_jsonl(
                        os.path.join(run_dir, "traces", f"eval_iter_{iter_idx:06d}_ep_{t_idx:03d}.jsonl"),
                        [
                            {
                                "step_idx": s.get("step_idx"),
                                "mode_id": s.get("mode_id"),
                                "action": s.get("action_key"),
                                "reward": s.get("reward"),
                                "reward_terms": s.get("reward_terms"),
                                "done": s.get("done"),
                                "state_hash_before": s.get("state_hash_before"),
                                "state_hash_after": s.get("state_hash_after"),
                                "effect_flag": bool(((s.get("transition_event") or {}).get("grid_delta") or {}).get("changed_cells_count", 0) > 0),
                                "novel_flag": bool(s.get("reward_terms", {}).get("r_novel", 0.0) > 0.0),
                                "loop_flag": bool(s.get("reward_terms", {}).get("r_loop", 0.0) < 0.0),
                            }
                            for s in ep.get("steps", [])
                        ],
                    )
                    for s in ep.get("steps", []):
                        assert_no_non_rl_trace_entry(s)

                eval_win = float(eval_report.get("EvalWinRate", 0.0))
                if eval_win >= switch_threshold:
                    summary["consecutive_eval_win_hits"] = int(summary.get("consecutive_eval_win_hits", 0)) + 1
                else:
                    summary["consecutive_eval_win_hits"] = 0

                if summary.get("stage") == "exploration" and summary["consecutive_eval_win_hits"] >= switch_k:
                    summary["stage"] = "win"

                if args.phase != 0:
                    candidate = dict(eval_report)
                    candidate["iter"] = iter_idx

                if games_eval_easy:
                    logger.info("eval_easy_start iter=%s episodes=%s", iter_idx, int(resolved_eval_episodes))
                    easy_batch = _collect_batch(
                        agent,
                        modules,
                        games=games_eval_easy,
                        op_mode=args.op_mode,
                        seed_base=eval_seed_base + 10_000_000,
                        episodes=int(resolved_eval_episodes),
                        max_actions=resolved_max_steps,
                        stochastic=False,
                        workers=max(1, int(args.workers)),
                        pool=shared_pool,
                        iter_idx=iter_idx,
                        rollout_cfg_hash=str(rollout_cfg_hash),
                        rollout_cfg_payload=rollout_cfg_payload,
                        render_terminal=bool(args.render_terminal),
                    )
                    easy_metrics, _ = _aggregate_metrics(easy_batch, cfg)
                    _write_json(
                        os.path.join(run_dir, "metrics", f"eval_easy_iter_{iter_idx:06d}.json"),
                        {"iter": iter_idx, **_prefixed(easy_metrics, "EvalEasy")},
                    )
                    logger.info(
                        "eval_easy_done iter=%s win_rate=%.6f mean_return=%.6f",
                        iter_idx,
                        float(easy_metrics.get("WinRate", 0.0)),
                        float(easy_metrics.get("MeanReturn", 0.0)),
                    )

                _write_json(os.path.join(run_dir, "metrics", "summary.json"), summary)
        run_completed = True
    finally:
        if shared_pool is not None:
            if run_completed:
                shared_pool.close()
            else:
                shared_pool.terminate()
            shared_pool.join()
            logger.info("shared_pool_closed completed=%s", run_completed)
        wandb_run.finish()
        logger.info("run_finish mode=%s completed=%s", args.mode, run_completed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

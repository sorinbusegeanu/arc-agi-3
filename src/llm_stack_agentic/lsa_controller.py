from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .lsa_bootstrap_explorer import run_bootstrap_explorer
from .lsa_bootstrap_explorer_types import normalize_available_actions, normalize_obs


def run_episode(
    *,
    game_id: str,
    episode_id: str,
    seed: int,
    controller_config: Optional[Dict[str, Any]],
    env_adapter: Any,
    agents: Optional[Dict[str, Any]] = None,
    services: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _normalize_config(controller_config or {}, game_id=game_id, seed=seed)
    agents = agents or {}
    services = services or {}

    outdir = cfg["outdir"]
    os.makedirs(outdir, exist_ok=True)

    blackboard = _init_blackboard()
    blackboard["errors"] = []
    blackboard["episode_action_log"] = []

    try:
        raw_obs = env_adapter.reset(episode_id, game_id, seed)
    except Exception as exc:  # noqa: BLE001
        return _episode_error(
            episode_id,
            game_id,
            seed,
            done_reason="env_error",
            errors=[f"reset_failed:{exc}"],
            outdir=outdir,
        )

    obs = normalize_obs(env_adapter.to_canonical_obs(raw_obs))
    avail = normalize_available_actions(env_adapter.get_available_actions(raw_obs))
    blackboard.update(
        {
            "obs": obs.to_dict(),
            "available_actions": avail.to_dict(),
            "step_idx": 0,
            "segment_index": 0,
            "branch_key": "main",
            "ignore_mask": [],
            "env_report": _placeholder_env_report(episode_id, game_id, seed, errors=["not_generated"]),
            "poi_history": [],
            "macro_buffer_handle": None,
        }
    )

    mode = "REPLAY_CHAIN"
    round_idx = 0
    total_steps = 0
    won = False
    done_reason = "max_actions"
    macro_chain: List[Dict[str, Any]] = []
    final_action_sequence: List[Dict[str, Any]] = []
    traces: Dict[str, Any] = {}

    while True:
        if total_steps >= cfg["max_actions_total"]:
            done_reason = "max_actions"
            mode = "TERMINATE"
        if round_idx >= cfg["max_rounds"]:
            done_reason = "max_rounds"
            mode = "TERMINATE"

        if mode == "TERMINATE":
            break

        if mode == "REPLAY_CHAIN":
            replay_result = _replay_chain(cfg, env_adapter, agents, blackboard)
            traces["replay_result"] = replay_result
            if replay_result.get("won"):
                won = True
                done_reason = "win"
                final_action_sequence = replay_result.get("final_action_sequence", [])
                macro_chain = replay_result.get("macro_chain", [])
                mode = "TERMINATE"
            else:
                mode = "BOOTSTRAP"
            continue

        if mode == "BOOTSTRAP":
            trace_id = _trace_id(episode_id, seed, round_idx, "bootstrap")
            probe_trace, bootstrap_report = run_bootstrap_explorer(
                episode_id=episode_id,
                game_id=game_id,
                seed=seed,
                env_adapter=env_adapter,
                probe_steps=cfg["probe_steps"],
                policy_config=cfg.get("bootstrap_policy_config") or None,
                logging_config={"trace_id": trace_id, "timestamp_step": blackboard["step_idx"]},
            )
            traces["probe_trace"] = _write_artifact(outdir, "probe_trace.json", probe_trace, services)
            traces["bootstrap_report"] = _write_artifact(outdir, "bootstrap_report.json", bootstrap_report, services)

            # Reset env to round start after bootstrap.
            raw_obs = env_adapter.reset(episode_id, game_id, seed)
            obs = normalize_obs(env_adapter.to_canonical_obs(raw_obs))
            avail = normalize_available_actions(env_adapter.get_available_actions(raw_obs))
            blackboard["obs"] = obs.to_dict()
            blackboard["available_actions"] = avail.to_dict()
            blackboard["step_idx"] = 0

            mode = "DESCRIBE"
            continue

        if mode == "DESCRIBE":
            env_report = _call_visual_describer(
                episode_id=episode_id,
                game_id=game_id,
                seed=seed,
                probe_trace=_load_artifact(traces.get("probe_trace"), services) or {},
                agents=agents,
                cfg=cfg,
                round_idx=round_idx,
                blackboard=blackboard,
            )
            blackboard["env_report"] = env_report
            traces["env_report"] = _write_artifact(outdir, "env_report.json", env_report, services)
            mode = "ROUTE_POIS"
            continue

        if mode == "ROUTE_POIS":
            poi_tasks, active_ignore_mask = _route_pois(cfg, blackboard)
            blackboard["ignore_mask"] = active_ignore_mask
            traces["poi_tasks"] = _write_artifact(outdir, "poi_tasks.json", {"tasks": poi_tasks}, services)
            blackboard["poi_tasks"] = poi_tasks
            mode = "EXPLORE_POIS"
            continue

        if mode == "EXPLORE_POIS":
            poi_results = _dispatch_poi_tasks(
                cfg,
                env_adapter,
                episode_id,
                game_id,
                seed,
                round_idx,
                blackboard,
            )
            for result in poi_results:
                poi_id = result.get("poi_id", "unknown")
                name = f"poi_run_result_{poi_id}.json"
                _write_artifact(outdir, name, result, services)
            blackboard["poi_results"] = poi_results
            mode = "DETECT_CP_AND_STORE"
            continue

        if mode == "DETECT_CP_AND_STORE":
            _detect_cp_and_store(cfg, agents, blackboard, outdir, services)
            round_idx += 1
            mode = "REPLAY_CHAIN"
            continue

        if mode == "FAIL_ANALYZE":
            _fail_analyze(cfg, agents, blackboard, outdir, services)
            mode = "ROUTE_POIS"
            continue

        # Fallback to terminate on unknown mode.
        done_reason = "unknown_mode"
        mode = "TERMINATE"

    summary = {
        "won": bool(won),
        "done_reason": done_reason,
        "total_steps": int(total_steps),
        "final_action_sequence": final_action_sequence,
        "macro_chain": macro_chain,
        "all_traces": traces,
        "metrics_summary": {},
        "schema_version": "EpisodeRunV1",
        "episode_id": episode_id,
        "game_id": game_id,
        "seed": int(seed),
    }
    _write_artifact(outdir, "episode_summary.json", summary, services)
    return summary


def _init_blackboard() -> Dict[str, Any]:
    return {
        "obs": None,
        "available_actions": None,
        "step_idx": 0,
        "ignore_mask": [],
        "env_report": None,
        "poi_history": [],
        "macro_buffer_handle": None,
        "segment_index": 0,
        "branch_key": "main",
        "episode_action_log": [],
        "episode_transition_log": [],
        "errors": [],
    }


def _normalize_config(cfg: Dict[str, Any], *, game_id: str, seed: int) -> Dict[str, Any]:
    outdir = cfg.get("outdir")
    if not outdir:
        outdir = os.path.join("runs", "lsa_controller", f"{game_id}_{seed}")
    env_parallel_mode = str(cfg.get("env_parallel_mode", "sequential"))
    if env_parallel_mode != "sequential":
        raise ValueError(f"env_parallel_mode '{env_parallel_mode}' not supported in v1 controller")
    return {
        "outdir": outdir,
        "max_actions_total": int(cfg.get("max_actions_total", 200)),
        "probe_steps": int(cfg.get("probe_steps", 4)),
        "top_k_pois_per_round": int(cfg.get("top_k_pois_per_round", 3)),
        "per_poi_step_budget": int(cfg.get("per_poi_step_budget", 12)),
        "parallel_poi_workers": int(cfg.get("parallel_poi_workers", 1)),
        "max_rounds": int(cfg.get("max_rounds", 5)),
        "stagnation_patience_steps": int(cfg.get("stagnation_patience_steps", 50)),
        "env_parallel_mode": env_parallel_mode,
        "bootstrap_policy_config": cfg.get("bootstrap_policy_config"),
        "default_poi_fallback": bool(cfg.get("default_poi_fallback", True)),
        "model_config": _load_model_config(cfg.get("model_config")),
        "debug": bool(cfg.get("debug", False)),
        "debug_log_path": cfg.get("debug_log_path") or os.path.join("runs", "debug.log"),
        "png_factor": int(cfg.get("png_factor", 1)),
    }


def _load_model_config(model_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if model_config:
        return model_config
    path = os.path.join(os.path.dirname(__file__), "model_config_default.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _episode_error(
    episode_id: str,
    game_id: str,
    seed: int,
    done_reason: str,
    errors: List[str],
    outdir: str,
) -> Dict[str, Any]:
    summary = {
        "schema_version": "EpisodeRunV1",
        "episode_id": episode_id,
        "game_id": game_id,
        "seed": int(seed),
        "won": False,
        "done_reason": done_reason,
        "total_steps": 0,
        "final_action_sequence": [],
        "macro_chain": [],
        "all_traces": {},
        "metrics_summary": {"errors": errors},
    }
    _write_json_atomic(os.path.join(outdir, "episode_summary.json"), summary)
    return summary


def _placeholder_env_report(episode_id: str, game_id: str, seed: int, errors: List[str]) -> Dict[str, Any]:
    return {
        "schema_version": "EnvReportV1",
        "agent_name": "lsa_visual_describer",
        "episode_id": episode_id,
        "game_id": game_id,
        "seed": int(seed),
        "model_name": "unknown",
        "generation_params": {},
        "game_description": "",
        "sprite_character": {},
        "ignore_regions": [],
        "poi_list": [],
        "exit_hypotheses": [],
        "errors": errors,
    }


def _call_visual_describer(
    *,
    episode_id: str,
    game_id: str,
    seed: int,
    probe_trace: Dict[str, Any],
    agents: Dict[str, Any],
    cfg: Dict[str, Any],
    round_idx: int,
    blackboard: Dict[str, Any],
) -> Dict[str, Any]:
    describer = agents.get("visual_describer")
    if describer is None:
        return _placeholder_env_report(episode_id, game_id, seed, ["visual_describer_missing"])
    return describer.run(
        episode_id=episode_id,
        game_id=game_id,
        seed=seed,
        probe_trace=probe_trace,
        max_pois=5,
        model_config=cfg.get("model_config"),
        logging_config={
            "trace_id": _trace_id(episode_id, seed, round_idx, "describe"),
            "timestamp_step": blackboard["step_idx"],
            "debug": cfg.get("debug", False),
            "debug_log_path": cfg.get("debug_log_path"),
            "frame_dir": os.path.join(cfg["outdir"], "vl_frames"),
            "png_factor": cfg.get("png_factor", 1),
        },
    )


def _route_pois(cfg: Dict[str, Any], blackboard: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Any]:
    env_report = blackboard.get("env_report") or {}
    poi_list = env_report.get("poi_list") or []
    tasks: List[Dict[str, Any]] = []
    for idx, poi in enumerate(poi_list[: cfg["top_k_pois_per_round"]]):
        tasks.append(
            {
                "task_id": f"poi_task_{idx}",
                "poi_id": poi.get("id", f"poi_{idx}"),
                "target_xy": (poi.get("x"), poi.get("y")),
                "intent_tag": poi.get("intent", "unknown"),
                "step_budget": cfg["per_poi_step_budget"],
                "policy_mode": "simple",
                "priority": poi.get("priority", idx + 1),
            }
        )
    if not tasks and cfg.get("default_poi_fallback", True):
        obs = blackboard.get("obs") or {}
        w = int(obs.get("w", 0))
        h = int(obs.get("h", 0))
        target = (w // 2 if w else 0, h // 2 if h else 0)
        tasks.append(
            {
                "task_id": "poi_task_default",
                "poi_id": "poi_default",
                "target_xy": target,
                "intent_tag": "default_scan",
                "step_budget": cfg["per_poi_step_budget"],
                "policy_mode": "simple",
                "priority": 1,
            }
        )
    ignore_mask = env_report.get("ignore_regions") or []
    return tasks, ignore_mask


def _dispatch_poi_tasks(
    cfg: Dict[str, Any],
    env_adapter: Any,
    episode_id: str,
    game_id: str,
    seed: int,
    round_idx: int,
    blackboard: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    tasks = blackboard.get("poi_tasks", [])
    for task in tasks:
        task_seed = _task_seed(seed, episode_id, round_idx, str(task.get("poi_id")))
        result = _run_single_poi_task(cfg, env_adapter, episode_id, game_id, task_seed, task)
        results.append(result)
    return results


def _run_single_poi_task(
    cfg: Dict[str, Any],
    env_adapter: Any,
    episode_id: str,
    game_id: str,
    seed: int,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    trajectory: List[Dict[str, Any]] = []
    try:
        raw_obs = env_adapter.reset(episode_id, game_id, seed)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "PoiRunResultV1",
            "poi_id": task.get("poi_id"),
            "reached": False,
            "reach_step": None,
            "trajectory": [],
            "nav_metrics": {"errors": [f"reset_failed:{exc}"]},
            "end_state_hash": None,
        }
    for t in range(int(task.get("step_budget", 0))):
        obs = normalize_obs(env_adapter.to_canonical_obs(raw_obs))
        avail = normalize_available_actions(env_adapter.get_available_actions(raw_obs))
        action = _select_poi_action(task, obs.to_dict(), avail.to_dict(), t)
        if action is None:
            break
        raw_next, reward, done, info = env_adapter.step(action)
        trajectory.append(
            {
                "t": t,
                "obs": obs.to_dict(),
                "action": action,
                "done": bool(done),
                "available_actions": avail.to_dict(),
            }
        )
        raw_obs = raw_next
        if done:
            break

    return {
        "schema_version": "PoiRunResultV1",
        "poi_id": task.get("poi_id"),
        "reached": False,
        "reach_step": None,
        "trajectory": trajectory,
        "nav_metrics": {"distance_trace": [], "stuck": False, "repeats": 0},
        "end_state_hash": _hash_obs(trajectory[-1]["obs"]) if trajectory else None,
    }


def _select_poi_action(task: Dict[str, Any], obs: Dict[str, Any], avail: Dict[str, Any], t: int) -> Optional[Dict[str, Any]]:
    target = task.get("target_xy")
    coord_enabled = bool(avail.get("coord_enabled"))
    coord_action_id = avail.get("coord_action_id")
    if coord_enabled and target and t == 0:
        x, y = target
        return {"type": "coord", "id": int(coord_action_id or 0), "x": int(x), "y": int(y)}
    mask = avail.get("discrete_mask") or []
    for idx, enabled in enumerate(mask):
        if enabled:
            return {"type": "discrete", "id": int(idx)}
    return None


def _detect_cp_and_store(
    cfg: Dict[str, Any],
    agents: Dict[str, Any],
    blackboard: Dict[str, Any],
    outdir: str,
    services: Dict[str, Any],
) -> None:
    cp_detector = agents.get("change_point_detector")
    segment_memory = agents.get("segment_memory")
    poi_results = blackboard.get("poi_results", [])
    for result in poi_results:
        if cp_detector is None:
            continue
        cp_result = cp_detector.run(
            trajectory=result.get("trajectory"),
            active_ignore_mask=blackboard.get("ignore_mask"),
            cfg=cfg.get("cp_config", {}),
        )
        poi_id = result.get("poi_id", "unknown")
        if isinstance(cp_result, dict):
            _write_artifact(outdir, f"cp_segment_{poi_id}.json", cp_result, services)
        if segment_memory is not None and isinstance(cp_result, dict):
            segment_memory.run(segment=cp_result.get("segment"), cp_signature=cp_result.get("cp_signature"))


def _fail_analyze(
    cfg: Dict[str, Any],
    agents: Dict[str, Any],
    blackboard: Dict[str, Any],
    outdir: str,
    services: Dict[str, Any],
) -> None:
    failure_analyser = agents.get("failure_analyser")
    if failure_analyser is None:
        return
    result = failure_analyser.run(
        probe_traces=blackboard.get("probe_traces"),
        poi_run_results=blackboard.get("poi_results"),
        segments=blackboard.get("segments"),
        terminal_reason="unknown",
        env_report=blackboard.get("env_report"),
        poi_history=blackboard.get("poi_history"),
    )
    _write_artifact(outdir, "failure_analyser.json", result, services)


def _replay_chain(
    cfg: Dict[str, Any],
    env_adapter: Any,
    agents: Dict[str, Any],
    blackboard: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "ReplayResultV1",
        "won": False,
        "macro_chain": [],
        "final_action_sequence": [],
        "errors": ["not_implemented"],
    }


def _trace_id(episode_id: str, seed: int, round_idx: int, name: str) -> str:
    payload = f"{episode_id}:{seed}:{round_idx}:{name}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _task_seed(seed: int, episode_id: str, round_idx: int, poi_id: str) -> int:
    payload = f"{seed}:{episode_id}:{round_idx}:{poi_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _hash_obs(obs: Dict[str, Any]) -> str:
    payload = json.dumps(obs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_artifact(outdir: str, name: str, payload: Any, services: Dict[str, Any]) -> str:
    path = os.path.join(outdir, name)
    storage = services.get("storage")
    if storage and hasattr(storage, "write_json"):
        storage.write_json(path, payload)
    else:
        _write_json_atomic(path, payload)
    return path


def _load_artifact(path: Optional[str], services: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    storage = services.get("storage")
    if storage and hasattr(storage, "read_json"):
        return storage.read_json(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json_atomic(path: str, payload: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)

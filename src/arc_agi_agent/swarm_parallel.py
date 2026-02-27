from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from .swarm_agent_registry import build_default_agents, default_call_order
from .swarm_orchestrator import SwarmOrchestratorConfig, run_game
from .trajectory_summarizer import summarize as summarize_trajectory


def _prepare_paths() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if "ENVIRONMENTS_DIR" not in os.environ:
        os.environ["ENVIRONMENTS_DIR"] = os.path.join(base_dir, "environment_files")
    extra = [
        os.path.join(base_dir, "src"),
        os.path.join(base_dir, "other_repos", "arc-agi"),
        os.path.join(base_dir, "other_repos", "ARCEngine"),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    paths = [p for p in existing.split(os.pathsep) if p]
    for path in extra:
        if path not in paths:
            paths.append(path)
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def _job_dir(base_outdir: str, game_id: str, seed: int) -> str:
    return os.path.join(base_outdir, game_id, str(seed))


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_job(args: Tuple[str, int, Dict[str, Any]]) -> Dict[str, Any]:
    game_id, seed, job_cfg = args
    _prepare_paths()

    from arc_agi import Arcade, OperationMode

    base_outdir = job_cfg["base_outdir"]
    op_mode = job_cfg["op_mode"]
    max_steps = job_cfg["max_steps"]
    probe_steps = job_cfg["probe_steps"]
    debug = bool(job_cfg.get("debug", False))
    run_id = f"{game_id}_{seed}"
    outdir = _job_dir(base_outdir, game_id, seed)

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    trace_path = os.path.join(outdir, "decision_trace.jsonl")
    if os.path.exists(trace_path):
        os.remove(trace_path)

    arcade = Arcade(operation_mode=OperationMode(op_mode))
    env = arcade.make(game_id, seed=seed, render_mode=None)
    if env is None:
        return {"game_id": game_id, "seed": seed, "ok": False, "error": "env_create_failed", "outdir": outdir}

    agents = build_default_agents()
    cfg = SwarmOrchestratorConfig(
        max_steps_total=max_steps,
        probe_steps=probe_steps,
        exploit_steps=max(0, max_steps - probe_steps),
        debug=debug,
    )

    started = time.time()
    exit_reason = "completed"
    error_msg = None
    try:
        blackboard = run_game(
            env,
            game_id,
            seed,
            agents,
            cfg=cfg,
            outdir=outdir,
            call_order=default_call_order(),
        )
        action_schema_path = os.path.join(outdir, "action_schema.json")
        _write_json(action_schema_path, blackboard.action_schema)

        if os.path.exists(trace_path):
            summarize_trajectory(
                planner_trace=trace_path,
                outdir=outdir,
                action_schema=blackboard.action_schema,
                ctx={"game_id": game_id, "seed": seed, "run_id": blackboard.run_id},
            )
    except Exception as exc:
        exit_reason = "error"
        error_msg = str(exc)
        try:
            with open(os.path.join(outdir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(error_msg)
        except Exception:
            pass
        return {
            "game_id": game_id,
            "seed": seed,
            "ok": False,
            "error": error_msg,
            "outdir": outdir,
        }
    finally:
        ended = time.time()
        run_meta = {
            "run_id": run_id,
            "game_id": game_id,
            "seed": seed,
            "config": {"max_steps": max_steps, "probe_steps": probe_steps, "op_mode": op_mode},
            "call_order": default_call_order(),
            "started_at": started,
            "ended_at": ended,
            "exit_reason": exit_reason,
            "error": error_msg,
        }
        _write_json(os.path.join(outdir, "run_meta.json"), run_meta)

    return {"game_id": game_id, "seed": seed, "ok": True, "outdir": outdir}


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel swarm coordinator")
    parser.add_argument("--games", required=True, help="Comma-separated game ids (e.g. ls20,bt11) or 'all'")
    parser.add_argument("--start-seed", type=int, required=True, help="Start seed (incremented)")
    parser.add_argument("--seeds-per-game", type=int, help="Number of seeds per game (default: workers)")
    parser.add_argument("--workers", type=int, required=True, help="Worker process count")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per run")
    parser.add_argument("--probe-steps", type=int, default=900, help="Probe steps per run")
    parser.add_argument("--debug", action="store_true", help="Enable detailed audit logging")
    parser.add_argument("--op-mode", choices=["offline", "online"], default="offline")
    parser.add_argument(
        "--outdir",
        default="/home/zodrak/zod/runs/swarm_parallel",
        help="Base output dir for batch (default: /home/zodrak/zod/runs/swarm_parallel)",
    )

    args = parser.parse_args()
    _prepare_paths()
    if args.games.strip().lower() == "all":
        from arc_agi import Arcade, OperationMode

        arcade = Arcade(operation_mode=OperationMode(args.op_mode))
        envs = arcade.get_environments()
        games = []
        seen = set()
        for env_info in envs:
            base_id = env_info.game_id.split("-", 1)[0]
            if base_id not in seen:
                seen.add(base_id)
                games.append(base_id)
        if not games:
            raise SystemExit("No environments available from Arcade")
    else:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
        if not games:
            raise SystemExit("No games provided")

    batch_outdir = f"{args.outdir}_{args.start_seed}"
    os.makedirs(batch_outdir, exist_ok=True)
    seeds_per_game = args.seeds_per_game if args.seeds_per_game is not None else args.workers
    batch_meta = {
        "games": games,
        "start_seed": args.start_seed,
        "seeds_per_game": seeds_per_game,
        "workers": args.workers,
        "config": {
            "max_steps": args.max_steps,
            "probe_steps": args.probe_steps,
            "op_mode": args.op_mode,
            "debug": args.debug,
        },
    }
    _write_json(os.path.join(batch_outdir, "batch_meta.json"), batch_meta)

    jobs: List[Tuple[str, int, Dict[str, Any]]] = []
    for game_id in games:
        for i in range(seeds_per_game):
            seed = args.start_seed + i
            jobs.append(
                (
                    game_id,
                    seed,
                    {
                        "base_outdir": batch_outdir,
                        "op_mode": args.op_mode,
                        "max_steps": args.max_steps,
                        "probe_steps": args.probe_steps,
                        "debug": args.debug,
                    },
                )
            )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    aggregate_path = os.path.join(batch_outdir, "aggregate_lessons.jsonl")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        for res in results:
            lessons_path = os.path.join(res["outdir"], "lessons.json")
            payload = {"game_id": res["game_id"], "seed": res["seed"], "ok": res["ok"], "outdir": res["outdir"]}
            if os.path.exists(lessons_path):
                payload["lessons"] = json.load(open(lessons_path))
            f.write(json.dumps(payload) + "\n")

    stats = {
        "total_runs": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
    }
    _write_json(os.path.join(batch_outdir, "aggregate_stats.json"), stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

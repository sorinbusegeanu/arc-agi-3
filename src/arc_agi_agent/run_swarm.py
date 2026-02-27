from __future__ import annotations

import argparse
import json
import os
import sys

from .swarm_agent_registry import build_default_agents, default_call_order
from .swarm_orchestrator import SwarmOrchestratorConfig, run_game
from .trajectory_summarizer import summarize as summarize_trajectory


def _prepare_paths() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    arc_agi_path = os.path.join(base_dir, "other_repos", "arc-agi")
    arcengine_path = os.path.join(base_dir, "other_repos", "ARCEngine")
    arcengine_pkg_path = os.path.join(arcengine_path, "arcengine")

    for path in (arc_agi_path, arcengine_path, arcengine_pkg_path):
        if path not in sys.path:
            sys.path.insert(0, path)

    if "ENVIRONMENTS_DIR" not in os.environ:
        os.environ["ENVIRONMENTS_DIR"] = os.path.join(base_dir, "environment_files")


def main() -> int:
    parser = argparse.ArgumentParser(description="Swarm Orchestrator runner")
    parser.add_argument("--game", required=True, help="Game id")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--max-steps", type=int, default=40, help="Max steps total")
    parser.add_argument("--probe-steps", type=int, default=10, help="Probe steps budget")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable detailed audit logging")
    parser.add_argument(
        "--op-mode",
        choices=["offline", "online"],
        default="offline",
        help="Operation mode for env (default: offline)",
    )

    args = parser.parse_args()
    _prepare_paths()

    env_dir = os.environ.get("ENVIRONMENTS_DIR")
    if not env_dir:
        raise SystemExit("ENVIRONMENTS_DIR is not set")
    if not os.path.isdir(env_dir):
        raise SystemExit(f"ENVIRONMENTS_DIR does not exist: {env_dir}")
    game_dir = os.path.join(env_dir, args.game)
    if not os.path.isdir(game_dir):
        raise SystemExit(f"Game directory not found in ENVIRONMENTS_DIR: {game_dir}")

    os.makedirs(args.outdir, exist_ok=True)

    from arc_agi import Arcade, OperationMode

    arcade = Arcade(operation_mode=OperationMode(args.op_mode))
    env = arcade.make(args.game, seed=args.seed, render_mode=None)
    if env is None:
        raise SystemExit(f"Failed to create environment for {args.game}")

    cfg = SwarmOrchestratorConfig(
        max_steps_total=args.max_steps,
        probe_steps=args.probe_steps,
        exploit_steps=max(0, args.max_steps - args.probe_steps),
        debug=args.debug,
    )
    agents = build_default_agents()
    trace_path = os.path.join(args.outdir, "decision_trace.jsonl")
    if os.path.exists(trace_path):
        os.remove(trace_path)

    blackboard = run_game(env, args.game, args.seed, agents, cfg=cfg, outdir=args.outdir, call_order=default_call_order())
    action_schema_path = os.path.join(args.outdir, "action_schema.json")
    with open(action_schema_path, "w", encoding="utf-8") as f:
        json.dump(blackboard.action_schema, f, indent=2)
    if os.path.exists(trace_path):
        summarize_trajectory(
            planner_trace=trace_path,
            outdir=args.outdir,
            action_schema=blackboard.action_schema,
            ctx={"game_id": args.game, "seed": args.seed, "run_id": blackboard.run_id},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

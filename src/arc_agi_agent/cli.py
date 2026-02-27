from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict

from .config import FPAnalystConfig
from .fp_analyst import FPAnalyst
from .logger import get_logger
from .action_schema import build_action_schema_from_env, parse_action_schema_data
from .full_explorer import run as run_full_explorer
from .full_explorer_config import FullExplorerConfig
from .goal_detector import estimate as estimate_goal
from .goal_detector_config import GoalDetectorConfig
from .grid_utils import bbox_area
from .planner import plan_next
from .planner_config import PlannerConfig
from .planner_types import PlannerInputs, PlannerState
from .trajectory_summarizer import summarize as summarize_trajectory
from .trajectory_summarizer_config import TrajectorySummarizerConfig
from .mechanic_classifier import classify as classify_mechanics
from .mechanic_classifier_config import MechanicClassifierConfig
from .rule_proposer import propose as propose_rules
from .rule_proposer_config import RuleProposerConfig
from .simple_explorer import run as run_simple_explorer
from .simple_explorer_config import SimpleExplorerConfig

logger = get_logger(__name__)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _serialize_report(report: Any) -> Dict[str, Any]:
    payload = asdict(report)
    graph = payload.get("transition_graph")
    if graph and hasattr(report, "transition_graph"):
        graph["edges"] = [asdict(edge) for edge in report.transition_graph.edges.values()]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="ARC-AGI agent runner")
    parser.add_argument(
        "--agent",
        required=True,
        choices=[
            "fp_analyst",
            "simple_explorer",
            "full_explorer",
            "rule_proposer",
            "mechanic_classifier",
            "goal_detector",
            "planner",
            "trajectory_summarizer",
        ],
        help="Agent name",
    )
    parser.add_argument("--input", help="Path to frame JSON (fp_analyst only)")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--save-viz", action="store_true", help="Save PNG visualizations if PIL is available")
    parser.add_argument("--format", default="both", choices=["ascii", "json", "both"], help="Output format (fp_analyst)")
    parser.add_argument("--input_fp", action="append", help="Path to fp_analyst report.json (repeatable)")
    parser.add_argument("--simple", help="Path to simple_explorer report.json")
    parser.add_argument("--full", help="Path to full_explorer report.json")
    parser.add_argument("--action-schema", dest="action_schema", help="Path to action_schema JSON")
    parser.add_argument("--trace", help="Path to trace jsonl (goal_detector)")
    parser.add_argument("--mechanic", help="Path to mechanic_classifier report.json")
    parser.add_argument("--hypotheses", help="Path to rule_proposer report.json")
    parser.add_argument("--goal", help="Path to goal_detector report.json")
    parser.add_argument("--planner-trace", help="Path to planner decision trace jsonl")
    parser.add_argument("--simple-trace", help="Path to simple explorer trace jsonl")
    parser.add_argument("--full-trace", help="Path to full explorer trace jsonl")
    parser.add_argument("--fp-dir", help="Directory containing fp_step_<step_idx>.json")
    parser.add_argument("--action-schema", dest="action_schema", help="Path to action_schema JSON")

    parser.add_argument("--game", help="Game id (simple_explorer/full_explorer) or 'all'")
    parser.add_argument("--seed", type=int, default=0, help="Seed (simple_explorer/full_explorer)")
    parser.add_argument("--max-steps", type=int, default=80, help="Max steps (simple_explorer/full_explorer)")
    parser.add_argument(
        "--op-mode",
        choices=["offline", "normal", "online"],
        default="offline",
        help="Operation mode for simple_explorer (default: offline)",
    )

    args = parser.parse_args()

    if args.agent == "fp_analyst":
        if not args.input or not args.outdir:
            raise SystemExit("--input and --outdir are required for fp_analyst")
        os.makedirs(args.outdir, exist_ok=True)
        observation = _load_json(args.input)
        config = FPAnalystConfig(save_images=args.save_viz, output_dir=args.outdir)
        analyst = FPAnalyst(config=config)
        report = analyst.analyze(observation)
        if args.format in ("ascii", "both"):
            ascii_output = analyst.render(report, mode="ascii")
            _write_text(os.path.join(args.outdir, "report.txt"), ascii_output)
        if args.format in ("json", "both"):
            _write_json(os.path.join(args.outdir, "report.json"), asdict(report))
        if report.viz_artifacts.save_paths:
            logger.info("Saved images: %s", report.viz_artifacts.save_paths)
        return 0

    if args.agent == "simple_explorer":
        if not args.game:
            raise SystemExit("--game is required for simple_explorer")
        if args.game != "all":
            output_dir = args.outdir or os.path.join("runs", "simple_explorer", f"{args.game}_{args.seed}")
            os.makedirs(output_dir, exist_ok=True)

        arc_agi_path = "/home/zodrak/zod/other_repos/arc-agi"
        arcengine_path = "/home/zodrak/zod/other_repos/ARCEngine"
        if arc_agi_path not in sys.path:
            sys.path.insert(0, arc_agi_path)
        arcengine_pkg_path = os.path.join(arcengine_path, "arcengine")
        if arcengine_path not in sys.path:
            sys.path.insert(0, arcengine_path)
        if arcengine_pkg_path not in sys.path:
            sys.path.insert(0, arcengine_pkg_path)
        from arc_agi import Arcade, OperationMode

        if "ENVIRONMENTS_DIR" not in os.environ:
            os.environ["ENVIRONMENTS_DIR"] = "/home/zodrak/zod/environment_files"

        arcade = Arcade(operation_mode=OperationMode(args.op_mode))
        fp_analyst = FPAnalyst()
        cfg = SimpleExplorerConfig(max_steps=args.max_steps, save_viz=args.save_viz)

        game_ids: list[str]
        if args.game == "all":
            envs = arcade.get_environments()
            game_ids = []
            seen = set()
            for env_info in envs:
                base_id = env_info.game_id.split("-", 1)[0]
                if base_id not in seen:
                    seen.add(base_id)
                    game_ids.append(base_id)
            if not game_ids:
                raise SystemExit("No environments available from Arcade")
        else:
            game_ids = [args.game]

        for game_id in game_ids:
            output_dir = args.outdir or os.path.join("runs", "simple_explorer", f"{game_id}_{args.seed}")
            os.makedirs(output_dir, exist_ok=True)
            env = arcade.make(game_id, seed=args.seed, render_mode=None)
            if env is None:
                logger.error("Failed to create environment for %s", game_id)
                continue
            report = run_simple_explorer(
                env=env,
                game_id=game_id,
                seed=args.seed,
                fp_analyst=fp_analyst,
                cfg=cfg,
                ctx={"output_dir": output_dir},
            )
        _write_json(os.path.join(output_dir, "report.json"), _serialize_report(report))
        return 0

    if args.agent == "full_explorer":
        if not args.game:
            raise SystemExit("--game is required for full_explorer")
        output_dir = args.outdir or os.path.join("runs", "full_explorer", f"{args.game}_{args.seed}")
        os.makedirs(output_dir, exist_ok=True)

        arc_agi_path = "/home/zodrak/zod/other_repos/arc-agi"
        arcengine_path = "/home/zodrak/zod/other_repos/ARCEngine"
        if arc_agi_path not in sys.path:
            sys.path.insert(0, arc_agi_path)
        arcengine_pkg_path = os.path.join(arcengine_path, "arcengine")
        if arcengine_path not in sys.path:
            sys.path.insert(0, arcengine_path)
        if arcengine_pkg_path not in sys.path:
            sys.path.insert(0, arcengine_pkg_path)
        from arc_agi import Arcade, OperationMode

        if "ENVIRONMENTS_DIR" not in os.environ:
            os.environ["ENVIRONMENTS_DIR"] = "/home/zodrak/zod/environment_files"

        arcade = Arcade(operation_mode=OperationMode(args.op_mode))
        fp_analyst = FPAnalyst()
        cfg = FullExplorerConfig(max_steps=args.max_steps, save_viz=args.save_viz)

        if args.game == "all":
            envs = arcade.get_environments()
            game_ids = []
            seen = set()
            for env_info in envs:
                base_id = env_info.game_id.split("-", 1)[0]
                if base_id not in seen:
                    seen.add(base_id)
                    game_ids.append(base_id)
            if not game_ids:
                raise SystemExit("No environments available from Arcade")
        else:
            game_ids = [args.game]

        for game_id in game_ids:
            output_dir = args.outdir or os.path.join("runs", "full_explorer", f"{game_id}_{args.seed}")
            os.makedirs(output_dir, exist_ok=True)
            env = arcade.make(game_id, seed=args.seed, render_mode=None)
            if env is None:
                logger.error("Failed to create environment for %s", game_id)
                continue
            report = run_full_explorer(
                env=env,
                game_id=game_id,
                seed=args.seed,
                fp_analyst=fp_analyst,
                cfg=cfg,
                ctx={"output_dir": output_dir},
            )
            _write_json(os.path.join(output_dir, "report.json"), _serialize_report(report))
        return 0

    if args.agent == "rule_proposer":
        if not args.input_fp:
            raise SystemExit("--input_fp is required for rule_proposer")
        if not args.action_schema:
            raise SystemExit("--action-schema is required for rule_proposer")
        if not args.outdir:
            raise SystemExit("--outdir is required for rule_proposer")
        os.makedirs(args.outdir, exist_ok=True)

        fp_reports = [_load_json(path) for path in args.input_fp]
        simple_report = _load_json(args.simple) if args.simple else None
        full_report = _load_json(args.full) if args.full else None
        action_schema_data = _load_json(args.action_schema)
        action_schema = parse_action_schema_data(action_schema_data)
        cfg = RuleProposerConfig()
        report = propose_rules(
            initial_fp_reports=fp_reports,
            simple_report=simple_report,
            full_report=full_report,
            action_schema=action_schema,
            cfg=cfg,
            ctx={"outdir": args.outdir},
        )
        _write_json(os.path.join(args.outdir, "rule_proposer_report.json"), _serialize_report(report))
        return 0

    if args.agent == "mechanic_classifier":
        if not args.input_fp:
            raise SystemExit("--input_fp is required for mechanic_classifier")
        if not args.outdir:
            raise SystemExit("--outdir is required for mechanic_classifier")
        os.makedirs(args.outdir, exist_ok=True)

        fp_reports = [_load_json(path) for path in args.input_fp]
        simple_report = _load_json(args.simple) if args.simple else None
        full_report = _load_json(args.full) if args.full else None
        action_schema = _load_json(args.action_schema) if args.action_schema else None
        cfg = MechanicClassifierConfig()
        report = classify_mechanics(
            fp_reports=fp_reports,
            simple_report=simple_report,
            full_report=full_report,
            action_schema=action_schema,
            cfg=cfg,
            ctx={"outdir": args.outdir},
        )
        _write_json(os.path.join(args.outdir, "mechanic_classifier_report.json"), _serialize_report(report))
        return 0

    if args.agent == "goal_detector":
        if not args.input_fp:
            raise SystemExit("--input_fp is required for goal_detector")
        if not args.outdir:
            raise SystemExit("--outdir is required for goal_detector")
        os.makedirs(args.outdir, exist_ok=True)

        fp_reports = [_load_json(path) for path in args.input_fp]
        cfg = GoalDetectorConfig()
        report = estimate_goal(
            fp_reports=fp_reports,
            trace_path=args.trace,
            cfg=cfg,
            ctx={"outdir": args.outdir},
        )
        _write_json(os.path.join(args.outdir, "goal_detector_report.json"), _serialize_report(report))
        return 0

    if args.agent == "planner":
        if not args.game:
            raise SystemExit("--game is required for planner")
        if not args.outdir:
            raise SystemExit("--outdir is required for planner")
        if args.op_mode == "offline":
            raise SystemExit("planner requires online/normal operation mode")
        os.makedirs(args.outdir, exist_ok=True)

        arc_agi_path = "/home/zodrak/zod/other_repos/arc-agi"
        arcengine_path = "/home/zodrak/zod/other_repos/ARCEngine"
        if arc_agi_path not in sys.path:
            sys.path.insert(0, arc_agi_path)
        arcengine_pkg_path = os.path.join(arcengine_path, "arcengine")
        if arcengine_path not in sys.path:
            sys.path.insert(0, arcengine_path)
        if arcengine_pkg_path not in sys.path:
            sys.path.insert(0, arcengine_pkg_path)
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction

        if "ENVIRONMENTS_DIR" not in os.environ:
            os.environ["ENVIRONMENTS_DIR"] = "/home/zodrak/zod/environment_files"

        arcade = Arcade(operation_mode=OperationMode(args.op_mode))
        env = arcade.make(args.game, seed=args.seed, render_mode=None)
        if env is None:
            raise SystemExit(f"Failed to create environment for {args.game}")
        obs = env.reset()
        if obs is None:
            raise SystemExit("planner failed to reset environment")
        fp_analyst = FPAnalyst()
        planner_state = PlannerState()
        cfg = PlannerConfig()

        mechanic_report = _load_json(args.mechanic) if args.mechanic else None
        hypotheses_report = _load_json(args.hypotheses) if args.hypotheses else None
        simple_report = _load_json(args.simple) if args.simple else None
        full_report = _load_json(args.full) if args.full else None
        goal_report = _load_json(args.goal) if args.goal else None

        inputs = PlannerInputs(
            mechanic_prior=mechanic_report,
            hypotheses_report=hypotheses_report,
            simple_report=simple_report,
            full_report=full_report,
            goal_report=goal_report,
            transition_graph=(simple_report or {}).get("transition_graph") if simple_report else None,
        )

        trace_path = os.path.join(args.outdir, "decision_trace.jsonl")
        with open(trace_path, "w", encoding="utf-8") as trace_file:
            for step_idx in range(args.max_steps):
                fp_report = fp_analyst.analyze(obs)
                if not fp_report.state_summary.grid_summaries:
                    raise SystemExit("planner requires at least one grid in fp_report")
                grid = fp_report.state_summary.grid_summaries[0]
                action_schema = build_action_schema_from_env(
                    env.action_space, width=grid.width, height=grid.height
                )
                action_dict, planner_state, decision_trace = plan_next(
                    observation=obs,
                    planner_state=planner_state,
                    inputs=inputs,
                    action_schema=action_schema,
                    fp_report_current=asdict(fp_report),
                    cfg=cfg,
                )
                state_before = fp_report.debug.grid_hash
                action_obj = GameAction.from_name(action_dict["action_id"])
                if action_dict["type"] == "coord":
                    obs_next = env.step(action_obj, data={"x": action_dict["x"], "y": action_dict["y"]})
                else:
                    obs_next = env.step(action_obj)
                if obs_next is None:
                    break
                fp_report_next = fp_analyst.analyze(obs_next, prev_observation=obs)
                diff = fp_report_next.diff_summary
                bbox_area = bbox_area(diff.changed_bbox) if diff and diff.changed_bbox else 0
                trace_entry = {
                    "step_idx": step_idx,
                    "state_before": state_before,
                    "action": action_dict,
                    "state_after": fp_report_next.debug.grid_hash,
                    "reward": getattr(obs_next, "levels_completed", None),
                    "reward_delta": getattr(obs_next, "levels_completed", None)
                    - getattr(obs, "levels_completed", None)
                    if getattr(obs_next, "levels_completed", None) is not None and getattr(obs, "levels_completed", None) is not None
                    else None,
                    "terminal": getattr(obs_next, "state", None) is not None
                    and getattr(getattr(obs_next, "state", None), "name", str(getattr(obs_next, "state", ""))).upper()
                    in {"WIN", "WON", "SUCCESS", "GAME_OVER", "LOSE", "LOST", "FAIL"},
                    "info": {"state": getattr(obs_next, "state", None).name if getattr(obs_next, "state", None) is not None and hasattr(getattr(obs_next, "state", None), "name") else None},
                    "counters": {
                        "levels_completed": getattr(obs_next, "levels_completed", None),
                        "win_levels": getattr(obs_next, "win_levels", None),
                    },
                    "fp_diff": {
                        "changed_cells": diff.changed_cells_count if diff else 0,
                        "changed_bbox_area": bbox_area,
                        "event_signatures": [sig.kind for sig in diff.event_signatures] if diff else [],
                    },
                }
                trace_file.write(json.dumps(trace_entry) + "\n")
                obs = obs_next
        return 0

    if args.agent == "trajectory_summarizer":
        if not args.outdir:
            raise SystemExit("--outdir is required for trajectory_summarizer")
        os.makedirs(args.outdir, exist_ok=True)
        cfg = TrajectorySummarizerConfig()
        summarize_trajectory(
            planner_trace=args.planner_trace,
            simple_trace=args.simple_trace,
            full_trace=args.full_trace,
            fp_dir=args.fp_dir,
            action_schema=_load_json(args.action_schema) if args.action_schema else None,
            proposer=_load_json(args.hypotheses) if args.hypotheses else None,
            classifier=_load_json(args.mechanic) if args.mechanic else None,
            goal=_load_json(args.goal) if args.goal else None,
            cfg=cfg,
            ctx={"outdir": args.outdir},
            outdir=args.outdir,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

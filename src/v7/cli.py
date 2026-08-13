from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from v7.derivation.scientific import EpisodeEvidence
from v7.environment.arc_adapter import registered_game_ids
from v7.environment.runner import ArcGameRunConfig, run_arc_game
from v7.evaluation import write_evidence_report
from v7.experiment import V7ExperimentConfig, parse_games, run_experiment
from v7.runtime import V7Runtime, V7RuntimeConfig


def _episode(row: dict[str, object]) -> EpisodeEvidence:
    return EpisodeEvidence(
        context_signature=int(row["context_signature"]), action_id=int(row["action_id"]), outcome_signature=int(row["outcome_signature"]), success=bool(row["success"]),
        prediction_error=float(row.get("prediction_error", 0.0)), future_option_delta=float(row.get("future_option_delta", 0.0)),
        source_game=None if row.get("source_game") is None else str(row["source_game"]),
        source_context=None if row.get("source_context") is None else str(row["source_context"]),
        source_global_step=None if row.get("source_global_step") is None else int(row["source_global_step"]),
    )


def run_events(root: str | Path, events_path: str | Path, *, no_restore: bool = False) -> dict[str, int]:
    runtime = V7Runtime(V7RuntimeConfig.from_path(root, restore=not no_restore))
    count = 0
    try:
        with Path(events_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    runtime.observe(_episode(json.loads(line)))
                    count += 1
        result = runtime.commit()
        return {"events": count, "generation": int(result.state.generation_id), "memories": len(result.view.nodes)}
    finally:
        runtime.close()


def doctor(env_root: str | None = None) -> dict[str, object]:
    try:
        import arc_agi  # noqa: F401
        sdk_import = True
        try:
            sdk_version = version("arc-agi")
        except PackageNotFoundError:
            sdk_version = "unknown"
    except Exception as exc:
        sdk_import = False
        sdk_version = None
        sdk_error = f"{type(exc).__name__}: {exc}"
    else:
        sdk_error = None
    games = registered_game_ids(env_root)
    return {"arc_agi_sdk": sdk_import, "arc_agi_version": sdk_version, "sdk_error": sdk_error, "local_games": len(games), "sample_games": list(games[:10])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arc-agi3-v7")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", required=True); run.add_argument("--events", required=True); run.add_argument("--no-restore", action="store_true")
    game = sub.add_parser("game")
    game.add_argument("--root", required=True); game.add_argument("--game", required=True); game.add_argument("--steps", type=int, default=1000); game.add_argument("--seed", type=int, default=0); game.add_argument("--env-root", default=None); game.add_argument("--commit-every", type=int, default=250); game.add_argument("--epsilon", type=float, default=0.10); game.add_argument("--op-mode", default="normal", choices=("normal","online","offline","competition")); game.add_argument("--render-mode", default=None); game.add_argument("--no-restore", action="store_true")
    experiment = sub.add_parser("experiment")
    experiment.add_argument("--root", required=True); experiment.add_argument("--games", nargs="+", required=True, help="Game IDs or v6-compatible sets including diverse, broad, foundation, transformation, context, role_transfer, future_enable, future_block, future_reversible, future_terminate, bridge, transfer_validation, falsification, all"); experiment.add_argument("--steps-per-game", type=int, default=1000); experiment.add_argument("--epochs", type=int, default=1); experiment.add_argument("--seed", type=int, default=0); experiment.add_argument("--env-root", default=None); experiment.add_argument("--commit-every", type=int, default=250); experiment.add_argument("--epsilon", type=float, default=0.10); experiment.add_argument("--op-mode", default="normal", choices=("normal","online","offline","competition"))
    report = sub.add_parser("report"); report.add_argument("--root", required=True); report.add_argument("--output", default=None)
    health = sub.add_parser("doctor"); health.add_argument("--env-root", default=None)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_events(args.root, args.events, no_restore=args.no_restore), sort_keys=True)); return 0
    if args.command == "game":
        result = run_arc_game(args.root, ArcGameRunConfig(game_id=args.game, steps=args.steps, seed=args.seed, env_root=args.env_root, commit_every=args.commit_every, epsilon=args.epsilon, restore=not args.no_restore, op_mode=args.op_mode, render_mode=args.render_mode))
        print(json.dumps(asdict(result), sort_keys=True)); return 0
    if args.command == "experiment":
        try:
            games = parse_games(args.games, env_root=args.env_root)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        result = run_experiment(args.root, V7ExperimentConfig(games=games, steps_per_game=args.steps_per_game, epochs=args.epochs, seed=args.seed, env_root=args.env_root, commit_every=args.commit_every, epsilon=args.epsilon, op_mode=args.op_mode))
        print(json.dumps(asdict(result), sort_keys=True)); return 0
    if args.command == "report":
        print(json.dumps(write_evidence_report(args.root, args.output), sort_keys=True)); return 0
    if args.command == "doctor":
        result = doctor(args.env_root); print(json.dumps(result, sort_keys=True)); return 0 if result["arc_agi_sdk"] else 1
    return 2

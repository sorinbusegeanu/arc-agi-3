import argparse
import concurrent.futures
import csv
import json
import random
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from arcengine import GameAction, GameState

import arc_agi
from arc_agi import OperationMode


def to_serializable(value):
    """Convert ARC objects/enums into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            k: to_serializable(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def safe_name(game_id: str) -> str:
    return game_id.replace("/", "__")


def run_random_agent_on_game(
    game_id: str,
    run_id: int,
    max_steps: int,
    use_csv: bool,
    output_dir: Path,
    base_seed: int,
    environments_dir: Path,
) -> dict:
    # One Arcade instance per worker for thread-safety.
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    env = arc.make(game_id, seed=base_seed)
    if env is None:
        return {
            "game_id": game_id,
            "run_id": run_id,
            "status": "failed_to_create_environment",
            "steps": 0,
            "duration_s": 0.0,
            "steps_per_sec": 0.0,
            "wins": 0,
            "resets": 0,
        }

    def is_reset_action(action: GameAction) -> bool:
        name = getattr(action, "name", str(action))
        value = getattr(action, "value", "")
        return str(name).upper() == "RESET" or str(value).upper() == "RESET"

    reset_action = next((a for a in env.action_space if is_reset_action(a)), None)
    non_reset_actions = [a for a in env.action_space if not is_reset_action(a)]
    latest_state = GameState.NOT_PLAYED

    ext = "csv" if use_csv else "jsonl"
    log_path = output_dir / f"{safe_name(game_id)}.run{run_id}.{ext}"
    wins = 0
    resets = 0
    executed_steps = 0
    start_time = time.perf_counter()

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        csv_writer = None
        if use_csv:
            csv_writer = csv.DictWriter(
                log_file,
                fieldnames=[
                    "timestamp",
                    "step",
                    "event",
                    "game_id",
                    "run_id",
                    "action",
                    "action_data",
                    "observation",
                ],
            )
            csv_writer.writeheader()

        for step in range(max_steps):
            executed_steps += 1
            action_data = {}
            if latest_state == GameState.NOT_PLAYED:
                action = reset_action if reset_action is not None else random.choice(env.action_space)
            else:
                pool = non_reset_actions if non_reset_actions else env.action_space
                action = random.choice(pool)
                if action.is_complex():
                    action_data = {
                        "x": random.randint(0, 63),
                        "y": random.randint(0, 63),
                    }

            obs = env.step(action, data=action_data)

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": step,
                "event": "step",
                "game_id": game_id,
                "run_id": run_id,
                "action": getattr(action, "name", str(action)),
                "action_data": action_data,
                "observation": to_serializable(obs),
            }

            if use_csv:
                csv_writer.writerow(
                    {
                        "timestamp": record["timestamp"],
                        "step": record["step"],
                        "event": record["event"],
                        "game_id": record["game_id"],
                        "run_id": record["run_id"],
                        "action": record["action"],
                        "action_data": json.dumps(record["action_data"]),
                        "observation": json.dumps(record["observation"]),
                    }
                )
            else:
                log_file.write(json.dumps(record) + "\n")

            if obs and obs.state == GameState.WIN:
                wins += 1
                break
            if obs and obs.state == GameState.GAME_OVER:
                resets += 1
                reset_obs = env.reset()
                reset_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "event": "reset_after_game_over",
                    "game_id": game_id,
                    "run_id": run_id,
                    "action": "",
                    "action_data": {},
                    "observation": to_serializable(reset_obs),
                }
                if use_csv:
                    csv_writer.writerow(
                        {
                            "timestamp": reset_record["timestamp"],
                            "step": reset_record["step"],
                            "event": reset_record["event"],
                            "game_id": reset_record["game_id"],
                            "run_id": reset_record["run_id"],
                            "action": "",
                            "action_data": json.dumps(reset_record["action_data"]),
                            "observation": json.dumps(reset_record["observation"]),
                        }
                    )
                else:
                    log_file.write(json.dumps(reset_record) + "\n")
                latest_state = (
                    reset_obs.state if reset_obs is not None else GameState.NOT_PLAYED
                )
                continue
            if obs is not None:
                latest_state = obs.state

    end_time = time.perf_counter()
    duration_s = end_time - start_time
    steps_per_sec = executed_steps / duration_s if duration_s > 0 else 0.0
    scorecard = arc.get_scorecard()
    score = getattr(scorecard, "score", None) if scorecard else None

    return {
        "game_id": game_id,
        "run_id": run_id,
        "status": "ok",
        "steps": executed_steps,
        "duration_s": duration_s,
        "steps_per_sec": steps_per_sec,
        "wins": wins,
        "resets": resets,
        "score": score,
        "log_path": str(log_path),
    }


def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-steps",
        type=int,
        default=1000,
        help="Maximum number of steps per game",
    )
    parser.add_argument(
        "-csv",
        action="store_true",
        help="Save per-game step data as CSV (default is JSONL)",
    )
    parser.add_argument(
        "-workers",
        type=int,
        default=8,
        help="Number of parallel random agents (swarm workers)",
    )
    parser.add_argument(
        "-outdir",
        type=Path,
        default=Path("swarm_runs"),
        help="Directory for per-game logs and summary",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=0,
        help="Base seed for environment creation",
    )
    parser.add_argument(
        "-envdir",
        type=Path,
        default=repo_root / "environment_files",
        help="Path to local ARC environment_files directory",
    )
    parser.add_argument(
        "-executor",
        choices=["process", "thread"],
        default="thread",
        help="Parallel executor backend",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.envdir.exists():
        print(f"Environment directory not found: {args.envdir}")
        return
    discovery_arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(args.envdir),
    )
    env_infos = discovery_arc.get_environments()
    game_ids = [env.game_id for env in env_infos]

    if not game_ids:
        print("No environments available.")
        return

    workers = max(1, args.workers)
    run_count = max(len(game_ids), workers)
    run_specs = [(run_id, game_ids[run_id % len(game_ids)]) for run_id in range(run_count)]
    print(
        f"Launching random-agent swarm across {len(game_ids)} games with "
        f"{workers} workers, {args.steps} steps/game, {run_count} total runs."
    )

    results = []
    run_start = time.perf_counter()

    executor_cls = (
        concurrent.futures.ProcessPoolExecutor
        if args.executor == "process"
        else concurrent.futures.ThreadPoolExecutor
    )
    with executor_cls(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_random_agent_on_game,
                game_id,
                run_id,
                args.steps,
                args.csv,
                args.outdir,
                args.seed + run_id,
                args.envdir,
            )
            for run_id, game_id in run_specs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{result['status']}] run={result['run_id']} {result['game_id']} "
                f"steps={result['steps']} sps={result['steps_per_sec']:.2f} "
                f"wins={result['wins']} resets={result['resets']}"
            )

    run_end = time.perf_counter()
    total_duration_s = run_end - run_start
    total_steps = sum(result["steps"] for result in results)
    total_wins = sum(result["wins"] for result in results)
    failed = sum(1 for result in results if result["status"] != "ok")
    aggregate_sps = total_steps / total_duration_s if total_duration_s > 0 else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "games_total": len(game_ids),
        "runs_total": run_count,
        "runs_failed": failed,
        "runs_completed": run_count - failed,
        "wins": total_wins,
        "total_steps": total_steps,
        "total_duration_s": total_duration_s,
        "aggregate_steps_per_sec": aggregate_sps,
        "steps_per_game": args.steps,
        "workers": workers,
        "csv": args.csv,
        "results": results,
    }

    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Swarm done: games={len(game_ids)}, runs={run_count}, failed={failed}, wins={total_wins}, "
        f"steps={total_steps}, total_s={total_duration_s:.2f}, "
        f"aggregate_steps/sec={aggregate_sps:.2f}"
    )
    print(f"Summary written to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()

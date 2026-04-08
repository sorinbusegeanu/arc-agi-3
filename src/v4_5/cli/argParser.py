from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v4.5 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--runtime-mode", default="offline")
        cmd.add_argument("--solver-version", default="v4.5")
        cmd.add_argument("--output-dir", default=None)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--advisor", default="null")
        cmd.add_argument("--video", action="store_true")
        cmd.add_argument("--render-terminal", action="store_true")
        cmd.add_argument("--debug", action="store_true")

    run_game = sub.add_parser("run-game")
    add_shared(run_game)
    run_game.add_argument("--game-id", required=True)
    run_game.add_argument("--max-steps", type=int, default=None)
    run_game.add_argument("--max-levels", type=int, default=None)

    run_games = sub.add_parser("run-games")
    add_shared(run_games)
    run_games.add_argument("--games", nargs="+", required=True)
    run_games.add_argument("--max-workers", type=int, default=1)
    run_games.add_argument("--per-game-timeout-seconds", type=float, default=60.0)
    run_games.add_argument("--max-steps", type=int, default=None)
    run_games.add_argument("--max-levels", type=int, default=None)

    run_benchmark = sub.add_parser("run-benchmark")
    add_shared(run_benchmark)
    run_benchmark.add_argument("--active-only", action="store_true")
    run_benchmark.add_argument("--games", nargs="*")
    run_benchmark.add_argument("--use-multiprocessing", action="store_true")
    run_benchmark.add_argument("--max-workers", type=int, default=1)
    run_benchmark.add_argument("--per-game-timeout-seconds", type=float, default=60.0)

    list_games = sub.add_parser("list-games")
    list_games.add_argument("--db-path", default=None)

    show_run = sub.add_parser("show-run")
    show_run.add_argument("--db-path", default=None)
    show_run.add_argument("--run-id", required=True)

    show_best = sub.add_parser("show-best")
    show_best.add_argument("--db-path", default=None)
    show_best.add_argument("--game-id", required=True)
    show_best.add_argument("--level-index", type=int, default=None)
    return parser

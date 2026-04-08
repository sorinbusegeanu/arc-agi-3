from __future__ import annotations

from v4_5.cli.argParser import build_parser
from v4_5.cli.listGames import list_games
from v4_5.cli.outputFormatter import format_benchmark_summary
from v4_5.cli.outputPaths import reset_debug_log
from v4_5.cli.runBenchmark import run_benchmark_command
from v4_5.cli.runGame import run_single_game
from v4_5.cli.runGames import run_multiple_games
from v4_5.cli.showBest import show_best
from v4_5.cli.showRun import show_run
from v4_5.cli.types import BenchmarkCliConfig, MultiGameRunConfig, SingleGameRunConfig


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "debug", False):
        reset_debug_log()
    if args.command == "run-game":
        result = run_single_game(
            SingleGameRunConfig(
                game_id=args.game_id,
                runtime_mode=args.runtime_mode,
                solver_version=args.solver_version,
                output_dir=args.output_dir,
                seed=args.seed,
                advisor=args.advisor,
                max_steps=args.max_steps,
                max_levels=args.max_levels,
                video=bool(args.video),
                render_terminal=bool(args.render_terminal),
                debug=bool(args.debug),
            )
        )
        return 0
    if args.command == "run-games":
        result = run_multiple_games(
            MultiGameRunConfig(
                game_ids=tuple(args.games),
                runtime_mode=args.runtime_mode,
                solver_version=args.solver_version,
                output_dir=args.output_dir,
                seed=args.seed,
                advisor=args.advisor,
                max_workers=args.max_workers,
                per_game_timeout_seconds=args.per_game_timeout_seconds,
                max_steps=args.max_steps,
                max_levels=args.max_levels,
                video=bool(args.video),
                render_terminal=bool(args.render_terminal),
                debug=bool(args.debug),
            )
        )
        return 0
    if args.command == "run-benchmark":
        result = run_benchmark_command(
            BenchmarkCliConfig(
                runtime_mode=args.runtime_mode,
                solver_version=args.solver_version,
                output_dir=args.output_dir,
                seed=args.seed,
                advisor=args.advisor,
                active_only=bool(args.active_only),
                game_ids=None if not args.games else tuple(args.games),
                use_multiprocessing=bool(args.use_multiprocessing),
                max_workers=args.max_workers,
                per_game_timeout_seconds=args.per_game_timeout_seconds,
                video=bool(args.video),
                render_terminal=bool(args.render_terminal),
                debug=bool(args.debug),
            )
        )
        return 0
    if args.command == "list-games":
        print(format_benchmark_summary({"games": list_games(db_path=args.db_path)}))
        return 0
    if args.command == "show-run":
        print(format_benchmark_summary(show_run(run_id=args.run_id, db_path=args.db_path)))
        return 0
    if args.command == "show-best":
        print(format_benchmark_summary({"best": show_best(game_id=args.game_id, level_index=args.level_index, db_path=args.db_path)}))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

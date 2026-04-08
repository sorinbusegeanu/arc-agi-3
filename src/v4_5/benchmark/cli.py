from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from v4_5.benchmark.catalog.gameCatalog import GameCatalog
from v4_5.benchmark.db.store import BenchmarkStore, default_db_path
from v4_5.benchmark.runner.benchmarkRunner import BenchmarkRunner
from v4_5.benchmark.runner.benchmarkTypes import BenchmarkRunRequest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v4.5 benchmark suite.")
    parser.add_argument("--db-path", type=Path, default=default_db_path(), help="SQLite benchmark DB path.")
    parser.add_argument("--game", dest="games", action="append", help="Run one or more specific game ids.")
    parser.add_argument("--label", default="benchmark_cli", help="Benchmark run label.")
    parser.add_argument("--solver-version", default="v4.5", help="Solver version label to persist.")
    parser.add_argument("--runtime-mode", default="offline", help="Runtime mode label to persist.")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of per-game worker processes.")
    parser.add_argument("--video", action="store_true", help="Capture per-screen frames and encode a per-game video artifact.")
    parser.add_argument("--notes", default=None, help="Optional run notes.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export.")
    parser.add_argument("--list-games", action="store_true", help="List catalog games and exit.")
    parser.add_argument("--list-active", action="store_true", help="List active benchmark games and exit.")
    return parser.parse_args(argv)


def _print_rows(rows: list[dict[str, object]], *, heading: str) -> None:
    print(heading)
    for row in rows:
        print(json.dumps(row, separators=(",", ":")))


def _print_catalog_rows(rows: list[dict[str, object]], *, heading: str) -> None:
    print(heading)
    for row in rows:
        print(
            json.dumps(
                {
                    "game_id": row.get("game_id"),
                    "category": _category_from_notes(row.get("notes")),
                    "in_benchmark": int(row.get("in_benchmark", 0) or 0),
                },
                separators=(",", ":"),
            )
        )


def _category_from_notes(notes: object) -> str:
    text = str(notes or "")
    for part in text.split(";"):
        if part.startswith("category="):
            return part.split("=", 1)[1]
    return ""


def _print_compact_run_report(store: BenchmarkStore, game_rows: list[dict[str, object]]) -> None:
    for row in game_rows:
        catalog_row = store.fetch_one("SELECT notes FROM games WHERE game_id = ?", (row["game_id"],))
        category = _category_from_notes(None if catalog_row is None else catalog_row.get("notes"))
        win_text = "yes" if int(row.get("terminal_success", 0) or 0) else "no"
        print(
            f'{row["game_id"]} | {category} | steps={int(row.get("total_steps_executed", 0) or 0)} | '
            f'levels={int(row.get("levels_solved", 0) or 0)} | win={win_text}'
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    store = BenchmarkStore(db_path=args.db_path)
    catalog = GameCatalog(store)
    catalog.sync_catalog_from_seed()

    if args.list_games:
        _print_catalog_rows(catalog.list_all_games(), heading="games")
        return 0
    if args.list_active:
        _print_catalog_rows(catalog.list_active_benchmark_games(), heading="active_games")
        return 0

    runner = BenchmarkRunner(store=store, catalog=catalog)
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        summary = runner.run(
            BenchmarkRunRequest(
                run_label=args.label,
                solver_version=args.solver_version,
                runtime_mode=args.runtime_mode,
                game_ids=None if not args.games else tuple(args.games),
                notes=args.notes,
                output_csv=not args.no_csv,
                use_multiprocessing=True,
                max_workers=max(1, int(args.max_workers)),
                output_root=str((args.db_path.parent / "output").resolve()),
                video=bool(args.video),
            )
        )
    finally:
        logging.disable(previous_disable)

    game_rows = store.fetch_all(
        "SELECT * FROM benchmark_game_results WHERE run_id = ? ORDER BY game_id",
        (summary.run_id,),
    )
    _print_compact_run_report(store, game_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

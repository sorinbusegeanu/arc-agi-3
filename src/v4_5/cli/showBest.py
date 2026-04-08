from __future__ import annotations

from pathlib import Path

from v4_5.benchmark.db.store import BenchmarkStore, default_db_path


def show_best(*, game_id: str, level_index: int | None = None, db_path: str | None = None) -> dict | None:
    store = BenchmarkStore(db_path=Path(db_path) if db_path else default_db_path())
    if level_index is None:
        return store.fetch_one("SELECT * FROM game_best_results WHERE game_id = ?", (game_id,))
    return store.fetch_one("SELECT * FROM level_best_results WHERE game_id = ? AND level_index = ?", (game_id, level_index))

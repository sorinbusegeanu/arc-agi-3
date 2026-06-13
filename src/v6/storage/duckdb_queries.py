from __future__ import annotations

from pathlib import Path
from glob import glob


def load_run_table(
    base_path: str | Path,
    table_name: str,
    *,
    game: str | None = None,
    sampler: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
):
    con = _duckdb_connection()
    files = _table_files(base_path, table_name, game=game, sampler=sampler, seed=seed, steps=steps)
    if not files:
        raise FileNotFoundError(f"no Parquet files for table {table_name}")
    return con.execute("SELECT * FROM read_parquet(?)", [files]).fetchdf()


def query_summary(base_path: str | Path, **filters):
    return load_run_table(base_path, "run_summary", **filters)


def query_future_effects(base_path: str | Path, **filters):
    return load_run_table(base_path, "future_effects", **filters)


def query_contingencies(base_path: str | Path, **filters):
    return load_run_table(base_path, "contingencies", **filters)


def query_role_validation_inputs(base_path: str | Path, **filters):
    con = _duckdb_connection()
    future = _table_files(base_path, "future_effects", **filters)
    contingencies = _table_files(base_path, "contingencies", **filters)
    return con.execute(
        """
        SELECT f.*, c.context_level, c.action, c.transformation_family, c.support_count, c.confidence
        FROM read_parquet(?) f
        LEFT JOIN read_parquet(?) c
          ON CAST(f.contingency_id AS BIGINT) = CAST(c.id AS BIGINT)
        """,
        [future, contingencies],
    ).fetchdf()


def _duckdb_connection():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB queries require duckdb. Install with: pip install duckdb") from exc
    return duckdb.connect(database=":memory:")


def _table_files(
    base_path: str | Path,
    table_name: str,
    *,
    game: str | None = None,
    sampler: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
) -> list[str]:
    path = Path(base_path)
    parts = [
        f"game={game}" if game is not None else "game=*",
        f"sampler={sampler}" if sampler is not None else "sampler=*",
        f"seed={int(seed)}" if seed is not None else "seed=*",
        f"steps={int(steps)}" if steps is not None else "steps=*",
    ]
    root = path.joinpath(*parts)
    files = sorted(glob(str(root / f"{table_name}_part_*.parquet")))
    plain_files = sorted(glob(str(root / f"{table_name}.parquet")))
    return files + plain_files

from __future__ import annotations

from v4_5.benchmark.catalog.gameCatalog import GameCatalog
from v4_5.benchmark.db.store import BenchmarkStore, default_db_path
from v4_5.benchmark.runner.benchmarkRunner import BenchmarkRunner
from v4_5.benchmark.runner.benchmarkTypes import BenchmarkRunRequest
from v4_5.cli.outputPaths import resolve_run_output_dir
from v4_5.cli.types import BenchmarkCliConfig


def run_benchmark_command(config: BenchmarkCliConfig) -> dict:
    db_path = default_db_path()
    store = BenchmarkStore(db_path=db_path)
    catalog = GameCatalog(store)
    catalog.sync_catalog_from_seed()
    output_dir = resolve_run_output_dir(config.output_dir)
    runner = BenchmarkRunner(store=store, catalog=catalog, output_dir=str(output_dir), debug=bool(config.debug))
    request = BenchmarkRunRequest(
        run_label="benchmark_cli",
        solver_version=config.solver_version,
        runtime_mode=config.runtime_mode,
        game_ids=config.game_ids,
        output_csv=True,
        use_multiprocessing=bool(config.use_multiprocessing),
        max_workers=config.max_workers,
        per_game_timeout_seconds=config.per_game_timeout_seconds,
        explicit_game_ids=None if config.active_only else config.game_ids,
        output_root=str(output_dir),
        seed_base=config.seed,
        video=bool(config.video),
        render_terminal=bool(config.render_terminal),
        debug=bool(config.debug),
    )
    summary = runner.run(request)
    return summary.to_dict()

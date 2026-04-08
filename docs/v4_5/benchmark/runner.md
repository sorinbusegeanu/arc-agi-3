## v4.5 Benchmark Runner

The benchmark runner is a separate benchmark layer that wraps existing v4 evaluation entry points through a thin adapter.

Behavior:

- accepts a predefined list of games from the catalog
- runs only games marked `in_benchmark = true` unless explicitly overridden
- can run one game or many games
- stores all run results in the benchmark database
- computes derived summaries after each run

For every run:

1. create one `benchmark_runs` row
2. invoke the existing evaluation path for each selected game
3. normalize the runner output into benchmark-specific game and level results
4. persist all normalized rows
5. update derived best-result tables
6. write a compact JSON summary report

Historical run rows are append-only. Old benchmark runs are never overwritten.

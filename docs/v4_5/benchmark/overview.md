## v4.5 Benchmark Overview

The v4.5 benchmark framework is a separate evaluation layer for tracking solver progress over time without changing live solver behavior.

It provides:

- a predefined game catalog
- optional inclusion of each catalog game in the active benchmark suite
- per-run persistence to SQLite
- historical comparison over time
- per-level best step count tracking
- per-game best solved-level count tracking
- best cumulative steps for solved levels
- the ability to query benchmark history later

The benchmark framework is for evaluation only.

It must not alter live solver semantics, runtime policy behavior, or existing v4 solving logic.

It does not replace current runners. It calls existing evaluation paths through a thin adapter and stores normalized benchmark results separately.

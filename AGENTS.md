# Repository Guidelines

## Project Structure & Module Organization
Primary work in this repository currently centers on `src/v6/`. Key areas:
- `src/v6/main.py`: core runtime and step loop.
- `src/v6/environment/`, `context/`, `contingency/`, `prediction/`, `transformation/`: environment adapters and learning logic.
- `src/v6/memory/`: SQLite stores, compact-memory fold/restore, substrate, and lifecycle logic.
- `src/v6/evaluation/`: sampling, continuous research, and diagnostics.
- `src/v6/tests/`: pytest coverage, including `test_v6_core.py` and `test_v6_higher_order.py`.

Generated outputs live under `runs/`. Treat them as runtime artifacts, not source.

## Build, Test, and Development Commands
Use `PYTHONPATH=src` for all local runs.

```bash
PYTHONPATH=src pytest src/v6/tests/test_v6_core.py -q
PYTHONPATH=src pytest src/v6/tests/test_v6_higher_order.py -q
PYTHONPATH=src pytest src/v6/tests -q
PYTHONPATH=src python -m v6.cli interaction-sampling-v05c --games tt01 --samplers random_baseline --seeds 0 --steps 100 --output-dir runs/v6/smoke
PYTHONPATH=src python -m v6.cli continuous-research-run ...
```

Prefer focused test files while iterating, then run the broader `src/v6/tests` suite before submitting.

## Coding Style & Naming Conventions
Python style is conventional: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and type hints on new code. Match the existing style in the touched module. Keep logic deterministic, bounded, and backward-compatible with existing SQLite schemas by using additive migrations (`ALTER TABLE` / compatibility helpers) instead of destructive changes.

Use `rg` for code search and keep comments sparse and technical.

## Testing Guidelines
Tests use `pytest`. Add focused regression tests in `src/v6/tests/` alongside the subsystem you change. Name files `test_*.py` and test functions `test_*`. For storage/reporting changes, verify both behavior and persisted schema/rows.

## Commit & Pull Request Guidelines
Recent history contains very terse commit subjects (`update`), but contributors should use specific imperative messages, e.g. `Fix H10 attention saturation diagnostics`. PRs should include:
- scope of changed modules
- commands run
- exact test results
- notes on schema changes or runtime artifact impact

## Runtime & Data Hygiene
`runs/` can grow quickly. Avoid committing generated SQLite, JSON, or smoke-run outputs. When changing reports or memory fold behavior, preserve idempotence and avoid duplicating derived rows on rerun.

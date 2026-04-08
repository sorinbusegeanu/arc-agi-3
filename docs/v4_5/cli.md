# v4.5 CLI

v4.5 exposes one top-level CLI.

Agents are internal roles, not standalone commands.

## Subcommands

- `run-game`
  - runs one full solver stack for one game
- `run-games`
  - runs multiple games in parallel with one full solver stack per game
- `run-benchmark`
  - uses the benchmark framework
- `list-games`
  - lists catalog games
- `show-run`
  - shows one stored benchmark run
- `show-best`
  - shows the best known game or level result

## Concurrency

- concurrency is allowed only at game level
- each game runs in its own Python process
- no same-game concurrency
- no same-level concurrency
- no agent-level concurrency

## Execution

- `run-game` uses one fully composed v4.5 runtime bundle
- `run-games` uses the shared worker process infrastructure
- `run-benchmark` keeps benchmark DB ownership in the benchmark layer

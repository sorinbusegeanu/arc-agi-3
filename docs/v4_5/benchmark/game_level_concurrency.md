# Game-Level Concurrency

## Scope

Concurrency is game-level only.

- one process per game task
- no threads
- no same-level or same-game concurrency

## Worker Rules

- worker processes must not write to SQLite
- workers write isolated result artifacts
- one failure in one game must not stop unrelated completed game results from being merged
- each worker must have isolated:
  - session
  - ledger
  - memory
  - POI registry
  - trajectory queue
  - output directory

## Parent Rules

- parent process performs all DB writes after worker completion
- parent supports:
  - `max_workers`
  - explicit `game_id` list
  - timeout per game
  - fail-fast false by default

## Multiprocessing

- default process start method must be safe for cross-platform multiprocessing
- workers run in separate Python processes
- parent validates result files before merge

# Hydra v9.7.6

`src/v9` is the independent implementation of the ARC-AGI-3 Hydra memory
architecture and research contract v0.7.0.

Its runtime, CLI, canonical graph, M0-M7 hierarchy, environment contracts,
developmental cognition, persistence, evidence, and reports are all owned by
v9. Production modules do not import historical runtime packages or depend on
runtime patch ordering.

Run the native CLI with:

```bash
PYTHONPATH=src python -m v9 --help
PYTHONPATH=src python -m v9 smoke --root runs/v9/smoke --events 100
PYTHONPATH=src python -m v9 continuous-run --games synthetic --steps-per-game 100
PYTHONPATH=src python -m v9 continuous-run --root runs/v9/step1 --games step1 --steps-per-game 1000
```

Native run roots use an immutable `ScientificConfigId` and v9 snapshot schema.
Predecessor roots are intentionally not migrated; use a fresh v9 root.

Core environments are ARC, FrozenLake, Chess, Sudoku, and the controlled
synthetic symbolic environment. BabyAI/MiniGrid is available through the
`babyai` optional dependency. ALFRED uses an injected live backend so its large
datasets remain outside the core installation.


## Developmental curriculum

The runtime resolves `--games step1` through `--games step14` from
`docs/v9/curriculum.yaml`. Presets include `gym_foundation`, `gym_broad`,
`semantics`, and `cross_family`.

Step-specific adapter type, environment kwargs, grounding condition, symbol
suppression, and validation mode are preserved by the runtime resolver.

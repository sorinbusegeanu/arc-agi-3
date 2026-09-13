# Hydra v9.5

`src/v9` is the independent implementation of the ARC-AGI-3 Hydra memory
architecture and research contract v0.6.3.1.

Its runtime, CLI, canonical graph, M0-M7 hierarchy, environment contracts,
developmental cognition, persistence, evidence, and reports are all owned by
v9. Production modules do not import historical runtime packages or depend on
runtime patch ordering.

Run the native CLI with:

```bash
PYTHONPATH=src python -m v9 --help
PYTHONPATH=src python -m v9 smoke --root runs/v9/smoke --events 100
PYTHONPATH=src python -m v9 continuous-run --games synthetic --steps-per-game 100
```

Native run roots use an immutable `ScientificConfigId` and v9 snapshot schema.
Predecessor roots are intentionally not migrated; use a fresh v9 root.

Core environments are ARC, FrozenLake, Chess, Sudoku, and the controlled
synthetic symbolic environment. BabyAI/MiniGrid is available through the
`babyai` optional dependency. ALFRED uses an injected live backend so its large
datasets remain outside the core installation.

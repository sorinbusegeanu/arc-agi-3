# Hydra v9.7.9

`src/v9` is the independent implementation of the ARC-AGI-3 Hydra memory
architecture and research contract v0.7.0.

Its runtime, CLI, canonical graph, M0-M7 hierarchy, environment contracts,
developmental cognition, persistence, evidence, and reports are all owned by
v9.

v9.7.9 adds bounded developmental residency for concrete low-level evidence:

```text
M0 episodic evidence                  bounded resident working set
M1 grounded contingencies             bounded resident working set
M1 normalized relations               persistent consolidated support
M2-M7                                 persistent consolidated cognition
```

Superseded M0 and M1-grounded records are physically deleted once their
normalized or higher-order replacement exists and representative coverage is
preserved. Deletion removes canonical payloads, incident edges, graph indexes,
version entries, lifecycle references, replay references, and training-reservoir
references. Continuous ingestion-time compaction keeps resident low-level memory
near its configured target, and startup compaction brings oversized compatible
snapshots back into the same envelope before sampling starts.

HGT graph construction and epoch preprocessing use bounded working sets. Runtime
telemetry exposes resident M0/M1-grounded counts, compaction backlog, deletion
throughput, RSS/USS/swap, and the memory-governor state.

Run the native CLI with:

```bash
PYTHONPATH=src python -m v9 --help
PYTHONPATH=src python -m v9 smoke --root runs/v9/smoke --events 100
PYTHONPATH=src python -m v9 continuous-run --games synthetic --steps-per-game 100
PYTHONPATH=src python -m v9 continuous-run --root runs/v9/step1 --games step1 --steps-per-game 1000
```

Native run roots use a versioned `ScientificConfigId` and v9 snapshot schema.
The complete v9.7.9 resident-memory contract is persisted in the scientific
manifest.

Core environments are ARC, FrozenLake, Chess, Sudoku, and the controlled
synthetic symbolic environment. BabyAI/MiniGrid is available through the
`babyai` optional dependency. ALFRED uses its installed ALFWorld backend and
keeps the large downloaded datasets outside the core installation.

## Developmental curriculum

The runtime resolves `--games step1` through `--games step14` from
`docs/v9/curriculum.yaml`. Presets include `gym_foundation`, `gym_broad`,
`semantics`, and `cross_family`.

Step-specific adapter type, environment kwargs, grounding condition, symbol
suppression, and validation mode are preserved by the runtime resolver.

Step 14 uses ALFWorld's fast TextWorld simulator by default. Select the embodied
AI2-THOR backend with `--alfred-mode thor`; on a remote shell, pass the available
X display explicitly when needed, for example `--alfred-x-display 0.0`. Start
embodied runs with one actor because every actor owns a Unity simulator process.

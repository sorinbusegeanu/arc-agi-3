# Hydra v9.7.8

`src/v9` is the independent implementation of the ARC-AGI-3 Hydra memory architecture and research contract v0.7.0.

Its runtime, CLI, canonical graph, M0-M7 hierarchy, environment contracts, developmental cognition, persistence, evidence, reports, symbolic grounding and HGT training are owned by v9.

Run the native CLI with:

```bash
PYTHONPATH=src python -m v9 --help
PYTHONPATH=src python -m v9 smoke --root runs/v9/smoke --events 100
PYTHONPATH=src python -m v9 continuous-run --games synthetic --steps-per-game 100
PYTHONPATH=src python -m v9 continuous-run --root runs/v9/step1 --games step1 --steps-per-game 1000
```

Native run roots use an immutable `ScientificConfigId` and v9 snapshot schema. Predecessor roots are intentionally not migrated; use a fresh v9 root.

Core environments are ARC, FrozenLake, Chess, Sudoku, and the controlled synthetic symbolic environment. BabyAI/MiniGrid is available through the `babyai` optional dependency. ALFRED uses its installed ALFWorld backend and keeps the large downloaded datasets outside the core installation.

## Symbolic grounding

Symbols are treated as opaque identities. The runtime stores canonical `SymbolOccurrence` identity/provenance and derives structural relations including symbol order, recurrence, action/change ordering, boundary/progress/outcome coincidence and cross-modal alignment. These relations carry support, contradiction, temporal offsets and causal watermarks.

Grounding maturity is G0-G5. G3+ validated grounding may influence retrieval and actor policy. G4 is validated novel composition and G5 is symbol-mediated learning. Shuffled alignments are negative controls and never grant behavioral authority.

The HGT graph uses canonical `symbol_identity` as the SYMBOL authority, adds explicit `SYMBOL` and `CONTEXT` nodes, and filters legacy symbolic semantic tuples before graph construction. Grounding-specific HGT objectives cover prediction, relevant-memory retrieval, world-to-symbol generalization, held-out composition, symbol-conditioned action ranking, shuffled-alignment discrimination and grounding calibration.

Run the matched H16 C0-C3 control experiment with:

```bash
PYTHONPATH=src python -m v9.research.h16_cli \
  --output runs/v9/h16/evidence.json \
  --seeds 0,1,2,3 \
  --interaction-budget 128
```

The saved H16 evidence records matched seed/config/budget/evaluation identity and source run-state metadata. C0 is interaction-only, C1 symbols-only, C2 aligned interaction+symbols, and C3 shuffled alignment.

See `docs/v9.7.8_SYMBOLIC_GROUNDING_CONFORMANCE.md` for the implementation and acceptance matrix.

## Developmental curriculum

The runtime resolves `--games step1` through `--games step14` from `docs/v9/curriculum.yaml`. Presets include `gym_foundation`, `gym_broad`, `semantics`, and `cross_family`.

Step-specific adapter type, environment kwargs, grounding condition, symbol suppression, and validation mode are preserved by the runtime resolver.

Step 14 uses ALFWorld's fast TextWorld simulator by default. Select the embodied AI2-THOR backend with `--alfred-mode thor`; on a remote shell, pass the available X display explicitly when needed, for example `--alfred-x-display 0.0`. Start embodied runs with one actor because every actor owns a Unity simulator process.

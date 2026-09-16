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

Native run roots use an immutable `ScientificConfigId`, symbolic grounding schema 3 and persisted symbolic graph schema 3.

Core environments are ARC, FrozenLake, Chess, Sudoku, and the controlled synthetic symbolic environment. BabyAI/MiniGrid is available through the `babyai` optional dependency. ALFRED uses its installed ALFWorld backend and keeps the large downloaded datasets outside the core installation.

## Symbolic grounding

Symbols are opaque identities. The runtime stores canonical `SymbolOccurrence` identity, position, stream, codec, macro/micro time and provenance. Symbolic M0 also records nearby interaction context, action, normalized transformation, progress and outcome.

The M1N layer derives symbol order, recurrence, action/change ordering, boundary/progress/outcome coincidence, cross-modal correspondence, symbol-to-interaction prediction and interaction-to-symbol generalization. Relations carry support, contradiction, temporal offsets/ranges, context, provenance and causal watermarks. Modality-neutral transformation descriptors allow WORLD, SYMBOL and CROSS_MODAL evidence to form shared M2 families and M3 roles.

Grounding maturity is G0-G5. Validated G3+ grounding can influence retrieval, interaction prediction, future-option scoring, M6 outcome selection, M7 strategy selection/replanning and actor action ranking. G4 represents validated novel composition and G5 represents symbol-mediated learning. Shuffled alignments provide contradiction/control evidence.

The HGT graph uses canonical `symbol_identity` as the SYMBOL authority and explicit `SYMBOL` and `CONTEXT` nodes. SYMBOL features contain identity, occurrence/position/time statistics, context diversity, phase distribution and grounding evidence. Typed graph relations cover observation, temporal structure, provenance, structural correspondence, participation, support, contradiction, grounding and transfer validation.

The seven grounding HGT objectives are:

1. symbol-conditioned relevant-memory retrieval;
2. cross-modal correspondence prediction;
3. symbol-conditioned interaction-consequence prediction;
4. world-to-symbol generalization;
5. held-out cross-modal composition;
6. shuffled-alignment discrimination;
7. grounding-confidence calibration.

Run the matched H16 C0-C3 experiment with:

```bash
PYTHONPATH=src python -m v9.research.h16_cli \
  --root runs/v9/h16/runtime \
  --output runs/v9/h16/evidence.json \
  --seeds 0,1,2,3 \
  --interaction-budget 128
```

The H16 runner persists the exact matched opportunity sequence for every seed before executing C0-C3. C0 is interaction-only, C1 symbols-only, C2 aligned interaction+symbols, and C3 uses shuffled symbols with matched symbol statistics. Saved evidence records configuration, model, symbolic schema, matched-state digest, condition, trial and causal-effect provenance.

See `docs/v9.7.8_SYMBOLIC_GROUNDING_CONFORMANCE.md` for the implementation and acceptance matrix.

## Developmental curriculum

The runtime resolves `--games step1` through `--games step14` from `docs/v9/curriculum.yaml`. Presets include `gym_foundation`, `gym_broad`, `semantics`, and `cross_family`.

Step-specific adapter type, environment kwargs, grounding condition, symbol suppression, and validation mode are preserved by the runtime resolver.

Step 14 uses ALFWorld's fast TextWorld simulator by default. Select the embodied AI2-THOR backend with `--alfred-mode thor`; on a remote shell, pass the available X display explicitly when needed, for example `--alfred-x-display 0.0`. Start embodied runs with one actor because every actor owns a Unity simulator process.

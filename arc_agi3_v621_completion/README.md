# ARC-AGI3 v6.2.1 completion drop-in

Built against:

```text
repository: sorinbusegeanu/arc-agi-3
branch: main
commit: d8d530893d5e5ca4bacc5bfe2bb15de7390367c1
```

## Install

Extract in the repository root, then run:

```bash
python apply_v621_completion.py
```

The installer is marker-checked and fails closed if the current source no longer matches the expected v6.2 structure. It creates `.v62_backup` copies of modified canonical files on first install.

## Completes the remaining v6.2 gaps

1. **Compact-memory persistence and restore**
   - preserves v6.2/v6.2.1 score, promotion, development, lifecycle, concept-transfer and world-model-relation state;
   - applies the v6.2.1 migration to compact state databases automatically;
   - restores extension state back into the live substrate.

2. **Worker/snapshot parity**
   - wraps snapshot query engines with `V621SnapshotMemoryQueryEngine`;
   - preloads promotion state, M5 relational models and M6 strategies once at worker initialization;
   - performs action selection from RAM after preload;
   - filters rejected/superseded/forgotten abstractions and applies the same M5/M6 bonuses as the SQLite path.

3. **M1-M6 lifecycle**
   - adds retention, replay priority, dormant state and forgetting for higher-level memories;
   - forgotten memories are excluded from memory-guided queries;
   - lifecycle state is persisted in `memory_level_lifecycle_v621`.

4. **Cached abstraction-based future options**
   - removes full `interactions` rescans on every before/after estimate;
   - incrementally refreshes the transition cache;
   - combines exact-state reachability with structural-state reachability.

5. **Stricter M4/M5/M6 semantics**
   - M4 concepts require structurally compatible multiple roles plus cross-game/context evidence;
   - accepted M4 concepts require direct runtime concept-transfer attempts;
   - M5 world models persist explicit relations such as `precedes`, `enables`, `constrains`, `shared_outcome`, `co_context`, and `shared_family`;
   - accepted M5 models require predictive relations;
   - M6 strategies require success, known cost, equivalent outcome, cost advantage and successful reuse.

6. **Controller-only runtime routing**
   - carrier, contradiction, efficiency, lifecycle, prediction, promotion and selected-action query paths route through `V621MemoryController`;
   - compact restore routes lifecycle imports through the controller when present.

7. **Sampler memory priors**
   - samplers still produce their proposed action;
   - memory evaluates all available actions;
   - memory overrides the sampler only when its best action exceeds the sampler action by `memory_sampler_prior_margin` (default `0.15`).

## Added files

```text
src/v6/memory/v621_runtime.py
src/v6/memory/v621_compact.py
src/v6/memory/migrations/v621.py
src/v6/tests/test_v621_memory_completion.py
```

## Patched files

```text
src/v6/main.py
src/v6/memory/compact_memory.py
src/v6/memory/compact_memory_restore.py
src/v6/memory/migrations/__init__.py
```

## Focused test

```bash
PYTHONPATH=src pytest -q src/v6/tests/test_v621_memory_completion.py
```

Then run the existing v6.1/v6.2 and full v6 test suites.

## v6.2.1 startup migration hotfix

The package now makes the v6.2/v6.2.1 migrations self-bootstrapping on a fresh
`current_state.sqlite`. `memory_versions` is created before schema-version writes,
so `continuous-research-run` can initialize a new compact-memory directory without
requiring a prior `MemorySubstrate` construction.


## Installer reruns

The installer is safe to rerun on an already-patched tree. If a target file already contains the v6.2.1 changes, that patch step is treated as a successful no-op.

## Installer repair revision
The installer is idempotent and repairs a malformed lifecycle restore block produced by earlier v6.2.1 installer revisions before source validation.

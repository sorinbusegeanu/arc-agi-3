# ARC-AGI3 v6.2 memory-runtime drop-in

Built for the current v6.1 `src/v6` architecture.

## Install

From the repository root, copy/extract this package and run one command:

```bash
python apply_v62_memory_runtime.py
```

The installer keeps `src/v6/main.py.v61_backup` on first install.

## Implements the five remaining architecture items

1. `V62MemoryController` becomes the mandatory runtime facade. Existing Hydra trackers remain compatibility computation caches, while canonical persisted state remains in `MemorySubstrate`.
2. Hierarchical ISF/fitness is propagated from M0 through M6. Developmental stage also changes the live interaction ISF weights.
3. Live future-option estimation uses observed transition reachability from memory; the old action-branching estimator is only a cold-start fallback.
4. M4 concepts require multi-role transfer evidence; M5 world models require multi-concept relational structure.
5. Promotion uses level-specific multi-dimensional evidence gates, and memory-guided prediction/action selection is enabled by default. M5/M6 evidence contributes to action ranking.

## Added files

```text
src/v6/memory/v62_runtime.py
src/v6/memory/migrations/v62.py
src/v6/tests/test_v62_memory_runtime.py
```

The installer patches `src/v6/main.py` and `src/v6/memory/migrations/__init__.py` rather than adding a monkeypatch/sitecustomize layer.

## Focused test

```bash
PYTHONPATH=src pytest -q src/v6/tests/test_v62_memory_runtime.py
```

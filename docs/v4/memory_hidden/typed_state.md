Status: implemented and verified
Scope: typed-state contract for `src/v4/memory_hidden/typedState.py`
Source of truth: `src/v4/memory_hidden/typedState.py`
Last verified against: repo state on 2026-03-29

# Typed state

`MemoryHiddenTypedStateV4` contains:
- common movement state
- revealed safe cells
- visible number cells and adjacent mine counts
- unrevealed frontier cells
- known mines
- forbidden cells
- goal cell
- local consistency facts derived from visible counts
- bounded parsed-memory reference counts

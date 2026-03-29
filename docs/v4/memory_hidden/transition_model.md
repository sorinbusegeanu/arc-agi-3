Status: implemented and verified
Scope: transition model for `src/v4/memory_hidden/transitionModel.py`
Source of truth: `src/v4/memory_hidden/transitionModel.py`
Last verified against: repo state on 2026-03-29

# Transition model

Implemented deterministic consequences:
- safe movement inside walkable cells
- blocked out-of-bounds and wall moves
- terminal failure on known mine / forbidden cell
- terminal success on the goal cell

The model does not invent hidden mine placements beyond count-derived local consequences already present in the typed state.

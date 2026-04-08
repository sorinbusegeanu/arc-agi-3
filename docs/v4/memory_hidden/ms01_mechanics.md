Status: implemented; gate-covered, live regression still mixed
Scope: `ms01` mechanics as implemented in `src/v4/memory_hidden`
Source of truth: `src/v4/memory_hidden/familyAdapters.py`, `src/v4/memory_hidden/solverPolicy.py`
Last verified against: repo state on 2026-03-30; targeted manual live regression for `ms01`

# ms01 mechanics

Implemented `ms01` treatment:
- movement-only hidden-information family
- walls and goal loaded from environment metadata
- revealed clue cells read from direct observation
- local consistency facts derived from visible counts only
- frontier choices filtered against count-derived contradictions
- frontier routing now keeps multiple locally consistent anchor candidates and biases away from immediately repeating the same dead anchor

Current boundary:

- the manual live regression row is still a low-expansion no-plan failure, so `ms01` is not yet live-verified end-to-end

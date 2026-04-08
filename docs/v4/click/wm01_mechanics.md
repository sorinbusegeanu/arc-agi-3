Status: implemented; gate-covered, live regression still mixed
Scope: `wm01` mechanics as implemented in `src/v4/click`
Source of truth: `/home/zodrak/zod/src/v4/click/familyAdapters.py`, `/home/zodrak/zod/src/v4/click/stateBuilder.py`, `/home/zodrak/zod/src/v4/click/solverPolicy.py`
Last verified against: current repo state on 2026-03-30; targeted manual live regression for `wm01`

# `wm01` Mechanics

Implemented `wm01` treatment:

- live hole positions are loaded from the local env level config
- active mole targets are detected directly from the current sampled board
- click payloads are derived from the env-backed hole geometry, not from raw mole coordinates alone
- when the surfaced current level index is out of range, config lookup first tries the authoritative live level index and then only the last valid level config that is still consistent with the live episode boundary
- if no valid config index can be selected, the builder fails closed with a specific `wm01_level_index` tag
- family-local live anti-repeat handling rotates across active mole targets and recent clickable cells instead of repeating the same dead click target indefinitely

Current boundary:

- gate and family tests are still the main verification surface
- the manual live regression runner no longer reports `unavailable`; it now keeps `wm01` on a classified repeated-cycle / step-budget path, so the family is not yet live-verified end-to-end

Status: implemented and verified
Scope: movement doc: phase3 movement summary
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Phase 3 Movement Summary

## Family-By-Family Mechanic Proved

- `ul01`: exact key pickup and locked-door dependency
- `fs01`: exact switch-bit and door-state planning
- `tp01`: exact symmetric teleporter-pair resolution
- `ic01`: exact forced straight-line slide with stop-before-block semantics
- `va01`: exact coverage-state progression over visit-all movement
- `pb01`: exact one-block push legality and push-state planning

## Exact Typed-State Additions Per Family

- `ul01`: key positions, key bit, door positions
- `fs01`: switch positions, activated-switch bitmask, explicit door-state bit
- `tp01`: teleporter endpoints, teleporter pair list, directional pair map
- `ic01`: exact slide surface, explicit red hazard set, slide mode
- `va01`: coverage-eligible cell set, current coverage mask
- `pb01`: pushable-block position, explicit target cell, real step limit

## Exact Transition Semantics Added Per Family

- `ul01`: pickup, locked-door block, unlock-pass completion
- `fs01`: latch-on-entry switch activation and explicit door opening
- `tp01`: immediate symmetric warp on teleporter entry
- `ic01`: one action to one slide-resolved landing cell
- `va01`: deterministic coverage update and revisit no-op
- `pb01`: ordinary move, legal push, blocked push, exact block-target completion

## Search-State Expansion Added Per Family

- `ul01`: avatar plus key and door state
- `fs01`: avatar plus switch and door bits
- `tp01`: avatar plus fixed teleporter mapping
- `ic01`: avatar plus fixed slide surface and blockers
- `va01`: avatar plus coverage mask
- `pb01`: avatar plus pushable-block position

## What Was Intentionally Excluded From Phase 3

- LLM, VLM, or RL control inside the action loop
- blackboard or POI layers
- hypotheses or mechanic graphs
- durable memory in the runtime path
- legacy `v3_1` branch and merge machinery

## What Future Tracks Must Still Prove

- broader non-movement mechanics
- higher-complexity planning beyond exact movement families
- any future advisory or learned layer staying outside the authoritative Stage 2 control path

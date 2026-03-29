Status: implemented and verified
Scope: click perception doc: transition model
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Purpose**
The Phase 4 click transition model applies one click action to one typed click state and returns one deterministic successor state where the family rules are deterministic.

**Successor Function**
Input:
- one `ClickTypedStateV4`
- one click action

Output:
- one successor `ClickTypedStateV4`
- minimal debug annotations

**Click Legality Semantics**
The authoritative legality surface still comes from the environment contract. Phase 4 candidate generation is narrower: it produces only clicks represented by the typed state.

**Family-Specific State Updates**
- `pt01`: clicking a rotatable tile rotates that tile 90 degrees clockwise; non-clicked tiles remain unchanged.
- `sy01`: clicking a valid right-side cell toggles a placed block at that cell; invalid left-side or divider clicks are exact no-ops.
- `ff01`: clicking inside a closed region marks that region filled; invalid clicks do not fabricate fills.
- `sq01`: clicking the required next color advances progress; wrong clicks reset progress exactly to the family rule used by the implementation.
- `wm01`: visible-step hit or miss handling uses only directly visible mole state.
- `mm01`: hidden-tile reveal and pair-match updates are driven by explicit slot state.

**Terminal-State Updates**
Terminal updates are family-specific and exact:
- `pt01` when all rotatable tiles match target rotations
- `sy01` when the placed right-side block set exactly equals the reflected target set
- `ff01` when every enclosure is filled
- `sq01` when the required click sequence is completed
- `wm01` and `mm01` only where the typed state explicitly represents the visible success condition used by the local solver slice

**Determinism Requirements**
Family transition semantics must be driven by explicit typed-state fields extracted from real environment or config data, not by solver-internal per-level lookup tables.

**Invariants**
- one click in, one successor out
- no environment stepping inside the transition model
- no blackboard or symbolic promotion in successor state

**Validation Against Real Env**
Each family gate checks the transition model against real local environment behavior on verified cases.

For `sy01`, the local rule is fixed:
- the reflection relation is a fixed vertical divider defined by the level
- clicks act on the right side only
- clicking an empty valid cell places a block
- clicking an occupied valid cell removes it
- completion is exact set equality between placed right-side cells and reflected target cells

**Prohibited Shortcuts**
- hidden mechanic tables
- learned next-click predictions
- planner metadata inside the successor state

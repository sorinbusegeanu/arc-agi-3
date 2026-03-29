Status: implemented and verified
Scope: movement doc: family coverage
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Family Coverage

## `ul01`

- mechanic type: key pickup and locked-door dependency
- required typed-state fields: avatar, key bits, door state, bounds, legal actions
- required transition semantics: pickup, locked-door block, unlock-pass completion
- expected search method: BFS
- gate test expectations: typed-state build, checked transition semantics, legal exact plan, one real level solved in Stage 2 loop
- deferred: nothing beyond multi-level rollout tuning

## `fs01`

- mechanic type: order-free latching yellow pressure plates; all activated plates remove one gray collidable door
- required typed-state fields: switch positions, activated switch bitmask, closed-door positions, explicit door-state bit, target cells
- required transition semantics: latch-on-entry activation, no deactivation on revisit, closed-door blocking, door removal only after the full switch set is active
- required search-state representation: avatar position plus switch/door state, not position only
- expected search method: BFS
- gate test expectations: exact switch semantics on checked local paths, augmented-state search proof, one real `fs01` level solved end to end in the Stage 2 loop
- deferred: deeper multi-level optimization only

## `tp01`

- mechanic type: fixed symmetric teleporter pairs with immediate warp on entry
- required typed-state fields: teleporter endpoint positions, explicit teleporter pair list, explicit directional pair map, target cells, walls
- required transition semantics: tentative move, immediate paired-endpoint warp, then goal evaluation from the post-warp position
- required search-state representation: avatar position plus fixed teleporter mapping
- expected search method: BFS or A* with a safe heuristic
- gate test expectations: real config-backed pair extraction, checked warp semantics, teleporter-aware search, one real `tp01` level solved in the Stage 2 loop
- deferred: later teleporter variants such as directed or single-use links

## `ic01`

- mechanic type: frictionless straight-line slide until the next cell would be out of bounds, a wall, or a red hazard
- required typed-state fields: slide mode, exact slide surface, explicit wall cells, explicit red hazard cells, target cells
- required transition semantics: full-ray slide, stop-before-OOB, stop-before-wall, stop-before-red-hazard, exact landing-cell terminal evaluation
- required search-state representation: avatar position plus exact slide surface, walls, hazards, and goal cells
- expected search method: BFS
- gate test expectations: checked real-env slide semantics, slide-resolved search proof, one real `ic01` level solved in the Stage 2 loop
- deferred: larger ice boards and more advanced heuristics

## `va01`

- mechanic type: visit-all walkable coverage with legal revisits
- required typed-state fields: exact coverage-eligible cell set, current covered-state set, walls
- required transition semantics: deterministic neighbor move, first-visit coverage add, revisit no-op for coverage, exact full-coverage completion
- required search-state representation: avatar position plus current covered-state over the fixed eligible-cell set
- expected search method: bounded BFS
- gate test expectations: checked real-env coverage semantics, coverage-aware search proof, one real `va01` level solved in the Stage 2 loop
- deferred: stronger coverage heuristics and larger boards

## `pb01`

- mechanic type: one-block push puzzle with fixed walls, one non-collidable target, and a real step limit
- required typed-state fields: block position, explicit target cell, bounds, wall blocker cells, optional step limit
- required transition semantics: legal push, blocked push, ordinary movement, joint successor positions, exact completion on block-target overlap
- required search-state representation: avatar position plus block position over fixed blockers and target cells
- expected search method: BFS
- gate test expectations: exact env-backed typed-state build, checked push semantics, push-aware search proof, one real `pb01` level solved in the Stage 2 loop
- deferred: stronger deadlock reasoning and larger push spaces

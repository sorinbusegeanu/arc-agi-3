Status: implemented and verified
Scope: movement doc: family coverage
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; targeted movement tests for `pb02`, `pb03`, `fs02`, and `fs03`

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

## `fs02`

- mechanic type: OR-switch door family implemented from the local `fs02` env, where any one orange plate removes the door permanently
- required typed-state fields: switch positions, occupied-switch bits, door state, target cells, and explicit switch logic mode
- required transition semantics: enter-any-switch opens the door, the opened door remains removed, and closed-door cells block motion before opening
- required search-state representation: avatar position plus switch/door state
- expected search method: BFS
- gate test expectations: exact env-backed typed-state build, checked OR-door semantics, legal plan proof, and one real `fs02` level solved before the next-level controller boundary
- deferred: no additional `fs02` semantics beyond the local env-backed permanent-door-removal rule

## `fs03`

- mechanic type: threshold switch-door family implemented from the local `fs03` env, with persistent distinct-switch activation and permanent door removal once the threshold is met
- required typed-state fields: switch positions, occupied-switch bits, activated-switch bits, threshold `k`, door state, and target cells
- required transition semantics: new distinct switch activations latch, the door opens when `activated >= k`, and closed-door cells block motion before opening
- required search-state representation: avatar position plus activated-switch state and door state
- expected search method: BFS
- gate test expectations: exact env-backed threshold extraction, checked persistent activation semantics, legal plan proof, and one real `fs03` level solved before the next-level controller boundary
- deferred: no generic boolean-expression engine or non-persistent threshold semantics

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

## `pb02`

- mechanic type: two-crate push puzzle with two real goals and success only when both crates are on goals
- required typed-state fields: explicit crate set, explicit goal set, exact solved-goal occupancy, bounds, walls, and step limit
- required transition semantics: single-crate push only, blocked push into walls or other crates, exact two-goal completion, and no multi-crate push
- required search-state representation: avatar position plus canonicalized crate positions over fixed blockers and goals
- expected search method: exact A* with an admissible push-goal heuristic
- gate test expectations: exact two-crate parsing, checked multi-crate push legality, exact planner success, a real-env plan replay that wins level 0, and live replanning that reconstructs hidden goal occupancy only when the carry state remains fully consistent

## `pb03`

- mechanic type: one-crate push puzzle with one real goal and one decoy lose pad, implemented from the local `pb03` env
- required typed-state fields: explicit crate position, true goal cells, decoy lose-pad cells, bounds, walls, and step limit
- required transition semantics: single-crate push only, immediate terminal loss on crate-to-decoy push, exact success on crate-to-goal push
- required search-state representation: avatar position plus crate position over fixed blockers, goals, and decoy cells
- expected search method: exact A* with an admissible push-goal heuristic
- gate test expectations: exact decoy parsing, checked decoy-loss semantics, exact planner success, and one real `pb03` level solved in the Stage 2 loop
- deferred: no generic special-tile system beyond the explicit `pb03` decoy rule

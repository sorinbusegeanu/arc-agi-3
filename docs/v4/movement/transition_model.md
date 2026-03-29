Status: implemented and verified
Scope: movement doc: transition model
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Transition Model

## Purpose

The Phase 3 transition model is the exact local successor function for movement-family typed states.

## Successor Function

Input:

- one `MovementTypedStateV4`
- one primitive movement action

Output:

- one deterministic successor typed state
- one minimal transition annotation for search and debugging

## Common Movement Semantics

- cardinal primitive actions only
- one-step motion for `ul01`, `fs01`, `tp01`, `va01`, and `pb01`
- full-slide motion for `ic01`
- out-of-bounds motion is blocked

## Collision And Blocking Semantics

- walls block movement
- closed doors block movement
- hazards block `ic01` slide rays
- push blocks block avatar motion unless a legal push successor exists

## Family-Specific State Updates

### Key Pickup And Door Unlock

- entering a key cell removes the key and sets the key bit
- entering a locked door without the key is blocked
- entering a door with the key removes the door and completes `ul01` level state

### Switch Toggle And Door State Change

- entering an inactive `fs01` switch latches that exact switch active
- revisiting an already active `fs01` switch leaves switch state unchanged
- the `fs01` door remains closed until all switch bits are active
- when all switch bits are active, the door state flips open and the closed-door cells are removed from the blocker set
- closed-door traversal is blocked before switch update
- open-door traversal is legal after the door-state update has already removed that blocker
- the deterministic update order is: bounds and blocker check, avatar move, switch-state update, door-state update, traversable/blocker recompute, terminal recompute
- walls, targets, legal-action surface, and unrelated family fields remain unchanged across an `fs01` step

### Teleporter Movement

- entering a `tp01` teleporter endpoint resolves immediately to its paired endpoint
- the local `tp01` rule is symmetric: either endpoint of a pair warps to the other endpoint
- warp is mandatory on enter, not on stand
- post-warp movement stops immediately for that action
- deterministic update order is: bounds check, wall check, tentative move, teleporter resolution, terminal recompute
- goal handling is checked after warp resolution because the real environment evaluates completion from the post-warp avatar position
- teleporter pairs and wall layout remain unchanged across a `tp01` step

### Ice Or Sliding Continuation

- one action consumes the full straight-line slide ray
- slide continues until the next cell would be out of bounds, a gray wall, or a red hazard
- the successor stops on the last legal cell before that blocking cell
- the yellow goal does not block a slide
- goal handling is evaluated from the final landing cell after slide resolution
- the deterministic update order is: bounds and blocker scan along the chosen ray, final landing-cell commit, terminal recompute
- traversable slide surface, walls, hazards, and legal-action surface remain unchanged across an `ic01` step

### Coverage Updates

- moving onto a new coverage-eligible cell adds it to the covered set
- revisits are legal and leave the covered set unchanged
- completion occurs exactly when covered cells equal the coverage-eligible set
- the deterministic update order is: bounds and wall check, avatar move, coverage update, terminal recompute
- wall layout and coverage-eligible cells remain unchanged across a `va01` step

### Push Mechanics

- ordinary movement without a push updates only the avatar position
- entering a block cell attempts a one-cell push in the same direction
- a push is legal only when the cell beyond the adjacent block is in bounds and not blocked by a wall or another block
- blocked push resolution leaves avatar and block state unchanged for that action
- successful push moves the block first and then places the avatar in the block's previous cell
- completion is evaluated from the updated block position against the explicit typed-state target cells
- the deterministic update order is: bounds and wall check, adjacent-block detection, push legality check, block update, avatar update, terminal recompute
- walls, target cells, bounds, and the legal-action surface remain unchanged across a `pb01` step

## Terminal-State Updates

- `ul01`, `fs01`, `tp01`, and `ic01` produce typed-state success on level completion
- `va01` produces typed-state success when all walkable cells are covered
- `pb01` produces typed-state success when the block reaches the target and failure when the local step limit is exceeded

## Determinism Requirements

- equal input state and equal primitive action must always produce equal successor state
- no environment stepping is allowed inside the transition model
- no learned components are allowed
- family transition semantics must be driven by explicit typed-state fields extracted from real observation or static environment metadata, not by solver-internal per-level lookup tables

## Invariants

- transition output is a new typed state, not an in-place mutation
- action legality still comes from the current legal-action surface
- dynamic family bits remain inside family-specific fields

## Validation Against Real Env

- checked local unit tests compare model successors against real env semantics for `ul01`, `fs01`, `tp01`, `ic01`, `va01`, and `pb01`
- other family tests validate exact local mechanics on known paths and local layouts

## Prohibited Shortcuts

- no score-based terminal inference
- no search-time environment stepping
- no POI or mechanic-graph annotations inside successor state

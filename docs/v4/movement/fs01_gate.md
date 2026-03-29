Status: implemented and verified
Scope: movement doc: fs01 gate
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `fs01` Gate

## Purpose

Phase 3B proves exact movement solving on the first family whose search state is not position-only.

## Mechanic Scope

- yellow non-collidable floor switches
- one gray collidable door while closed
- one green target
- order-free latch activation across all switches in the level

## Required Typed-State Additions

- switch positions
- activated switch bitmask
- door positions while the door is still closed
- explicit door-state bit and open/closed flag
- target cells and static bounds

## Required Transition Semantics

- primitive movement remains cardinal and deterministic
- closed-door traversal is blocked
- entering an unactivated switch latches that switch active
- revisiting an already activated switch does not clear it
- the door opens only when all level switches are active
- target entry solves the typed state only after the door is open

## Required Search-State Augmentation

- visited-state identity must include switch activation state and door state
- the same avatar position under different switch/door configurations must be treated as different search nodes
- no position-only shortcut is allowed for `fs01`

## Required Solver-Policy Behavior

- build exact `fs01` typed state from Stage 2 parsed state
- search over augmented state, not only over avatar position
- return only legal primitive actions or a short legal prefix
- replan after every executed action
- no LLM, VLM, RL, blackboard, POIs, hypotheses, or mechanic graph in the action loop

## Pass Criteria

- `fs01` typed-state build succeeds on at least one verified real local level
- `fs01` transition-model checks match real env semantics on checked local paths
- search distinguishes equal positions with different switch/door state
- `MovementSolverPolicyV4` solves at least one verified real `fs01` level end to end inside the Stage 2 loop
- no legacy v3.1 runtime path is active

## Failure Buckets

- builder
- adapter
- transition
- search
- policy

## Non-Goals

- generic switch-door abstractions for later families
- inferred linkage not directly supported by `fs01`
- multi-family heuristic planning

## Exit Rule For Moving To `tp01`

`tp01` work begins only after this gate passes, because `fs01` is the first proof that the Phase 3 solver handles exact non-positional mechanic state in search and policy execution.

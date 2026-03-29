Status: implemented and verified
Scope: closed loop doc: local memory
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# LocalMemoryV4

## Purpose

`LocalMemoryV4` is a small session-local memory used only to support immediate closed-loop control.

## Scope

It is bounded, deterministic, and local to the active session. It is updated only after an authoritative step result exists.

## Allowed Contents

- recent transitions
- recent actions
- recent step results
- visited-state hashes
- retry counts
- cooldown markers
- tested action-outcome facts
- revealed and unknown cells for hidden-state games
- small task-local notes tied to direct observations

## Forbidden Contents

- durable priors
- hypothesis registry
- mechanic graph state
- promoted POIs
- cross-round symbolic knowledge
- planner score caches
- ranker state

## Update Timing

Memory updates occur only after:

1. transition record creation
2. step-result derivation

## Pruning Rules

- recent records are truncated to a small fixed bound
- visited-state hashes are bounded
- notes are bounded
- cooldowns and retries are kept as small keyed maps

## Reset Rules

- memory initializes empty at session start
- memory may be fully reset between sessions
- no cross-session carry is part of Stage 2

## Serialization Rules

- serialization is optional and lightweight
- only session-local bounded state is serialized
- no durable or cross-session persistence contract is implied

## Non-Goals

`LocalMemoryV4` is not:

- a durable memory layer
- a symbolic planner memory
- a mechanic knowledge base
- a long-horizon learning store


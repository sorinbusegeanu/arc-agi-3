Status: implemented and verified
Scope: Stage 2 runtime overview
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Stage 2 Overview

## Purpose

Stage 2 defines the first minimal live closed loop on top of the `v4` agent contract. It is intentionally small: one environment instance, one continuous step loop, one local memory, one append-only ledger, and one stop-condition surface.

## Scope

Stage 2 covers:

- one live environment session
- authoritative observation extraction
- short-horizon parsed state
- primitive or short-plan action selection
- single-step execution
- local-memory update
- append-only step ledger
- terminal and budget stop conditions

## Runtime Shape

The runtime owns exactly one live environment instance per session and repeatedly performs one step of control against that instance.

There is:

- no probe pass
- no directed pass
- no branch fanout
- no branch winner selection
- no merge phase
- no reconcile phase

## Loop Steps

1. observe the current authoritative environment output
2. parse a short-horizon `ParsedStateV4`
3. choose one primitive action or one short interruptible plan
4. execute at most one action
5. build the authoritative transition record
6. derive the authoritative step result
7. update local session memory
8. append one step ledger record
9. evaluate stop conditions
10. repeat if not stopped

## Component Boundaries

- `agentContract`: authoritative environment-facing models, extractors, validators
- `runtime`: live session control, ledgering, stop evaluation
- `state`: short-horizon parsing only
- `policy`: immediate control decision only
- `memory`: small session-local memory only

## Authoritative Path

The authoritative path is:

1. raw environment output
2. `V4Observation`
3. `V4TransitionRecord`
4. `V4StepResult`

Nothing inferred by parsing, policy, or memory becomes authoritative state.

## Control Path

The control path is:

1. current authoritative observation
2. parsed short-horizon state
3. policy decision
4. one executed action
5. post-step authoritative result

If a short plan is proposed, the runtime still re-enters the loop after at most one executed action.

## Local Memory Path

The memory path is strictly session-local and post-step only.

Memory may store:

- recent transitions
- recent actions
- recent step results
- visited-state hashes
- retry counts
- cooldown markers
- tested action-outcome facts
- revealed or unknown cells where directly supported
- small observation-tied notes

## Logging Path

Stage 2 logging is step-based and append-only through the session ledger.

Each step ledger record may include:

- the authoritative pre-step observation
- a parsed-state summary
- the chosen decision summary
- the executed action
- the transition record
- the step result
- the memory update summary
- the failure bucket, if any
- the stop-condition status

## Stop Conditions

Stage 2 supports only:

- terminal win
- terminal fail
- hard step budget exhausted
- explicit invalid-state abort

## Explicit Removals From v3.1

Stage 2 removes these from the runtime path:

- probe planning
- directed planning
- blackboard merge
- mechanic graph maintenance
- hypothesis registry updates
- subgoal-chain runtime
- helper-worker branches
- ranker path
- durable-memory flush and reconcile

## Success Criteria

- one live environment instance is used for the whole session
- every executed step yields one transition record and one step result
- failures are localizable to one Stage 2 bucket
- local memory updates happen only after step-result derivation
- no v3.1 branch, merge, or hypothesis logic remains in the runtime path

## Non-Goals

Stage 2 is not:

- a world-model runtime
- a blackboard runtime
- a mechanic-inference runtime
- a hypothesis-management runtime
- a durable-learning runtime
- a multi-branch planner runtime


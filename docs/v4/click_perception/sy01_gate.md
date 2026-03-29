Status: implemented and verified
Scope: click perception doc: sy01 gate
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Purpose**
Phase 4B proves exact click solving for the real local `sy01` symmetry game on top of the Stage 2 loop.

**Mechanic Scope**
`sy01` uses a fixed vertical divider. The left side contains the source pattern. The solver clicks on the right side to place or remove blocks until the placed set exactly matches the reflected target set.

**Required Typed-State Additions**
- fixed reflection axis column
- explicit source-to-target reflection mapping
- reflected target cell set
- current placed right-side block set
- completion-relevant exact set state

**Required Transition Semantics**
- one click produces one deterministic successor
- valid right-side click toggles the clicked cell
- left-side or divider click is a no-op
- completion occurs only when placed cells equal reflected target cells exactly

**Required Search-State Representation**
Search state must include mirrored-state progress, not clicked-cell identity only. The same click cell under different placed-block sets is a different solver state.

**Required Solver-Policy Behavior**
- build exact `sy01` typed state from direct observation plus local config
- search over symmetry-resolved successors
- return only legal click actions or a short plan prefix
- replan after each executed click
- no LLM, VLM, or RL dependency

**Pass Criteria**
- `sy01` typed-state build passes on at least one verified real level
- `sy01` transition checks match the real env on checked cases
- search handles mirrored-state successors correctly
- `ClickSolverPolicyV4` solves at least one verified real `sy01` level end to end inside the Stage 2 loop
- no legacy runtime surfaces are used

**Failure Buckets**
- builder
- adapter
- transition
- selection/search
- policy

**Non-Goals**
- generic symmetry reasoning beyond the real local fixed-axis mechanic
- hidden-state inference
- blackboard, POIs, hypotheses, or mechanic graphs

**Exit Rule For Moving To `ff01`**
`ff01` work starts only after the `sy01` gate passes with exact symmetry-backed typed state, exact transition semantics, and a real Stage 2 end-to-end solve.

Status: implemented and verified
Scope: Phase 4 click/perception overview
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Purpose**
Phase 4 adds an exact click/perception solver track on top of the Stage 2 single-session loop. The track is environment-backed and action-loop local: no LLM, VLM, RL, blackboard, POIs, hypotheses, mechanic graph, or durable memory.

**Scope**
Implemented Phase 4 families:
- `pt01`
- `sy01`
- `ff01`
- `sq01`
- `wm01`
- `mm01`

Excluded for this phase:
- non-click movement families
- mixed movement plus learned perception tracks
- any runtime that needs branch/merge or advisory symbolic layers

**Typed Click/Perception State**
`ClickTypedStateV4` is solver state, not authoritative env state. It keeps:
- exact clickable cell candidates
- the visual grid needed for exact control
- legal click surface from the current observation
- small family-specific fields such as rotation state, mirror targets, fill regions, sequence progress, visible mole cells, and memory-tile visibility state

**Family Adapters**
Each family adapter builds typed state from:
- direct current observation content
- static environment metadata and local level config
- allowed Stage 2 local-memory facts only when needed

No adapter may depend on blackboard data, POIs, hypotheses, or learned family classification.

**Exact Click Transition Model**
The transition model applies one click action to one typed state and returns one deterministic successor for deterministic families. Family-specific updates cover:
- `pt01`: 90 degree clockwise tile rotation
- `sy01`: mirrored place/remove toggles
- `ff01`: closed-region fill updates
- `sq01`: required click-order progression or reset
- `wm01`: visible-step hit or miss handling only from directly visible state
- `mm01`: reveal and pair-match updates

**Search/Selection Layer**
The click search layer supports:
- exact candidate generation from current typed state
- bounded BFS or A* where exact planning is feasible
- deterministic greedy selection where one-step exact control is enough

Plans remain short and interruptible. The loop re-evaluates after every executed click.

**Solver Policy**
`ClickSolverPolicyV4` conforms to the Stage 2 policy surface. It:
- consumes `ParsedStateV4`
- builds a family-specific `ClickTypedStateV4`
- runs exact click search or exact deterministic selection
- returns one legal click or a short plan prefix

**Integration With Stage 2 Loop**
Phase 4 uses the unchanged Stage 2 order:
1. observe
2. parse
3. build typed click state
4. choose one click or short click prefix
5. execute one click
6. record transition and step result
7. update bounded local memory
8. append ledger
9. evaluate stop conditions

**Failure Buckets**
- builder
- adapter
- transition
- selection
- policy

**Current Verification State**
- dedicated gate tests exist for `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, and `mm01`
- Phase 4 consolidation and regression coverage exists
- click solver stays inside the Stage 2 single-loop contract
- no forbidden runtime dependencies appear

**Non-Goals**
- learned click policies
- planner/ranker layers
- symbolic perception promotion
- durable memory or post-run consolidation inside the action loop

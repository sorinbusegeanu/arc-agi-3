Status: implemented and verified
Scope: click perception doc: search policy
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Purpose**
Phase 4 search and selection chooses exact click candidates from typed state and returns only short, interruptible click plans.

**Candidate Generation**
Candidates come only from typed state:
- exact clickable cells
- exact family-specific click targets

**Search/Selection Algorithms**
- bounded BFS or A* where exact planning is feasible
- deterministic greedy exact selection where one-step control is enough

**Plan Representation**
Plans are short ordered click actions. No opaque candidate bundles are allowed.

**Action-Prefix Execution Rule**
Execute at most a short prefix and re-enter the Stage 2 loop after each executed click.

**Replan Rule**
Re-evaluate after every executed click. No blind long fixed click sequences.

**Legality Rule**
Returned clicks must still be legal under the current authoritative observation surface.

**Failure Handling**
Failures must localize to:
- builder
- adapter
- transition
- selection
- policy

**Non-Goals**
- learned rankers
- branch-and-merge runtimes
- symbolic perception layers

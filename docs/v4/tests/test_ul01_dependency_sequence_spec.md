Status: implemented and verified
Scope: tests doc: test ul01 dependency sequence spec
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# `ul01` Dependency-Sequence Test Spec

The tests must be based on the real local `ul01` level layout and rule implementation.

They must verify, using direct environment truth only:

- the game contains a dependency sequence that is more than pure directional walking
- the required action order is respected by a deterministic local policy or exact action script
- terminal success is reached only when the true dependency order is satisfied
- terminal or non-terminal outcomes are taken only from the environment truth surface

The tests must not use:

- POIs
- hypotheses
- blackboard state
- planner abstractions

Coverage requirement:

- if the local level data makes it cheap and reliable, require deterministic win-path coverage for all levels
- otherwise require at least one verified deterministic win path for level 1 and one separate proof that incorrect ordering does not falsely pass

For `ul01`, incorrect ordering must include at least one case where the agent tries to pass the door before collecting the key and is blocked or remains non-successful according to the real environment.

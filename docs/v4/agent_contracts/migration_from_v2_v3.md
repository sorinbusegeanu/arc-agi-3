Status: implemented and verified
Scope: agent contracts doc: migration from v2 v3
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Migration From v2 And v3 To v4

## Why v4 Narrows The Authoritative Path

The repeated failure mode in earlier stacks was the same: too much inferred structure entered the core control path too early.

Across the earlier local stacks:

- the symbolic runtimes concentrated heuristic POIs, hypotheses, chain state, and planner abstractions in the runtime loop
- the VLM loop coupled perception and control too tightly
- the RL path still relied on heuristic reward and proposal layers

`v4` responds by shrinking the authoritative path to the direct environment contract.

## What Remains Authoritative From Old Stacks

These survive into `v4` as authoritative because they come directly from the environment or wrapper metadata:

- the raw returned frame object
- the raw per-step `state`
- the raw per-step progress counters
- the raw per-step legal action list
- the exact executed action id and payload
- static wrapper metadata when directly exposed

## What Becomes Advisory Only

The following move out of the authoritative path and become advisory layers:

- POIs
- inferred objects and canonicalized entities
- mechanic graphs
- hypotheses
- planner beliefs
- route scores
- proposal rankings
- learned value estimates
- symbolic promotions
- inferred topology

These may still exist in planner, memory, analysis, or evaluation code, but they can no longer masquerade as environment truth.

## What Is Deleted From The Base Control Contract

The base control contract no longer includes:

- heuristic world-model state
- symbolic chain progression as fact
- planner rationale as part of execution truth
- reward shaping terms as environment truth
- inferred legality
- inferred terminality

The base control contract is now only:

1. authoritative observation
2. authoritative action
3. authoritative post-observation
4. derived terminal signal from raw state

## Migration Impact On Planner, Runtime, Memory, And Logging

Planner:

- planners must consume `v4` observations and produce `v4` actions
- planner beliefs must remain sidecar state, not authoritative fields

Runtime:

- runtime loops must preserve the observation-action-observation boundary
- wrappers and adapters become the only environment-facing translation layer

Memory:

- memory can still store inferred structure, but those stores are explicitly advisory
- memory writes must not overwrite authoritative step truth with promoted abstractions

Logging:

- transition logs should center on raw observations, executed action, legality, and terminal status
- heuristic analysis should move to separate extensions or secondary artifacts

## Backward-Compatibility Stance

`v4` is intentionally not backward-compatible with earlier mixed-authority contracts at the semantic level.

Adapters may import legacy observations, actions, or logs into `v4`-shaped records only when:

- authoritative fields map directly to observed environment data
- non-authoritative fields are either dropped or carried in clearly advisory sidecars

If a legacy field combined observation with inference, `v4` treats it as non-authoritative by default.

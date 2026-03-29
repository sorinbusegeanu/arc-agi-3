Status: outdated relative to current repo state
Scope: Historical Stage 2 next-gate planning note
Source of truth: `/home/zodrak/zod/tests/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Stage 2 Next Gate

## Why `ul01` And `tt01` Are The Next Proving Games

They add real structure that the directional tutorial games do not cover:

- `ul01` adds dependency ordering with key-before-door progression
- `tt01` adds target collection plus blocking layout or hazard structure

## What New Properties They Validate Beyond `ez01`–`ez04`

- dependency-sensitive correctness
- progress under nontrivial layout constraints
- rejection of bad or incomplete action orderings
- longer non-terminal loop stability

## Why Fault-Injection Tests Are Required Before Phase 3

Before Phase 3, the Stage 2 runtime must prove it rejects corrupted observations, invalid actions, and inconsistent terminal derivations cleanly and locally rather than letting bad authoritative inputs leak deeper into control.

## Pass Criteria For The Next Gate

- `ul01` and `tt01` Stage 2 tests pass
- deterministic dependency or progress proofs pass
- invalid-action and corrupted-observation rejection tests pass
- terminal-mapping mismatch rejection tests pass
- no legacy v3.1 runtime path is active

## What Still Remains Unproven Afterward

- richer mechanic discovery
- hidden-state control
- broader long-horizon planning
- any Phase 3 or symbolic-runtime behavior

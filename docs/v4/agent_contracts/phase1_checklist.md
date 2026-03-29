Status: outdated relative to current repo state
Scope: agent contracts doc: phase1 checklist
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Agent Contract Phase 1 Checklist

## Required Docs

- [ ] `README.md` exists and defines the source-of-truth rule
- [ ] `observation_format.md` traces every authoritative observation field to a real environment source
- [ ] `action_format.md` documents legal and illegal action handling
- [ ] `authoritative_state_fields.md` separates authoritative from advisory fields
- [ ] `terminal_signal.md` documents terminal mapping from real raw states
- [ ] `transition_record.md` documents replay-sufficient transition structure
- [ ] `per_step_result_record.md` documents the reduced per-step result surface
- [ ] `environment_metadata.md` documents only real static metadata
- [ ] `serialization.md` documents JSON serialization rules
- [ ] `validation_errors.md` documents the error model
- [ ] `migration_from_v2_v3.md` explains what leaves the authoritative path
- [ ] `examples.md` shows grounded compact examples

## Required Source Files

- [ ] `src/v4/agentContract/types.py`
- [ ] `src/v4/agentContract/validators.py`
- [ ] `src/v4/agentContract/adapters.py`
- [ ] `src/v4/agentContract/environmentMetadata.py`
- [ ] `src/v4/agentContract/errors.py`
- [ ] `src/v4/agentContract/extract.py`
- [ ] `src/v4/agentContract/README.md`

## Required Tests

- [ ] observation tests cover valid extraction and strict rejection paths
- [ ] action tests prove legal and illegal validation
- [ ] terminal tests prove mapping from real `GameState` values
- [ ] transition tests prove invariants and replay sufficiency
- [ ] adapter integration tests exercise local engine or wrapper surfaces

## Completion Gates

- [ ] every authoritative field is traced to a real environment source
- [ ] no invented authoritative fields remain
- [ ] legal and illegal action validation is proven
- [ ] terminal mapping is proven against real environment states
- [ ] transition record replay sufficiency is proven
- [ ] serialization rules are documented
- [ ] error model is documented and implemented
- [ ] adapters are tested against local environments
- [ ] all authoritative models are separated from advisory layers

## Explicit Non-Goals

- [ ] no planner logic is included in the authoritative contract
- [ ] no POI extraction is included in the authoritative contract
- [ ] no reward shaping is included in the authoritative contract
- [ ] no mechanic inference is included in the authoritative contract
- [ ] no hypothesis graph is included in the authoritative contract

## Sign-Off Criteria

- [ ] docs and code agree on the same authoritative field set
- [ ] tests pass in the local environment setup or skip cleanly when dependencies are unavailable
- [ ] advisory layers are kept outside the base control contract
- [ ] Phase 1 scope ends at the environment-facing contract boundary, not at planner integration


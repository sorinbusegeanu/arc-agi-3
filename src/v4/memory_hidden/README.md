# Memory Hidden Package

## Purpose

`src/v4/memory_hidden` implements the exact fail-closed Phase 5 hidden-information movement track.

## Boundary

- deterministic `ms01` only
- no learned model, blackboard, hypotheses, durable memory, LLM/VLM/RL
- uses only current/previous observation, environment metadata, and parsed local-memory reference already exposed through `ParsedStateV4`
- aborts instead of inventing hidden-cell knowledge

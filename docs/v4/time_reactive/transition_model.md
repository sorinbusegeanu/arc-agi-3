Status: implemented and verified
Scope: `sv01` transition model
Source of truth: `/home/zodrak/zod/src/v4/time_reactive/transitionModel.py`
Last verified against: unknown

# Transition Model

Implemented deterministic consequences:

- movement and wait primitives
- hunger decay every step
- warmth decay outside warm zones
- food restore on food cells
- warm-zone protection against warmth decay
- terminal failure on resource exhaustion
- terminal success on surviving the bounded timer

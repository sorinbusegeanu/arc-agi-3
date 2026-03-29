Status: implemented and verified
Scope: `rs01` transition model
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/transitionModel.py`
Last verified against: unknown

# Transition Model

Implemented deterministic consequences:

- grid movement on walkable cells
- blocking on walls and bounds
- collecting a safe-color target as legal progress
- collecting a non-safe-color target as terminal failure
- updating remaining-target groups and collected counts
- marking success when no remaining targets are left

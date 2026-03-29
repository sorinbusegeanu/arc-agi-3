Status: implemented and verified
Scope: `sv01` mechanics
Source of truth: `/home/zodrak/zod/src/v4/time_reactive/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/sv01/63be02fb/sv01.py`
Last verified against: unknown

# `sv01` Mechanics

Current implementation treats `sv01` as a movement + wait survival family with explicit hunger, warmth, and timer decay.

Implemented now:

- hunger and warmth are read from the rendered bars
- food and warm-zone cells are extracted from the grid
- wait is legal when action `5` is exposed
- the transition model decrements resources exactly according to the current package rules

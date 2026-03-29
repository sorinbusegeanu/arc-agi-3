Status: implemented and verified
Scope: `rs01` mechanics
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/rs01/63be02fb/rs01.py`
Last verified against: unknown

# `rs01` Mechanics

Current implementation treats `rs01` as a movement-only family where the top-of-screen signpost color is the authoritative safe target color.

Implemented now:

- safe target color is extracted from the live observation
- targets are grouped by color
- wrong-color collection is terminal loss
- safe-color collection removes the target from the remaining target set
- search avoids intentionally entering visible wrong-color targets

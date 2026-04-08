Status: implemented; gate-covered, live regression still mixed
Scope: `rs01` mechanics
Source of truth: `/home/zodrak/zod/src/v4/rule_switch/*`, `/home/zodrak/zod/other_repos/arc-interactive/environment_files/rs01/63be02fb/rs01.py`
Last verified against: current repo state on 2026-03-30; targeted manual live regression for `rs01`

# `rs01` Mechanics

Current implementation treats `rs01` as a movement-only family where the top-of-screen signpost color is the authoritative safe target color.

Implemented now:

- safe target color is extracted from the live observation
- targets are grouped by color
- wrong-color collection is terminal loss
- safe-color collection removes the target from the remaining target set
- search avoids intentionally entering visible wrong-color targets
- policy-side live fallback now skips invalid successor reconstructions and treats “current safe color fully collected” as a goal-compatible phase state before surfacing a safe-color no-plan failure

Current boundary:

- the manual live regression row remains a real search failure, but it no longer collapses into a successor-validation abort

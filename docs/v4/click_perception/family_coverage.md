Status: implemented and verified
Scope: click perception doc: family coverage
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Phase 4 Coverage**

| family | mechanic type | required typed-state fields | required transition semantics | expected search/selection method | gate expectations | deferred |
| --- | --- | --- | --- | --- | --- | --- |
| `pt01` | tile rotation | rotation tiles, target rotations | exact 90 degree clockwise rotation | bounded exact search | solve at least one real level end to end | later harder click families |
| `sy01` | fixed-axis spatial reflection with right-side place/remove | reflection axis, explicit reflection pairs, mirror targets, placed cells | exact right-side toggle and invalid-click no-op behavior | bounded exact search over mirrored-state progress | solve at least one real level end to end with symmetry-aware state | broader symmetry track |
| `ff01` | closed-region fill | explicit fill regions, filled set | exact region fill updates | deterministic region selection | dedicated gate plus bounded live-level completion | larger topology families |
| `sq01` | click order | sequence, progress, visible color cells | exact sequence advance/reset plus pending-advance handling | deterministic next-click selection | dedicated gate plus bounded live-level completion | richer memory/order tracks |
| `wm01` | visible reaction clicking | visible mole cells, click radius | visible-step hit/miss update | deterministic visible-step selection | dedicated gate plus bounded live-level completion | timing-heavy or stochastic planning |
| `mm01` | reveal and match | slot colors, hidden/revealed/matched slots | exact reveal and pair-match updates | deterministic reveal-then-match selection | dedicated gate plus bounded live-level completion | larger partial-observation memory tracks |

Status: implemented and verified
Scope: click perception doc: sy01 mechanics
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**Exact Local Mechanic**
The real local `sy01` implementation defines a fixed vertical divider at `x = 5` on an `11 x 11` grid. Pattern cells live on the left side. Their mirrored targets are computed by reflecting across that divider with `mirrored_x = GRID_WIDTH - 1 - x`.

Clicking acts on the right side only. Clicking an empty valid right-side cell places a player block there. Clicking a right-side cell that already contains a player block removes it. Clicking the left side or the divider does not place a mirrored block.

Completion is exact set equality between the current placed right-side blocks and the reflected target set derived from the level pattern. Failure can occur when the move budget is exhausted.

| mechanic field | observed source | state representation | update rule | notes |
| --- | --- | --- | --- | --- |
| reflection relation | local `sy01.py` constants and `_get_mirror_position` | fixed vertical reflection across `x = 5` | static per level | explicit local rule |
| mirror axis | divider sprites and `CENTER_X` | one fixed vertical column | static per level | divider is visible in the level |
| source pattern | level `pattern_positions` | tuple of left-side cells | static per level | columns `0..4` only |
| target cells | reflected source pattern | explicit right-side target set | static per level | target is exact set, not score-based |
| valid click behavior | `_click_at` | right-side place/remove toggle | empty valid cell places, occupied valid cell removes | clicked side changes directly |
| invalid click behavior | `_click_at` | no-op | left-side or divider click does not place/remove mirrored state | still consumes an env step |
| completion condition | `_check_win()` | exact set equality | success when placed set equals reflected target set | no partial completion |
| terminal implications | step loop and move budget | success or failure | success on exact match, failure on move exhaustion | success advances after short end animation |

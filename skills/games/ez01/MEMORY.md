# Game Memory

## 1. Current State
- From run 1d1c91a7: level ?; ?/? completed; state ?.

## 2. Confirmed Mechanics
- none

## 3. Successful Strategies
- none

## 4. Known Exceptions
- none

## 5. Active Hypotheses
- none
- Legacy review retained pending structured migration:
  ## Patterns
  - "up" moves blue (color 9) by dy=−8, dx=0 during gap-closing steps; displacement score 1.0, margin 0.125. Confirmed across T2, T4–T6, T8–T12, T14–T18, T20–T25. Yellow (color 11) never moves during these steps.
  - Intermediate separation events (stacked → apart) produce 258 changed pixels, both colors 9 and 11 shift, and the candidate-displacement algorithm returns margin 0.0 with two tied candidates at score 0.5. Observed at T3, T7, T13, T19.
  - Intermediate completions change exactly two indicator pixels: one 3→11 and one 11→14 in the top-left row. Confirmed at T3 (level 0→1), T7 (level 1→2), T13 (level 2→3), T19 (level 3→4).
  - The indicator row contains exactly 5 positions; completed levels turn green (14), the current level is yellow (11), remaining are gray (3). This confirms a 5-level goal.
  - Gap-to-close press counts: level 0 = 1, level 1 = 3, level 2 = 5, level 3 = 5, level 4 = 6. The 2n+1 progression (1,3,5,7,9) breaks at levels 3 and 4.
  - Blue square starting x-position by level: 40, 40, 48, 48, 48. Starting y-position: 24–32, 32–40, 24–32, 40–48, 24–32.
  - T13 separation shows yellow moving up-and-right while blue moves down. T3, T7, T19 separations appear purely vertical.
  - **T26 (WIN) is structurally different from intermediate completions.** Only 145 changed pixels (not 258). Yellow (11) disappears entirely (before 65, after 0): the yellow square turns blue (11→9, 64 px) and the bottom-left yellow dot vanishes (11→5, 1 px). The original blue square disappears (9→5, 64 px). A 16-pixel green bar appears at bottom-left (5→14). The indicator row goes from 4 green + 1 yellow to 5 green. No bidirectional separation occurs; instead yellow transforms into blue and the original blue is removed.
  - T26 candidate_displacement returns score 0.015, margin 0.0 — essentially meaningless for this transition.
  
  ## Useful approaches

## 6. Failed Approaches
- none

## 7. Next Actions
- none

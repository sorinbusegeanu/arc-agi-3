# Game Memory

## 1. Current State
- level 0; 0/5 completed; state NOT_FINISHED.

## 2. Confirmed Mechanics
- confirmed: A stationary blue square (color 11) serves as the level goal and acts as a physical obstacle. The player cannot move into the blue square (blocked). The level completes when the player is adjacent to the blue square and takes a directional action that separates them. The blue square is positioned in the bottom-left corner of the level.
  Applies: level 0 of ez01
- confirmed: Level progress is tracked by a row of small squares in the top-left corner; completed levels turn yellow.
- confirmed: The player (yellow square, color 9) moves exactly 8 pixels per directional action. From center: up is open, left is blocked, down succeeds once then blocked, right is open. The player successfully navigated right→right→up→left→down→left, proving a path exists around obstacles. From center, left and down are both blocked (confirmed across multiple attempts including 3 additional consecutive blocked tries). The player is stuck at center with only up and right available; the blue goal in the bottom-left corner is not adjacent to the player at this position.
  Applies: level 0 of ez01

## 3. Successful Strategies
- none

## 4. Known Exceptions
- none

## 5. Active Hypotheses
- none

## 6. Failed Approaches
- none

## 7. Next Actions
- none

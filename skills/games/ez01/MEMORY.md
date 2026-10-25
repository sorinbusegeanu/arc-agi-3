# Game Memory

## 1. Current State
- From run bd014adb: level 1; 1/5 completed; state NOT_FINISHED.

## 2. Confirmed Mechanics
- [P-e8566ad270a4887b] confirmed, confidence 0.95: The player (yellow square, color 9) moves exactly 8 pixels per directional action (up/down/left/right). Attempting to move into a boundary or obstacle produces 0 pixel change (blocked).
  Refs: + bd014adb:T18, bd014adb:T19, bd014adb:T20, +16

## 3. Successful Strategies
- none

## 4. Known Exceptions
- none

## 5. Active Hypotheses
- [P-4b1094ee143c3071] candidate, confidence 0.67: A stationary blue square (color 11) serves as the level goal. The level completes when the player is adjacent to the blue square and takes a directional action that separates them.
  Refs: + bd014adb:T10
- [P-91d3d858815bb42f] candidate, confidence 0.67: Level progress is tracked by a row of small squares in the top-left corner; completed levels turn yellow.
  Refs: + bd014adb:T10

## 6. Failed Approaches
- none

## 7. Next Actions
- none

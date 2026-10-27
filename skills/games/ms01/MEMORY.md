## Established mechanics (Level 1)
- Blue square is the movable agent; green square (bottom-right) is a hazard—contact triggers GAME_OVER. Light-blue dot (bottom-left) is static and non-goal.
- Up and left from the top-left starting position produce zero change (screen boundary). Down and right produce visible movement with a gray trail.
- The agent can traverse the left column downward, the bottom edge rightward, the top edge rightward, and the right edge downward.
- Interior traversal at column 2 (one cell right of the left column) is safe from bottom to top; red and yellow objects at column 2 are passable—neither hazards nor goals.
- The interior hazard zone extends at least four columns right of the left column when entered from the bottom edge; moving up from the bottom at columns 3 and 4 triggers GAME_OVER.
- Entering the interior from the top edge at column 3 is safe for the first two rows but triggers GAME_OVER on the third row down.
- Top-edge traversal rightward is safe through at least column 5; entering the interior from the top edge at column 5 is safe for at least one row down.

## Failed hypotheses (Level 1)
- Approaching the light-blue dot from above (via left column) did NOT trigger completion.
- Traversing the bottom edge rightward, the top edge rightward, and the right edge downward did NOT trigger completion.
- Moving left from a position adjacent to the green square (along the bottom) triggers GAME_OVER; the hazard zone extends at least one cell to the left of the visible green square.
- From the bottom area one step right of the left column, moving left or up produces zero change.
- Moving right from the bottom area into the green square's column triggers GAME_OVER.
- Moving up from the bottom edge into the interior at columns 3 and 4 triggers GAME_OVER.
- Entering the interior from the top edge at column 3 and moving down three rows triggers GAME_OVER.
- Passing through red and yellow objects at column 2 did NOT trigger completion or GAME_OVER.
- Moving left from an interior position (reached via left column then rightward) triggers GAME_OVER, confirming the hazard zone extends into the interior from the left side as well.

## Next experiment (Level 1, unsolved)
- Question: Is the right-side interior (columns 5+) safe to enter from the top edge beyond the first row, and does it contain additional objects or a completion trigger?
- Competing explanations: (a) the hazard is confined to columns 3–4 and the right interior is safe with hidden objects; (b) the entire interior beyond column 2 is hazardous; (c) completion requires visiting a specific combination of objects or reaching a specific untested cell.
- Test: From a position one row below the top edge at column 5, move down. If safe, continue exploring the right interior for new objects or completion triggers. If GAME_OVER, the hazard extends to column 5 and the safe interior corridor is narrower than columns 3–4.

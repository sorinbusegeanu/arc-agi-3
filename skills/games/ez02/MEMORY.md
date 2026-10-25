## Mechanics
- Yellow square is the player in the first level; blue square is the player in subsequent levels. Overlap completes the level.
- Level 4: Blue responds to input (up and left move it; right is blocked in some states), yellow stays fixed as the goal.

## Strategy notes
- First level: Move toward the goal then adjust to achieve overlap; verified route stored for matching start states.
- Second level: Blue responds to input, yellow stays fixed. Right is blocked in some states. Navigate blue around yellow using mixed vertical and horizontal moves; verified route stored for matching start states.
- Third level: Blue responds to input, yellow stays fixed. Right is blocked in some states. Left moves bring blue toward yellow. When blue is adjacent but not overlapping, try up or down to achieve true overlap; verified route stored for matching start states.
- Fourth level: Blue is the player, yellow is the fixed goal. Right is blocked; use left and up to close distance. Verified route stored for matching start states.

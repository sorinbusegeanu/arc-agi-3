# Global Memory

- Track player and target objects separately even when they appear adjacent; use color-coded pixel transitions to distinguish which object moved.
- When an action produces 0 changed pixels, the player is blocked by a boundary or obstacle; try a different direction rather than repeating.
- Measure displacement (dy, dx) from pixel-diff data to confirm movement direction and step size rather than inferring from frames alone.
- Progress indicators (e.g., small squares in a corner) are reliable completion signals; compare before/after frames to detect level transitions.
- A level may require reaching a target AND then taking an additional action; do not assume first contact completes the level. If moving into the target is blocked, the completion action may be moving AWAY from it.
- When repeatedly blocked in one direction, systematically test all four directions to map the accessible region before concluding the target is unreachable.
- If one direction succeeds repeatedly but another is always blocked, the player may be in a corridor; backtrack and try perpendicular directions to find alternate paths to the target.
- When stuck after moving away from the target, backtrack to the last position where movement toward the target was possible rather than continuing to probe blocked directions from the dead-end position.
- If the player is adjacent to a target and the only available movement is into the target (blocked), the correct action is to move in the opposite direction to separate from the target.
- When multiple consecutive moves in one direction lead to a dead-end (edge + blocked perpendicular), reverse the entire sequence of moves to return to a junction with more options rather than probing remaining blocked directions.
- A closed cycle of moves (e.g., up-right-down-left) that returns to the same position confirms the player is in an open area but may still be at a dead-end if two directions are blocked; use the cycle to verify position, then explore the unblocked directions to find a path around obstacles.
- When trapped in a small oscillation loop where a cycle of moves returns to the same position, do not repeat the cycle or re-probe blocked directions; instead commit to moving in the goal's direction for multiple consecutive steps to break out of the local area.
- In a narrow corridor where only two opposite directions work and the goal direction is blocked, do not oscillate between the two open directions; instead move perpendicular to the corridor (up or down) to find an opening that leads around the obstacle toward the goal.
- A direction may succeed for one step then become blocked (wall one step away); do not assume a direction is permanently open or permanently blocked—re-test after position changes and note that single-step success followed by blockage indicates a nearby wall requiring a different route.

# Global Memory

- Track player and target objects separately even when they appear adjacent; use color-coded pixel transitions to distinguish which object moved.
- When an action produces 0 changed pixels, the player is blocked by a boundary or obstacle; try a different direction rather than repeating.
- Measure displacement (dy, dx) from pixel-diff data to confirm movement direction and step size rather than inferring from frames alone.
- Progress indicators (e.g., small squares in a corner) are reliable completion signals; compare before/after frames to detect level transitions.
- A level may require reaching a target AND then taking an additional action; do not assume first contact completes the level.
- When repeatedly blocked in one direction, systematically test all four directions to map the accessible region before concluding the target is unreachable.

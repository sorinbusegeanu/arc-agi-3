## Established mechanics
- Level 0: completed with a single "up" action from the initial state.
- Level 1: From the initial state, "up" and "left" produce zero visible change (agent already at top-left). "down" moves the agent to the bottom-left. "right" from the bottom-left position completes the level.
- Level 2: From the initial state, "up" and "left" produce zero visible change. "down" moves the agent to the bottom-left. "right" from the bottom-left position completes the level.
- Level 3: From the initial state, "up" and "left" produce zero visible change. "down" moves the agent downward but does not complete the level. "right" from the initial state completes the level. "right" from the bottom-left position (after "down") produced visible change but did not complete the level.
- Level 4: From the initial state, "up" moves the agent to the top-left. "left" from the top-left produces zero visible change. "down" from the top-left moves the agent back down. "right" from the top-left completes the level. "right" from the initial state (bottom-left) produced visible change but did not complete the level.

## Verified strategies
- Level 0 (matching initial state): press "up" once to complete.
- Level 1 (matching initial state): press "down" then "right" to complete. "up" and "left" are no-ops from the initial state.
- Level 2 (matching initial state): press "down" then "right" to complete. "up" and "left" are no-ops from the initial state.
- Level 3 (matching initial state): press "right" once to complete. "up" and "left" are no-ops. "down" then "right" does not complete the level (unlike levels 1–2).
- Level 4 (matching initial state): press "up" then "right" to complete. "left" from the top-left is a no-op. "right" from the initial state does not complete the level.

## Efficiency note
- On levels 1–3 initial states, "up" and "left" produced no pixel change and were ineffective.
- Level 3's completion condition differs from levels 1–2: a single "right" from the initial state suffices; the "down"-then-"right" sequence that works on levels 1–2 does not complete level 3.
- Level 4 mirrors Level 1's pattern but inverted vertically: "up" then "right" completes it, while "right" from the initial state does not.

## Established mechanics
- Blue square is the player; each directional action moves it one step (confirmed levels 0–4).
- Gray square is a goal: entering it completes the level (confirmed levels 0–4).
- Gray entry acceptance is direction-specific and can depend on other objects' positions. Level 1: rejected right-entry from left, accepted up-entry from below. Level 2: accepted down-entry from above, rejected right-entry from left at bottom. Level 3: initially rejected both left and below approaches, but accepted right-entry from left after the yellow was pushed away from its initial position. Level 4: initially rejected right-entry from left and down-entry from above, but accepted down-entry from above after the yellow was pushed away from its initial position.
- Yellow square is pushable: moving the player into it displaces the yellow (levels 0–2). In level 1, yellow was pushed off-screen and the level still completed.
- (level 3) Pushing yellow upward from below relocated it to the bottom-right corner rather than one step up. The yellow is now at both the right and bottom edges, unpushable in any direction from that position.
- (level 4) Pushing yellow upward from below (at top-right corner) relocated it to the bottom-right corner rather than one step up. Consistent with level 3's anomalous corner push.
- Gray is stationary across all observed frames.

## Failed hypotheses
- (level 1) Right into gray from left: no-op; gray was later entered from below.
- (level 2) Right at right edge: no-op. Boundary.
- (level 2) Right when adjacent to gray at bottom: no-op. Gray rejected entry from left at bottom.
- (level 2) Down at bottom edge: no-op. Boundary.
- (level 3) Right into gray from left (before yellow was moved): no-op. State-specific to when yellow was in its initial position.
- (level 3) Up into gray from below (before yellow was moved): no-op. State-specific to when yellow was in its initial position.
- (level 3) Right at right edge: no-op. Boundary.
- (level 3) Up when adjacent to gray from below (before yellow was moved): no-op. Gray rejected entry.
- (level 4) Right at right edge: no-op. Boundary.
- (level 4) Down into gray from above (before yellow was moved): no-op. Gray rejected entry from above when yellow was at top-right.

## Verified strategies
- (level 3) Push the yellow away from its initial position (upward push relocated it to bottom-right corner, making it immovable), then navigate to the left of the gray and enter it from the left.
- (level 4) Navigate to the right edge and move up to reach the yellow at top-right corner. Push the yellow upward (relocated to bottom-right corner). Navigate back down along the right edge and enter gray from above.

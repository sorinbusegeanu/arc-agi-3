# Game Memory


## 2. Confirmed Mechanics
- confirmed: In the recorded ez01 run, up moved the player upward by one 8-pixel cell when the destination was open.
  Applies: Observed ez01 states where an upward move changes the player sprite.

## 3. Successful Strategies
- confirmed: On level 2 (2 completed), this observed route succeeded: up → up → up → up → up → up.
  Applies: Current level is 2 with 2 levels completed.; Use only from the recorded visible start state; cached replay verifies the full frame.
  State: level=2, levels_completed=2
  Goal: Complete level 2.
  Preconditions: ["Current level is 2 with 2 levels completed.", "Use only from the recorded visible start state; cached replay verifies the full frame."]
  Policy: Use the saved exact-state route (up → up → up → up → up → up) only when the visible start state matches; stop when progress changes or WIN appears.
  Success: ["levels_completed increases, or game_state becomes WIN."]
- confirmed: On level 0 (0 completed), this observed route succeeded: up → up.
  Applies: Current level is 0 with 0 levels completed.; Use only from the recorded visible start state; cached replay verifies the full frame.
  State: level=0, levels_completed=0
  Goal: Complete level 0.
  Preconditions: ["Current level is 0 with 0 levels completed.", "Use only from the recorded visible start state; cached replay verifies the full frame."]
  Policy: Use the saved exact-state route (up → up) only when the visible start state matches; stop when progress changes or WIN appears.
  Success: ["levels_completed increases, or game_state becomes WIN."]
- confirmed: On level 3 (3 completed), this observed route succeeded: up → up → up → up → up → up.
  Applies: Current level is 3 with 3 levels completed.; Use only from the recorded visible start state; cached replay verifies the full frame.
  State: level=3, levels_completed=3
  Goal: Complete level 3.
  Preconditions: ["Current level is 3 with 3 levels completed.", "Use only from the recorded visible start state; cached replay verifies the full frame."]
  Policy: Use the saved exact-state route (up → up → up → up → up → up) only when the visible start state matches; stop when progress changes or WIN appears.
  Success: ["levels_completed increases, or game_state becomes WIN."]
- confirmed: Repeated up completed all five levels and reached WIN in the recorded 25-action ez01 run.
  Applies: ez01 levels beginning in one of the five recorded level-start states.; Cached replay is allowed only when the full current state matches a recorded route start.
  State: game_state=NOT_FINISHED
  Goal: Complete ez01.
  Preconditions: ["Start at a previously observed ez01 level state."]
  Policy: Repeat up while the observed state continues to progress; stop when levels_completed advances or WIN appears.
  Success: ["The environment reports game_state WIN."]
- confirmed: On level 4 (4 completed), this observed route succeeded: up → up → up → up → up → up → up.
  Applies: Current level is 4 with 4 levels completed.; Use only from the recorded visible start state; cached replay verifies the full frame.
  State: level=4, levels_completed=4
  Goal: Complete level 4.
  Preconditions: ["Current level is 4 with 4 levels completed.", "Use only from the recorded visible start state; cached replay verifies the full frame."]
  Policy: Use the saved exact-state route (up → up → up → up → up → up → up) only when the visible start state matches; stop when progress changes or WIN appears.
  Success: ["levels_completed increases, or game_state becomes WIN."]
- confirmed: On level 1 (1 completed), this observed route succeeded: up → up → up → up.
  Applies: Current level is 1 with 1 levels completed.; Use only from the recorded visible start state; cached replay verifies the full frame.
  State: level=1, levels_completed=1
  Goal: Complete level 1.
  Preconditions: ["Current level is 1 with 1 levels completed.", "Use only from the recorded visible start state; cached replay verifies the full frame."]
  Policy: Use the saved exact-state route (up → up → up → up) only when the visible start state matches; stop when progress changes or WIN appears.
  Success: ["levels_completed increases, or game_state becomes WIN."]

# V10 Implementation Plan — Small Review Addition

**25 October 2026 · Implements DESIGN.md v0.4**  
**Status:** Planned. Replace the previous broad implementation plan with these three changes.

## 1. Add review triggers to the existing loop

**Files:** src/v10/live.py and src/v10/cli.py.

- Add a configurable review interval, default 50 executed game actions.
- Count actions since the last review attempt.
- After each action, trigger review on a level-completion increase or the interval.
- Coalesce simultaneous triggers into one review.
- Pause/discard the remaining planned actions and request a fresh action decision after review.
- Reset the interval counter after an attempt. Do not reset or advance the environment during review.
- Preserve terminal handling: save a final-level review when possible, then end the run.

## 2. Make the review call and save memory

**Files:** reuse the existing model client, history/frame helpers, and game-memory code. Keep the review helper alongside the owning runtime unless a separate module is clearly useful.

- Build a review prompt from existing MEMORY.md, recorded actions and transformations so far, relevant frames, transition IDs, level/reset boundaries, and outcome feedback.
- Ask the same model for compact consolidated Markdown covering patterns, useful approaches, failures/exceptions, and uncertain ideas.
- Require evidence references and distinguish observations from guesses in the prompt.
- Save a non-empty result using the existing memory path; preserve previous memory if the call or save fails.
- Log trigger, covered transitions, result/error, and available usage/latency.
- Reuse existing persistence and logging utilities. Do not introduce another ledger or a new knowledge framework.

## 3. Supply updated memory and verify

**Files:** existing action-context builder and relevant v10 tests.

- Ensure the next action prompt reads the newly saved memory through the existing memory-loading path.
- Keep the directional action contract unchanged.
- Add focused tests for level completion, 50-action/default and configured intervals, simultaneous triggers, counter reset after failure, old-memory preservation, and updated-memory visibility.
- Verify review causes no game action/reset and terminal completion does not resume play.
- Run relevant existing v10 tests, then an ez01 run under the current model and action settings.
- Inspect the saved memory and next prompt; report levels solved and added review cost. Do not promise an improvement over the reported three-of-five baseline.

## Completion

The addition is done when review runs at the requested boundaries, useful findings are saved to MEMORY.md, the next decision receives them, and review failures leave the working agent able to continue.

Do not include unrelated fixes, control changes, mandatory predictions, memory promotion policies, cross-game transfer, new evaluation infrastructure, or architectural refactoring in this task.

# V10 Implementation Plan — Small Review Addition

**25 October 2026 · Implements DESIGN.md v0.4**  
**Status:** Implemented. Replaces the previous broad implementation plan with these three changes.

## 1. Add review triggers to the existing loop

**Files:** src/v10/live.py and src/v10/cli.py.

- Add a configurable review interval, default 50 executed game actions.
- Count actions since the last review attempt.
- After each action, trigger review on a level-completion increase or the interval.
- Coalesce simultaneous triggers into one review.
- Discard the remaining planned actions and request a fresh action decision after review.
- Reset the interval counter after an attempt. Do not reset or advance the environment during review.
- Preserve terminal handling: save a final-level review when possible, then end the run.

## 2. Make the review call and save memory

**Files:** reuse the existing model client, history/frame helpers, and game-memory code. Keep the review helper alongside the owning runtime unless a separate module is clearly useful.

- Build a review prompt from existing MEMORY.md, recorded actions and transformations so far, relevant frames, transition IDs, level/reset boundaries, and outcome feedback.
- Ask the same model for compact consolidated Markdown covering patterns, useful approaches, failures/exceptions, and uncertain ideas. Use a review-specific response format through the existing client; add only a small per-call format option if necessary. Keep action-call JSON handling unchanged.
- Distinguish observations from guesses in the prompt. Keep evidence references in ledger metadata; exclude them from memory prose and action context.
- Save a non-empty Markdown result using the existing memory path; preserve previous memory if the call or save fails. No new review schema or promotion rules are needed.
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

Limit helper changes to what is needed to connect review, persistence, and action context. Existing unrelated issues do not become prerequisites.

The addition is implemented: reviews run at the requested boundaries, validated findings are saved to MEMORY.md, the next decision receives them, and review failures leave the working agent able to continue.

## Verification record

- Focused suite: `PYTHONPATH=src /home/zodrak/cuda_python14_t/bin/python -m pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py` — 69 passed.
- 50-action run: `runs/v10/ez01_review_v04_50_20261025`. It completed 1/5 levels, triggered a review on level completion after action 2, then used the remaining action budget. Review usage was 4,362 tokens and 23.95 seconds. The model returned the prior legacy memory layout; the code initially accepted it. That malformed review was retired from the authoritative ledger, restoring the previous memory view.
- Follow-up format check: `runs/v10/ez01_review_format_check_20261025`. Two actions completed one level and triggered a review. The model returned all four required sections; status was saved, with 3,662 tokens and 15.96 seconds review latency.
- The 50-action result is below the prior 3/5 baseline and does not demonstrate an improvement. The short follow-up validates review formatting and persistence, not game-solving performance.

Do not include unrelated fixes, control changes, mandatory predictions, memory promotion policies, cross-game transfer, new evaluation infrastructure, or architectural refactoring in this task.

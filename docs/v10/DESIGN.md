# V10 — Add a Review Session to the Working ez01 Agent

**Design v0.4 · 25 October 2026**  
**Status:** Proposed addition. Supersedes the larger v0.3 proposal.  
**Plan:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## 1. Purpose

The user reports that the current agent uses up, down, left, and right and solves three of ez01's five levels. Preserve that working behavior.

Add a review session that identifies game-specific patterns from recorded experience, saves useful findings in MEMORY.md, and makes them available to subsequent action decisions and levels.

## 2. When review runs

Review after either:

- A level is solved.
- A configurable number of executed game actions since the last review; default: 50.

Check after each action. If both triggers occur together, run one review. Discard the remaining action sequence, review, then request a fresh decision using the updated memory.

Reset the interval counter after a review attempt, including a failed attempt, so a failure does not trigger another review on every following action. A later solved level can still trigger a new review.

Review does not advance or reset the game. A solved final level may be reviewed and saved, but playing does not resume after a terminal outcome.

## 3. Review input and task

Use the same configured model in a separate call with a review-specific prompt. Supply:

- Existing MEMORY.md, if present.
- The ordered recorded actions and measured visual transformations so far in the current run, with transition IDs and level/reset boundaries.
- Relevant recorded before/after frames to help interpret those transformations.
- Level-completion and other environment feedback.

Use existing logs and frame selection helpers. The reviewer should consider the complete recorded action history; it need not receive every screenshot.

Ask it to identify recurring action effects, conditions, exceptions, successful approaches, and repeated failures. It should distinguish supported observations from tentative explanations, cite transition IDs, and say when evidence is insufficient. Direction names describe commands; actual movement must be inferred from observations.

The review call returns Markdown using a review-specific response format; the action call keeps its existing JSON format. Reuse the existing client with a small per-call format option if needed.

The output is a compact, consolidated replacement for MEMORY.md. Preserve useful earlier findings, merge repetition, and revise or qualify claims contradicted by new evidence. Do not invent game rules or outcomes.

## 4. Memory and continued play

Use a short Markdown structure:

- **Patterns:** recurring effects, conditions, and evidence.
- **Useful approaches:** what helped solve a level and when it may apply.
- **Failures and exceptions:** what failed or limits a pattern.
- **Uncertain ideas:** possible explanations or tests still needed.

Keep findings specific to ez01. Carry useful mechanics into later levels without treating old coordinates or object positions as current facts. Cross-game reuse is outside this addition.

Save a non-empty Markdown review result through the existing game-memory path, preserving the previous memory on failure. No new structured review schema or promotion rules are required. Reuse the existing memory-loading path so the next action decision actually receives the saved content; avoid a second competing memory source.

If the model call fails or its result cannot be saved, log the failure, retain the previous memory, and continue the existing game loop. Record the review trigger and covered transition range for inspection.

## 5. Scope and validation

Keep the existing directional mapping, action-only response, game controls, action budgets, model settings, and normal execution behavior. The action-selection model is not asked to produce analysis or memory updates.

Implementation consists of a trigger, a review call, memory persistence, and supplying that memory to subsequent decisions. Small changes to existing helpers needed for this integration are allowed; unrelated baseline repairs are not prerequisites. No new planner, prediction framework, knowledge-status system, cross-game library, or unrelated baseline refactor is part of this change.

Verify both triggers, combined-trigger handling, failure recovery, and memory visibility in the next decision. Then run ez01 with review enabled and inspect whether later decisions use useful patterns. Compare with the current three-of-five result under the same game/model/action settings; report review overhead and actual outcomes without assuming improvement.

# V10 — Replay, Live Play, and Incremental Game Memory

**Design v0.6 · 25 October 2026**
**Status:** Implemented in `src/v10`.

## 1. Purpose and boundaries

V10 records reproducible game transitions, asks the configured model for bounded action plans, validates each action in Python, and reviews recorded experience to improve later decisions. Action selection remains action-only: the model returns legal action names and arguments. Review is a separate JSON call and cannot execute game actions.

Game and global `MEMORY.md` files contain model-authored Markdown and are authoritative for durable action guidance. Python owns validation, atomic persistence, event checkpoints, and exact successful-route replay. Existing append-only `memory.jsonl` files remain readable as historical records and retain run state and review checkpoints; structured ledger entries are not rendered into or duplicated in action prompts.

## 2. Live action loop

At startup, the runner checks verified trajectories for the current level and the current starting state separately. An exact starting-state match skips bootstrap and uses the normal route-reuse path. If a trajectory exists for the level but starts from another screen, bootstrap is also skipped; the model is asked to choose from the current screen and receives the saved route as guidance, marked as a different starting state. Bootstrap runs only when no verified trajectory exists for the level. It executes the standard legal directional sequence (`up`, `left`, `down`, `right`) and records its transitions as exploration, not as a successful route. It then performs one budgeted reset. A single model call receives the bootstrap observations and ordered screens, saved memory, and the current initial-screen image. That call is explicitly told that the game has reset and its proposed exploratory action sequence must start from the initial screen, not continue from the bootstrap's last screen. Invalid output falls back to the standard directional routine. This special instruction is used only for that startup call; all later action prompts follow the normal flow. The default reset budget is one to cover the startup reset; `--max-resets 0` disables reset-based startup.

The model-facing movement names are lowercase `up`, `down`, `left`, and `right`; Python maps them to environment action IDs. The action prompt reads the saved game and global Markdown directly, alongside the current observation and recent transitions scoped to the active level. `level` is a zero-based index; the prompt also states the corresponding one-based display number and levels completed. The action images contain only the active level's starting screen and its chronological action frames, including every intermediate environment frame. Screenshots from earlier levels are omitted; durable memory and verified-route guidance carry forward reusable findings. The final image is explicitly marked as the current screen from which the next action starts. These rules apply to normal decisions; the one-time startup exploration prompt also includes its bootstrap sequence and restored starting screen. Review uses its separate informative-frame selector. The action prompt does not receive raw JSONL or a duplicate structured-strategy payload. Current progress belongs in observation context, not durable memory. The action response contains no memory patches, predictions, or explanations.

The harness validates each action and stops a plan on a level change, terminal state, action error, budget exhaustion, or illegal next action. Review interrupts a remaining plan so a later action request sees the newly saved Markdown.

## 3. Review and Markdown persistence

Review runs after level completion, `WIN`, an unexpected legacy pattern prediction, the configured action interval (default 50 executed actions), or stagnation. Stagnation counts repeated observed states, not merely actions that have not yet completed a level. Repetition history is cleared after a review and at each level boundary; at least three fresh actions must occur before it can trigger again. Simultaneous triggers are coalesced. If a level completed entirely by exact replay of a verified route and game Markdown already exists, its redundant prose review is logged as skipped; `WIN` always receives final consolidation. A success-triggered review may summarize strategy guidance only from verified trajectories and their scope; exploratory reviews must not describe the current sequence as a successful strategy. Transition context distinguishes the level before an action from the level after it, so a boundary action is not attributed as a solution for the newly started level. Review does not advance or reset the game.

Each review receives the existing game and global Markdown, new ordered observations since the last successful review checkpoint, a compact summary of repeated states and ineffective actions, selected review images, and verified successful routes from `level_solutions.jsonl`. Routes are included with environment-recorded outcome and level scope. The reviewer may summarize equivalent routes into one strategy while preserving their level/state conditions. Exact route execution remains Python-owned.

The review response is JSON containing exactly two non-empty strings: `game_memory_markdown` and `global_memory_markdown`. Game memory should contain a few concise mechanics, demonstrated and useful approaches, failures, and unresolved experiments. Global memory should contain only reusable problem-solving methods. The reviewer is instructed to consolidate repetition, qualify uncertainty, resolve contradictions, preserve proven strategies and scope, and omit exact action sequences, action counts, pixel statistics, state fingerprints, evidence IDs, internal schemas, and run-specific progress from the Markdown.

Python checks both outputs against configurable character limits (defaults: 2,000 for game memory and 1,000 for global memory). Oversized, empty, malformed, or truncated output gets one repair request. It is rejected if still invalid; prose is never truncated to fit. Each valid file is atomically replaced. Invalid output or a failed save leaves that file's previous contents intact. A successful pair of saves is followed by an append-only review checkpoint, making repeat reviews of the same event idempotent. Failures are logged in `reviews.jsonl` and do not stop live play.

The existing Markdown files are supplied to later reviews for consolidation. Existing ledger rows and saved successful routes remain available as migration input; historical JSONL is retained and is not rewritten into a competing Markdown view.

## 4. Successful route storage and replay

The game directory contains append-only `level_solutions.jsonl`. It stores observed state/action/next-state edges and complete trajectories only when level completion or `WIN` was observed. Successful routes are persisted immediately at the completion transition, independently of review success. Before startup clears a run directory, complete winning v10 run logs for that game are discovered and imported after validating their manifest, metrics, evidence chain, frames, legal actions, progress boundaries, and terminal win. Import is idempotent and invents no transitions.

State fingerprints include the visible grid, level/progress fields, game/terminal state, and legal action names. At startup, a verified trajectory for the exact current level and starting fingerprint is replayed before any model plan. When several verified successes exist for that same start, the shortest successful trajectory is selected; a longer new completion is retained only when it improves the best route. The append-only ledger retains earlier revisions. If a route exists for the level but not the current start, bootstrap is skipped and the shortest same-level route is passed as guidance to the model without replay. Bootstrap runs only when no verified route exists for the level. Cached actions are converted to the lowercase direction contract before validation. Each action is checked against the expected source and resulting state; divergence or illegality stops replay and returns control to model selection. Routes count against the normal action budget and are never replaced by reviewer prose.

## 5. Persistent storage and prompt loading

Persistent files default to `skills/games/<game-id>/` and `skills/global/MEMORY.md`. Shared action instructions live in `skills/SKILL.md`. `V10_STATE_DIR` or `--state-dir` changes the persistent root; `--skill` overrides shared action instructions. `--reset-memory` on the live CLI removes the complete `games/` memory and solution subtree plus `global/MEMORY.md` before a run, while preserving `SKILL.md` and unrelated global files. `--game-memory-max-chars` and `--global-memory-max-chars` configure accepted review-output limits.

Before each action request, the harness reads the saved game `MEMORY.md` and shared global `MEMORY.md` as Markdown. Game guidance takes priority for game rules; global guidance is limited to reusable exploration methods. The exact saved Markdown is preserved across restarts and is not regenerated during startup, state recording, action execution, replay analysis, or prompt construction.

`memory.jsonl` remains append-only for historical knowledge revisions, run-specific progress, legacy review prose, and review checkpoints. It is not the authoritative action-memory document and is not sent to action prompts. `level_solutions.jsonl` is the authoritative store for exact route edges, starting-state compatibility, and verified success.

## 6. Run history and interruption

Recent transition context is bounded independently from plan horizon. Run-specific observations and progress stay associated with their originating run. On interruption, completed transitions and verified successes remain saved, metrics record `stop_reason=interrupted`, and shutdown does not launch a review or mark an unfinished route successful.

## 7. Validation

Focused regression suite:

```sh
PYTHONPATH=src python -m pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py tests/test_v10_solutions.py
```

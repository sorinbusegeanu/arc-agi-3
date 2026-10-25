# V10 — Replay, Live Play, and Incremental Game Memory

**Design v0.5 · 25 October 2026**
**Status:** Implemented in `src/v10`.

## 1. Purpose and boundaries

V10 records reproducible game transitions, asks the configured model for a bounded action plan, validates each action in the harness, and reviews recorded experience to improve later decisions. Action selection remains action-only: the model returns legal action names and arguments. Review is a separate JSON call and cannot execute game actions.

Game knowledge is persistent and game-scoped. Run state and observations remain associated with their originating run. Existing append-only `memory.jsonl` files remain readable; `MEMORY.md` is a deterministic generated view, not an independent source of truth.

## 2. Live action loop

The runner observes the game, supplies images, legal actions, recent outcomes, and compact applicable memory, then requests up to the configured plan horizon. The harness validates and executes each action. Level changes, terminal states, errors, budget exhaustion, and invalid next actions interrupt the remainder of a plan. Review also interrupts a remaining plan so the next model request sees the updated memory.

The model-facing movement names are `up`, `down`, `left`, and `right`; the harness maps them to environment action IDs. The action prompt receives memory as read-only context. The action response does not contain memory patches, predictions, or explanations.

## 3. Review lifecycle

Review runs after a level completion or the configured action interval (default 50 executed actions). Simultaneous triggers are combined. Review does not advance or reset the game. It receives the current persistent patterns, generated memory view, ordered observations and relevant frames from the run, plus prediction audits. It returns JSON changes, not a replacement Markdown document.

The reviewer identifies new patterns, new supporting or contradicting evidence, condition refinements, conditional variants, retirements, and successful strategies. Each call receives only run observations not covered by its latest checkpoint, alongside the existing patterns; coverage remains cumulative so repeated identical reviews are recognizable. Existing pattern IDs are reused for revisions. A new conditional variant has its own stable ID. Review changes only records it touches; other patterns are not rewritten. Evidence references must identify observations in the supplied new history. The harness qualifies them with the originating run ID and deduplicates them.

Review coverage checkpoints make an identical review idempotent. A checkpoint records the run and cumulative evidence IDs covered. Retrying a review with the same coverage does not append another pattern revision or duplicate evidence. Review failures retain the prior ledger and view, are logged in `reviews.jsonl`, and do not stop live play.

Legacy whole-document review entries are retained until the reviewer reports that all useful findings have been migrated into structured records. Old rows are never rewritten; migration and retirement append ledger revisions.

## 4. Pattern ledger

Each structured pattern is independently revised in the append-only JSONL ledger and contains:

- Stable `P-...` ID, `game_id`, type (`mechanic`, `strategy`, `exception`, or `hypothesis`), and concise claim.
- Applicability conditions and an optional state selector over `game_state`, `level`, `levels_completed`, `win_levels`, and `terminal`.
- Separate supporting and contradicting evidence references.
- Harness-computed confidence, status (`candidate`, `confirmed`, `contradicted`, or `retired`), and pattern revision number.
- Strategy-specific goal, preconditions, action policy, expected observations, and success conditions where applicable.

IDs are derived deterministically from the game, type, normalized claim, and initial applicability conditions. Revising a record preserves its ID and increments its pattern revision. Explicit retirements append a revision; no historical row is deleted. Conditional variants remain separate records so their evidence and applicability can be evaluated independently.

Confidence uses unique evidence references and a documented Laplace-smoothed support ratio:

`confidence = (support_count + 1) / (support_count + contradiction_count + 2)`

The result is rounded to four decimals. A pattern is `contradicted` when it has at least one contradiction and contradictions are at least as numerous as supports. Otherwise it is `confirmed` after at least two supports and confidence of at least 0.75. All other non-retired patterns are `candidate`. The model cannot set confidence or status.

## 5. Strategy and prediction handling

A strategy describes a goal, preconditions, a reusable action policy, expected observations, and success conditions. Strategies are game-scoped and can include conditional expectations keyed to observed state. A review may store a strategy only when its action sequence is supported by successful recorded experience.

When a confirmed mechanic or strategy supplies one unambiguous measurable expectation for the selected action and current state, the harness evaluates that expectation against the resulting transition. It appends a prediction audit with expected and actual observations and a matched, failed, or inconclusive verdict. A matched transition adds supporting evidence; a failed transition adds contradicting evidence. The harness deterministically updates confidence and status. An unexpected transition triggers an incremental review. Inconclusive measurements do not change evidence counts.

Prediction expectations use the existing measurable-prediction validator and evaluator. Conflicting confirmed expectations are not arbitrarily selected; the harness skips prediction when applicable patterns disagree.

## 6. Cross-run memory and retrieval

Persistent files are stored under `skills/games/<game-id>/` by default. The shared/global action instructions are `skills/SKILL.md`. `V10_STATE_DIR` or `--state-dir` can override the game-memory root; the shared skill remains in the project skills directory unless `--skill` overrides it. The append-only `memory.jsonl` ledger is authoritative. `MEMORY.md` is regenerated from active records after changes and is capped at 12,000 characters. Older unstructured entries are rendered compatibly until migrated or retired.

Each run stores progress under a run-qualified state record. A new run can retrieve prior game patterns and strategies but does not receive another run's current progress as its own. Evidence references also include run IDs so local transition identifiers do not collide.

Before each action request, retrieval filters patterns by game and state selector. Confirmed mechanics and strategies rank first, then applicable exceptions. Hypotheses are included only when no confirmed mechanic or strategy is available. Contradicted and retired patterns are excluded from action context. The existing context-entry limit remains in force; detailed provenance stays in the ledger rather than expanding the action prompt.

## 7. Generated `MEMORY.md`

The renderer deterministically produces these sections:

1. Current State
2. Confirmed Mechanics
3. Successful Strategies
4. Known Exceptions
5. Active Hypotheses
6. Failed Approaches
7. Next Actions

Pattern entries show their IDs, confidence, status, applicability, and concise supporting/contradicting references (for example, `run-id:T26`). Detailed observations and every historical revision remain in JSONL. Claims do not contain inline evidence prose such as “evidence: T26”.

## 8. Backward compatibility and validation

The reducer continues reading prior v10 ledger rows and preserves their revision history. New metadata is appended without changing existing run interfaces. Legacy review prose is available to the migration review and remains in the generated view until successfully migrated. The ez01 regression fixture distinguishes a gap-closing transition (`T2`/`T25`), intermediate level completion (`T3`, not terminal), and the final `WIN` transition (`T26`).

Focused regression suite:

```sh
PYTHONPATH=src python -m pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py
```

# V10 — Replay, Live Play, and Incremental Game Memory

**Design v0.5 · 25 October 2026**
**Status:** Implemented in `src/v10`.

## 1. Purpose and boundaries

V10 records reproducible game transitions, asks the configured model for a bounded action plan, validates each action in the harness, and reviews recorded experience to improve later decisions. Action selection remains action-only: the model returns legal action names and arguments. Review is a separate JSON call and cannot execute game actions.

Game knowledge is persistent and game-scoped. Run state and observations remain associated with their originating run. Existing append-only `memory.jsonl` files remain readable as internal review records; generated `MEMORY.md` is the model-facing game memory.

## 2. Live action loop

The runner observes the game, supplies images, legal actions, recent outcomes, and compact applicable memory, then requests up to the configured plan horizon. The harness validates and executes each action. Level changes, terminal states, errors, budget exhaustion, and invalid next actions interrupt the remainder of a plan. Review also interrupts a remaining plan so the next model request sees the updated memory.

The model-facing movement names are `up`, `down`, `left`, and `right`; the harness maps them to environment action IDs. The action prompt receives memory as read-only context. The action response does not contain memory patches, predictions, or explanations.

## 3. Review lifecycle

Review runs after level completion, a `WIN`, an unexpected prediction, the configured action interval (default 50 executed actions), or stagnation. Stagnation requires at least three actions without level progress plus repeated visible states or repeated state/action pairs; at least three fresh actions must occur before it can trigger again. Simultaneous triggers are combined into one review, and a failed review can retry after fresh experience. Review does not advance or reset the game. It receives the current persistent game patterns, game `MEMORY.md`, shared global `MEMORY.md`, ordered new observations, a compact summary of repeated states and ineffective actions, and varied frames covering movement, blocked actions, interactions, and progress.

The reviewer identifies repeated effects, successful approaches, repeated failures, contradictions, and a useful next experiment. It separates observations from explanations: unchanged pixels alone do not establish walls or total reachability, and proximity alone does not establish a completion rule. Exploration tactics may be recorded as tentative hypotheses before a level is solved and are kept distinct from demonstrated solutions. It also identifies meaningful, reusable problem-solving techniques for global memory. Memory patterns do not store evidence references, evidence counts, or numeric confidence. Global Markdown is deduplicated and bounded deterministically on save.

Each call receives only run observations not covered by its latest checkpoint, alongside the existing patterns; coverage is stored as an ordered event count so repeated identical reviews are recognizable without copying observation IDs into memory. Existing pattern IDs are reused for revisions. A new conditional variant has its own stable ID. Review changes only records it touches; other patterns are not rewritten. Observation IDs remain in run history and are omitted from reviewer event text and memory records.

Review coverage checkpoints make an identical review idempotent. A checkpoint records the run and number of ordered review events covered. Retrying a review with the same coverage does not append another pattern revision. Review failures retain the prior ledger and view, are logged in `reviews.jsonl`, and do not stop live play.

The harness gives the reviewer one repair attempt when the response is malformed or a proposed pattern fails schema validation. Valid changes from a mixed accepted/rejected response are persisted, but the coverage checkpoint is withheld until a later review accepts the rejected changes or otherwise completes the set. This prevents failed proposals from hiding observations from the next review. Repair calls are included in request, token, and latency totals.

Legacy whole-document review entries are retained until the reviewer reports that all useful findings have been migrated into structured records. Old rows are never rewritten; migration and retirement append ledger revisions.

## 4. Pattern ledger

Each structured pattern is independently revised in the append-only JSONL ledger and contains:

- Stable `P-...` ID, `game_id`, type (`mechanic`, `strategy`, `exception`, or `hypothesis`), and concise claim.
- Applicability conditions and an optional state selector over `game_state`, `level`, `levels_completed`, `win_levels`, and `terminal`.
- Qualitative status (`candidate`, `confirmed`, `contradicted`, or `retired`) and pattern revision number.
- Strategy-specific goal, preconditions, action policy, expected observations, and success conditions where applicable.

IDs are derived deterministically from the game, type, normalized claim, and initial applicability conditions. Revising a record preserves its ID and increments its pattern revision. IDs are internal to the ledger and review prompt so the reviewer can update a specific pattern; they are omitted from the game-facing Markdown. Explicit retirements append a revision; no historical row is deleted. Conditional variants remain separate records with their own applicability conditions.

The reviewer assigns qualitative status from the ordered observations. The harness validates the status value and preserves revisions; it does not calculate confidence from observation counts.

## 5. Strategy and prediction handling

A strategy describes a goal, preconditions, a reusable action policy, and success conditions. Expected measurable observations are optional. Strategies are game-scoped and can include conditions keyed to observed state. A review may store a strategy only when its action sequence is supported by successful recorded experience. Verified route completions and wins immediately append a state-scoped strategy to game memory, so it is available to the next session. Applicable successful strategies are called out before the rest of the action context.

When a confirmed mechanic or strategy supplies one unambiguous measurable expectation for the selected action and current state, the harness evaluates that expectation against the resulting transition. It appends a prediction audit with expected and actual observations and a matched, failed, or inconclusive verdict to run history. A failed prediction marks the affected pattern contradicted; a matched prediction leaves it unchanged. An unexpected transition triggers an incremental review. Prediction audits are not copied into game memory.

Prediction expectations use the existing measurable-prediction validator and evaluator. Conflicting confirmed expectations are not arbitrarily selected; the harness skips prediction when applicable patterns disagree.

## 6. Cross-run memory and retrieval

Persistent files are stored under `skills/games/<game-id>/` by default. Shared cross-game guidance is stored in `skills/global/MEMORY.md`; shared action instructions are `skills/SKILL.md`. `V10_STATE_DIR` or `--state-dir` overrides the persistent root, including both `games/` and `global/`; `--skill` overrides the shared action instructions. Game `MEMORY.md` is regenerated from internal review records after changes and is capped at 12,000 characters. The JSONL ledger supports review, validation, and rendering; it is not sent to action or replay prompts. Global memory is a single simple Markdown file, atomically replaced only when its contents change, and is capped at 3,000 characters both on write and in action prompts. Older unstructured game entries remain compatible until migrated or retired.

Each run stores progress under a run-qualified state record. A new run can retrieve prior game patterns and strategies but does not receive another run's current progress as its own. Evidence references also include run IDs so local transition identifiers do not collide.

The game directory also contains append-only `level_solutions.jsonl`. It stores observed state/action/next-state edges and complete action trajectories only when a level completion or `WIN` was observed. The recorded successful `runs/v10/ez01_20` history is imported after validating its contiguous evidence and frame chain, legal action order and arguments, progress boundaries, terminal win, and run metrics. Import is idempotent and writes no inferred transitions. State fingerprints include the visible grid, level/progress fields, game/terminal state, and legal action names. At each decision point, the runner searches only for routes starting at the exact current fingerprint and matching level key. Cached internal action IDs are translated to the established lowercase direction contract before validation. A verified path is replayed without an action-model call, up to the configured plan horizon, and every action still counts against the action budget. Each step is checked against the expected source and resulting state; if it diverges or is no longer legal, replay stops and normal model selection resumes from the observed state. These records do not enter model prompts.

Recent transition context is bounded independently from plan horizon. On interruption, completed transitions and verified successes remain saved, metrics record `stop_reason=interrupted`, and shutdown does not start a review or mark an incomplete route successful.

Before each action request, the harness includes the generated game `MEMORY.md` and shared global `MEMORY.md` as Markdown text. The game view presents active game knowledge; global memory guides exploration and does not override game-specific observations. The action prompt does not receive raw JSONL entries or the unrendered legacy review claim. Game Markdown is capped at 12,000 characters and global Markdown at 3,000 characters.

## 7. Generated `MEMORY.md`

The renderer deterministically produces these sections:

1. Current State
2. Confirmed Mechanics
3. Successful Strategies
4. Known Exceptions
5. Active Hypotheses
6. Failed Approaches
7. Next Actions

Pattern entries contain the claim, applicability conditions, qualitative status, and useful strategy details. They omit internal pattern IDs, confidence values, and observation references. Detailed observations and every historical revision remain in JSONL, with observation identifiers confined to run history.

## 8. Backward compatibility and validation

The reducer continues reading prior v10 ledger rows and preserves their revision history while omitting legacy evidence references and confidence fields from memory views. New metadata is appended without changing existing run interfaces. Legacy review prose is available to the migration review and remains in the generated view until successfully migrated. The ez01 regression fixture distinguishes gap-closing transitions, intermediate level completion, and the final `WIN` transition.

Focused regression suite:

```sh
PYTHONPATH=src python -m pytest -q tests/test_v10_replay.py tests/test_v10_analysis.py tests/test_v10_memory_integrity.py tests/test_v10_live.py tests/test_v10_solutions.py
```

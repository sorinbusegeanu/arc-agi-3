# Implementing V10 discovery tools

Companion: [tool_design.md](tool_design.md).

## Task

Implementation status: the four MVP tools are available by default in action-selection calls. Use `--no-discovery-tools` to disable them. This document records their contracts and validation requirements.

Preserve `v10.reasoned_action/1`, review's two-Markdown response, existing review inputs and triggers, route validation, and startup/recovery behavior. Ordinary prediction mismatches remain diagnostic.

## Code locations

| Responsibility | Owner |
|---|---|
| Action schema and aliases | `src/v10/analysis.py` |
| Action execution and predicate evaluation | `src/v10/tools/game.py` |
| Native game environment forks | `src/v10/arcade_env.py` |
| Discovery schemas, validation, branches, and dispatch | `src/v10/tools/discovery.py` |
| Frame storage | `src/v10/frames.py` |
| Image measurements | `src/v10/tools/vision.py` |
| History queries | `src/v10/tools/history.py` |
| Markdown memory persistence | `src/v10/tools/memory.py` |
| Live scheduling, accounting, and review | `src/v10/live.py` |
| CLI settings | `src/v10/cli.py` |

Keep discovery-specific schemas and dispatch in `src/v10/tools/discovery.py`. Reuse existing owners for measurement, history, persistence, action budgeting, and environment execution.

## 1. Inspection and history

Implement `inspect_region` using `FrameStore`, `pixel_diff`, and `candidate_displacement`. Require an explicit half-open raw-grid region and reject crops wider or taller than 20 pixels before returning arrays.

- Resolve evidence IDs to recorded frames.
- Validate raw-grid regions and return optional crops.
- For regional comparisons, calculate the change ratio using the region's area.
- Return current-only data for state records and before/after data for transitions.
- Preserve existing caller behavior when adding regional measurements.

Wrap `History.query` for `search_history`. Add pre-action level filtering where needed, using recorded state links. Apply filters before limiting results. Normalize direction labels and canonical action IDs. Enforce the current visible cutoff and the 20-result maximum.

## 2. Request dispatch

Define the four function schemas beside the v10 discovery dispatcher. Use native Strata function calls and the provider call ID as the idempotency request ID.

Accept the four discovery functions. Validate fields, types, finite values, evidence references, string/list limits, and call IDs. Return structured errors without executing invalid actions. Do not use a `finish` pseudo-tool; an ordinary action decision ends discovery for that selection.

Do not impose per-level quotas. Bound discovery planning to three tool-call rounds and four hypothesis cases per action-selection exchange, with a ten-completion hard ceiling as a final guard. Bound payloads through function schemas and Strata's output/context limits. Completed-request retries return the recorded result without another tool execution. Reuse the current model client, request logs, token limits, context handling, and usage accounting.

## 3. Isolated experiment execution

Normalize model-facing direction names and recorded replay actions to the canonical IDs exposed by the fork's current legal actions. Replay the ordered reset/action prefix from the initial state, compare the resulting observation fingerprint—including frame, level, and progress—to the requested base evidence, and execute the candidate only after an exact match.

Persist experiment records in run-local `discovery.jsonl`, keyed by run and request ID:

| State | Contents |
|---|---|
| started | Dispatch has begun; each isolated case has a branch ID and replay target |
| experiment_scored | Case results, branch evidence IDs, measurements, predicate comparisons, verdicts |
| rejected | Validation or known pre-dispatch failure |
| branch_error / replay_mismatch | A branch failed or its observable state did not match the target |

Store a canonical request digest. Same ID and payload returns the existing result; a different payload rejects. Persist `started` before branch actions. Never automatically redispatch an incomplete started experiment.

Fork the current environment and replay the run-local reset/action prefix for each candidate. Compare the observable fingerprint (frame, legal actions, level/progress/state) with the target and execute only after a match. Keep branch evidence under `discovery_branches/`; do not add branch transitions to live route learning or review history or charge them against the live action cap. Track branch resets and transitions separately.

Build one outcome object for all hypotheses and reuse `evaluate_prediction`. Restrict MVP predicates to the paths in the design. Across level changes, mark visual measurements unavailable while retaining environment outcomes.

An observable replay mismatch prevents candidate execution. Hidden environment state that is not exposed through observations cannot be certified by this fingerprint.

## 4. Live integration

Add the CLI flag, pass it to `run_live`, and record it in the manifest.

Make tools available during every model action-selection call, including startup/bootstrap. The model may return a native tool call or its ordinary action decision.

Include tool-use guidance in the shared skill and current evidence, legal actions, and remaining live-action allowance in the action prompt. The model can return an action without testing. After a probe, rebuild current context and invalidate pending responses tied to the previous state. Resume normal selection or existing terminal handling.

Read/model failures resume ordinary selection. With the feature disabled, preserve existing prompts, call counts, and behavior.

## 5. Memory integration

Implement `game_memory.read` using the current game/global Markdown readers.

Append validated proposals to `discovery.jsonl`. Supply pending proposals and experiment results to the next existing review. Keep its response and save path unchanged. Track consumed proposals through review checkpoints; unsuccessful reviews leave them pending.

Serialize experiment events explicitly in review context and preserve their evidence references. Store raw measurements in logs; let review produce the existing qualitative Markdown.

## 6. Validation

Add `tests/test_v10_discovery.py` and extend existing live/memory tests where appropriate.

Cover:

- Regional measurements, crops, missing frames, and invalid regions.
- History aliases, filter ordering, reset boundaries, level attribution, and cutoffs.
- Invalid requests, stale states, illegal actions, and exhausted budgets.
- Predictions saved before execution and all alternatives scored against one transition.
- Matched, failed, inconclusive, and cross-level outcomes.
- Duplicate requests, conflicting payloads, and uncertain execution.
- Probe accounting, route learning, review triggers, and terminal handling.
- Queued proposals, failed reviews, and deduplicated evidence.
- Feature-disabled behavior and unchanged normal prediction handling.

Run focused tests, then the repository's V10 regression suite. Runtime changes must follow the patch-version requirements in `AGENTS.md`.

## Completion

Deliver the four tools, CLI documentation, and passing relevant tests. Report any unavailable live validation.

For a live comparison, use isolated memory stores with matched model settings, seeds, starting memory, and action budgets. Retain requests, responses, decisions, discovery records, and metrics. Compare solving progress and action efficiency alongside added model cost. Keep separate existing evaluation plans unchanged.

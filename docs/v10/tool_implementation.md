# Implementing V10 discovery tools

Companion: [tool_design.md](tool_design.md).

## Task

Implement the four MVP tools behind `--discovery-tools`, off by default. Read current source and `AGENTS.md` before editing.

Preserve `v10.reasoned_action/1`, review's two-Markdown response, existing review inputs and triggers, route validation, and startup/recovery behavior. Ordinary prediction mismatches remain diagnostic.

## Code locations

| Responsibility | Owner |
|---|---|
| Action schema and aliases | `src/v10/analysis.py` |
| Action execution and predicate evaluation | `src/v10/tools/game.py` |
| Frame storage | `src/v10/frames.py` |
| Image measurements | `src/v10/tools/vision.py` |
| History queries | `src/v10/tools/history.py` |
| Markdown memory persistence | `src/v10/tools/memory.py` |
| Live scheduling, accounting, and review | `src/v10/live.py` |
| CLI settings | `src/v10/cli.py` |

Add one `src/v10/tools/discovery.py` module for request validation, dispatch, and experiment state. Reuse existing owners for measurement, history, persistence, and execution.

## 1. Inspection and history

Implement `inspect_region` using `FrameStore`, `pixel_diff`, and `candidate_displacement`.

- Resolve evidence IDs to recorded frames.
- Validate raw-grid regions and return optional crops.
- For regional comparisons, calculate the change ratio using the region's area.
- Return current-only data for state records and before/after data for transitions.
- Preserve existing caller behavior when adding regional measurements.

Wrap `History.query` for `search_history`. Add pre-action level filtering where needed, using recorded state links. Apply filters before limiting results. Normalize direction labels and canonical action IDs. Enforce the current visible cutoff and the 20-result maximum.

## 2. Request dispatch

Implement `validate_discovery_request` and `dispatch_discovery_request` using the design's JSON envelope.

Accept the four tools and `finish`. Validate fields, types, finite values, evidence references, string/list limits, and request IDs. Return structured errors without executing invalid actions.

Track per-phase limits:

- Two reads across inspection, history, and memory.
- One hypothesis test.
- One memory proposal.
- Five model completions, each returning one request.

Rejected requests consume their category slot. Completed-request retries return the recorded result without another tool execution. Reuse the current model client, request logs, token limits, context handling, and usage accounting.

## 3. Experiment execution

Persist experiment records in run-local `discovery.jsonl`, keyed by run and request ID:

| State | Contents |
|---|---|
| prepared | Validated request, predictions, base evidence, measurement scope |
| started | Dispatch is about to occur |
| completed | Transition ID, measurements, predicate comparisons, verdicts |
| rejected | Validation or known pre-dispatch failure |
| unknown_execution | Action may have executed without a reliable result |

Store a canonical request digest. Same ID and payload returns the existing result; a different payload rejects. Persist and flush `started` before dispatch. Never automatically redispatch an incomplete started experiment.

Have the dispatcher return a validated experiment action to the live loop. Execute it through the existing pathway, including action legality, ineffective-action checks, budgets, `learn_transition`, route learning, counters, level changes, reviews, and terminal handling. Extract a small shared helper if needed instead of copying the action loop.

Build one outcome object for all hypotheses and reuse `evaluate_prediction`. Restrict MVP predicates to the paths in the design. Across level changes, mark visual measurements unavailable while retaining environment outcomes.

If dispatch or result persistence leaves execution uncertain, stop automatic advancement and report `unknown_execution`. Normal initialization remains responsible for a new run; this feature adds no automatic crash recovery.

## 4. Live integration

Add the CLI flag, pass it to `run_live`, and record it in the manifest.

Offer one phase per level per run at the first normal model-selection boundary after startup. Resets do not grant additional phases. Skip discovery during verified replay/bootstrap.

Use a dedicated discovery prompt describing tools, available evidence, legal actions, and remaining limits. The model can finish without testing. After a probe, rebuild current context and invalidate pending responses tied to the previous state. Resume normal selection or existing terminal handling.

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

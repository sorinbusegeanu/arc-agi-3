# Implementing V10 discovery tools in Codex

Status: implementation instructions, not completed runtime work.
Companion: [tool_design.md](tool_design.md).
Baseline inspected: c374a062134c807783df2e4fd8a66c5327f3392d.

## Task

Implement the four tools described in tool_design.md as a small optional addition to V10. Read current source and AGENTS.md first; if the checkout has advanced, adapt the integration points without reverting newer behavior.

Preserve the current reasoned action schema, review's two-Markdown response, existing review evidence sequence and image-selection policy, exact-state replay safety, and existing startup/recovery behavior. No migration back to action-only responses. Ordinary prediction mismatches remain diagnostic and must not interrupt plans.

Do not implement future roadmap features, a new planner, native provider tool calling, cross-game mechanic inference, or hidden-state access.

## Existing owners to inspect

| Responsibility | Current owner |
|---|---|
| Normal action contract and aliases | src/v10/analysis.py: validate_reasoned_action, model_action_aliases, model_action_name |
| Action execution and predicate scoring | src/v10/tools/game.py: GameTools.act, validate_decision, validate_measurable_prediction, evaluate_prediction |
| Frame loading and provenance | src/v10/frames.py: FrameStore |
| Pixel/color measurement | src/v10/tools/vision.py |
| Recorded history | src/v10/tools/history.py: History.query, History.get |
| Authoritative Markdown memory | src/v10/tools/memory.py: read_global_memory, save_game_memory, save_global_memory |
| Live scheduling/accounting | src/v10/live.py: run_live, prepare_action_turn, request_action_plan, learn_transition |
| Review integration | src/v10/live.py: _perform_review, _review_events, perform_triggered_review |
| Settings | src/v10/cli.py |
| Shared model guidance | skills/SKILL.md |

Some legacy structured-memory/prediction helpers remain in the repository. Do not make them a second durable owner or assume old comments describe the current Markdown workflow.

A single src/v10/tools/discovery.py is justified for the new request dispatcher and experiment lifecycle. Keep measurements in vision.py, filtering in history.py, and environment execution in the existing live pathway. Do not add fixup modules or monkey patches.

## Step 1: exact inspection and bounded retrieval

Add selected-color measurements in vision.py. Return exact before/after counts and bounded coordinates for requested colors in a validated region. Reuse FrameStore and recorded transition/state links; never accept model-supplied filesystem paths.

Check regional measurement semantics: counts and change_ratio must use the requested region's area. Existing regional masking may retain full-frame dimensions; do not accidentally use full-frame area as the regional denominator. Preserve documented behavior for existing callers or add an explicit regional measurement helper rather than silently changing their interpretation.

Implement inspect_region as a wrapper around these measurements plus existing pixel differences, displacement and optional crops. Return current-only measurements for state evidence. Test original-grid versus enlarged-image coordinates.

Extend History.query only where needed for pre-action level filtering. Apply all filters before truncation; scanning only the last 20 records before filtering can miss valid matches. Respect the harness cutoff and preserve source metadata.

History may contain direction labels or canonical action IDs. Normalize both sides through existing alias helpers, then return consistent model-facing names. Do not infer pre-action level from a numeric decrement or conflate it with levels_completed.

## Step 2: dispatcher and validation

Implement validate_discovery_request and dispatch_discovery_request in the discovery owner. Use plain JSON, independent of Strata's native tool-call support.

Accept only the four named tools and finish. Define finish as:
~~~json
{"request_id":"D5","tool":"finish","arguments":{}}
~~~

Validate each tool's exact fields, bounded strings/lists, JSON types excluding booleans where integers are required, finite numbers, evidence scope and limits. Return structured errors. A bad request never executes an action.

Track limits per discovery phase: two read calls total across inspect_region/search_history/game_memory read; one test; one proposal; five model completions maximum. Each completion returns one request. A rejected request consumes its category slot. Retries of completed request IDs return cached results without consuming another action or tool slot, but still consume model-call budget.

Reuse current client.complete, logging, usage accounting and context-window handling. Do not change model, token settings or provider protocol.

## Step 3: prospective experiment lifecycle

Use a run-local discovery.jsonl ledger with records keyed by run_id and request_id:
- prepared: validated request, immutable hypotheses, base evidence and measurement scope.
- started: persisted immediately before dispatch.
- completed: actual transition ID, measurements and verdicts.
- rejected: validation or known pre-dispatch failure.
- unknown_execution: dispatch may have happened but no reliable joined result exists.

Store a canonical request digest. Same ID/same request returns the existing result; same ID/different request rejects. Load prior records before deciding whether dispatch is allowed. An incomplete started experiment must never be automatically dispatched again.

A run restart may create a new run ID; never treat the new ID as permission to replay an unresolved old action automatically. Existing restart/initialization behavior remains authoritative. Do not add automatic resume-from-ledger support in this MVP.

Use test_hypotheses only to validate/register the action and request execution from live.py. The dispatcher must not directly call the raw environment. Pass it through the same legal-action, known-ineffective-action, budget and accounting path as ordinary active play. It must not bypass checks just because the test contains one action.

Reuse the common post-action path for learn_transition, action counts, level progress, route learning, terminal handling, normal reviews and next observation. Extract a small shared helper if necessary after reading callers; avoid copying the large action loop. Explicitly identify the decision source as discovery in logs.

Persist prepared/started before acting and completed after joining the transition. Flush the started record before dispatch. If dispatch or post-dispatch persistence fails ambiguously, report unknown_execution and stop automatic advancement; never fall back by issuing the same or another action while the state is unresolved. Do not advertise distributed exactly-once semantics.

Extend the predicate validator with selected-color count paths only. Keep the path allowlist and existing operator/type checks. Displacement predicates are deferred. Mark visual predicates unavailable across level changes; still evaluate reliable environment outcomes. Build one actual-outcome object and use it for every hypothesis.

Do not auto-confirm prose, auto-update Markdown or terminate a normal plan because a hypothesis failed.

## Step 4: schedule the optional phase

Add --discovery-tools (default false), pass it to run_live, and record it in the manifest.

At the first normal model-selection boundary on a new level, after required startup and outside verified replay/bootstrap, run at most one discovery phase. Track the level encounter independently from reset_generation so repeatedly resetting the same level does not grant unlimited phases. Record the phase as offered even if the model finishes or a call fails.

Before the phase there must be no executable queued model plan. After a test, rebuild action context and invalidate any cached pending model response tied to the previous observation. Keep verified routes intact; do not broaden their applicability.

Use a dedicated discovery system prompt in the discovery module. Supply current legal actions, state/evidence handles, memory, measurements and remaining limits. Explain that requests return observations, predictions precede actions, no-ops are state-specific, and color labels are hypotheses. Ask for short claims and measurable alternatives, not chain-of-thought.

Allow the model to finish immediately. On a read/model/validation failure, preserve memory and resume normal selection within the remaining budgets. On successful test, return the observed state to ordinary selection. Normal review/terminal handling may end the phase before further requests. Do not add a discovery-specific review trigger.

With the feature disabled, do not add prompts, model calls, memory writes or behavioral changes.

## Step 5: integrate memory without replacing its owner

game_memory read returns current game or global Markdown. Propose appends a validated run-local proposal to discovery.jsonl and returns queued. No write/retire operations directly mutate durable memory.

Include queued proposals and experiment results in the next existing _perform_review request, alongside the existing complete chronological text history and current image selection. Keep the review's response exactly the existing two fields. Preserve current memory validation, save limits, failure behavior and global evidence requirements.

Use explicit experiment event serialization. Do not label every new audit record as an action error. Deduplicate proposals/evidence, and use the existing review checkpoint conventions so consumed inputs are not repeatedly presented as new. A failed review must not silently mark pending proposals consumed.

Preserve qualitative Markdown. Raw regions, exact counts, predicates and provenance belong in logs/review metadata. Do not amend VA01/global memory as part of implementation merely to match the example.

## Step 6: focused validation

Use existing tests/test_v10_live.py, tests/test_v10_analysis.py and tests/test_v10_memory_integrity.py. Add tests/test_v10_discovery.py for the new behavior.

Required checks:
- Exact regional color counts, bounded coordinates, optional crops, valid current-only inspection, invalid regions and unknown colors.
- Action aliases, filters before limits, pre-action level boundaries, reset scope and no future evidence.
- Duplicate/unknown fields, stale base state, illegal/reset actions, non-finite values and exhausted budgets cause no action.
- Prepared predictions precede dispatch; all alternatives share the actual transition.
- Same request retry dispatches once; changed request ID payload rejects; incomplete started records do not redispatch.
- False/missing/true predicates and cross-level visual unavailability produce correct verdicts.
- A toy coverage environment distinguishes final coverage away from a goal from later goal entry when coverage is independently observable. A second fixture renders visited/unvisited cells identically: color inspection must not claim coverage is known. Do not encode VA01-specific colors or mechanics into production.
- Probe execution follows current accounting, route learning, review triggers and terminal rules exactly once.
- Proposals do not write Markdown; next normal review sees them; failed review keeps them pending.
- Feature off preserves current contracts and call counts; ordinary prediction mismatch remains diagnostic; verified routes stay governed by strict state matching.

Run focused tests, then the repository-required V10 checks. Follow AGENTS.md: runtime implementation increments the patch version in pyproject.toml and V10_VERSION and synchronizes related references/tests. This documentation-only change does not bump runtime versions.

## Completion and evaluation

Deliver implementation with the feature off by default, passing relevant tests, updated usage documentation, and no duplicate owners. Report what was tested and any unavailable live environment.

For the initial behavioral smoke test, use paired VA01 runs with identical model settings, seed, starting memory and action budget, isolated writable memory stores, and retained raw model requests/responses plus decisions, discovery evidence and metrics. The current VA01 memory says trail and background are visually indistinguishable: verify that the model acknowledges this limit and does not claim exact coverage from color counts. This tests the motivating coverage case; it is not a broad benchmark claim.

Report levels completed, actions, ineffective repeats, resets, model calls, tokens and elapsed time. Check whether later evidence confirms discovered claims. Do not replace the separate UL01 evaluation prescribed by v10_reasoning_improvement_plan.md or claim policy improvement from divergent offline replay.

Suggested Codex instruction: “Implement docs/v10/tool_design.md following docs/v10/tool_implementation.md. Preserve current reasoned-action and Markdown-review contracts. Start with the four MVP tools only, off by default. Run the specified focused checks and report live-validation limits.”

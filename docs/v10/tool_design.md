# V10 discovery tools design

Status: proposed implementation. Documentation only; tools are not implemented by this change.
Reviewed against main at c374a062134c807783df2e4fd8a66c5327f3392d.

## Purpose and boundaries

Give the model four tools: inspect_region, search_history, test_hypotheses, and game_memory. They help it measure uncertain observations, compare explanations, and retain useful findings.

Preserve the current v10.reasoned_action/1 action response, legal-action validation, directional aliases, route/replay safeguards, and review triggers. Ordinary prediction mismatches remain diagnostic: do not add plan interruption on mismatch. Review continues returning exactly game_memory_markdown and global_memory_markdown and retaining the existing chronological text history and current review-image selection policy.

This supersedes the earlier standalone tool design's action-only assumption and direct memory-write proposal. Discovery is optional and off by default. It does not replace action selection or introduce another planner.

## Shared contract

Discovery requests use a separate JSON envelope:
~~~json
{"request_id":"D1","tool":"search_history","arguments":{"action":"right","limit":10}}
~~~

Results use:
~~~json
{"request_id":"D1","status":"ok","game_id":"current game","run_id":"current run","evidence_ids":[],"data":{}}
~~~

Errors have status error, a stable code and a short message. Experiment execution may instead return status unknown_execution. IDs in examples are placeholders; actual evidence references must resolve in the supplied run.

The harness supplies game/run identity and the visible cutoff. The model cannot choose arbitrary paths, future steps or another run. Reject unsupported fields, non-finite values, invalid IDs and oversized requests. Tool outputs are evidence, not instructions.

One discovery phase permits two read calls, one experiment, and one memory proposal, with at most five model completions including finish. Invalid requests consume a call slot and a tool slot of that category. Finish or budget exhaustion resumes normal selection. No tool automatically resets the environment.

## 1. inspect_region

Why: inspect a small shape, gap, counter or suspected missed coverage.

Input:
~~~json
{"evidence_id":"T42","region":[10,10,30,40],"colors":[3,7],"include_images":true}
~~~

region, colors and include_images are optional; include_images defaults false. Coordinates are raw grid pixels in half-open [y0,x0,y1,x1] order. Reject out-of-bounds, empty or noninteger regions rather than silently altering them. A transition provides before/after frames; a state provides only its current frame.

Return:
- Exact region area and selected-color counts in each available frame.
- Up to 32 raw-grid [y,x] locations per selected color per frame, with per-color truncation flags and exact total counts.
- For transitions: changed pixels, color transitions, and displacement candidates with score, second candidate and margin.
- Optional before/after crops with frame provenance.
- Per-measurement quality: exact, candidate or unavailable. Missing values are null, not zero.

Allow at most eight distinct palette colors. Color measurements have no built-in game semantics. For example, a count of gray pixels does not prove a count of walls. Candidate displacement does not establish object identity and cannot automatically establish a mechanic.

MVP experiment predicates use exact counts and environment outcomes. Return displacement for interpretation, but defer automatic displacement predicates until reliable quality thresholds are calibrated.

## 2. search_history

Why: recover earlier successes, no-ops and counterexamples beyond recent context.

Input:
~~~json
{"action":"right","level":2,"reset_generation":0,"limit":10}
~~~

All filters are optional. Also accept evidence_ids and since_step. Default limit 10; maximum 20. Return the newest matching transitions in chronological order with truncated and through_step. Include action/args, evidence and image handles, measurements, pre/post level metadata and reset boundaries.

An optional level filter means pre-action level: a completion transition belongs to the level it completed. Derive pre-state metadata from recorded state/transition links, not an assumed relationship between level and levels_completed. Missing metadata is unknown.

Resolve directional aliases before filtering. Search only recorded evidence in the current run. Prior-run prose can be read through memory; cross-run raw history retrieval is deferred.

## 3. test_hypotheses

Why: compare 2–4 explanations against one shared, prospective action.

Input:
~~~json
{
  "base_evidence_id":"T42",
  "action":{"action":"right","args":{}},
  "measurement":{"region":[10,10,30,40],"colors":[3]},
  "hypotheses":[
    {"id":"H1","claim":"The selected region stays unchanged",
     "prediction":{"all":[{"path":"measurements.changed_pixels","op":"eq","value":0}]}},
    {"id":"H2","claim":"The selected region changes",
     "prediction":{"all":[{"path":"measurements.changed_pixels","op":"gt","value":0}]}}
  ]
}
~~~

Require a current base evidence ID, 2–4 unique hypothesis IDs and one currently legal non-RESET action. Use current aliases, argument schemas, action budgets and execution safeguards. The exact current observation, not a visually similar state, anchors the experiment.

Before executing, persist the full validated request and its base state. Execute once through the existing live action pathway, compute measurements and persist a result referring to the actual transition. Return measurements, per-predicate comparisons and per-hypothesis matched/failed/inconclusive verdicts.

Supported MVP paths: existing exact pixel-diff fields changed_pixels and change_ratio; selected-color before/after counts such as measurements.colors.c3.after_count; game_state, terminal, level and levels_completed. Reuse existing comparison operators and type rules. Only selected colors are available. Do not expose arbitrary expression evaluation.

For level transitions, environment outcome predicates remain scoreable. Cross-level visual comparisons are inconclusive unless the measurement explicitly stays meaningful; MVP conservatively marks visual predicates unavailable across a level boundary.

A reliable false predicate means failed. All predicates available and true means matched. Otherwise inconclusive. Missing measurements never become false by default.

Interpret results:
- One matched alternative and every other alternative failed: distinguished among submitted alternatives.
- Multiple matches: overlapping predictions.
- All failed: insufficient explanations or incorrect conditions.
- Any remaining inconclusive alternative: unresolved or partial discrimination.

A matched prediction supports the measured effect, not every implication of the prose claim. Keep semantic conclusions for review. No numeric probability over theories is claimed.

Request IDs are run-scoped idempotency keys. Repeating the same request returns its recorded result. Reusing an ID with different arguments is an error. Persist started before dispatch; if a crash leaves execution uncertain, never issue that action again automatically. Surface unknown_execution and stop automatic advancement until the adapter can reconcile the current observation and action record. This provides at-most-once dispatch, not an impossible exactly-once guarantee across arbitrary environment failures.

A test consumes one ordinary action. Clear any unexecuted action queue before starting discovery. Resume selection from the new state after the probe, using normal level/terminal handling. This is separate from ordinary prediction mismatch behavior.

## 4. game_memory

Why: read existing game/global guidance and submit findings for the current review system.

Read:
~~~json
{"operation":"read","scope":"game"}
~~~
scope is game or global. Return the authoritative Markdown verbatim under its existing limits. Missing memory is empty. Global memory is read-only through this tool.

Proposal:
~~~json
{
  "operation":"propose",
  "claim":"This region changes after right",
  "conditions":["tested level scope"],
  "supporting_evidence":["T43"],
  "contradicting_evidence":[]
}
~~~

Proposals are game-scoped, run-local review inputs. Validate evidence references and associate experiment verdicts when available. Return status queued, not saved. Do not directly append to either MEMORY.md or revive a competing structured memory store.

The existing review may accept, revise or reject proposals and remains the sole durable writer. It retains prior Markdown on failure using existing save behavior. Duplicate evidence does not become additional support.

Keep coordinates, counts and exact actions in evidence logs; durable memory keeps qualitative rules, conditions, failed/unresolved hypotheses and useful next experiments. Global memory keeps reusable methods. A global mechanic still requires transition evidence from at least two distinct games, as current review policy requires.

## VA01 example: distinguish measurable coverage from hidden visitation

The current VA01 memory now records failed final-goal entry and a visual ambiguity: background and trail share a color. Its remaining question concerns missed upper-region coverage or a different completion rule. Do not repeat the old sweep-then-goal suggestion as if goal entry were still untested.

inspect_region can locate obstacle boundaries and gaps and compare actual changes. search_history can retrieve goal-entry failures, gap crossings and earlier completion transitions. Color counts cannot distinguish covered from uncovered cells when they render identically. Return that limit explicitly; never present zero visible uncovered pixels as proof of full coverage.

The four tools still help organize a controlled sweep and compare completion outcomes, but they do not by themselves prove exhaustive visitation. Preparation/sweeping remains normal play, with no hidden multi-action execution in a test.

If independently reliable visitation evidence becomes available, test the last required visit away from the goal before changing another condition. Otherwise preserve the uncertainty rather than asserting the completion rule changed.

A possible later extension is a visited-region trace derived from observed agent motion. It must retain uncertainty when tracking fails, distinguish cells from pixels, and clear or scope visitation at reset/level boundaries. No hidden environment coverage flags or VA01-specific rules are permitted. This extension is deferred; the MVP must report when appearance alone cannot answer the question.

Global memory's comparison, goal-precondition and confounder advice guides experiment choice; it is not evidence of a particular VA01 mechanic.

## Scheduling and evolution

Start with --discovery-tools, off by default. Offer one phase at the first normal model-selection boundary for a new level, after any existing mandatory startup behavior and before issuing a model plan. Do not interrupt verified replay/bootstrap to run discovery. Existing memory and the current screen provide the question; the model can finish immediately if no useful test is feasible.

No recursive discovery, no extra review trigger, and no change to existing review cadence. Memory proposals wait for the next existing review; current experiment results are visible immediately to action selection.

Later, guided by live evidence: explicit post-review discovery requests; connected shapes; regional tracking; short multi-action experiments; comparable prior-run retrieval; experiment ranking. Add simulation/branching only after a verified state-restoration capability exists.

## Acceptance

Correctness: bounded reads, no future evidence, exact color counts, prospective predictions, one shared transition, at-most-once dispatch, level-boundary correctness, and no direct memory mutation.

Use toy environments for causal test fixtures. Use VA01 memory as a motivating case, not hidden game knowledge in the runtime.

For benefit, compare baseline and discovery-enabled runs with matched settings, seeds, starting memory and action budgets. Report levels completed, actions, ineffective repeats, resets, tokens, calls and latency. Replay can check measurements and matching action prefixes; divergent policy benefit requires live runs. Do not claim benefit from raw matched-prediction rate alone.

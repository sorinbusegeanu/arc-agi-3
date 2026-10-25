# V10 — ez01 Action Selection and Evidence Review

**Design specification v0.3 · 25 October 2026**  
**Status:** Proposed target architecture. This document supersedes v0.2; it does not claim the changes below are implemented.  
**Implementation plan:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## 1. Objective and scope

Test whether a simple Qwen/Strata agent can improve its performance on ez01 by periodically reviewing its own recorded actions and visual transformations, extracting game-specific patterns, and reusing compact memory on subsequent levels.

Keep immediate action selection small. Move pattern discovery and memory consolidation into a separate review session using the same model with a different prompt. These are sequential roles, not concurrent agents.

The first release targets ez01 directional controls and transfer between levels of that game. Do not preprogram ez01 solutions, object meanings, routes, or win conditions. Cross-game transfer, click interfaces, learned world models, external planners, and automatic skill rewriting are later work.

Success means better live progress or completion under matched action and inference budgets, fewer repeated failures, traceable memory, and bounded cost. Convincing retrospective explanations alone do not establish learning.

## 2. Inspected baseline

The inspection covered d37a478 and the directional-name changes in b225637. The latter exposes up/down/left/right to the model and translates them to canonical ARC action IDs. The existing runtime already has a compact action-only response, sequential action execution, terminal/level/legal-action/budget checks, durable transition logs, a persistent memory ledger, and compact memory reads.

The active action loop does not currently create new knowledge or audit prospective predictions. The memory policy helpers and prediction utilities are not evidence that those behaviors run. The new review stage below is proposed work.

Known baseline concerns to verify during implementation include the missing read_jsonl import in prior-decision analysis, incomplete action argument validation, lost native intermediate frames, and missing episode-wide inference limits. Click-specific schemas are deferred until the ez01 experiment works.

## 3. Architecture and ownership

```text
environment -> adapter -> durable transitions and frames
                    |
                    v
         action context + compact MEMORY
                    |
                    v
          Qwen action-selection session
                    |
              validated actions
                    |
         execute one action and observe
                    |
          review trigger at a boundary
                    v
     frozen evidence -> Qwen review session
                    |
          proposed knowledge changes
                    v
       harness validation -> memory ledger
                    |
             regenerate MEMORY.md
```

| Owner | Responsibility |
|---|---|
| Environment adapter | Normalize observations, preserve original/intermediate frames, translate controls |
| Action runner | Build context, validate actions, enforce budgets, execute and interrupt plans |
| Review orchestrator | Freeze an evidence boundary, schedule bounded review batches, log coverage and results |
| Reviewer | Identify patterns, alternatives, exceptions, scope, and useful next tests |
| Memory manager | Validate provenance, assign IDs, deduplicate, preserve contradictions, gate status changes, render memory |
| Evaluation | Compare live outcomes and costs with isolated memory states |

Use normal imports and explicit composition. Extend the existing authoritative modules. A review module is justified as a new responsibility; patch installers, replacement runners, and version-suffixed production modules are not.

## 4. Action interface

The model-facing movement vocabulary is up, down, left, right. Preserve the inspected canonical mapping:

| Model name | Canonical ID |
|---|---|
| up | ACTION1 |
| down | ACTION2 |
| left | ACTION3 |
| right | ACTION4 |

This names a command; it does not assert that an object actually moves in that direction. Movement outcomes remain empirical. Keep one authoritative mapping and log its version. Preserve canonical IDs for execution and record the model-facing alias alongside them. Future adapters may supply different control vocabularies.

Action selection returns only:

```json
{"actions":[{"action":"up","args":{}}]}
```

Reject additional response fields, unknown actions, invalid arguments, and plans beyond the configured horizon. Reset is a separate budgeted pseudo-action returned alone with a non-empty justification. Empty actions ends the run and is logged explicitly.

Default horizon is one while establishing ez01 behavior; horizons two to four remain configurable comparison settings. Revalidate against the newest observation before every action. Interrupt on terminal state, level change, illegal action, action error, exhausted budget, or a pending review trigger. An interrupted sequence is discarded, not resumed automatically.

No predictions, memory patches, or written reasoning are required from the action response. After invalid output, allow one repair. Default to stopping after a second failure; retain random fallback only as an explicit experimental option with a logged seed and counter.

## 5. Evidence and action context

Each transition records run ID, game ID, level before/after, reset generation, action index, alias and canonical ID, validated arguments, pre/post frame references, intermediate frame references, exact visual measurements, environment feedback, and action success/error. Persist action intent before dispatch, then append outcome or error; never rewrite an earlier record.

Evidence is addressed by run-qualified IDs. Preserve original frames and palette fidelity. Durable evidence and memory must survive disposable run cleanup; use unique run directories or protect retained evidence from deletion. A memory reference must remain resolvable after the next run starts.

Action context contains current and previous actual images, recent transitions from the current reset generation, legal directional controls, remaining budgets, and up to five relevant memory entries. Include coordinate conventions and image dimensions. Clear transient level state at boundaries; keep prior-level knowledge with its original scope.

Initial configurable limits: two images, five memory entries, 8,000 characters of serialized memory, and 24,000 characters for the complete user prompt. Enforce limits deterministically and record omitted IDs. Character limits supplement rather than claim to equal tokenizer limits. Log actual usage where available.

## 6. Review scheduling and coverage

Run a review at the next action boundary when any condition holds:

- A level completes: review the completed level before requesting another action.
- Twelve actions have executed since the last completed review.
- Eight consecutive actions leave the final pixel grid unchanged: label this a no-visible-change signal, not proof of no progress.
- Before a deliberate reset, or at terminal/normal budget exit, when unreviewed transitions exist and review budget remains.
- The user explicitly requests review.

Thresholds are initial experimental settings, not ez01 mechanics. Coalesce simultaneous triggers. Do not reset or advance the environment during review. Limit one automatic review per evidence boundary. If a review cannot run, log the skipped reason and continue or stop according to the normal run state.

The review must account for all transitions so far without resending every screenshot on every call:

1. Maintain a review cursor and append-only coverage records.
2. Supply new transition summaries plus existing knowledge, exceptions, and prior coverage.
3. Batch at most 32 new transitions and eight selected actual images per review call. Select before/after pairs for representative changes, repeated effects, contradictions, and level outcomes.
4. If the interval exceeds the batch size, process sequential batches. Store provisional results per batch; mark the review complete only when every transition through the frozen boundary is covered.
5. At level completion reconcile the level's compact transition index and active knowledge, using bounded additional batches when needed. Logs distinguish summaries inspected from images inspected.

A cursor advances only after the corresponding accepted or explicitly rejected review result is persisted. Previously reviewed evidence can be revisited for contradictions. Never claim the model visually inspected every frame when it only received summaries.

## 7. Review contract and memory admission

The review response is a separate versioned JSON contract:

```json
{
  "review_boundary":"run-id:T24",
  "patches":[{
    "op":"upsert",
    "id":null,
    "kind":"pattern",
    "claim":"A concise conditional observation.",
    "scope":{"game_id":"ez01","levels":[1],"reset_generations":[0]},
    "conditions":["Observable conditions under which the claim applies."],
    "support_ids":["run-id:T12","run-id:T18"],
    "contradiction_ids":[],
    "alternative":"Another explanation consistent with these observations.",
    "next_test":"A future observation that could distinguish the explanations.",
    "use":"How this should influence later action selection."
  }]
}
```

This is a logical example, not a claim about ez01 mechanics. Define an executable schema during implementation. Initial kinds are pattern, mechanic_candidate, strategy, goal_candidate, and failure. The harness assigns new stable IDs when id is null. Updates must refer to existing IDs. Retire operations require an existing ID, reason, and evidence citations. Bound patches to eight per call and each text field to 500 characters.

Reject invented, future, wrong-game, or out-of-boundary citations and invalid operations. Qualify evidence by run, level, and reset generation. Keep raw reviewer output and rejection reasons in the review log. One invalid response receives one repair; repeated failure makes no memory change.

The harness owns status and counters:

- Candidate: a well-formed, evidence-linked proposal; a citation does not prove semantic correctness.
- Repeated: at least two distinct supporting transitions, no unresolved cited contradiction. This is descriptive support, not causal proof.
- Supported: a candidate made a recorded prediction before a later test, and the measured outcome supports that prediction.
- Contradicted: preserve the counterexample and suppress unconditional use pending review.
- Retired: retain history and provenance but exclude from normal action context.

Deduplicate repeated citations; replaying the same transition does not add support. Reject claim replacement that drops existing contradictions. A substantive claim revision creates a new version and resets its support status. Ranking must not hide applicable contradictions behind positive entries.

For supported status, the reviewer can nominate at most one next test using the existing allowlisted predicate representation. The harness records the hypothesis ID, applicable conditions, proposed action, predicate, and evidence boundary before execution. Action selection may choose a different action; score only an actually applicable executed test. Evaluate exact predicates deterministically; ambiguous identity or semantic outcomes remain inconclusive. Retrospective matching never counts as prospective confirmation. A broad pixel predicate is evidence only for its narrow measured claim.

This prospective test is optional review metadata, never a new mandatory field in the action response.

## 8. MEMORY.md and persistence

Keep the append-only machine ledger authoritative. Generate MEMORY.md atomically after accepted changes with:

1. Applicable supported/repeated mechanics and their limits.
2. Candidate patterns and competing explanations.
3. Current strategy or goal candidates.
4. Known failures, contradictions, and recovery clues.
5. Pending tests and concise evidence references.

Render counts and statuses from the ledger, not model-supplied confidence. Cap active entries at 50 and rendered MEMORY.md at 12,000 characters. Retire or archive superseded entries while retaining the complete ledger; reject additions if no valid compaction is available. Action context is a smaller relevance-ranked subset under section 5 limits.

Store memory per game outside disposable outputs. Include schema, review prompt, model, mapping and renderer versions; memory revision; review boundary; source run references; support and contradiction IDs. Review logging and memory updates must be idempotent for the same review ID. Freeze the input memory revision for each review; serialize writers.

## 9. Transfer between levels and games

Within ez01, carry game-scoped knowledge forward, but mark a new level as untested for that entry. Do not carry old coordinates, routes, or transient object positions forward as current facts. Recheck applicability against the new image. Successful prospective tests can extend scope; failures add exceptions.

Do not write a cross-game memory library in this release. A future stage may propose abstract templates such as blocked movement or delayed effects. Such templates must strip game-specific action mappings, colors, coordinates, and object identities, and enter another game as candidates requiring new evidence.

## 10. Budgets and observability

Expose separate action and review settings. Initial review limits: 32 transitions, eight images, eight patches, 48,000 prompt characters and 4,096 output tokens per call; at most two HTTP attempts per completion and one corrective completion. Count all attempts.

Require configurable episode action/reset limits, a total HTTP-attempt limit, cumulative reported-token limit, per-request timeout, and a wall-clock deadline. Count reviews, batch continuations, repairs, and fallbacks against the same episode totals. Reserve review allowance for a level boundary if configured. Missing usage is unknown, not zero; request/time limits still apply.

Log action/review tokens and latency separately, review triggers/coverage, omitted context, schema failures, memory changes, duplicate support, contradiction counts, fallback actions, actions to first level completion, levels solved, terminal outcome, and artifact paths. Log image/reasoning token details when exposed; never invent them.

## 11. Evaluation and release gate

Use matched ez01 seeds, initial states, model settings, action limits and total inference ceilings. Every arm starts with its own empty memory namespace or the same frozen starting snapshot.

Compare: A, no review/no memory; B, review with proposals logged but withheld from action selection; C, review with admitted memory supplied to action selection. B and C share review scheduling. Report actual spend as well as common ceilings; equal limits do not imply equal work.

Measure completion, levels solved, actions per level, repeated state/action failures, review overhead, memory size, contradictions, and prospective-test outcomes. Report level-to-level transfer separately from repeated exposure to the same level. Offline action agreement is a debugging measure only.

Initial smoke gate: complete deterministic fixture tests and at least three paired ez01 seeds with all artifacts retained. This is feasibility evidence, not a statistically established improvement. Expand evaluation before claiming generalization. If memory gives no benefit or causes regressions, inspect cited entries and tune review/ranking before adding architecture.

## 12. Delivery boundaries

Implement the plan in IMPLEMENTATION_PLAN.md in stages. Preserve the action-only baseline as a selectable evaluation condition. Adapt the current GameMemory rather than creating another authoritative store. Keep one control mapping and one context builder per responsibility. No implementation phase is complete until its tests and live evidence meet the stated exit criteria.

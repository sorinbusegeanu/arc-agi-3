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

<<<<<<< HEAD
Use normal imports and explicit composition. Extend the existing authoritative modules. A review module is justified as a new responsibility; patch installers, replacement runners, and version-suffixed production modules are not.
=======
| Component | Owns | Must not do |
|---|---|---|
| Qwen via Strata | Choose a legal action sequence from the supplied state and history | Claim execution, predict outcomes, update memory, or fabricate action names |
| Agent harness | Model call orchestration, legal-action validation, step budgets, interrupts, retries, state continuity | Infer game rules silently |
| Python vision tools | Pixel masks, regions, temporal comparisons, candidate tracking with uncertainty | Declare an object a player, goal, or HUD without evidence |
| Evidence store | Immutable state, action transitions, original frames, measurements, and run decisions | Rewrite past transitions |
| Memory manager | Stable IDs, provenance, and compact read-only context generation | Depend on memory edits from the action-selection response |
| SKILL.md | Shared action-only response contract | Store per-game colors, coordinates, or action mappings |
| MEMORY.md | Persistent per-game context and prior observations | Duplicate all frames or unrestricted free-form reasoning |
>>>>>>> 5b9552a (update)

## 4. Action interface

The model-facing movement vocabulary is up, down, left, right. Preserve the inspected canonical mapping:

| Model name | Canonical ID |
|---|---|
| up | ACTION1 |
| down | ACTION2 |
| left | ACTION3 |
| right | ACTION4 |

<<<<<<< HEAD
This names a command; it does not assert that an object actually moves in that direction. Movement outcomes remain empirical. Keep one authoritative mapping and log its version. Preserve canonical IDs for execution and record the model-facing alias alongside them. Future adapters may supply different control vocabularies.
=======
1. Harness assembles the current and previous game frames, recent action outcomes, legal actions, budgets, and up to five read-only memory entries.
2. Qwen chooses up to the configured **1–4 action plan horizon** and returns only action names and arguments. The four movement actions are presented as `up`, `down`, `left`, and `right`.
3. Harness validates the action list and arguments. Invalid outputs are rejected or repaired with bounded retry; no illegal action is executed.
4. Execute actions one at a time; collect original frames and metadata, compute measurements, and record each transition.
5. On a level change, terminal state, illegal next action, action error, or exhausted budget, stop the remaining plan and consult Qwen again when appropriate.
6. Repeat with the latest state and outcomes. The harness maintains evidence and persistent memory; the model does not write memory or predict outcomes.
>>>>>>> 5b9552a (update)

Action selection returns only:

```json
{"actions":[{"action":"up","args":{}}]}
```

<<<<<<< HEAD
Reject additional response fields, unknown actions, invalid arguments, and plans beyond the configured horizon. Reset is a separate budgeted pseudo-action returned alone with a non-empty justification. Empty actions ends the run and is logged explicitly.
=======
| Input | Format | Notes |
|---|---|---|
| General instruction | `SKILL.md` or distilled prompt | Stable and cacheable when supported |
| Game frames | Current and previous actual frame, at most two | Do not send initial-frame or delta-image stacks |
| Current state | Game status, level, legal action names and argument information | Movement IDs are presented as `up`, `down`, `left`, and `right`; other actions retain their names. The harness maps movement names to environment IDs. |
| Recent outcomes | Ordered action, arguments, and measured result | Bounded by the configured plan horizon |
| Game memory | Up to five compact entries, read-only | Existing memory is context; the model does not edit it |
| Budgets | Live remaining action/reset counts and plan horizon | Offline replay has no live remaining-budget estimate |
>>>>>>> 5b9552a (update)

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
<<<<<<< HEAD
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
=======
  "actions": [
    {"action":"left","args":{}},
    {"action":"up","args":{}}
  ]
}
```

The `v10.action_set/2` response must contain only the top-level `actions` array. Each action object must contain only a legal action name and its `args`; extra fields are rejected. The four standard movement IDs are presented to the model as `up`, `down`, `left`, and `right`, and the model must return those same lowercase names. The harness converts them back to the environment's `ACTION1`–`ACTION4` IDs before validation and execution. Other legal actions retain their listed names. The array may contain zero to the configured horizon, with a maximum horizon of four. An empty list ends a live run. If reset budget is available, `RESET` is listed as a pseudo-action and must be the only returned action, with its justification in `args`. The harness validates every action against the latest legal-action schema, records transitions and measurements, and interrupts the remaining plan on a level change, terminal state, illegal next action, action error, or exhausted budget. Offline analysis compares proposed actions with the recorded continuation as action matches only; this is not a measure of policy quality.

The response schema is enforced by the runtime validator. Environment action IDs and argument schemas come from the current observation; the model-facing prompt uses direction names for the four movement actions.
>>>>>>> 5b9552a (update)

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

<<<<<<< HEAD
Implement the plan in IMPLEMENTATION_PLAN.md in stages. Preserve the action-only baseline as a selectable evaluation condition. Adapt the current GameMemory rather than creating another authoritative store. Keep one control mapping and one context builder per responsibility. No implementation phase is complete until its tests and live evidence meet the stated exit criteria.
=======
**Action horizon:** Configure a plan horizon of 1–4 actions. Interrupt on terminal or level changes, action failure, illegal next actions, or budget exhaustion.

## 10. Action contract example

For a requested horizon of four, a valid response can contain one to four actions:

```json
{"actions":[{"action":"up","args":{}},{"action":"left","args":{}}]}
```

The harness maps `up` and `left` to their environment action IDs, then validates them against the current legal-action schemas. It executes the first action, observes the resulting state, and executes the second only if it remains legal and the game has not changed level or ended. It records measured outcomes but does not compare them against model-authored predictions.

An invalid response receives one corrective prompt. If the repair is also invalid, the runner logs the rejection and attempts a seeded random legal action when its required arguments can be generated safely; otherwise, it stops. Offline analysis compares the proposed list to the recorded continuation and labels each item as matched, mismatched, or unscorable. This is action match only, not policy quality.

## 11. Harness behavior and failure handling

- **Input integrity:** Preserve ordered action-to-frame mapping, including intermediate frames and resets; verify image dimensions and palette fidelity.
- **Action legality:** Validate action IDs, coordinate ranges, required arguments, and remaining budget before execution.
- **State consistency:** Associate all evidence with game/session, level, action index and reset generation; do not merge transitions across resets as continuous state.
- **JSON robustness:** Strict action-only schema validation; one bounded repair attempt; then a seeded random legal fallback if safe arguments can be generated, otherwise stop.
- **Offline scoring:** Report action matches only; do not infer prediction accuracy or policy quality from a fixed trace.
- **Memory integrity:** Preserve evidence provenance, deduplicate repeated entries, and cap active memory size; archive retired details with references.
- **Tool budgets:** Cap history query size, number of images, model calls per game step, tool retries, wall-clock time, and total actions.
- **Safety of autonomous action:** Treat the model response as a proposed action list; only the environment adapter can assert state transitions.
- **Observability:** Record prompt version, input images, token usage, latency, action plans, transition evidence, memory size and action counts.

## 12. Project layout

```text
src/v10/
├── cli.py              # replay, analyze, ablate, optimize, and live commands
├── strata_client.py    # multimodal JSON inference and optional HTTP logs
├── analysis.py         # action-only validation, offline context, trace matching
├── live.py             # bounded multi-action live runner and repair/fallback flow
├── SKILL.md            # shared action-only model instructions
└── tools/
    ├── game.py         # legal action validation, environment actions, transitions
    ├── history.py      # append-only evidence and prefix-bounded reads
    └── memory.py       # persistent game ledger and compact read-only context

runs/v10/<run>/
├── evidence.jsonl      # state and action transition records
├── decisions.jsonl     # model plans and execution results
├── frames/             # original and derived images
├── metrics.json        # live action, level, usage, and timing metrics
└── model_*.jsonl       # optional raw HTTP request/response logs
```

The shared game memory ledger and generated view live outside disposable run directories under `$V10_STATE_DIR/games/<game-id>/`.

## 13. Implementation sequence

**Phase 1 — Replay harness:** Implemented. Reads recorded traces, stores state/transition evidence and images, and prevents future transitions from entering the prompt.

**Phase 2 — Offline analysis:** Implemented. Runs Qwen via Strata on a replay prefix and compares its proposed action sequence with subsequent recorded actions without leaking them into the prompt. This measures action match only, not policy quality.

**Phase 3 — Live single-agent loop:** Implemented. Runs interruptible 1–4-action plans, records transition evidence, and offers reset only when budgeted.

**Phase 4 — Ablation:** Implemented. Compares pixels+actions with pixels+actions plus up to five read-only memory entries under matched budgets; can also compare prior action plans.

**Phase 5 — Optimization:** Implemented. Sweeps plan horizon, output-token budget, memory limit, and context profile on a fixed offline cut.

**Persistence invariant:** `SKILL.md` is the shared, game-independent source and is referenced by digest rather than copied into each run. The append-only game ledger and generated `MEMORY.md` live in a durable per-game state directory outside disposable run outputs. Every evidence reference carries its run ID. Model context is compacted from the persistent ledger without rewriting its history; evaluation arms are read-only with respect to that ledger.

## 14. Evaluation

| Metric | Interpretation |
|---|---|
| Levels solved / WIN | Primary task outcome |
| Actions to first progress / completion | Interaction efficiency |
| Offline action match | Agreement with the recorded continuation; not policy quality |
| Invalid actions / resets / interrupted plans | Control reliability and wasted budget |
| Input/output/image tokens, latency, GPU usage | Runtime cost |
| Active memory size and growth | Sustainability |
| Performance with vs without memory | Whether learned knowledge actually helps |

**Evaluation safeguards:** Compare at equal action and compute budgets; use unseen episodes/levels where possible; do not evaluate counterfactual actions using a fixed log. Offline traces report whether proposed actions match the recorded continuation; they do not measure policy quality.

## 15. Open decisions and risks

1. **Strata/Qwen interface:** Continue measuring image encoding cost, JSON reliability, reasoning-token accounting, and inference latency on the deployed model and hardware.
2. **Visual representation:** Determine whether the current/previous frame pair and structured measurements provide enough context at acceptable cost.
3. **Hidden state:** Visually identical frames can have different future dynamics; memory must represent relevant history and uncertainty.
4. **Goal discovery:** Visual salience is not proof of a target; require feedback and explicit competing goal hypotheses.
5. **Context anchoring:** Prior action plans may bias later choices; compare with and without prior plan summaries.
6. **Experiment cost:** Some actions consume scarce resources or irreversible progress; information gain must be balanced against risk.
7. **Memory quality:** Avoid runaway accumulation; compact repeated evidence and archive obsolete hypotheses without losing provenance.
8. **Generalization:** The method should work for click-based, directional, temporal and other unknown interfaces; no genre-specific assumptions in the skill.

## 16. Future work outside the current v10 baseline

HGT training, learned latent world models, multi-agent exploration, cross-game permanent pattern transfer, autonomous skill rewriting, broad graph databases, and guaranteed optimal planning remain possible future research directions.

---

**Current baseline:** **Qwen/Strata JSON action selection + Python validation and game execution + `SKILL.md` + read-only per-game memory context + append-only transition evidence + configurable 1–4 action plans.** Evaluate live progress and action efficiency; use offline action matches as a trace comparison only.
>>>>>>> 5b9552a (update)

# V10 — Single-Agent Action Planning

**Design specification v0.2 · 25 October 2026**

**Status:** Runtime phases 1–5 are implemented. Live task performance remains under evaluation.

## 1. Objective

Build a minimal, general-purpose agent that learns to play an unfamiliar visual game through repeated cycles of observation, action planning, outcome measurement, and revision. The agent receives pixels, available actions, and environment feedback; no game rules, object meanings, or goals are preprogrammed.

**Primary hypothesis:** A single multimodal LLM with compact per-game memory and precise Python tools can discover useful mechanics and exploit them without an HGT, multi-agent architecture, or trained game-specific world model.

**Success criteria:** measurable progress or level completion; improved action efficiency versus a no-memory baseline; bounded inference cost and memory growth. Offline action matches indicate agreement with a recorded continuation, not policy quality.

## 2. Design principles

1. **One action-selection agent.** Qwen vision-language model served through Strata uses the supplied state and history to choose a short action sequence.
2. **Tools provide measurements and effects.** Python tools observe and act, compute exact pixel changes, retrieve evidence, and persist structured records. They do not invent game semantics.
3. **Short adaptive action horizon.** Propose up to **1–4 actions**, executed one at a time; interrupt on level changes, terminal events, illegal actions, errors, or exhausted budgets. Longer sequences are a future optimization.
4. **Continue by default.** An analysis boundary never implies a reset. Reset is a deliberate, costed action requiring justification.
5. **Measure action outcomes.** Record each environment response and its measured visual changes.
6. **Keep evidence and interpretation distinct.** Any persistent knowledge is harness-managed context; the action response does not create or update it.
7. **No future leakage.** At time *t*, the agent may see only evidence available through *t*.
8. **Evidence is durable, context is compact.** Full logs are append-only; the LLM receives a concise current memory and relevant recent frames.
9. **General method, local facts.** `SKILL.md` is game-independent; `MEMORY.md` is specific to the active game.
10. **Measure before adding complexity.** No HGT, vector database, external planner, or extra LLM stages in the current v10 baseline.

## 3. System architecture

```text
                     ┌────────────────────────────┐
                     │ ARC-AGI-3 game environment │
                     │ pixels · actions · outcome │
                     └──────────────┬─────────────┘
                                    │
                     ┌──────────────▼─────────────┐
                     │ Python agent/harness       │
                     │ budget · tools · evidence  │
                     │ transition evidence · safety│
                     └──────────────┬─────────────┘
                                    │ context + images
                     ┌──────────────▼─────────────┐
                     │ Qwen-VL via Strata          │
                     │ choose actions · args      │
                     └──────────────┬─────────────┘
                                    │ JSON action list
                     ┌──────────────▼─────────────┐
                     │ Harness validates/executes │
                     │ one action at a time       │
                     └──────────────┬─────────────┘
                                    │
                            observe and repeat

 Persistent assets: SKILL.md | per-game MEMORY.md | evidence.jsonl | frames/
```

### Responsibilities

| Component | Owns | Must not do |
|---|---|---|
| Qwen via Strata | Choose a legal action sequence from the supplied state and history | Claim execution, predict outcomes, update memory, or fabricate action IDs |
| Agent harness | Model call orchestration, legal-action validation, step budgets, interrupts, retries, state continuity | Infer game rules silently |
| Python vision tools | Pixel masks, regions, temporal comparisons, candidate tracking with uncertainty | Declare an object a player, goal, or HUD without evidence |
| Evidence store | Immutable state, action transitions, original frames, measurements, and run decisions | Rewrite past transitions |
| Memory manager | Stable IDs, provenance, and compact read-only context generation | Depend on memory edits from the action-selection response |
| SKILL.md | Shared action-only response contract | Store per-game colors, coordinates, or action mappings |
| MEMORY.md | Persistent per-game context and prior observations | Duplicate all frames or unrestricted free-form reasoning |

## 4. Interaction cycle

**Bootstrap:** Load the shared `SKILL.md`; load the persistent per-game memory read-only; observe the game; supply the current state, legal actions, action budget, and current frame to Qwen through Strata.

**Repeated cycle:**

1. Harness assembles the current and previous game frames, recent action outcomes, legal actions, budgets, and up to five read-only memory entries.
2. Qwen chooses up to the configured **1–4 action plan horizon** and returns only action IDs and arguments.
3. Harness validates the action list and arguments. Invalid outputs are rejected or repaired with bounded retry; no illegal action is executed.
4. Execute actions one at a time; collect original frames and metadata, compute measurements, and record each transition.
5. On a level change, terminal state, illegal next action, action error, or exhausted budget, stop the remaining plan and consult Qwen again when appropriate.
6. Repeat with the latest state and outcomes. The harness maintains evidence and persistent memory; the model does not write memory or predict outcomes.

The model does not call game tools directly. It returns JSON to the Python runner, which validates and executes one action at a time. The runner checks the latest observation before each action and interrupts the remainder of a plan when needed.

## 5. Inputs to each Qwen decision

| Input | Format | Notes |
|---|---|---|
| General instruction | `SKILL.md` or distilled prompt | Stable and cacheable when supported |
| Game frames | Current and previous actual frame, at most two | Do not send initial-frame or delta-image stacks |
| Current state | Game status, level, legal action IDs and argument information | Live includes action schemas; offline replay uses recorded action IDs and observed args |
| Recent outcomes | Ordered action, arguments, and measured result | Bounded by the configured plan horizon |
| Game memory | Up to five compact entries, read-only | Existing memory is context; the model does not edit it |
| Budgets | Live remaining action/reset counts and plan horizon | Offline replay has no live remaining-budget estimate |

**Image selection:** Send the current actual game frame and, when available, the immediately previous actual frame. Do not send a history stack or delta overlays to action selection.

**Token budgeting:** Live defaults to a 4,096-token output cap; offline analysis and ablation default to 1,024. Reasoning effort defaults to `medium`; a separate reasoning-token budget is optional. Image token cost depends on the deployed Strata/Qwen path; use returned usage/latency and opt-in request/response logs rather than planning estimates.

## 6. LLM action contract

The model receives the current and previous actual game frames when available, legal actions, recent action outcomes, live budgets, and up to five read-only memory entries. It may reason internally, but the action loop does not request a written analysis report or memory updates. The configured plan horizon is at most four actions. Offline analysis has no live budget values.

```json
{
  "actions": [
    {"action":"ACTION3","args":{}},
    {"action":"ACTION1","args":{}}
  ]
}
```

The `v10.action_set/2` response must contain only the top-level `actions` array. Each action object must contain only a legal action ID and its `args`; extra fields are rejected. The array may contain zero to the configured horizon, with a maximum horizon of four. An empty list ends a live run. If reset budget is available, `RESET` is listed as a pseudo-action and must be the only returned action, with its justification in `args`. The harness validates every action against the latest legal-action schema, records transitions and measurements, and interrupts the remaining plan on a level change, terminal state, illegal next action, action error, or exhausted budget. Offline analysis compares proposed actions with the recorded continuation as action matches only; this is not a measure of policy quality.

The response schema is enforced by the runtime validator. Legal action IDs and argument schemas come from the current environment observation.

## 7. Python runner interfaces

The model interface is a Strata chat completion with system instructions, a user prompt, up to two actual game frames, and JSON output. The model does not receive callable tools. Python validates the action list, then dispatches actions through the environment adapter. The internal interfaces are:

| Tool | Inputs | Outputs | Rules |
|---|---|---|---|
| `GameTools.observe` | environment observation | Current frame, state, legal actions, level/terminal metadata | Read-only; does not advance game |
| `GameTools.act` | action ID, validated args | Result, new frame IDs, intermediate frames, measurements, transition evidence | Single action; consumes budget |
| `History` | bounded evidence reads | State and transition records with frame links | Append-only; prefix reads prevent future leakage |
| `GameMemory.compact` | entry limit and visible step | Compact persistent game context | Read-only from the action-selection loop; maximum five entries |
| Vision helpers | before/after frames | Pixel changes and candidate displacement measurements | Measurements only; no game semantics |

`StrataClient` requests JSON output and retries once without server-side JSON enforcement if the endpoint rejects structured output. The v10 validator then checks the response; the live runner allows one corrective model call before using a seeded random legal fallback when one can be safely formed.

## 8. Knowledge model

The persistent ledger retains typed knowledge entries for continuity with existing game memory. These categories describe stored context; they are not fields the model must produce during action selection. The live and offline action-selection loops read memory only.

### 8.1 Epistemic levels

- **Observation:** Direct, reproducible pixel/action/outcome evidence.
- **Pattern:** Repeated relation with conditions, support count, and exceptions.
- **Hypothesis:** Explanatory claim with alternatives and a falsifiable prediction.
- **Supported mechanic:** Hypothesis with successful predictive tests in relevant contexts; never equate support with universal validity.
- **Strategy:** Conditional plan tied to tested mechanics, outcomes, costs and applicability.
- **Goal hypothesis:** Candidate win/progression condition, promoted only through environment feedback or repeatable tests.

### 8.2 Per-game `MEMORY.md`

This is an internal knowledge organization reference, not a required model response. The action-selection loop supplies existing memory read-only and does not ask the model to build or update it.

Recommended compact sections:

1. **Current state:** last evidence ID, level, known budgets, controllable candidates.
2. **Supported mechanics:** stable IDs, preconditions, predictions, supporting/contradicting tests.
3. **Open hypotheses:** competing explanations, cheapest next tests.
4. **Patterns:** descriptive/temporal/relational patterns and exceptions.
5. **Goal candidates:** evidence for/against; terminal semantics if established.
6. **Current strategy:** mode, near-term objective, recovery triggers.
7. **Action history:** tried actions and their observed outcomes. Older prediction entries may remain in existing ledgers, but the current action loop neither requests nor writes them.
8. **Known failures/contradictions:** what should not be assumed.

Keep evidence IDs, not embedded screenshots. Maintain an append-only machine-readable ledger as the authoritative record; `MEMORY.md` is a concise, regenerated view. Memory may be supplied read-only to action selection, with at most five entries. The model is not asked to propose memory edits.

### 8.3 `SKILL.md`

Reusable, game-independent instructions for choosing legal actions, returning a bounded action sequence, and handling reset. No `ls20`-specific action mapping or color labels.

### 8.4 Reasoning and decision history

Preserve **structured action plans and outcomes** in `decisions.jsonl`. Record model version, prompt/schema version, visible evidence IDs, chosen actions, actual outcomes, and action-match results for offline runs. Raw request and response bodies are logged only when the user enables the corresponding CLI flags.

## 9. Action selection and control

The LLM proposes a decision, but the harness enforces budgets and interrupts.

Choose actions that make progress or clarify the current state, based on the supplied context. The model need only return the action sequence; it is not asked to label its mode or explain its reasoning.

The harness executes each action separately and observes the result before continuing the plan.

The harness stops a plan when the game changes level, reaches a terminal state, rejects an action, or exhausts a budget. A new decision then uses the latest observation.

**Reset policy:** Continue by default. When reset budget is available, the runner lists `RESET` as a legal pseudo-action. The model may return it alone with a justification in `args`; the harness enforces reset permission and budget. Never reset solely because a plan ended.

**Action horizon:** Configure a plan horizon of 1–4 actions. Interrupt on terminal or level changes, action failure, illegal next actions, or budget exhaustion.

## 10. Action contract example

For a requested horizon of four, a valid response can contain one to four actions:

```json
{"actions":[{"action":"ACTION1","args":{}},{"action":"ACTION3","args":{}}]}
```

The harness validates both actions against the current legal-action schemas. It executes the first action, observes the resulting state, and executes the second only if it remains legal and the game has not changed level or ended. It records measured outcomes but does not compare them against model-authored predictions.

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

# Hydra Next — Single-Agent Active Mechanic Discovery

**Design specification v0.1 · 24 October 2026**  
**Status:** Proposed; not yet validated with Qwen/Strata in a live ARC-AGI-3 environment.

## 1. Objective

Build a minimal, general-purpose agent that learns to play an unfamiliar visual game through repeated cycles of observation, hypothesis formation, experiment, prediction, action, and revision. The agent receives pixels, available actions, and environment feedback; no game rules, object meanings, or goals are preprogrammed.

**Primary hypothesis:** A single multimodal LLM with compact per-game memory and precise Python tools can discover useful mechanics and exploit them without an HGT, multi-agent architecture, or trained game-specific world model.

**Success criteria:** measurable progress or level completion; accurate held-out transition predictions; improved action efficiency versus a no-memory baseline; bounded inference cost and memory growth. Plausible textual explanations alone are not success.

## 2. Design principles

1. **One reasoning agent.** Qwen vision-language model served through Strata handles interpretation, hypotheses, experiments, and plans.
2. **Tools provide measurements and effects.** Python tools observe and act, compute exact pixel changes, retrieve evidence, and persist structured records. They do not invent game semantics.
3. **Short adaptive action horizon.** Initially propose **1–4 actions**, executed one at a time; interrupt immediately on significant surprises or terminal events. Longer sequences are a future optimization.
4. **Continue by default.** An analysis boundary never implies a reset. Reset is a deliberate, costed action requiring justification.
5. **Predict before acting.** Record observable predictions and alternatives before executing each diagnostic action.
6. **Separate facts from beliefs.** Observations, correlations, hypotheses, validated rules, and goal guesses have different epistemic status.
7. **No future leakage.** At time *t*, the agent may see only evidence available through *t*.
8. **Evidence is durable, context is compact.** Full logs are append-only; the LLM receives a concise current memory and relevant recent frames.
9. **General method, local facts.** `SKILL.md` is game-independent; `MEMORY.md` is specific to the active game.
10. **Measure before adding complexity.** No HGT, vector database, external planner, or extra LLM stages in v0.1.

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
                     │ prediction audit · safety  │
                     └──────────────┬─────────────┘
                                    │ context + images
                     ┌──────────────▼─────────────┐
                     │ Qwen-VL via Strata          │
                     │ discover · test · plan     │
                     └──────────────┬─────────────┘
                                    │ typed decision
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
| Qwen via Strata | Semantic observations, competing explanations, experiment choice, conditional plans | Claim execution or fabricate evidence |
| Agent harness | Call orchestration, legal-action validation, step budgets, interrupts, retries, state continuity | Infer game rules silently |
| Python vision tools | Pixel masks, regions, temporal comparisons, candidate tracking with uncertainty | Declare an object a player, goal, or HUD without evidence |
| Evidence store | Immutable actions, original frames, action outcomes, prediction/decision records | Rewrite past transitions |
| Memory manager | Stable IDs, merge/update/retire with provenance, compact context generation | Promote hypotheses to facts merely because the LLM says so |
| SKILL.md | General reasoning protocol and tool usage | Store per-game colors, coordinates, or action mappings |
| MEMORY.md | Current game's state, mechanic candidates, goals, failures, plans | Duplicate all frames or unrestricted free-form reasoning |

## 4. Interaction cycle

**Bootstrap:** Load `SKILL.md`; create empty game memory; call `game.observe`; supply initial image, available actions, game status, and budget to Qwen.

**Repeated cycle:**

1. Harness assembles the latest **1–4 action transitions**, relevant prior predictions, current image, optional delta visualizations, concise memory, and tool inventory.
2. Qwen answers: What changed? What repeats? What is still unknown? Which hypotheses explain it? Which previous predictions succeeded/failed? What goal is plausible? What is the cheapest informative experiment? What should happen next?
3. Qwen emits **one schema-conformant decision**: evidence-linked observations, proposed memory changes, priority question, mode, and 1–4 actions with observable predictions and stop conditions.
4. Harness validates the decision and action IDs. Invalid outputs are rejected or repaired with bounded retry; no illegal action is executed.
5. Harness records predictions **before** actions. Execute **one action**; collect original frames, intermediate frames and metadata; compute measurements; check stop conditions.
6. If an unexpected transition, level change, terminal state, resource threshold, or failed prediction occurs, stop the remaining plan and reconsult Qwen.
7. Audit predicted versus actual observations; append evidence and decisions; apply allowed memory patches; repeat.

**Important distinction:** A 1–4 action *plan* does not mean the LLM must consume 1–4 separate tool calls. A tool-calling loop can return after each action, or the harness can execute a bounded plan with local stop checks. Begin with one model decision per bounded plan and a deterministic interrupt mechanism.

## 5. Inputs to each Qwen decision

| Input | Format | Notes |
|---|---|---|
| General instruction | `SKILL.md` or distilled prompt | Stable and cacheable when supported |
| Current game memory | Compact Markdown or JSON | Active hypotheses, evidence counts, contradictions, current goals |
| Recent transitions | Ordered structured records | Action ID, pre/post evidence IDs, frame counts, outcomes |
| Original images | Initial/current plus relevant keyframes | Preserve visual meaning; do not replace with delta-only views |
| Delta images | One per informative action, when useful | Pixel changes, preferably with appeared/disappeared overlay |
| Exact measurements | JSON | Changed-pixel count, region boxes, candidate displacements, temporal changes |
| Prediction audit | Structured entries | Previous prediction, observation, match/fail/inconclusive |
| Game affordances | Tool schema and available actions | No assumptions about action names or coordinate semantics |
| Budgets | Counts/time/resources, if known | Reset and inference costs explicitly represented |

**Image selection:** For a four-action cycle, start with the initial or prior state image, four delta visualizations, and the current image (approximately six images). Include intermediate frames selectively when an action spans multiple frames or delta interpretation is ambiguous. Do not send all frames blindly.

**Token budgeting:** Earlier planning estimates were roughly 2,900–8,550 input tokens and 500–800 output tokens for a four-action cycle, but **image token cost is unmeasured** and depends on the exact Strata/Qwen multimodal path. Treat all numbers as provisional. Set a 1,024-token output cap initially; log actual input, image, reasoning, output tokens and latency if the runtime exposes them.

## 6. LLM reasoning contract

The analyst must answer these eight questions, allowing `unknown`:

1. **Observations:** What changed, and what stayed invariant? Cite exact transition/frame IDs.
2. **State candidates:** Which persistent regions, entities, temporal signals or HUD candidates are supported? Distinguish pixels from semantics.
3. **Patterns:** Which relations recur, under what conditions, and with which exceptions?
4. **Mechanics:** What competing causal explanations fit the observations? What would falsify each?
5. **Prediction audit:** Which prior predictions matched, failed or were inconclusive? What belief changes follow?
6. **Goal:** What objective is supported by outcome feedback? Which visually plausible targets remain guesses?
7. **Experiment:** What single unresolved question matters most, and what is the cheapest discriminating test?
8. **Decision:** Explore, exploit, or recover? Continue or reset? Which 1–4 legal actions, predictions and stop conditions?

**Fixed response schema (logical contract; validate with JSON Schema):**

```json
{
  "observations": [{"id":"O1","claim":"...","evidence_ids":["T4"]}],
  "patterns": [{"id":"P1","relation":"...","conditions":"...","evidence_ids":["T1","T2"],"exceptions":[],"status":"candidate"}],
  "hypotheses": [{"id":"H1","claim":"...","alternatives":["..."],"prediction":"...","falsification":"...","status":"candidate"}],
  "prediction_audit": [{"prediction_id":"PR1","verdict":"matched","evidence_ids":["T4"]}],
  "goals": [{"id":"G1","claim":"...","status":"unverified","evidence_ids":[]}],
  "priority_question": "...",
  "decision": {
    "mode":"explore",
    "continue_or_reset":"continue",
    "reset_justification":null,
    "actions":[{"action":"ACTION3","args":{},"prediction_id":"PR2","expected_observation":"...","measurable_prediction":{"all":[{"path":"measurements.changed_pixels","op":"gt","value":0}]},"stop_if":["unexpected_transition","level_change"]}]
  },
  "memory_patch": [{"op":"upsert","id":"H1","type":"hypothesis","claim":"...","evidence_ids":["T4"]}]
}
```

Each action must include `measurable_prediction`. Use a non-empty `all` list of `{path, op, value}` comparisons against allowlisted measurement or outcome fields, with `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, or inclusive `between`. If no supported measurement fits, use `null` and a non-empty `unmeasurable_reason`; that prediction is audited as inconclusive. Legacy flat pixel-diff predictions are normalized for existing callers. Frame IDs beginning with `F` are image handles and are never valid evidence citations. Memory patches accept only `upsert` and `retire`; the harness owns prediction verdicts.

This is a **representative schema**, not executable code. Real tool/action schemas must support game-specific arguments (e.g. coordinates), enforce enum values, bounds and valid evidence IDs, and forbid invented tool names. Avoid free-form numeric confidence until calibrated; retain support/contradiction counts and qualitative status.

## 7. Tools: initial MCP-compatible interface

Expose tools through a Python adapter; MCP compatibility is optional at runtime but useful for portability. Tool calls must be bounded, typed, and logged.

| Tool | Inputs | Outputs | Rules |
|---|---|---|---|
| `game.observe` | optional frame selection | Current frame ID/image, state, legal actions, level/terminal metadata | Read-only; does not advance game |
| `game.act` | action ID, validated args | Action result, new frame IDs, intermediate frames, metadata, transition ID | Single action; consumes budget; never speculative |
| `vision.diff` | two frame IDs, optional region | Exact changed pixels, boxes, masks/overlay | Report measurements, not causal labels |
| `history.query` | bounded filters, limit | Matching transition IDs, actions, measurements, frame links | No future records; enforce output cap |
| `memory.read` | IDs or concise summary | Current knowledge and evidence references | Read-only |
| `memory.propose_update` | typed patches | Accepted/rejected patches with reasons | Preserve prior evidence; harness controls promotion |

**Defer:** `vision.track` until necessary for robust identity/displacement; `experiment.compare` until comparison queries are frequent; specialized planning/search tools until baseline measurements justify them. A local deterministic tracking helper can still run inside preprocessing without being an LLM-visible tool.

**Tool interface caveat:** Verify Strata's *specific deployed model* supports multimodal image inputs and tool-call output together. If tool calling is unreliable, have Qwen emit schema-constrained JSON and let Python dispatch it. Do not assume that a runtime supporting vision automatically supports reliable agentic tool use.

## 8. Knowledge model

### 8.1 Epistemic levels

- **Observation:** Direct, reproducible pixel/action/outcome evidence.
- **Pattern:** Repeated relation with conditions, support count, and exceptions.
- **Hypothesis:** Explanatory claim with alternatives and a falsifiable prediction.
- **Supported mechanic:** Hypothesis with successful predictive tests in relevant contexts; never equate support with universal validity.
- **Strategy:** Conditional plan tied to tested mechanics, outcomes, costs and applicability.
- **Goal hypothesis:** Candidate win/progression condition, promoted only through environment feedback or repeatable tests.

### 8.2 Per-game `MEMORY.md`

Recommended compact sections:

1. **Current state:** last evidence ID, level, known budgets, controllable candidates.
2. **Supported mechanics:** stable IDs, preconditions, predictions, supporting/contradicting tests.
3. **Open hypotheses:** competing explanations, cheapest next tests.
4. **Patterns:** descriptive/temporal/relational patterns and exceptions.
5. **Goal candidates:** evidence for/against; terminal semantics if established.
6. **Current strategy:** mode, near-term objective, recovery triggers.
7. **Prediction register:** outstanding predictions and evaluation status.
8. **Known failures/contradictions:** what should not be assumed.

Keep evidence IDs, not embedded screenshots. Maintain an append-only machine-readable ledger as the authoritative record; `MEMORY.md` is a concise, regenerated view. A proposed update must not silently delete contradictory evidence.

### 8.3 `SKILL.md`

Reusable, game-independent instructions: how to compare transitions, detect patterns, formulate causal hypotheses, design controlled experiments, distinguish goals from visual guesses, choose explore/exploit/recover, handle reset, and cite evidence. No `ls20`-specific action mapping or color labels.

### 8.4 Reasoning and decision history

Preserve **structured decisions and predictions** in `decisions.jsonl`, with optional raw model reasoning stored separately when the runtime permits it. Do not routinely feed the full reasoning transcript back: it is verbose and can reinforce false hypotheses. Retrieve only selected historical rationales when auditing contradictions. Record model version, prompt/schema version, evidence IDs, chosen action, predicted effect, actual effect, and verdict.

## 9. Exploration–exploitation policy

The LLM proposes a decision, but the harness enforces budgets and interrupts.

**Explore** when action semantics, hidden state, goals or key mechanics remain uncertain and a cheap test could change the plan. Prefer one-variable comparisons, matched contexts where feasible, and actions that distinguish competing hypotheses.

**Exploit** when a supported mechanic and plausible goal justify a low-risk action sequence. Still attach predictions and verify each action.

**Recover** when predictions repeatedly fail, progress stalls, a resource becomes critical, or state becomes irrecoverable. Consider revising the state model, backtracking, or resetting.

**Reset policy:** Continue by default. Reset only if (a) the state is irrecoverable, (b) a controlled experiment requires a reproducible start, or (c) expected progress after reset outweighs reset and replay cost. The LLM must state its reason; the harness validates permission and budget. Never reset solely because a cycle ended.

**Action horizon:** Start with 1–2 actions for unknown mechanics and 2–4 for short tests or reliable progress. Interrupt at any point on terminal/level changes, action failure, prediction mismatch, or significant unexpected visual changes. A later version may extend to 4–10 actions once validated.

## 10. Concrete worked example: `ls20`, first four actions

**Source:** Earlier analysis of the supplied `ls20` game log; this is an illustrative retrospective reconstruction, not a Qwen/Strata live result.

| Step | Action | Measured changed pixels | Tracked marker effect |
|---|---|---:|---|
| 1 | ACTION1 | 52 | Up approximately 5 px |
| 2 | ACTION1 | 52 | Up approximately 5 px |
| 3 | ACTION1 | 52 | Up approximately 5 px |
| 4 | ACTION3 | 2 | No tracked displacement |

**Observation:** ACTION1 repeatedly correlates with a similar upward displacement of a candidate visual marker; ACTION3 at step 4 produces little visible change. A separate small region may still change.

**Pattern P1:** Repeated action-conditioned displacement. Evidence: T1–T3. Candidate, not yet a fully proven control rule.

**Hypothesis H1:** ACTION1 controls upward movement. Prediction: at another permissive location, ACTION1 again moves the tracked region upward by about 5 pixels. Alternative: repeated animation or mistaken identity.

**Hypothesis H2:** ACTION3 failed because movement was blocked. Prediction: ACTION3 will cause displacement in a different spatial context. Alternative: ACTION3 has another meaning or changes hidden state.

**Goal G1:** Unknown. A visually prominent region may be a destination, but there is no completion evidence in these four actions.

**Next experiment:** Choose a legal action that discriminates blocked movement from an alternative action meaning; compare the marker and other changed regions. Continue the existing game state unless geometry makes a matched comparison impossible. Record predicted displacement before executing.

**Limit of offline replay:** We can score predictions only for actions actually present in the recorded continuation. We cannot claim that an alternative experiment would succeed without a live environment or validated simulator.

## 11. Harness behavior and failure handling

- **Input integrity:** Preserve ordered action-to-frame mapping, including intermediate frames and resets; verify image dimensions and palette fidelity.
- **Action legality:** Validate action IDs, coordinate ranges, required arguments, and remaining budget before execution.
- **State consistency:** Associate all evidence with game/session, level, action index and reset generation; do not merge transitions across resets as continuous state.
- **JSON robustness:** Strict schema validation; one bounded repair attempt; on failure, take no action and surface error.
- **Prediction evaluation:** Distinguish exact measurable predicates (e.g. changed pixel count) from semantic predictions requiring further interpretation; mark ambiguous outcomes `inconclusive`.
- **Memory integrity:** Version patches, preserve conflicting hypotheses, deduplicate observations, and cap active memory size; archive retired details with references.
- **Tool budgets:** Cap history query size, number of images, model calls per game step, tool retries, wall-clock time, and total actions.
- **Safety of autonomous action:** Never assume an LLM tool response is a real game outcome; only the environment adapter can assert state transitions.
- **Observability:** Record per-cycle prompt version, images, tokens, latency, tool calls, prediction verdicts, memory size and action counts.

## 12. Project layout

```text
hydra-next/
├── agent.py                   # single agent/harness loop
├── strata_client.py           # multimodal inference and schema adapter
├── SKILL.md                   # reusable mechanic-discovery method
├── schemas/                   # validated tool/decision/memory contracts
├── tools/
│   ├── game.py                # observe, act, reset
│   ├── vision.py              # frame deltas and candidate tracking
│   ├── history.py             # bounded evidence queries
│   └── memory.py              # read and propose patches
├── tests/                     # deterministic runtime smoke tests
└── runs/
    └── <session-id>/
        ├── MEMORY.md          # concise current game knowledge
        ├── evidence.jsonl     # immutable environment transitions
        ├── decisions.jsonl    # plans, predictions, audits
        ├── frames/            # original and selected derived images
        └── metrics.json       # cost, quality and outcome counters
```

This is a suggested layout, not a claim about existing repository paths. Tool implementations are Python modules; `SKILL.md` documents how to use them and does not embed executable tools.

## 13. Implementation sequence

**Phase 1 — Replay harness:** Read one existing log, expose only evidence through the current step, generate exact delta images and structured transitions, and validate output JSON. No game actions are executed.

**Phase 2 — Offline analysis:** Run Qwen via Strata on the first 1–4 `ls20` actions; inspect hypotheses and predicted outcomes; score against subsequent *recorded* actions without leaking them into the prompt.

**Phase 3 — Live single-agent loop:** Add `game.observe`/`game.act`, interruptible 1–4-action plans, memory patches and prediction audits. Continue by default; reset only explicitly.

**Phase 4 — Ablation:** Compare pixels+actions alone, pixels+actions+memory, and memory+delta frames under matched budgets. Separately compare optional prior reasoning snippets.

**Phase 5 — Optimization:** Tune image selection, token budgets, horizon and memory compaction. Add advanced tools or HGT only if measured bottlenecks justify them.

**Persistence invariant:** `SKILL.md` is the shared, game-independent source and is referenced by digest rather than copied into each run. The append-only game ledger and generated `MEMORY.md` live in a durable per-game state directory outside disposable run outputs. Every evidence reference carries its run ID. Model context is compacted from the persistent ledger without rewriting its history; evaluation arms are read-only with respect to that ledger.

## 14. Evaluation

| Metric | Interpretation |
|---|---|
| Levels solved / WIN | Primary task outcome |
| Actions to first progress / completion | Interaction efficiency |
| Held-out action-effect prediction accuracy | Whether mechanics predict observations |
| Prediction calibration and exception rate | Whether knowledge is appropriately qualified |
| Experiment yield | Fraction of tests that discriminate important hypotheses |
| Invalid actions / resets / interrupted plans | Control reliability and wasted budget |
| Input/output/image tokens, latency, GPU usage | Runtime cost |
| Active memory size and growth | Sustainability |
| Performance with vs without memory | Whether learned knowledge actually helps |

**Evaluation safeguards:** Compare at equal action and compute budgets; use unseen episodes/levels where possible; do not evaluate counterfactual actions using a fixed log. Offline traces measure prediction and analysis quality, not the performance of a policy that would have chosen different actions.

## 15. Open decisions and risks

1. **Strata/Qwen interface:** Confirm exact vision encoding, image limits, JSON/tool-call reliability and reasoning-token accounting on the intended model and hardware.
2. **Visual representation:** Determine whether delta images, overlays, intermediate frames and structured tracking improve prediction accuracy enough to justify cost.
3. **Hidden state:** Visually identical frames can have different future dynamics; memory must represent relevant history and uncertainty.
4. **Goal discovery:** Visual salience is not proof of a target; require feedback and explicit competing goal hypotheses.
5. **Model self-confirmation:** Prior reasoning can anchor false mechanics; score against observations and preserve counterexamples.
6. **Experiment cost:** Some actions consume scarce resources or irreversible progress; information gain must be balanced against risk.
7. **Memory quality:** Avoid runaway accumulation; compact repeated evidence and archive obsolete hypotheses without losing provenance.
8. **Generalization:** The method should work for click-based, directional, temporal and other unknown interfaces; no genre-specific assumptions in the skill.

## 16. Explicitly out of scope for v0.1

HGT training, learned latent world models, multi-agent exploration, cross-game permanent pattern transfer, autonomous skill rewriting, broad graph databases, and guaranteed optimal planning. These remain possible future research directions, not prerequisites for the first experiment.

---

**Decision:** Build the smallest observable loop first: **Qwen/Strata + Python game/vision/history/memory tools + `SKILL.md` + per-game `MEMORY.md` + append-only evidence + strict schemas**. Evaluate whether predictions and decisions improve before adding architecture.

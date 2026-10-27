# V10 discovery tools

Status: proposed.

## Scope

Add four tools for inspecting observations, retrieving past evidence, testing explanations, and reading or proposing memory.

Keep the current reasoned-action response, action validation, route replay, and review workflow. Discovery uses a separate request format and is enabled with `--discovery-tools`, off by default.

## Request format

```json
{"request_id":"D1","tool":"search_history","arguments":{"action":"right","limit":10}}
```

Each result contains the request ID, status, game/run IDs, evidence references, and tool data. Errors contain a code and message. The harness sets the current game, run, and history cutoff.

## 1. inspect_region

Examine an area of a recorded frame or transition.

```json
{"evidence_id":"T42","region":[10,10,30,40],"include_images":true}
```

- `region` is optional and uses original grid coordinates: `[y0,x0,y1,x1]`, excluding the end coordinates.
- Omit the region to inspect the whole frame. Images are omitted unless requested.
- A state returns its current crop; a transition returns before/after crops and measurements.
- Return changed pixels, color transitions, and candidate displacement with its score and margin.
- Missing measurements are `null`. Displacement is an estimate, not an identified object's movement.

Optional color filtering can be added later. It is not required for the MVP.

## 2. search_history

Retrieve earlier actions and outcomes.

```json
{"action":"right","level":2,"reset_generation":0,"limit":10}
```

All filters are optional. Also accept `evidence_ids` and `since_step`. Default limit: 10; maximum: 20.

Return matching transitions with actions, arguments, measurements, image handles, and pre/post state metadata. Select the newest matches and return them chronologically, with `truncated` and `through_step`.

The level filter refers to the pre-action level. Normalize directional aliases when matching actions. MVP searches the current run.

## 3. test_hypotheses

Execute one action and compare 2–4 predictions against its outcome.

```json
{
  "base_evidence_id":"T42",
  "action":{"action":"right","args":{}},
  "measurement":{"region":[10,10,30,40]},
  "hypotheses":[
    {"id":"H1","claim":"The region stays unchanged",
     "prediction":{"all":[{"path":"measurements.changed_pixels","op":"eq","value":0}]}},
    {"id":"H2","claim":"The region changes",
     "prediction":{"all":[{"path":"measurements.changed_pixels","op":"gt","value":0}]}}
  ]
}
```

The harness validates the current base state, action, arguments, and predictions; saves the predictions; executes one action through the normal live pathway; then scores every prediction against the same transition.

MVP predicates use exact pixel-difference fields and environment outcomes: `measurements.changed_pixels`, `measurements.change_ratio`, `game_state`, `terminal`, `level`, and `levels_completed`. Reuse existing comparison operators. Treat visual comparisons across a level change as unavailable.

| Verdict | Meaning |
|---|---|
| matched | All required measurements satisfy the prediction. |
| failed | A reliable measurement contradicts a predicate. |
| inconclusive | Required measurements are missing or ambiguous. |

One match with all other alternatives failed distinguishes the submitted alternatives. Multiple matches indicate overlapping predictions; all failures require revised explanations. A match supports the measured prediction; broader conclusions go to review.

Tests consume the normal action budget and cannot request RESET. Request IDs prevent duplicate dispatch: return a recorded result on retry and reject changed arguments under the same ID. If execution is uncertain, return `unknown_execution` and stop automatic advancement until reconciled.

## 4. game_memory

Read existing guidance or queue a finding for review.

```json
{"operation":"read","scope":"game"}
```

Read scope is `game` or `global`; return the saved Markdown.

```json
{
  "operation":"propose",
  "claim":"This region changes after right",
  "conditions":["tested level scope"],
  "supporting_evidence":["T43"],
  "contradicting_evidence":[]
}
```

Validate the proposal and return `queued`. The next existing review considers it and remains responsible for saving game/global Markdown. Failed reviews leave proposals pending. Evidence references are deduplicated.

## Loop integration

Offer one discovery phase at the first normal model-selection boundary for a new level, after startup and outside verified replay/bootstrap.

Allow two read requests, one experiment, one proposal, and at most five model completions. The model may return `{"request_id":"D5","tool":"finish","arguments":{}}` at any time.

Start with no pending action plan. After a probe, rebuild context from its outcome and resume normal selection. Existing level, terminal, and review handling still applies. Ordinary action-prediction mismatches remain diagnostic.

## Evolution and evaluation

Possible additions: color filtering, connected shapes, object tracking, short action-sequence tests, experiment ranking, and prior-run retrieval. Add branching experiments only when state restoration is available and verified.

Validate tool contracts and action accounting with controlled environments. Compare baseline and discovery-enabled live runs using matched model settings, seeds, starting memory, and action budgets. Measure levels completed, actions, resets, repeated ineffective actions, model cost, and latency.

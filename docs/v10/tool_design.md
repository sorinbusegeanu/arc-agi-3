# V10 discovery tools

Status: implemented in v10.0.22; available by default in model-driven action-selection calls, including bootstrap. Each selection allows at most three tool-call planning rounds, four newly executed branch outcomes, and ten model completions total. Cached outcomes do not consume the branch limit. After three tool-call rounds or four newly executed branches, tools are withheld and the harness explicitly requires an action-decision JSON object. Bootstrap allows at most three invalid plan attempts; if they produce no valid action, the harness executes one legal directional or wait fallback, counts it as an ordinary game action, and resumes model selection. Use `--no-discovery-tools` to disable.

## Scope

Add four tools for inspecting observations, retrieving past evidence, testing explanations, and reading or proposing memory.

Keep the current reasoned-action response, action validation, route replay, and review workflow. Action-selection calls offer these functions through Strata's OpenAI-compatible tool interface with automatic tool choice. The model may call a function or return the existing reasoned-action object. Discovery is enabled by default and can be disabled with `--no-discovery-tools`.

## Request format

Strata returns a native `tool_calls` entry with a provider call ID, function name, and JSON arguments. The harness uses that call ID as the request ID, dispatches through the v10 discovery owner, and appends an assistant tool-call message followed by a matching `role: tool` result. Results contain status, game/run IDs, evidence references, and tool data; errors contain a code and message.

## 1. inspect_region

Examine an area of a recorded frame or transition.

```json
{"evidence_id":"T42","region":[10,10,20,20]}
```

Branch transition evidence is inspected with both identifiers returned in `evidence_refs`:

```json
{"evidence_id":"T2","branch_id":"a1b2c3d4e5f60718","region":[0,0,5,5]}
```

- `region` is required and uses original grid coordinates: `[y0,x0,y1,x1]`, excluding the end coordinates.
- Each crop is limited to 20 rows by 20 columns. Choose the smallest useful region; full-frame inspection is not supported. Pixel arrays are returned as structured data.
- A state returns its current crop; a transition returns before/after crops and measurements.
- When `branch_id` is supplied, both IDs must resolve within that branch. Branch references never fall back to live history.
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

Test one to four candidate actions from the same observed base, with one or more measurable predictions per case. Each `prediction` must use the `all` list of exact `{path, op, value}` objects; supported paths are `measurements.changed_pixels`, `measurements.change_ratio`, `game_state`, `terminal`, `level`, and `levels_completed`. Operators are `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, and `between` (the latter takes an ascending two-number array). The function schema and runtime validator enforce this contract.

```json
{
  "base_evidence_id":"T42",
  "cases":[
    {"case_id":"move-right","action":{"action":"right","args":{}},"hypotheses":[
      {"id":"H1","prediction":{"all":[{"path":"measurements.changed_pixels","op":"eq","value":0}]}},
      {"id":"H2","prediction":{"all":[{"path":"measurements.changed_pixels","op":"gt","value":0}]}}
    ]},
    {"case_id":"wait","action":{"action":"WAIT","args":{}},"hypotheses":[
      {"id":"H3","prediction":{"all":[{"path":"measurements.changed_pixels","op":"eq","value":0}]}},
      {"id":"H4","prediction":{"all":[{"path":"measurements.changed_pixels","op":"gt","value":0}]}}
    ]}
  ]
}
```

Each case runs in an isolated fork, replays the run-local reset/action prefix, normalizes recorded direction names to the environment's canonical action IDs, and must match the base frame and level/progress fingerprint before acting. This ensures a case based on later-level evidence is tested at that recorded level. The live game and its action cap are unchanged. Resets and branch transitions are logged separately.

MVP predicates use exact pixel-difference fields and environment outcomes: `measurements.changed_pixels`, `measurements.change_ratio`, `game_state`, `terminal`, `level`, and `levels_completed`. Reuse existing comparison operators. Treat visual comparisons across a level change as unavailable.

| Verdict | Meaning |
|---|---|
| matched | All required measurements satisfy the prediction. |
| failed | A reliable measurement contradicts a predicate. |
| inconclusive | Required measurements are missing or ambiguous. |

One match with all other alternatives failed distinguishes the submitted alternatives. Multiple matches indicate overlapping predictions; all failures require revised explanations. A match supports the measured prediction; broader conclusions go to review.

Candidate actions cannot request RESET. Request IDs prevent duplicate dispatch: return a recorded result on retry and reject changed arguments under the same ID. Observable replay mismatch prevents candidate execution. Hidden state not exposed through observations cannot be certified by this check.

Completed outcomes are cached by exact base evidence ID, canonical action, and normalized arguments. Repeated cases reuse the branch transition and evaluate their own predictions against its measured outcome. `discovery.jsonl` records reuse, and live metrics report `discovery_branch_reuses`. Replay mismatches, branch errors, and incomplete outcomes are never cached.

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

Make discovery available on every action-selection call, including after startup and bootstrap. The model chooses whether to request a tool or return an ordinary action decision. Bound tool/result round trips by model completions per action-selection exchange rather than per-level discovery quotas.

Discovery tools have no per-level call quotas. An action-selection exchange is limited to ten model completions, which bounds repeated tool/result round trips by their actual LLM cost. Function schemas bound case, prediction, history, and region arguments; model output and context limits bound the data sent to Strata. Tool calls are sequential. The model may return an ordinary action decision at any time. The four functions are offered on startup/bootstrap action requests and ordinary action-selection requests, but not on reviews or post-mortems.

Start with no pending action plan. After a probe, rebuild context from its outcome and resume normal selection. Existing level, terminal, and review handling still applies. Ordinary action-prediction mismatches remain diagnostic.

## Evolution and evaluation

Possible additions: color filtering, connected shapes, object tracking, short action-sequence tests, experiment ranking, and prior-run retrieval.

Validate tool contracts and action accounting with controlled environments. Compare baseline and discovery-enabled live runs using matched model settings, seeds, starting memory, and action budgets. Measure levels completed, actions, resets, repeated ineffective actions, model cost, and latency.

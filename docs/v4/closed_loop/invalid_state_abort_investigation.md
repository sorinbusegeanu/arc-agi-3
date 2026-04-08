Invalid State Abort Investigation

## Purpose

The current comparison runner must distinguish candidate-generation activation, exact family solver behavior, and invalid-state abort before meaningful policy progress.

## Required per-run fields

- game_id
- seed
- policy_mode
- selected_policy_name
- success
- stop_reason
- step_count
- trace_row_count
- first_failure_step_index
- first_failure_bucket
- first_failure_abort_site
- first_failure_abort_message
- first_failure_missing_field
- first_failure_required_fields
- first_failure_current_visible_fields

## Interpretation

- widespread `invalid_state_abort` across both policy modes indicates a shared typed-state or runtime-surface blocker
- `family_exact`-only invalid-state failures indicate a family-local builder or solver path issue
- failures with zero trace rows indicate startup or pre-decision failure

## Metadata propagation requirement

- invalid-state aborts must carry structured metadata from the abort site into the persisted ledger record
- a generic `action selection` bucket is insufficient when the abort originated earlier in state parsing or reconstruction
- investigation is not complete until `abort_site` and `abort_message` are visible in the exported analysis surface

## Certified planner exhaustion vs state failure

- `no certified plan available` is not a typed-state parse failure
- it must be classified under `policy_action_selection`
- exported analysis must distinguish true state invalidity from certified-planner exhaustion

## Next separation after invalid-state classification

- once certified-planner exhaustion is separated from true typed-state failure, the next question is whether family-exact runs fail by zero-action startup, noop-only execution, or bounded non-winning behavior
- exported analysis must expose executed action count and noop-only behavior for family-exact runs

## Next separation after execution surface

- once family-exact runs are confirmed to execute non-noop actions, the next question is whether they produce meaningful state change or only repetitive low-value behavior
- exported analysis must expose changed-step counts, total changed cells, and repeated-action streaks

## Next separation after progress surface

- when family-exact runs execute non-noop actions but show zero changed cells, the next question is whether the environment is returning unchanged observations, whether changed-cells computation is missing, or whether actions are being ignored by the environment
- exported analysis must expose observation presence, step-result presence, changed-cells field presence, and repeated identical observation hashes

## Next separation after observation surface

- when observations change and actions are applied but `changed_cells` is absent, the immediate blocker is in diff propagation or trace-row population rather than in the environment
- exported analysis must explicitly count rows where observation change occurred but `changed_cells` was missing

## Diff propagation fix target

- the current blocker is not environment application and not unchanged observations
- the blocker is absence of `changed_cells` population in Step 8 trace rows despite changing observations
- the immediate fix target is loop-controller trace-row population from pre/post observation diff

## Next separation after diff propagation fix

- once `changed_cells` is populated correctly, the next question is whether family-exact runs are making objective progress, reaching terminal signals, or merely changing state without approaching a win
- exported analysis must expose terminal or win signals, completion deltas, and stagnation classification

## Next separation after outcome surface

- terminal or win signals must be attributed to post-action observations, not just the initial observation
- once diff propagation is fixed, the next question is whether the solver is looping on the same action because it is also looping on the same goal or subgoal kind
- exported analysis must separate initial terminal-looking observations from true post-action terminal progression

## Next separation after family-exact outcome ambiguity

- when family-exact runs already look terminal at the initial observation, the next useful surface is whether the family solver is emitting any normalized decision signal at all
- exported analysis must show whether selected goal/subgoal kind fields are genuinely absent or merely not propagated into trace rows

## Next separation after decision-surface propagation

- once family-exact decision fields are present on every traced step, the next question is whether the solver is repeating the same goal or subgoal, or changing actions without changing decision intent
- exported analysis must expose decision switch counts and action-to-decision coupling

## Next separation after decision-intent stability

- when a family-exact solver keeps the same goal and subgoal across all steps, the next question is whether its underlying decision basis is also frozen or whether the solver is adapting locally while preserving the same top-level intent
- exported analysis must expose whether decision basis changes under stable goal or subgoal labels

## Next separation after basis stability

- `sv01` now appears top-level stable but locally adaptive, so the next question is whether it is moving the same target locator or genuinely retargeting under one intent label
- `tb01` appears fully frozen, so the next question is whether its family solver basis lacks any changing target locator, anchor, or construction target across steps
- exported analysis must expose target-locator stability separately from top-level goal stability

## Next separation after target-locator stability

- `sv01` now clearly retargets under one stable top-level intent, so its remaining issue is likely objective quality rather than frozen local state
- `tb01` is frozen at target-locator level, so the next question is whether its mode hint, route size, bridge anchor, bridge target, and construction target are also frozen
- exported analysis must separate target-locator freeze from deeper bridge/construction basis freeze

## Next separation after candidate-count surface

- `tb01` still shows a frozen basis and now also needs candidate-identity inspection
- stable candidate counts alone are insufficient because different candidates can still exist under the same count
- exported analysis must expose whether the ranked candidate set and selected candidate identity ever change across steps

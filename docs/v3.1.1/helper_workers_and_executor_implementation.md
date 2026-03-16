# Helper Workers And Executor Implementation

This document describes the current native `v3_1` implementation of:

- planning helper workers
- executor request construction
- env worker execution of the selected plan

It reflects the actual code in `src/v3_1`, not a target design.

## Main Files

Helper path:

- [helper_modes.py](/home/zodrak/zod/src/v3_1/planning/helper_modes.py)
- [planning_helper_worker.py](/home/zodrak/zod/src/v3_1/agents/planning_helper_worker.py)
- [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)

Execution path:

- [executor_service.py](/home/zodrak/zod/src/v3_1/execution/executor_service.py)
- [env_worker.py](/home/zodrak/zod/src/v3_1/execution/env_worker.py)
- [option_execution.py](/home/zodrak/zod/src/v3_1/execution/option_execution.py)
- [route_execution.py](/home/zodrak/zod/src/v3_1/execution/route_execution.py)
- [outcomes.py](/home/zodrak/zod/src/v3_1/execution/outcomes.py)
- [env_worker_agent.py](/home/zodrak/zod/src/v3_1/agents/env_worker_agent.py)
- [env_factory.py](/home/zodrak/zod/src/v3_1/execution/env_factory.py)

## Helper Workers

## Role

Helper workers are advisory only.

They do not:

- mutate blackboard
- mutate memory
- select the final plan
- perform execution

They take a planning context plus a candidate id list and return proposal deltas.

## Worker Shape

The Ray worker is minimal:

- [PlanningHelperWorker](/home/zodrak/zod/src/v3_1/agents/planning_helper_worker.py)

It exposes:

- `run(request)`

That method directly calls:

- `run_helper_mode(request)`

in [helper_modes.py](/home/zodrak/zod/src/v3_1/planning/helper_modes.py).

## Request Flow

Helper requests are created in:

- [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)

inside:

- `_dispatch_helpers(...)`

The orchestrator:

1. builds `candidate_ids` from the seed planner decision
2. identifies `high_retry_ids` from memory state
3. builds one shared payload:
   - `route_features`
   - `high_retry_ids`
   - `durable_priors`
4. decides which helper modes are enabled from feature flags
5. creates one `HelperTaskRequest` per mode
6. dispatches them to the helper worker pool
7. resolves them in parallel submission order
8. falls back to local `run_helper_mode(request)` if remote execution fails

## HelperTaskRequest contents

The helper request includes:

- session/run/game/round/pass ids
- `helper_mode`
- `plan_context_id`
- `blackboard_version`
- `memory_version`
- `policy_version`
- `ranker_version`
- `candidate_ids`
- `payload`

This means helper outputs are always tied to one planning context and one version set.

## Helper Modes

Current helper modes are all implemented in [helper_modes.py](/home/zodrak/zod/src/v3_1/planning/helper_modes.py).

### 1. `candidate_expansion`

Behavior:

- iterates over candidate ids
- gives each one a base `score_delta = 0.08`
- reads durable `candidate_outcomes`
- if prior successes exceed failures, adds another `0.03`

Output fields:

- `candidate_id`
- `score_delta`
- `source = "candidate_expansion"`

### 2. `route_analysis`

Behavior:

- reads `route_features`
- reads durable `recovery_patterns`
- for each candidate:
  - `score_delta = 0.05 * progress_potential`
  - `risk_delta = 0.1 * risk + 0.03 * prior_failures`

Output fields:

- `candidate_id`
- `score_delta`
- `risk_delta`
- `source = "route_analysis"`

### 3. `score_feature_computation`

Behavior:

- adds a flat `score_delta = 0.03`

Output fields:

- `candidate_id`
- `score_delta`
- `feature_name = "helper_feature_bonus"`
- `source = "score_feature_computation"`

### 4. `hypothesis_proposal`

Behavior:

- reads durable `trigger_patterns`
- returns a hypothesis-oriented bonus:
  - base `0.04`
  - plus `0.03` if trigger priors exist

Output fields:

- `candidate_id`
- `score_delta`
- `hypothesis`
- `source = "hypothesis_proposal"`

### 5. `pruning_suggestion`

Behavior:

- reads `high_retry_ids`
- reads durable `failure_patterns`
- returns only `risk_delta`
  - `+0.2` if high retry
  - `+0.05` if found in durable failure priors

Output fields:

- `candidate_id`
- `risk_delta`
- `source = "pruning_suggestion"`

## Helper Result Shape

All helper modes return `HelperTaskResult` through `_result(...)`.

Result fields include:

- planning context identity fields
- `proposal_id = "helper:<mode>:<plan_context_id>"`
- `proposals`

Each proposal is a lightweight advisory patch, not an authoritative plan.

## Where Helper Results Are Used

Helper results are consumed in planner reranking through:

- [reranking.py](/home/zodrak/zod/src/v3_1/planning/reranking.py)

Current usage:

- sum `score_delta` by candidate id
- sum `risk_delta` by candidate id
- apply:
  - `final_score = score + helper_boost - helper_penalty`

So helpers currently affect reranking, not belief construction or filtering directly.

## Executor And Env Worker

## High-Level Flow

The selected planner decision is turned into execution through:

1. planner produces `PlannerDecision`
2. executor service converts it into `ExecutorRequest`
3. env worker runs either:
   - probe episode
   - directed episode
4. raw episode is analyzed
5. blackboard and memory reconcile from the result

This flow is driven from:

- [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py)

## Executor Request Construction

Request construction happens in:

- [executor_service.py](/home/zodrak/zod/src/v3_1/execution/executor_service.py)

Function:

- `build_executor_request(decision, max_steps, mode, seed)`

It copies the selected candidate from decision metadata and creates one `ExecutorRequest`.

Important request fields:

- `candidate_id`
- `action`
- `max_steps`
- `mode`
- `action_id`
- `action_name`
- `action_family`
- `required_action_family`
- `target_entity_id`
- `target_centroid`
- `click_target_coordinates`
- `metadata`

The metadata currently contains:

- `candidate_class`
- `target_entity_id`
- `target_area_id`
- `skill_id`
- `target_centroid`
- `click_target_coordinates`
- `required_action_family`
- `rationale`
- `fallback_candidates`
- `seed`

This request is the bridge from planner selection to execution.

## Env Worker Ownership

The execution actor is:

- [EnvWorkerAgent](/home/zodrak/zod/src/v3_1/agents/env_worker_agent.py)

It owns one live:

- [EnvWorker](/home/zodrak/zod/src/v3_1/execution/env_worker.py)

The env worker owns:

- one persistent `NormalizedEnvAdapter`
- last observation
- last info
- reset counter

So the worker process persists, but each probe or directed episode currently begins with a reset.

## Environment Adapter Layer

The environment adapter is implemented in:

- [env_factory.py](/home/zodrak/zod/src/v3_1/execution/env_factory.py)

Important responsibilities:

- build the env from the configured factory
- normalize reset/step outputs
- normalize available actions
- normalize action ids, names, and families

The adapter exposes:

- `reset(seed)`
- `step(action)`
- `available_actions()`
- `env_metadata()`

It can also fall back to:

- `NullEnv`

for scaffolded execution.

## Action Normalization

Action normalization is centralized in [env_factory.py](/home/zodrak/zod/src/v3_1/execution/env_factory.py):

- `normalize_action_lookup(...)`

It maps raw ids or names into normalized fields:

- `action_id`
- `action_name`
- `action_family`

Families currently include:

- `move`
- `interact`
- `click_at`
- `undo`
- `reset`

## Probe Execution

Probe execution happens in:

- `EnvWorker._run_probe(...)`

### Probe Episode Steps

1. reset env
2. capture initial observation
3. for each step:
   - read available actions
   - choose an action using `choose_probe_action(...)`
   - step the env
   - record `RawStep`
   - track no-change aliases
4. stop on:
   - done
   - truncated
   - step budget

### Probe Policy

Probe action choice is implemented in:

- [option_execution.py](/home/zodrak/zod/src/v3_1/execution/option_execution.py)

Function:

- `choose_probe_action(...)`

Current probe policy:

- uses only currently available actions
- does not synthesize noop
- samples from a weighted distribution

Biases:

- same as last action: penalized
- 2-step oscillation: penalized
- actions that recently produced no change repeatedly: penalized
- unseen actions in current episode: boosted

This is not uniform random. It is a short-horizon anti-repeat stochastic policy.

## Directed Execution

Directed execution happens in:

- `EnvWorker._run_directed(...)`

### Directed Episode Steps

1. reset env
2. capture initial observation
3. for each step:
   - read available actions
   - derive route instruction from the selected planner action
   - if route instruction fails: stop immediately
   - if route instruction says stop: stop immediately
   - resolve one concrete action from available actions
   - step the env
   - record `RawStep`
   - append routed decision info to `routed_history`
   - if routed terminal action was executed: stop
   - if env reports done/truncated: stop
   - if movement produced no state change while routing: emit `stalled` failure and stop

Directed execution never degrades into probe mode.

## Route Computation

Route decisions are computed in:

- [route_execution.py](/home/zodrak/zod/src/v3_1/execution/route_execution.py)

Function:

- `route_instruction(decision_action, current_observation, info)`

### What It Uses

- planner-selected target centroid
- current avatar position from:
  - `info["avatar"]`
  - or avatar inferred from observation
- required action family

### Current Routing Logic

If target is missing:

- fail with `missing_target`

If avatar is missing:

- fail with `missing_avatar`

If `required_action_family == "click_at"`:

- emit terminal click instruction immediately

If already within terminal distance:

- if move-only or position-only:
  - return stop instruction
- if interact:
  - return terminal interact instruction

Else:

- inspect available movement actions
- keep only those that reduce Manhattan distance to target
- if none reduce distance:
  - return failed `blocked` or `unreachable`
- otherwise choose the best reducing move

So directed navigation is greedy local routing, recomputed every step.

## Directed Action Selection

Concrete directed action resolution happens in:

- `EnvWorker._select_directed_action(...)`

### Click Path

If `required_action_family == "click_at"`:

- find a click action in available actions
- attach target coordinates
- fail if unavailable

### Movement Or Terminal Interaction

Otherwise:

- match the routed desired action name first
- if terminal, allow matching by required action family
- fail if unavailable

This is intentionally strict:

- no synthetic noop
- no fallback probing
- no invented action if mapping is missing

## Option Execution Utilities

`option_execution.py` currently contains:

- `choose_probe_action(...)`
- `choose_directed_action(...)`

In the current env worker path, directed execution mainly uses:

- `route_instruction(...)`
- `EnvWorker._select_directed_action(...)`

`choose_directed_action(...)` exists as a stricter action-family helper, but the live env-worker path currently resolves directed actions inside `EnvWorker`.

## RawStep Recording

Both probe and directed episodes emit `RawStep` rows.

Each step includes:

- session/run/game ids
- episode id
- step idx
- observation
- raw action
- normalized action id/name/family
- reward
- done
- truncated
- info

The env worker then bundles the steps into a `RawEpisode`.

## Outcome Summarization

Outcome summarization happens in:

- [outcomes.py](/home/zodrak/zod/src/v3_1/execution/outcomes.py)

Function:

- `summarize_outcome(...)`

### Current Derived Signals

It computes:

- avatar positions
- initial and final distance to target
- progress
- unique positions
- noop steps
- stalled
- blocked
- route success
- route failure
- termination reason
- consequence summary

### Termination Reasons

Possible reasons include:

- `done`
- `step_budget_exhausted`
- explicit routed failure reason
- `blocked`
- `stalled`
- `noop`

### Success Rule

Current success is:

- probe:
  - env done with normal reward outcome
- directed:
  - env done and either:
    - route succeeded
    - or total reward became positive

This remains heuristic and environment-dependent.

## What The Orchestrator Actually Does

The coordinator loop in [orchestrator.py](/home/zodrak/zod/src/v3_1/runtime/orchestrator.py) currently uses execution in two stages per round:

### Probe Stage

1. build probe planning context
2. request probe planner decision
3. build probe executor request with `mode="probe"`
4. dispatch to env worker
5. analyze probe raw episode
6. merge probe delta into blackboard
7. reconcile memory from probe outcome

### Directed Stage

1. build final planning context
2. request seed decision
3. dispatch helper workers
4. request final planner decision with helper results
5. build directed executor request with `mode="directed"`
6. dispatch to env worker
7. analyze directed raw episode
8. merge directed delta into blackboard
9. reconcile memory from directed outcome

The same env worker actor is reused, but each episode currently starts with a reset.

## Current Strengths

- helper workers are context/version-scoped and non-authoritative
- helper dispatch is parallelized
- env execution is typed and actor-owned
- probe and directed execution are clearly separated
- directed routing is strict and does not collapse into probing
- terminal click/interact handling is explicit

## Current Limitations

- helper payload is still relatively thin
  - route features in helper payload are currently empty in orchestrator dispatch
- helper results only affect reranking, not earlier planner stages
- directed routing is greedy local routing, not full graph search
- env worker resets before every probe and directed episode, so persistent env actor ownership does not yet mean persistent trajectory state
- success/failure interpretation in `summarize_outcome(...)` is still heuristic
- `choose_directed_action(...)` exists but is not the main live path
- executor request metadata still reflects the older selected-candidate shape more than a dedicated execution contract

## Practical Summary

Today, helper workers are lightweight proposal producers, and the executor/env worker path is a strict two-mode trajectory runner:

- probe mode explores with biased random actions over current available actions
- directed mode repeatedly routes toward the selected target, executes only valid route-improving actions, and stops immediately on terminal action, block, stall, or failure

That gives the system a clean split:

- planner chooses intent
- helper workers advise
- executor converts intent into a typed request
- env worker turns that request into a raw episode
- analyzer and blackboard consume the result afterward

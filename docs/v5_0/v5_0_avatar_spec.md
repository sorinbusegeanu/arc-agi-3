# v5.0 Spec: Avatar Identification for ez01–ez04

## Goal

Build a new `v5_0` code line whose only runtime goal is:

- identify the controllable avatar in `ez01`, `ez02`, `ez03`, and `ez04`
- do so from short probe sequences and reset-aware evidence
- return ranked avatar candidates, not a forced full game interpretation

This version is intentionally narrow. It is not a full agent.

## Primary success criterion

For each of `ez01`, `ez02`, `ez03`, `ez04`, after one bootstrap probe session:

- produce `top_k_avatar_candidates`
- rank the true avatar at rank 1 in the normal case
- include confidence and evidence for the chosen candidate
- never depend on fixed color assumptions like `avatar=1`

## Explicit non-goals

Do not implement any of these in v5.0:

- full POI discovery
- full mechanic classification
- full traversable/blocking map inference
- HUD semantics
- planner integration
- hypothesis engine
- broad orchestrator stages
- learned model training
- full game solving

The only required output is robust avatar identification for the four easy movement games.

## Design principle

v4_5 failed because discovery tried to commit too early to too many symbolic outputs. v5.0 must be narrower and evidence-driven.

v5.0 must use this sequence:

1. reset the game
2. run a fixed bootstrap probe sequence
3. collect transition records
4. generate top-k avatar candidates from controlled movement evidence
5. score and rank candidates
6. emit an artifact-rich report

No later stage may assume POIs or mechanic type.

## Probe protocol

### Required default probe sequence

Use a short deterministic sequence:

- `LEFT`
- `RIGHT`
- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`

This is enough for the four easy movement games.

### Probe requirements

For each step record, store:

- `step_index`
- `action`
- `pre_frame`
- `post_frame`
- `invalid_action`
- `blocked_action`
- `terminal`
- `levels_completed_before`
- `levels_completed_after`
- `reward_before` if available
- `reward_after` if available

### Reset discipline

Bootstrap must always start from a clean reset.

v5.0 does not need multi-reset logic yet, but the data structures must allow it later.

## Runtime outputs

v5.0 must emit:

### 1. Avatar candidate list

For each candidate:

- `candidate_id`
- `bbox`
- `center`
- `score`
- `support_step_indices`
- `support_actions`
- `observed_motion_vectors`
- `direction_agreement_score`
- `shape_consistency_score`
- `track_consistency_score`
- `value_histogram_pre`
- `value_histogram_post`
- `failure_flags`

### 2. Selected avatar result

- `selected_candidate_id`
- `selected_bbox`
- `selected_center`
- `confidence`
- `failure_reason` or `null`
- `ranking_margin_to_second`

### 3. Diagnostics

- per-step candidate counts
- per-step top scores
- total candidate count
- total track count
- dropped candidate reasons
- ambiguous ranking flag
- no-motion flag
- all-blocked flag

### 4. Artifacts

- bootstrap transition JSON
- avatar candidate report JSON
- summary JSON
- optional PNG montage of pre/post probe frames

## Required architecture

Create a new package namespace:

- `src/v5_0/`

Required modules:

- `src/v5_0/contracts/avatar_types.py`
- `src/v5_0/bootstrap/probe_plan.py`
- `src/v5_0/bootstrap/probe_runner.py`
- `src/v5_0/avatar/candidate_extractor.py`
- `src/v5_0/avatar/track_builder.py`
- `src/v5_0/avatar/scorer.py`
- `src/v5_0/avatar/service.py`
- `src/v5_0/io/artifact_writer.py`
- `src/v5_0/runtime/run_avatar_bootstrap.py`
- `src/v5_0/cli.py`

Keep the code small and single-purpose.

## Internal pipeline

### Step 1: Probe runner

Input:

- game id
- level id
- runtime/session adapter
- fixed probe sequence

Output:

- ordered tuple of transition records

### Step 2: Candidate extraction

For each probe step:

- compute changed cells between `pre_frame` and `post_frame`
- split changed cells into connected components
- derive per-component candidate features

Per-component features must include:

- bbox
- area
- pre center
- post center
- observed dx/dy
- action issued
- whether action was blocked
- pre non-background cells
- post non-background cells
- pre/post value histograms

### Step 3: Per-step scoring

Score each candidate on:

- directional agreement with issued action
- movement magnitude consistency
- shape consistency from pre to post
- compactness / plausibility

Blocked actions must not be treated as strong negative evidence.

### Step 4: Track building

Link per-step candidates into short tracks across the probe sequence.

Track linking must use:

- spatial continuity
- action consistency
- shape/value consistency

Track output must include:

- support step indices
- support actions
- aggregate confidence
- candidate bbox/center summary
- candidate value set summary

### Step 5: Final ranking

Produce ranked candidates.

Final ranking must prefer:

- candidates supported by multiple steps
- candidates whose observed motion aligns with issued actions
- candidates with stable local shape/value evidence
- candidates with better separation from alternatives

### Step 6: Failure handling

Return explicit failure reasons for:

- `no_moving_candidate`
- `ambiguous_avatar`
- `insufficient_support`
- `all_actions_blocked`
- `invalid_probe_capture`

Do not fabricate a winner when evidence is weak.

## Acceptance criteria

### Functional

For each of `ez01`, `ez02`, `ez03`, `ez04`:

- bootstrap completes without crash
- candidate report is produced
- true avatar is rank 1 in the normal path
- selected result has non-null bbox and center
- no fixed color ids are used for avatar detection

### Quality

- candidate ranking is deterministic for the same seed and same probe sequence
- the report explains why the top candidate won
- ambiguous cases return explicit ambiguity, not silent fallback

## Required tests

Create tests for:

- changed-cell component extraction
- per-step candidate scoring
- track linking
- ambiguity handling
- no-motion handling
- end-to-end bootstrap on `ez01`
- end-to-end bootstrap on `ez02`
- end-to-end bootstrap on `ez03`
- end-to-end bootstrap on `ez04`

Also add a regression test that ensures no avatar color constant is used.

## What may be mined from v4_5

v5.0 must not be a copy of v4_5. It may mine only narrow reusable parts.

### Safe to mine with adaptation

#### 1. Bootstrap sequence planning

Mine logic from:

- `v4_5/bootstrap/bootstrapSequenceBuilder.py`
- `v4_5/config/bootstrapConfig.py`
- `v4_5/contracts/bootstrapMediaTypes.py`

Use only for:

- representing the probe plan
- fixed action sequence configuration
- bootstrap bundle structure ideas

Do not carry over broad discovery coupling.

#### 2. Probe capture / export plumbing

Mine carefully from:

- `v4_5/bootstrap/bootstrapCapture.py`
- `v4_5/bootstrap/pngExporter.py`
- `v4_5/bootstrap/videoExporter.py`
- `v4_5/bootstrap/runtimeFactory.py`

Use only for:

- collecting pre/post frames
- writing image/video artifacts
- runtime/session construction patterns

Do not carry over HUD or POI analysis dependencies.

#### 3. Avatar data contracts

Mine and simplify from:

- `v4_5/contracts/avatarTypes.py`
- `v4_5/contracts/boardObject.py`

Use only for:

- candidate/result dataclasses
- bbox/center/value summaries

Do not keep fields that imply a full board object ontology.

#### 4. Avatar candidate extraction ideas

Mine algorithmic pieces from:

- `v4_5/perception/board_builder/avatarExtractor.py`
- `v4_5/agents/avatarDetector.py`

Use only for:

- changed component extraction
- direction agreement scoring
- shape similarity scoring
- track support logic
- diagnostics structure

This is the most important donor.

But rewrite it into isolated v5.0 modules:

- candidate extraction
- track building
- scoring
- final service

Do not keep it inside a broad perception/discovery system.

#### 5. Artifact logging patterns

Mine lightly from:

- `v4_5/logging/agentLogger.py`
- `v4_5/runtime/resultBuilder.py`
- `v4_5/cli/outputPaths.py`

Use only for:

- output directory structure
- JSON artifact naming
- simple logging helpers

### Do not mine into v5.0 core logic

These files must not be used as the basis of the new runtime logic:

- `v4_5/agents/discoveryAgent.py`
- `v4_5/agents/orchestratorAgent.py`
- `v4_5/adapters/stateAdapter.py`
- `v4_5/perception/board_builder/poiExtractor.py`
- `v4_5/perception/board_builder/backgroundExtractor.py`
- `v4_5/perception/board_builder/hudExtractor.py`
- `v4_5/agents/hypothesisAgent.py`
- `v4_5/plugins/*`
- `v4_5/perception/service.py`
- `v4_5/perception/board_builder/builder.py`
- `v4_5/perception/board_fusion/*`

Reason:

- too broad
- too coupled
- includes premature symbolic commitments
- includes brittle frame semantics

### Explicitly forbidden carryover

Do not carry over the frame-value assumptions from:

- `v4_5/adapters/stateAdapter.py`

Specifically forbidden assumptions:

- avatar = `1`
- poi = `2`
- click = `3`
- hazard = `9`

v5.0 avatar identification must be movement-evidence-based, not value-id-based.

## Implementation notes for Codex

### Keep the scope strict

Do not add:

- POI detection
- HUD detection
- traversability inference
- mechanic typing
- planner hooks
- LLM hooks
- neural components

### Prefer this decomposition

- `probe_runner.py` handles environment interaction only
- `candidate_extractor.py` handles per-step changed components only
- `track_builder.py` links candidates across steps only
- `scorer.py` contains scoring and ranking only
- `service.py` assembles the final report only

### Required output files per run

Write under one run directory:

- `bootstrap_transitions.json`
- `avatar_candidates.json`
- `avatar_summary.json`
- `probe_montage.png` if enabled

## Minimal milestone plan

### Milestone A

Single function that consumes transition records and returns ranked avatar candidates.

No runtime integration yet.

### Milestone B

Hook the function to a bootstrap probe runner for `ez01` only.

### Milestone C

Generalize to `ez01..ez04`.

### Milestone D

Add artifact writing and deterministic tests.

## Final deliverable for v5.0

A small deterministic subsystem that, for `ez01..ez04`, can answer:

- which object is most likely the controllable avatar
- why it was selected
- how strong the evidence is

Nothing else is required in v5.0.

Module spec: Executable_Hypothesis_Engine
0) Scope and non-goals

Scope: Track a ranked set of executable hypotheses and update their confidence using observed step transitions compiled into a canonical TransitionEvent.
Non-goals: Executing environment actions, long-horizon planning, learning/training, generating coordinate proposals (handled by Test Selector).

1) Inputs and data contracts
1.1 Required inputs

fp_reports[] (windowed) and/or per-step compiled TransitionEvent records (see §1.3)

action_schema (to normalize action keys)

ctx: game_id, seed, step_idx

1.2 Optional inputs

simple_explorer_report / full_explorer_report (for richer observed transitions)

rule_proposer_report (initial hypothesis seeds) 

04_Rule_Proposer

memory_evidence (read-only, future; not required for Gap 3)

1.3 Normalized internal representation

Codex must define a versioned, canonical struct:

TRANSITION_EVENT_V1

state_hash_before, state_hash_after

action_key (simple or coord)

event_signature_histogram (from FP_Analyst taxonomy)

delta_metrics: changed_cells, changed_bbox, palette_added/removed

meta_delta: available_actions_before/after, optional reward, optional terminal

TransitionEvent compilation is deterministic and must re-use FP_Analyst hashing and signature extraction. 

02_Simple_Explorer

2) Outputs (stable + machine-readable)

Executable_Hypothesis_Engine returns:

2.1 hypotheses[] (ranked)

Each hypothesis:

hypothesis_id (stable)

name, description (deterministic template, 1–3 sentences)

program_v1 (executable primitive composition; see §3)

params (discrete)

confidence (0–1 deterministic)

fit_stats:

transitions_scored

avg_likelihood

falsified boolean

predictions[] (testable statements in terms of TransitionEvent fields)

2.2 run_summary

window used, warnings, truncation notes (top-K)

3) Core hypothesis representation (executable)
3.1 Primitive program skeleton

A hypothesis program is a fixed skeleton (to avoid combinatorial explosion):

Intent -> Gates* -> Effects+ -> MetaEffects*

Intent maps action_id into an intent label (not assumed true; just a handle)

Gates constrain when effects apply (e.g., requires coord action; requires “selection exists” marker)

Effects predict event signatures + coarse delta bins

MetaEffects predict meta deltas (available actions changes, terminal/reward when present)

3.2 Primitive library (initial)

Must be small and game-mechanic oriented (not ARC transforms):

movement-like signature predictors

click/paint/toggle signature predictors

“global tick” signature predictors (e.g., gravity-like / step-advance)

undo/reversal signature predictor

meta gating primitives (mode/phase abstractions based on stable meta keys)

All primitives output only TransitionEvent-compatible predictions (no full grid reconstruction).

4) Scoring and posterior update (deterministic)

For each observed TransitionEvent e and hypothesis h:

compute pred = h.predict(features_before, action_key)

compute likelihood as weighted agreement over:

dominant event signature match

noop vs effect agreement

changed_cells bin agreement

bbox directionality/coarseness agreement

meta_delta agreement (available_actions, terminal/reward if present)

Update confidence by deterministic aggregation over last N scored transitions, with falsification if any hard constraint violated.

5) Interfaces and integration points

Expose:

seed_hypotheses(rule_proposer_report, cfg) -> hypotheses[]

update(hypotheses, transition_events, cfg) -> hypotheses[]

predict_all(hypotheses, state_features, candidate_actions) -> predicted_events

Consumers:

Test Selector consumes hypotheses[] + predict_all(...)

Planner may consume confidence and “best hypothesis id” as a prior

6) Configuration (explicit defaults)

topK_hypotheses = 8

window_N = 12 (scored transitions considered)

weights:

w_sig = 0.50

w_noop = 0.20

w_delta = 0.20

w_meta = 0.10

hard_falsify = true

All weights must be logged and deterministic. 

04_Rule_Proposer

7) Logging and failure handling

If TransitionEvent missing meta keys, set meta agreement to “unknown” and renormalize weights deterministically.

Always keep unknown.mechanic hypothesis alive (never falsified), ranked last unless all scores ≤ 0. 

04_Rule_Proposer

8) Deliverables Codex should implement (files/classes)

Executable_Hypothesis_Engine implementation

dataclasses: TransitionEventV1, ExecutableHypothesisV1, HypothesisEngineReport

primitive library: primitive_program_v1.py

minimal CLI:

--agent executable_hypothesis_engine --input_fp <...> [--simple <...>] [--full <...>] --outdir <...>

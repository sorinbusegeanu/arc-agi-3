Mechanic_Synthesizer
0) Scope and non-goals

Scope: generate candidate ExecutableHypothesisV1 programs from recent TransitionEventV1 sequences and current FP_Analyst features, using bounded deterministic search over a small mechanic-primitive DSL.
Non-goals: updating confidences/posteriors (engine does that), selecting probe actions (test selector), executing environment actions, any gradient-based learning/training.

1) Inputs and required contracts
1.1 Required inputs

events[]: recent TransitionEventV1 (window N, default 12)

fp_current: current FP_Analyst normalized features (objects, masks, hotspots, meta extraction)

available_actions_current (from meta)

existing_hypotheses[] (current active set from Hypothesis Engine)

1.2 Optional inputs

full_explorer_coord_hints (if present)

memory_evidence (future; not required for Gap 3)

1.3 Assumed invariants

TransitionEventV1 compilation is deterministic and includes: dominant signature(s), noop/effect, delta metrics, and available_actions deltas when present.

Action keys are normalized, including coord action identity for ACTION6(x,y).

2) Outputs (stable, machine-readable)

Mechanic_Synthesizer returns a MechanicSynthesisReport with:

2.1 candidates[] (ranked)

Each candidate is a complete ExecutableHypothesisV1:

hypothesis_id (deterministic, derived from program hash + param hash)

name (short)

description (1–2 sentences)

program_v1 (primitive composition skeleton)

params (discrete)

origin: "synth_v1"

priority_score (for insertion into top-K by engine)

2.2 diagnostics

why synthesis triggered (see §3)

which action families were explained/unexplained

search budget usage, pruning reasons

3) When synthesis triggers (deterministic policy)

Mechanic_Synthesizer is invoked by Orchestrator (or Hypothesis Engine) when any is true:

best_avg_likelihood < L_min over last N events (default 0.35)

active_nonfalsified_count == 0

ambiguity_high: top-2 hypotheses within Δ < 0.05 for M consecutive steps (default M=4)

unexplained_signature_rate > R_max (default 0.50), where “unexplained” means predicted signature is unknown or mismatched for the winning hypothesis.

4) Synthesis approach (bounded, deterministic)
4.1 Decompose the problem by action family

Maintain per-action-type “micro-models” inferred from events:

For each action_id in {ACTION1..5, ACTION7}:

collect events where that action executed

derive a dominant signature profile (e.g., mostly noop, mostly translation-like, mostly toggle-like)

derive delta bins (changed_cells bin, bbox size bin)

derive any meta effects (available_actions changes correlated with this action)

For ACTION6:

treat separately: events include coord; infer whether effects correlate with coord position class (on object / on background / edge / hotspot).

This yields a deterministic “action semantics draft” used to assemble programs.

4.2 Primitive program skeleton to synthesize

All synthesized programs must fit the canonical skeleton:

Intent(action_id) -> Gates* -> Effects+ -> MetaEffects*

Mechanic_Synthesizer fills this skeleton using the inference signals above.

4.3 Candidate construction rules (v1)

Mechanic_Synthesizer must generate candidates from a small menu of templates, but with data-driven parameters (so it is not a fixed catalog ceiling). For each action family, instantiate at most B candidates (beam width), default B=3.

Template families (examples; must be encoded as primitive programs):

Directional move model

ACTION1–4 map to dirs in one of the 4! permutations (bounded set via pruning)

Effect: TranslateAvatar(dir) OR TranslateCursor(dir) (distinguished by delta bbox size bin)

Gates: blocked-by-occupancy? (from noop vs non-noop conditional patterns)

Click-select / click-apply model

ACTION6 sets selection when clicking on object-like pixels; subsequent ACTION5 applies effect

Gates: RequiresSelection

Effects: PaintAt(selection) OR ToggleAt(selection) OR SpawnAt(selection)

Tool/mode cycling model

ACTION5 changes mode; subsequent actions’ signatures differ by mode

Introduce latent mode_id (small: 2–3) only if evidence supports mode dependence

Global tick model

Any action causes a consistent nonlocal signature (gravity/advance)

Effect: GlobalTick() + optional local intent effect

The actual list should be limited and explicitly versioned as MECH_SYNTH_TEMPLATESET_V1.

4.4 Pruning and determinism

Hard cap total candidates per invocation: max_candidates_total (default 12)

Deterministic ordering:

higher explained-event coverage

higher fit estimate (see §5)

simpler program (fewer gates/effects)

hypothesis_id lexical

No random sampling.

5) Fast pre-scoring (before engine)

Mechanic_Synthesizer must compute an internal priority_score for each candidate using only the provided events[]:

simulate/predict PredictedEvent for each event’s (state_features_before, action_key)

compute the same agreement components as Hypothesis Engine but cheaper (no posterior smoothing)

priority_score = mean likelihood over window minus complexity penalty

This does not replace Hypothesis Engine scoring; it only ranks synthesized candidates for insertion.

6) Interfaces and integration points
6.1 Public API

synthesize(events, fp_current, available_actions_current, existing_hypotheses, cfg, ctx) -> MechanicSynthesisReport

6.2 Consumers

Orchestrator: calls synthesizer when trigger conditions met; passes candidates to Hypothesis Engine as additional seeds.

Executable_Hypothesis_Engine: merges candidates, scores them, and updates top-K.

Discriminating_Test_Selector: benefits immediately because it can compare more diverse executable hypotheses.

7) Configuration (explicit defaults)

window_N = 12

L_min = 0.35

R_max = 0.50

ambiguity_delta = 0.05

ambiguity_M = 4

beam_per_action_family = 3

max_candidates_total = 12

max_mode_states = 3

complexity_penalty = 0.02 * (#gates + #effects + #meta_effects)

All values logged and deterministic.

8) Logging and failure handling

If insufficient events (<3), emit only “action semantics draft” candidates (e.g., noop-model + directional-move permutations pruned by observed bbox bins).

Always emit a fallback candidate:

Hypothesis: UnknownMechanic (predicts unknown signature, noop/effect uncertain), so pipeline never empty.

9) Deliverables Codex should implement

Module class: Mechanic_Synthesizer

dataclasses: MechanicSynthesisReport, SynthesisCandidate, ActionSemanticsDraft

deterministic program hashing/id generation

optional CLI for debugging:

--agent mechanic_synthesizer --events <...> --fp <...> --outdir <...>

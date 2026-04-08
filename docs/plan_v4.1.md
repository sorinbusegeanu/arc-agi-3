# Plan: Hybrid `v4` + Local-LLM ARC AGI 3 Agent

## Summary

Use `v4` as the deterministic execution kernel and add a local-LLM learning layer above it. With your budget profile:

- local inference on an `RTX 5070 Ti`
- per-call latency of seconds is acceptable
- initial budget target: about 20 minutes per game

the best design is a hybrid agent, not a pure symbolic agent and not an LLM-only agent.

The LLM should be used for abstraction, retrieval, hypothesis drafting, and plan ranking. The deterministic `v4` stack should remain responsible for:

- authoritative observation and action handling
- state parsing and belief tracking
- bounded experiment execution
- exact-family solving where available
- verification and fail-closed behavior

This gives you transfer across games without making the runtime brittle.

## Key Changes

### 1. Split the system into three layers

#### A. Deterministic execution kernel
Keep current `v4` runtime as the only environment-facing control loop:

- `agentContract`, `runtime`, `state`, `policy`
- `belief`, `hypothesis`, `temporal`, `composition`
- exact family solvers as specialist tools and teachers

#### B. Durable learning layer
Add persistent cross-session stores:

- `EpisodeStore`
  - traces, failures, solved runs, experiment logs
- `MechanicStore`
  - promoted mechanic rules with evidence and contradiction counts
- `SkillLibrary`
  - reusable skill templates with preconditions, bindings, and verification rules
- `CaseIndex`
  - retrieval over prior games, episodes, mechanics, and skills

#### C. Local-LLM advisor layer
Add one bounded advisor service that only consumes structured summaries and only emits structured proposals:

- mechanic hypothesis drafts
- experiment choices
- skill template drafts
- retrieval rebinding proposals
- plan rankings and failure analyses

The LLM never writes authoritative state and never directly controls primitive actions without deterministic verification.

### 2. Use the local LLM at four specific points

#### A. New-game bootstrap
On first contact with a game:

- summarize the observation/action vocabulary
- retrieve nearest prior cases
- ask the LLM for:
  - likely mechanic families
  - likely useful experiments
  - candidate reusable skills to try first

Output should be a small ranked set of symbolic hypotheses, not free-form strategy text.

#### B. Online ambiguity resolution
During play, call the LLM only when uncertainty is high:

- repeated no-progress cycles
- several competing mechanic hypotheses
- hidden-state ambiguity with multiple legal low-risk probes
- multiple retrieved skills that partially match

The LLM should rank or edit a small candidate set that the deterministic planner already prepared.

#### C. Skill induction after each episode
After a run:

- segment the trace into event atoms and subgoal shifts
- give the LLM a grounded summary
- ask it to propose:
  - reusable skill templates
  - mechanic candidates
  - failure signatures
  - transfer tags

All promoted skills must include explicit preconditions and a verification check.

#### D. Offline consolidation across many episodes
Periodically run a larger consolidation job:

- cluster similar traces
- merge overlapping mechanic candidates
- compress repeated successful behavior into generalized symbolic skills
- update retrieval embeddings and case summaries

This is where larger local models pay off most.

### 3. Make the LLM output constrained symbolic artifacts

Do not use free-form text as the core interface. Require structured records:

- `MechanicHypothesisDraft`
  - rule name, scope, triggering evidence, expected effects, confidence
- `SkillTemplateDraft`
  - intent, preconditions, role bindings, action sketch, verification rule, failure modes, transfer tags
- `ExperimentChoice`
  - target uncertainty, expected distinguishing outcome, bounded prefix
- `RetrievalRebind`
  - prior skill id, current object bindings, reasons for match
- `FailureAnalysis`
  - probable cause, missing evidence, suggested next experiments

Every field should be grounded in known objects, actions, events, or evidence refs.

### 4. Add a budget-aware control policy

Given your budget, use three modes:

#### Fast path
No LLM call.

Use:

- exact solver if a family is recognized and implemented
- deterministic retrieval-only skill execution
- bounded hand-built experiment templates

#### Advisor path
One small LLM call.

Use when:

- first game contact
- ambiguity persists after deterministic filtering
- loop risk rises
- repeated failure buckets recur

Budget:
- a few seconds per call
- limit to a small number of calls per episode
- reuse cached summaries aggressively

#### Consolidation path
Offline or post-episode larger LLM call.

Use for:
- skill extraction
- mechanic abstraction
- cross-game transfer updates
- episode compression

This should consume most of the 20-minute budget, not the per-step loop.

### 5. Add a retrieval-first transfer pipeline

The main transfer path should be:

1. encode current game into a structured summary
2. retrieve similar prior episodes, mechanics, and skills
3. ask the LLM to rebind the best-matching skills to current roles
4. verify those bindings against current parsed state
5. execute only short certified prefixes
6. update the case library from outcomes

This gives you transferable behavior across games without retraining a full policy every time.

### 6. Keep learning symbolic first, learned ranking second

Recommended priority order:

1. trace storage and event extraction
2. mechanic hypothesis promotion
3. skill template induction
4. retrieval and rebinding
5. bounded experiment planning
6. LLM ranking over candidate experiments and skills
7. optional learned scorers or embeddings
8. only later: learned world models or policy learning

This matches your stated goal of learning patterns and skills that transfer across games.

## Public Interfaces / Types To Add

Add a durable-learning API with these records:

- `EpisodeArtifact`
  - episode id, game id, outcome, trace refs, solved flag, failure bucket
- `EventAtom`
  - event kind, actors, targets, local context, evidence refs
- `MechanicHypothesisRecord`
  - hypothesis id, rule family, scope conditions, support refs, contradiction refs, promotion status
- `SkillTemplateRecord`
  - skill id, intent, preconditions, bindings, action sketch, verification predicate, failure signatures, transfer tags
- `CaseSummaryRecord`
  - game summary, structural features, linked mechanics, linked skills, solved/unsolved stats
- `LLMProposalRecord`
  - prompt context hash, structured proposal payload, later acceptance/rejection outcome
- `RetrievalMatchRecord`
  - query features, matched cases, chosen rebind, final usefulness

Add one local-LLM adapter interface:

- `LocalAdvisorModel`
  - `draft_mechanics(context) -> MechanicHypothesisDraft[]`
  - `draft_skills(context) -> SkillTemplateDraft[]`
  - `rank_experiments(context, candidates) -> ranked candidates`
  - `rebind_skill(context, retrieved_skill) -> RetrievalRebind`
  - `analyze_failure(context) -> FailureAnalysis`

## Test Plan

### Deterministic safety tests

- LLM proposals never bypass deterministic action legality checks
- LLM proposals never become authoritative state directly
- invalid or malformed LLM outputs are rejected safely
- exact-family solver regressions remain green

### Learning and transfer tests

- repeated play on one game improves via stored skills and mechanics
- a learned skill is reused on a structurally related game with different surface details
- retrieval plus rebinding beats a cold-start baseline on held-out games
- contradiction evidence suppresses bad mechanic proposals over time

### Budget tests

- online LLM calls stay within configured per-episode limits
- cached retrieval and summaries prevent repeated expensive calls
- a 20-minute per-game budget is mostly spent on:
  - first-contact reasoning
  - ambiguity resolution
  - post-episode consolidation
- per-step loop remains responsive because the LLM is not called on every action

### Offline consolidation tests

- multiple solved episodes produce merged skill templates
- duplicate mechanic candidates are merged instead of proliferating
- case retrieval remains stable after library growth

## Assumptions And Defaults

- optimize for a hybrid symbolic-plus-local-LLM agent
- `v4` remains the authoritative executor and verifier
- local LLM use is allowed and should be exploited primarily for abstraction and transfer
- target budget is roughly 20 minutes per game for the first version
- seconds-level model latency is acceptable, but only for sparse high-value calls
- prioritize symbolic skill learning and mechanic induction before RL or policy learning
- exact family solvers remain specialist teachers/fallbacks, not the final general architecture

## Practical Build Order

1. durable episode store and event extraction
2. case retrieval and embeddings over prior episodes/skills
3. local-LLM adapter with constrained structured outputs
4. post-episode skill/mechanic induction pipeline
5. online first-contact advisor for new games
6. bounded ambiguity-resolution advisor during play
7. retrieval-based skill rebinding and certified execution
8. consolidation jobs that merge and promote durable skills
9. later: learned scorers/world models if the symbolic-transfer path plateaus

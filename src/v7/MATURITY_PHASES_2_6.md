# v7 maturity phases 2-6

This document defines the logical behavior implemented after Phase 1. It preserves v7's immutable-generation/single-writer architecture while restoring mature v6 cognition mechanisms without importing v6 code or compatibility semantics.

## Phase 2 — abstraction maturity

### M2 transformation family

M2 identity is action-independent. The same normalized transformation may be produced by different actions and still belongs to the same family.

`M2 = transformation_signature`

Actions remain evidence attached through M1 members and indexes; they are not part of the family identity.

### Explicit carrier

Carrier is an explicit M3 subtype representing a recurring structural locus/entity pattern that participates in transformations. Carrier evidence is accumulated across episodes and must be supported by multiple observations and multiple contexts before it can support a functional role.

Carrier fitness uses empirical recurrence, contextual breadth, predictive discrimination and compression. Carrier is not indexed directly as a policy role.

### Contextual role instance and functional role

A contextual role instance preserves the concrete relation:

`carrier + transformation family + action + context`

A functional role abstracts multiple contextual instances according to their shared functional signature. Functional identity is independent of the exact context and carrier and captures transformation/effect class, future-option direction, terminal effect and change behavior.

Only the functional role is exposed to normal role-based cognition. Contextual role instances remain provenance/evidence nodes.

## Phase 3 — validated concepts and relational world models

### Concept lifecycle

M4 concepts move through evidence states:

1. candidate
2. structurally supported
3. transfer candidate
4. transfer validated
5. trusted
6. transfer rejected

Behavioral influence is status-weighted. Rejected concepts have zero influence. Candidate concepts have weak influence. Transfer-validated/trusted concepts have strong influence.

M5 derivation requires transfer-validated or trusted concepts.

### M5a transition model

M5a preserves the v7 empirical transition model:

`prior concepts + action -> current concepts`

### M5b relational model

Repeated M5a transitions additionally derive typed semantic relations between concepts:

- precedes
- enables
- constrains
- preserves
- opens options
- closes options
- causes progress
- causes regression
- shared outcome

These relations are stored as typed canonical graph edges and as M5 relational-model memories, then indexed back to the contexts/actions that provided the supporting transition evidence.

## Phase 4 — context intelligence

Decision context is a bounded lattice:

- C0: general context
- C1: behavioral history only
- C2: structural state only
- C3: behavioral history + structural state
- C4: exact-state temporal specialization

The behavioral context allows transfer across visually different states. Structural context protects against over-generalization. Combined context joins both. Exact context is a contradiction-driven specialization, not the default identity.

Backoff is evidence-driven. The policy prefers the least-specific supported context unless a more-specific context has enough evidence and either resolves contradiction or materially improves prediction confidence. Exact context is promoted only when coarse evidence is contradictory or the exact context has become independently well supported.

Canonical episode expansion omits C4 on ordinary low-surprise transitions; C4 is retained for prediction violations, terminal evidence, or explicit specialization.

## Phase 5 — developmental memory policy

Development stage is inferred from evidence already present in the immutable generation rather than from epoch number:

- CONTROL: no stable M1
- CONTINGENCY: stable M1, no reusable M2
- ABSTRACTION: M2 exists, no functional M3
- TRANSFER: functional M3 exists, no transfer-validated M4
- PLANNING: validated M4 exists, no executable M6
- STRATEGY: executable M6 exists

Stage changes behavior without changing canonical scientific identity:

- exploration is high in early stages and falls as validated knowledge appears;
- planning depth grows with maturity;
- exact-state specialization becomes harder once reusable abstractions exist;
- replay is focused on the memory levels most useful for the current stage;
- abstraction candidate budgets grow with evidence maturity.

A strategy match still overrides stage-level exploration when its confidence and risk gates pass.

## Phase 6 — scientific validation and ablation

All restored mechanisms are independently ablatable at the acting-policy boundary:

- persistent planning
- executable strategy reuse
- functional-role evidence
- relational M5 evidence
- developmental policy

The experiment writes cognition metrics in addition to raw game totals:

- repeat solution rate
- solution retention rate across epochs
- mean successful trajectory length
- steps to rediscover a previously solved game/level
- cross-game transfer success rate
- failure repetition rate
- per-epoch solved-game counts

Ablation helpers generate deterministic masks and comparison specifications. Ablations keep the same memory substrate unless the experiment explicitly asks to disable a derivation mechanism; this isolates the causal contribution of a policy mechanism from data-collection differences.

## Required invariants

- no production v7 import from v6;
- one canonical writer;
- immutable published generations;
- worker-local learning remains local;
- candidate searches remain bounded;
- higher-level memories cannot gain strong policy influence merely by existing;
- transfer evidence remains retrospective and separate from transfer prior;
- action selection remains deterministic for equal RNG state and equal immutable generation.

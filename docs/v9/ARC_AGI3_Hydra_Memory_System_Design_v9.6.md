# ARC-AGI-3 Hydra Memory System Design v9.6

**Version:** v9.6  
**Status:** Independent target design  
**Research contract:** `Research_problem_statement_v0631.md`  
**Design predecessors:** Hydra v9.5, v9.4, v9.3, v8.55/v8.56  
**Purpose:** define Hydra v9 as a clean, independently runnable implementation of the research architecture, extended with recursive deliberation for iterative internal refinement before action.

---

## 1. Architectural boundary

Hydra v9 is a new implementation under `src/v9/`.

Its production runtime has **zero imports from `v8`**.

```text
src/v9  -> Python/runtime dependencies
        -> environment libraries
        -> no src/v8 dependency
```

v8 is a predecessor and reference implementation. Its algorithms, tests, measured behavior, and data formats may be studied and selectively reimplemented in v9, but v8 code is not part of the v9 runtime dependency graph.

The boundary applies to:

- runtime and CLI;
- memory model and graph;
- proposal/mutation machinery;
- environment contracts;
- action selection;
- developmental cognition;
- lifecycle;
- persistence;
- research evidence and reports;
- symbolic grounding;
- tests.

Compatibility with v8 is isolated under `src/v9/migration/`.

---

## 2. Design objective

v9 learns from heterogeneous interactive environments, passive observations, ordered symbolic observations, interventions, structural recurrence, and causal transfer.

The developmental hierarchy is:

```text
multimodal observations + interventions
        ↓
M0 grounded episodes
        ↓
M1G grounded contingencies
        ↓
M1N normalized observable relations
        ↓
M2 recurrence/transformation families
        ↓
M3 carriers and functional roles
        ↓
M4 concept candidates / validated concepts
        ↓
M5 consequence structures / world models
        ↓
M6 outcome-equivalence abstractions
        ↓
M7 strategies / procedures
```

The implementation must preserve the distinction between:

```text
implemented mechanism
!=
empirically validated research claim
```

---

## 3. Core invariants

1. Canonical cognition is RAM-authoritative.
2. Raw experience creates only M0/M1 evidence.
3. M2-M7 arise from developmental operators over canonical lower memory.
4. Canonical identity is independent of mutable support, lifecycle, lineage, provenance, and parent connectivity unless explicitly declared as invariant identity content for that memory level.
5. Higher memory retains auditable grounded provenance.
6. Structural similarity proposes candidates; causal held-out evidence validates transfer.
7. Symbol IDs contain no supplied task semantics.
8. Interaction remains the causal grounding authority.
9. Native actions remain environment-local.
10. Memory, mutation, structural search, replay, validation, and symbol processing remain bounded.
11. Scientific configuration is immutable within an experimental condition.
12. v9 runtime correctness cannot depend on v8 being installed.
13. Adding a new environment or modality must not require domain-semantic changes to M2-M7.
14. Scientific evaluation observes cognition but does not silently change the cognitive rules during the same experimental condition.

---

## 4. Independent system architecture

```text
ENVIRONMENTS / DATA SOURCES
        ↓
src/v9/environments + src/v9/modalities
        ↓
Multimodal Event Timeline
        ↓
M0/M1 bounded ingestion
        ↓
Additive / stateful proposal queues
        ↓
Deterministic partition reducers
        ↓
Canonical RAM graph
        ↓
Immutable/versioned read views
        ↓
Developmental cognition
        ↓
Mutation proposals
        ↓
Read-set/version validation
        ↓
Atomic canonical mutation
```

Scientific evidence and reports consume immutable decision/evidence snapshots.

---

## 5. Target package organization

```text
src/v9/
    __init__.py
    __main__.py
    cli.py

    runtime/
        runtime.py
        config.py
        rings.py
        partitions.py
        reducers.py
        read_view.py
        publication.py
        snapshot.py
        lifecycle.py

    memory/
        model.py
        identity.py
        relations.py
        m0_episode.py
        m1_grounded.py
        m1_normalized.py
        m2_family.py
        m3_role.py
        m4_concept.py
        m5_consequence.py
        m6_outcome.py
        m7_strategy.py
        provenance.py
        residency.py

    environments/
        contract.py
        registry.py
        schemas.py
        arc/
        gym/
        chess/
        sudoku/
        synthetic_symbolic/
        babyai/
        alfred/

    modalities/
        contract.py
        symbols/
            vocabulary.py
            codec.py
            stream.py
            normalizer.py

    cognition/
        prediction.py
        compression.py
        roles.py
        similarity.py
        correspondence.py
        transfer.py
        concepts.py
        future_options.py
        world_model.py
        outcomes.py
        strategies.py
        planning.py
        replay.py
        grounding.py
        isf.py
        developmental_stage.py

    mutation/
        proposals.py
        versions.py
        read_sets.py
        transactions.py
        lineage.py
        context.py

    research/
        evidence.py
        experiments.py
        hypotheses.py
        grounding_h16.py
        reports.py

    migration/
        v8_snapshot.py
        v8_evidence.py

    tests/
```

No module outside `migration/` may import `v8`.

---

# 6. Environment and sensory boundary

Every interactive environment implements the v9-owned `EnvironmentCognitionAdapter` contract:

```text
identity()
reset()
observe()
step(native_action)
available_actions()
observation_schema()
action_schema()
encode_observation()
encode_action()
transition(before, after)
boundary_event()
optional_micro_trace()
optional_symbol_stream()
```

Canonical provenance uses:

```text
EnvironmentFamilyId
EnvironmentTypeId
EnvironmentConfigId
EnvironmentInstanceId
EpisodeId
```

Native action identity is scoped to the environment/action schema.

No global action namespace is assumed.

The adapter may expose declared non-semantic structure such as coordinates, adjacency, ordering, topology, or scalar domains, but not task-semantic labels.

---

# 7. Multimodal timeline

The event model preserves macro actions and passive observations as distinct causal events.

```text
world observation O_t
symbol observations S_t,*
        ↓
actor action A_t
        ↓
passive world/symbol observations
        ↓
settled world observation O_t+1
```

Only actor-selected interventions are actions.

Every event carries:

```text
EventUid
CausalWatermark
ProducerId
ProducerSequence
EnvironmentInstanceId
EpisodeId
ModalityId
```

Passive observations do not contain sentinel/no-op action IDs.

All decisions may use only evidence whose causal availability precedes the decision.

---

# 8. Symbolic/text representation

Text is an ordered sensory stream.

A symbol contains:

```text
SymbolVocabularyId
SymbolStreamId
SymbolId
SymbolPosition
CausalWatermark
SourceProvenance
```

Permitted input structure:

```text
identity
order
adjacency
sequence boundaries
stream provenance
temporal alignment
```

Not permitted in canonical cognition:

```text
embeddings
pretrained lexical similarity
POS tags
dependency parses
named entities
synonym classes
LLM-generated semantics
task-semantic parser output
```

Human-readable words may be used as raw symbol IDs, but Hydra receives no lexical meaning.

---

# 9. Canonical identity contract

Canonical identity is defined independently for every memory level.

Mutable evidence, validation, lifecycle, lineage, target trust, parent connectivity, and support counts are never silently included in identity.

## M0

```text
identity = EventUid
```

M0 represents a specific grounded event.

## M1G

Interaction identity:

```text
EnvironmentInstanceId
GroundedContextSignature
ExecutableActionToken
RealizedTransitionSignature
GroundedNextContextSignature
```

Passive-symbol identity:

```text
EnvironmentInstanceId
SymbolVocabularyId
SymbolStreamId
ObservedStructuralRelation
GroundedLocalContext
```

## M1N

```text
NormalizedPrimitiveKind
NormalizedStructuralSignature
RelationSignature
TemporalBucket
Magnitude/structure bucket when applicable
```

Environment-native identities are excluded unless required by the normalized primitive definition.

## M2

```text
InvariantFamilyDescriptor
NormalizedTransformation/recurrence class
```

Parent UIDs are provenance, not identity.

## M3

```text
InvariantFunctionalRoleDescriptor
```

This is derived from bounded typed relational/consequence structure, not raw carrier identity.

## M4

```text
InvariantConceptDescriptor
```

Concept candidate and validated concept share canonical identity. Validation changes state/evidence, not identity.

## M5

```text
InvariantConsequenceStructureDescriptor
```

## M6

```text
InvariantOutcomeEquivalenceDescriptor
```

Terminal labels, native reward, and strategy identity are excluded.

## M7

```text
TargetM6Uid
InvariantProcedure/strategy structural identity
ApplicabilityIdentity
```

Outcome-achievement reliability, primary valence, and efficiency are statistics, not identity.

Collision detection is mandatory for all packed/hash identities.

---

# 10. Memory hierarchy and admission

## 10.1 M0 — grounded episodes

M0 is the exact provenance root.

It retains environment/episode identity, causal ordering, observation/action/symbol evidence, boundary/primary-valence evidence, and optional payload references.

Large payloads are externalized from canonical hot nodes.

## 10.2 M1G — grounded local contingencies

M1G preserves local executable or observed relations.

It is environment/modality local.

## 10.3 M1N — normalized observable relations

M1N is the first mandatory cross-environment/cross-modal normalization boundary.

World, symbol, and cross-modal fact channels have independent budgets.

## 10.4 M2 — recurrence/transformation families

M2 requires:

```text
multiple distinct lower-level parents
minimum support
shared invariant structural family
positive compression benefit
bounded candidate search
provenance
```

Raw recurrence/frequency alone is insufficient.

## 10.5 M3 — carriers and functional roles

M3 admission requires:

```text
multiple supporting M2/M1N structures
recurring typed relational position
bounded relational descriptor
sufficient support diversity
```

A role may converge across modalities when invariant relational/consequence structure matches.

## 10.6 M4 — concepts

M4 has four distinct states:

```text
CONCEPT_CANDIDATE
TRANSFER_TEST_ELIGIBLE
VALIDATED_CONCEPT
FAILED/PROBATIONARY_CONCEPT
```

Candidate formation may use:

```text
compression
explanatory reach
transfer prior
structural correspondence
```

Validated concept status additionally requires successful held-out causal transfer according to the configured experimental contract.

Ordinary learning does not require every M4 candidate to be causally tested immediately.

## 10.7 M5 — consequence structures

Validated M4 evidence is required for mature M5 authority.

Local/probationary consequence candidates may exist for research or shadow evaluation, but they cannot acquire unrestricted planning authority until their upstream validation contract is satisfied.

## 10.8 M6 — outcome equivalence

M6 groups states/trajectory endpoints by learned consequence interchangeability.

Persistent classes require support, stability, contextual consistency, and bounded within-class divergence.

## 10.9 M7 — strategies/procedures

M7 requires causal grounded ancestry to its target M6 outcome.

Every M7 separates:

```text
outcome identity
achievement reliability
signed primary-valence evidence
trajectory cost/efficiency
context applicability
provenance
```

Execution always resolves to target-local available actions.

---

# 11. Context and lineage authority

## 11.1 Context identity

Context is an empirical structural partition, not arbitrary metadata.

A `ContextScopeId` is formed only when a bounded evidence-supported partition materially improves prediction, contradiction resolution, transfer validity, or action applicability.

Context formation records:

```text
ParentContextScopeId optional
PartitionDescriptor
EvidenceRefs
FormationWatermark
Support
Prediction/contradiction improvement
```

Contexts may be refined, merged, superseded, or retired while retaining provenance.

## 11.2 Lineage identity

A `LineageUid` identifies a causal developmental derivation path.

Lineage is separate from canonical node identity.

## 11.3 Effective state

The cognitive state visible to a reader is resolved as:

```text
CanonicalNodeState
+ applicable ContextScope
+ applicable LineageOverlay
+ dependency-edge authority
+ target-scoped trust/validation
= EffectiveCognitiveState
```

This resolution function is centralized and used consistently by:

```text
planning
transfer
replay
lifecycle
similarity/correspondence
reporting
```

No subsystem may independently reinterpret overlay semantics.

## 11.4 Dependency authority

Dependency edges can be:

```text
ACTIVE
SUSPENDED
REJECTED
```

When support becomes uncertain, only affected dependency edges are suspended.

Descendants remain active if sufficient independent support remains.

## 11.5 Minimal copy-on-write

Canonical nodes remain shared while invariant identity remains valid.

A canonical fork occurs only if a lineage/context-specific change modifies invariant identity and another authoritative scope still requires the previous identity.

Downstream nodes are reused when their identity remains independently valid.

---

# 12. Canonical mutation model

Workers operate against immutable/versioned read views.

Two mutation classes exist.

## 12.1 Additive commutative proposals

Used for:

```text
new evidence support
new canonical node with deterministic identity
new relation support
additive statistics
```

Equivalent proposals may be deterministically reduced.

## 12.2 Stateful mutation proposals

Used for:

```text
lifecycle transition
lineage/context overlay update
dependency suspension/reactivation
canonical fork
validation-state change
retirement
equivalence-set resolution
estimator publication
```

A stateful proposal declares:

```text
ProposalUid
SourcePeerStableId
CausalWatermark
MutationKind
TargetPartitions
ReadSet
WriteSet
EvidenceRefs
DeterministicOrderingKey
```

## 12.3 Read-set staleness

Global generation advancement alone does not make a proposal stale.

A proposal is stale when a materially consumed authoritative object changed.

```text
ReadSet = [
    ObjectUid,
    ObjectKind,
    Version
]
```

If all read versions still match, the proposal remains eligible even when unrelated graph mutations occurred.

## 12.4 Partition ownership

Each canonical node/edge/overlay maps deterministically to a partition.

Single-partition mutations are committed by the owning reducer.

Cross-partition mutations:

```text
identify partitions
sort IDs
prepare all
validate complete read set
commit atomically
or abort completely
```

Partial cross-partition commit is forbidden.

## 12.5 Deterministic ordering

Within a reducer batch:

```text
(
  CausalWatermark,
  MutationPriorityClass,
  SourcePeerStableId,
  ProposalUid
)
```

or another fixed equivalent ordering is used.

Thread scheduling must not change canonical state.

## 12.6 Retry/recompute

A stale rejected proposal is not automatically merged.

The originating worker may recompute against a current read view if the work remains relevant.

---

# 13. Developmental stage model

The canonical developmental capability stages are:

```text
Stage 0  interaction seeding
Stage 1  stable contingencies / causal prediction
Stage 2  structural abstraction
Stage 3  held-out transfer capability
Stage 4  consequence integration / initial planning
Stage 5  outcome equivalence + learned preference
Stage 6  alternative-strategy linkage
Stage 7  demonstrated replanning + outcome-conditioned efficiency
```

`Stage_t` is inferred from evidence available before an evaluation interval.

It is fixed during that interval.

Evidence produced during the interval may update only `Stage_t+1`.

Stage 7 requires behavioral evidence and cannot be inferred from mere M7 existence.

---

# 14. Interaction Significance Function

ISF controls developmental resource allocation, not behavioral utility.

The six channels are:

```text
PVI  primary-valence impact
OSI  option-structure impact
PE   empirical prediction error
LV   prospective learning value
TP   prospective transfer prior
EP   prospective explanatory potential
```

Every ISF/fitness decision persists:

```text
decision watermark
evidence availability watermark
raw component vector
normalized component vector
developmental stage snapshot
next developmental stage
score schema/version
graph generation
```

Later evidence cannot rewrite an earlier causal decision.

Stage-dependent weights are fixed for the interval.

---

# 15. Primary valence and future options

Primary valence is the minimal signed motivational channel:

```text
-1 negative boundary
 0 neutral
+1 positive boundary
```

It is separate from:

```text
native environment reward
future-option direction
outcome identity
strategy reliability
trajectory efficiency
symbolic sentiment
```

Future-option structure is learned structural evidence describing bounded discovered reachability.

Expansion is not intrinsically good; contraction is not intrinsically bad.

Delayed primary-valence credit may propagate over experienced grounded trajectories without changing structural memory identity.

---

# 16. Prediction and contradiction handling

Prediction error activates only after supported expectations exist.

Contradiction handling follows:

```text
prediction violation
    ↓
context search
    ↓
candidate partition/refinement
    ↓
retest
    ↓
retain / split / demote / suspend
```

Global concept destruction is not the first response to target/context-specific contradiction.

Negative evidence remains scoped to the relation, target, context, or lineage actually tested.

---

# 17. Structural candidate retrieval and indexing

Bounded scoring is insufficient unless candidate discovery is also bounded.

v9 therefore maintains incremental structural indices over M2-M4.

Candidate indices may include:

```text
memory level/type
typed relation signature
bounded degree signature
dependency signature
enable/block signature
future-option bucket
consequence bucket
context bucket
validated correspondence hints
```

For source `X`:

```text
retrieve bounded coarse candidates
    ↓
deduplicate
    ↓
deterministic rank/filter
    ↓
cap at K candidates
    ↓
progressive structural scoring
```

Index maintenance is incremental and versioned.

No all-pairs graph scan is allowed in the normal cognitive path.

---

# 18. Progressive structural search

Structural comparison operates over preregistered radii:

```text
r = 1, 2, 4, 8, ... <= r_max
```

Each comparison records:

```text
GraphGeneration
ObjectVersions
DescriptorGeneration
EstimatorGeneration
Radius
CandidateSet
```

## 18.1 Scale-stratified normalization

Normalization states:

```text
EMPTY
PROVISIONAL
AUTHORITATIVE
RECALIBRATING
```

Provisional storage contains:

```text
bounded streaming summaries
bounded diagnostic/sample reservoir
lifetime counters
```

It cannot grow linearly with experience.

Promotion to authoritative normalization requires preregistered support, coverage, stability, and observation-span criteria.

Failure to bootstrap does not stop M0-M3 learning.

## 18.2 Plausibility mapping

Candidate scores at radius `r` use a preregistered `beta_r`.

```text
p_r(candidate | source) = stable_softmax(beta_r * normalized_score)
```

`beta_r` is fixed within a scientific condition and calibrated outside hypothesis-evaluation episodes.

## 18.3 Ambiguity and stopping

Record:

```text
candidate count
score distribution
entropy
entropy delta
top-2 margin
compute cost
```

Expand only while additional radius adds useful discrimination or ambiguity remains within budget.

Persistent ambiguity with negligible information gain yields:

```text
StructuralEquivalenceSet
```

No arbitrary winner is selected.

Only later causal evidence may refine or resolve the set.

---

# 19. Transfer and causal validation scheduling

Causal validation is scientifically required for validated transfer claims, but it is not mandatory hot-path work for ordinary developmental learning.

Transfer validation is a budgeted service.

Runtime modes may include:

```text
learning_only
validation_budgeted
validation_full
```

The scientific configuration records the mode and budget.

## 19.1 Learning-only

Allows:

```text
M0-M3 development
M4 candidate formation
local prediction
planning from already validated structures
replay
lifecycle
```

No new claim requiring held-out causal validation is promoted to validated status.

## 19.2 Budgeted validation

Schedules a bounded number of matched trials according to:

```text
candidate priority
scientific information value
target diversity
runtime budget
available environment snapshot support
```

## 19.3 Full validation

Used for dedicated research runs where causal evidence is the objective.

Matched trials compare:

```text
same captured target state
same hidden environment state
same RNG state
same seed
same available actions
same horizon
memory-on vs matched memory-off
```

Targets used for validation are held out from formation provenance.

Matching reduces noise; it does not create positive evidence.

Validation cost and learning cost are reported separately.

---

# 20. Transfer trust and negative evidence

Transfer trust is target/context scoped.

A failed source→target intervention:

```text
reduces trust for that target/context relation
```

It does not globally invalidate the source abstraction unless independent evidence justifies that conclusion.

Repeated failures can move a candidate through:

```text
validated/active
→ probation
→ failed
→ quarantined
```

according to preregistered evidence thresholds.

Successful later evidence may reactivate a structure.

---

# 21. Grounding

Text is an ordered sensory stream.

Cross-modal temporal association does not establish meaning.

Grounding maturity is:

```text
G0 observed symbol
G1 recurrent symbolic structure
G2 cross-modal structural association
G3 predictive cross-modal relation
G4 causally validated interaction grounding
G5 held-out transferable grounding
```

These are orthogonal to Stage 0-7.

A `GROUNDS` relation requires interaction-relevant evidence.

Authority is separated as:

```text
TEMPORALLY_ALIGNED
    ↓
CROSS_MODAL_CORRESPONDENCE
    ↓
symbol-conditioned prediction
    ↓
causal grounding
    ↓
behavioral authority
```

Ungrounded or G3-only symbolic evidence may operate in shadow prediction but cannot directly alter action selection.

H16 uses:

```text
C0 interaction only
C1 symbols only
C2 aligned interaction + symbols
C3 shuffled interaction + symbols
```

---

# 22. Replay

Replay is active cognition without environment execution.

The replay scheduler uses bounded ISF/memory-fitness allocation.

Replay may reconsider:

```text
prediction
M2/M3 abstraction
context refinement
similarity/correspondence
grounding
lifecycle
```

Replay may generate new canonical proposals but cannot:

```text
execute actions
fabricate new M0 environment evidence
bypass validation contracts
perform unbounded graph scans
```

Replay telemetry includes selected, processed, new memories, revisions, and correspondences.

---

# 23. Lifecycle, forgetting, and compression

Lifecycle states are:

```text
CANDIDATE
PROBATION
ACTIVE
VALIDATED
QUARANTINED
RETIRE_PENDING
RETIRED
REACTIVATED
```

## 23.1 Hysteresis

Promotion/demotion requires sustained evidence windows.

## 23.2 Probation

Probation is evidence-opportunity aware.

A node is not retired merely because unrelated global activity advanced time.

Probation considers:

```text
developmental age
relevant evidence opportunities
independent support
pending validation
memory pressure
```

## 23.3 Dependency-safe retirement

Before retirement:

```text
check active incoming dependencies
check active outgoing dependencies
check lineage/context overlays
check higher-level claims
check scientific provenance obligations
```

If another authoritative scope still uses the canonical node, retire only the affected overlay/dependencies.

## 23.4 Consolidated-memory compression

The system tracks:

```text
persistent consolidated memory / cumulative experience
explanatory reach per persistent byte
transfer quality per persistent byte
prediction quality per persistent byte
```

In recurrently compressible environments with bounded irreducible novelty, v9 predicts declining persistent-memory growth ratio while preserving or improving predictive, explanatory, transfer, and planning quality.

Compression may retire payload or redundant low-level residency while preserving required provenance.

---

# 24. Snapshot consistency and restart equivalence

v9 defines its own snapshot schema and consistency protocol.

## 24.1 Snapshot cut

A snapshot is taken at a consistent publication boundary:

```text
request snapshot
    ↓
stop accepting a new canonical publication boundary
    ↓
drain/record in-flight accepted proposal state
    ↓
reach fixed canonical cut
    ↓
capture immutable graph + auxiliary state
    ↓
resume live cognition
    ↓
persist snapshot asynchronously
```

Ordinary cognition does not depend on synchronous disk I/O.

## 24.2 Snapshot contents

Snapshots persist:

```text
canonical graph
object/edge/overlay versions
environment/schema registries
symbol vocabularies/codecs
lineage/context overlays
dependency authority
normalization estimator generations
structural indices
equivalence sets
grounding state
transfer trials
developmental stage
ISF score schema/state
lifecycle/replay state
scientific evidence
ScientificConfigId
watermark
graph generation
```

## 24.3 Restart equivalence

Restore must reproduce:

```text
canonical identities
effective cognitive state
published strategy/action preferences
validation/trust state
developmental stage
normalization state
grounding state
replay/lifecycle state
```

subject only to explicitly declared ephemeral caches.

Native v9 restore cannot import or execute v8.

---


# 25. Recursive deliberation

v9.6 adds recursive deliberation as an explicit cognitive process between memory retrieval and action commitment.

The deliberation state is:

```text
X = current grounded state + relevant retrieved memory
Y = current candidate action / strategy / trajectory
Z = temporary reasoning workspace
```

`X` is fixed for one deliberation episode unless new causally available evidence arrives.

`Y` is the candidate being refined.

`Z` contains temporary reasoning state such as:

```text
retrieved M2/M3 correspondences
candidate M4 concepts
predicted M5 consequences
candidate M6 outcomes
candidate M7 strategies
contradictions
alternative correspondences
future-option estimates
uncertainty/ambiguity state
reasoning trace
```

`Z` is ephemeral working state. It is persisted only as telemetry/evidence when required for reproducibility or research analysis.

## 25.1 Deliberation loop

The reference flow is:

```text
current grounded state X
        ↓
retrieve relevant M0-M7
        ↓
generate candidate Y0
        ↓
initialize reasoning workspace Z0
        ↓
reason(X, Yn, Zn)
        ↓
Zn+1
        ↓
refine(Yn, Zn+1)
        ↓
Yn+1
        ↓
evaluate candidate improvement
        ↓
continue or stop
        ↓
execute best candidate encountered
```

The same reasoning operators may be applied repeatedly.

Each cycle may:

```text
retrieve additional relevant memory
expand or refine structural correspondence
simulate consequence structures
compare outcome alternatives
inspect contradictions
reconsider target/context applicability
compare strategies
estimate future-option effects
re-evaluate trajectory efficiency
```

The executed candidate is the best admissible candidate observed during the deliberation episode, not necessarily the last generated candidate.

## 25.2 Candidate representation

A candidate may represent one of:

```text
single action
short action sequence
M7 strategy
target M6 outcome + strategy
trajectory fragment
prediction requiring an intervention
```

Candidate type is explicit.

All executable candidates resolve to target-local available actions before execution.

## 25.3 Candidate evaluation

Every candidate cycle produces a bounded evaluation vector.

Recommended components:

```text
prediction_consistency
contradiction_penalty
expected_consequence_support
future_option_support
outcome_preference_support
strategy_reliability
trajectory_efficiency
structural_correspondence_support
transfer_trust
ambiguity
```

The exact score composition is versioned in `ScientificConfig`.

Candidate evaluation reuses existing memory evidence and does not create new causal evidence by itself.

## 25.4 Adaptive reasoning budget

Deliberation depth is dynamically allocated from current cognitive difficulty.

Budget inputs may include:

```text
prediction error
candidate ambiguity
learning value
transfer potential
explanatory potential
candidate confidence
context contradiction
available alternatives
```

The configured budget policy produces:

```text
minimum_cycles
maximum_cycles
compute_budget
```

Simple familiar situations may terminate after one cycle.

Novel, contradictory, ambiguous, or transfer-relevant situations may receive deeper deliberation.

The budget decision records the causal evidence available when it was made.

## 25.5 Stopping conditions

A deliberation episode stops when one or more configured conditions are satisfied:

```text
candidate unchanged for N cycles
candidate improvement < epsilon
ambiguity <= threshold
no admissible refinement remains
compute budget exhausted
maximum cycle count reached
```

Stopping rules are deterministic under the same published graph state, scientific configuration, and random seed.

The stop reason is recorded.

## 25.6 Directed replay integration

Replay may be invoked as a reasoning operator inside deliberation.

Directed replay starts from the current candidate and searches for memory that can change or strengthen it.

Typical directed replay paths include:

```text
candidate action
→ conflicting M1/M2 evidence
→ context refinement
→ revised candidate
```

```text
candidate strategy
→ M5 consequence simulation
→ predicted failure
→ alternative M7 strategy
```

```text
candidate transfer
→ M3/M4 correspondence
→ target-specific negative evidence
→ revised correspondence or local fallback
```

Directed replay uses the same bounded retrieval and mutation authority rules as ordinary cognition.

## 25.7 Reasoning operators

v9.6 defines a stable operator vocabulary for telemetry and later learning.

Initial operator families:

```text
RETRIEVE_SUPPORT
RETRIEVE_COUNTEREVIDENCE
EXPAND_CORRESPONDENCE
REFINE_CONTEXT
SIMULATE_CONSEQUENCE
COMPARE_OUTCOMES
COMPARE_STRATEGIES
ESTIMATE_FUTURE_OPTIONS
CHECK_TARGET_TRUST
CHECK_TRAJECTORY_EFFICIENCY
RESOLVE_CONTRADICTION
FALLBACK_LOCAL
```

Operator implementation may evolve while preserving versioned operator identity.

## 25.8 Reasoning-policy learning

Hydra records whether reasoning operations improved the candidate and whether the eventual behavior succeeded.

A reasoning-policy evidence record contains:

```text
problem/context descriptor
initial candidate type
operator sequence
candidate score delta
prediction delta
ambiguity delta
strategy change
behavioral outcome
reasoning cost
```

Repeated evidence may support a learned policy for selecting reasoning operators and allocating deliberation depth.

Reasoning-policy evidence is initially advisory and may later become action-selection authority only through an explicitly versioned design change.

## 25.9 Structural invariance reasoning

Structure-preserving transformations may be used as deliberate tests of higher-level abstraction.

Examples include:

```text
color permutation
rotation/reflection
translation
equivalent carrier substitution
equivalent symbol permutation
equivalent state relabeling
```

The transformed case should increasingly map to the same invariant M2/M3/M4 structures when the learned abstraction is correct.

Transformation tests produce invariance evidence and contradiction evidence for canonical identity and correspondence quality.

## 25.10 Deliberation and canonical memory

Deliberation may generate normal mutation proposals when it discovers supported:

```text
context refinement
correspondence revision
prediction violation
strategy reliability evidence
replay-derived abstraction candidate
```

All resulting persistent changes pass through the same proposal, read-set, validation, partition, lifecycle, and provenance rules as other cognition.

## 25.11 Deliberation telemetry

Record at minimum:

```text
DeliberationUid
GraphGeneration
ScientificConfigId
context
reasoning_budget
reasoning_cycles
initial_candidate
final_candidate
best_candidate_cycle
candidate_changes
candidate_score_by_cycle
candidate_score_delta
prediction_delta
ambiguity_delta
contradictions_resolved
strategy_changes
reasoning_operator_sequence
memory_uids_consulted
reasoning_stop_reason
final_vs_initial_candidate_improvement
reasoning_compute_cost
behavioral_outcome
```

Aggregate reporting includes:

```text
mean/median cycles
cycle distribution
improvement by cycle count
improvement by operator family
cost per successful improvement
fraction of actions changed by deliberation
fraction of changed actions that improve outcome
```

## 25.12 Scientific evaluation

Recursive deliberation is evaluated with declared ablations:

```text
deliberation OFF
1-cycle deliberation
fixed N-cycle deliberation
adaptive-budget deliberation
```

Primary comparisons include:

```text
initial candidate vs executed candidate
prediction quality before vs after deliberation
strategy quality before vs after deliberation
trajectory efficiency before vs after deliberation
behavioral success before vs after deliberation
reasoning cost vs improvement
```

The v9.6 research question is:

> Does repeated reasoning over learned memory structures progressively improve predictions, strategies, and actions, with adaptive computation concentrating effort on difficult situations?

Evidence must distinguish:

```text
candidate changed
candidate score improved
prediction improved
behavior improved
```

A changed candidate alone is not positive evidence.


# 26. Scientific evidence contract

Scientific reports distinguish:

```text
implemented mechanism
structural evidence
observational behavioral evidence
matched causal evidence
validated scientific claim
```

Every hypothesis report contains:

```text
raw decision
quality gate
dependency gate
final decision
blocker
evidence counts
ScientificConfigId
```

No structural proxy may substitute for a required causal trial.

---

# 27. Environment neutrality

All environments enter cognition through the same v9-owned contracts.

Initial families:

```text
ARC
FrozenLake / Gym
Chess
Sudoku
synthetic symbolic environments
BabyAI
```

Later:

```text
ALFRED
```

A new environment may add an adapter/schema.

It must not require domain-semantic modifications to M2-M7.

If it does, the environment boundary has failed.

---

# 28. Performance and boundedness

All expensive mechanisms have explicit budgets.

## Event ingestion

```text
events per queue
symbols per window
payload bytes
M1N facts per channel
pending passive events
```

## Mutation

```text
proposal queue depth
read-set size
cross-partition transaction size
```

## Structural search

```text
candidates per radius
number of radii
descriptor size
equivalence-set size
index buckets touched
```

## Replay/validation

```text
replay candidates
validation trials per interval
validation CPU/time budget
```

## Statistics

Storage is bounded by configured estimator keys and sample reservoir size, not cumulative experience.

---

# 29. v8 migration boundary

v8 is supported only as an external predecessor format.

```text
v8 snapshot/evidence
        ↓
src/v9/migration/*
        ↓
validated v9 representation
        ↓
native v9 snapshot
```

Migration may be offline or a one-time startup conversion.

Once conversion completes, ordinary execution uses only v9 code.

Ambiguous/incompatible state fails explicitly.

---

# 30. CLI and execution

The independent entrypoint is:

```bash
PYTHONPATH=src python -m v9 continuous-run ...
```

`src/v9/__main__.py` invokes the v9 CLI only.

`src/v9/cli.py` constructs the v9 runtime only.

v9 must run when `src/v8/` is absent.

The historical v8 command remains separate:

```bash
PYTHONPATH=src python -m v8 ...
```

---

# 31. Acceptance criteria for independence and correctness

v9 is complete only when:

1. no production v9 module imports v8;
2. `src/v8` can be unavailable and v9 imports successfully;
3. `python -m v9 --help` works without v8;
4. ARC smoke works without v8;
5. heterogeneous environment smoke works without v8;
6. snapshot/restart works without v8;
7. M0-M7 developmental smoke works without v8;
8. Stage 0-7 and six-channel ISF are active and persisted;
9. canonical identity tests exist for every M-level;
10. context/lineage effective-state resolution is deterministic;
11. additive and stateful mutation paths are deterministic;
12. read-set staleness validation ignores unrelated graph changes;
13. candidate retrieval is bounded and does not use all-pairs scans;
14. progressive similarity can produce explicit equivalence sets;
15. validation trials can be disabled or budgeted without disabling learning;
16. target/context-scoped negative transfer survives restart;
17. replay can produce developmental proposals without environment interaction;
18. lifecycle retirement preserves authoritative dependencies and provenance;
19. consolidated-memory growth and quality metrics are reported;
20. research evidence/report generation works without v8;
21. H16 synthetic grounding runs without v8;
22. all v9 tests pass with v8 unavailable.

---

# 32. Final target

Hydra v9 is a self-contained implementation of the v0.6.3.1 research architecture with recursive deliberation over learned memory structures.

It inherits **ideas and validated engineering lessons** from v8, not runtime dependencies.

```text
v8 = predecessor/reference/migration input
v9 = independent executable research system
```

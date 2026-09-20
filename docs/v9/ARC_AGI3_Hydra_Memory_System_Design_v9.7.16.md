# ARC-AGI-3 Hydra Memory System Design v9.7.16

**Version:** v9.7.16  
**Status:** Dual-mode research-contract-aligned, reproducible-reasoning, asynchronous-development, transactionally isolated target design  
**Research contract:** `Research_problem_statement_v070.md`  
**Design predecessors:** Hydra v9.7.15, v9.7.12, v9.7.11, v9.7.10, v9.7.9, v9.7.8, v9.7.7, v9.7.5, v9.7.4, v9.7.3, v9.7.2, v9.7.1, v9.7, v9.6, v9.5, v9.4
**Purpose:** extend Hydra with an HGT-style heterogeneous graph transformer and a reproducible epoch-view, crash-recoverable, performance-bounded sampling/memory-update runtime whose live working set, synchronization cost, IPC cost, snapshot cost, actor-visible inference state, training evidence, evidence durability, durable-storage footprint, and whole-host memory footprint remain controlled as persistent memory grows.

---

# 0.1 v9.7.16 dual-mode consistency and implementation delta

v9.7.16 closes the remaining integration gaps identified after v9.7.15.

The revision makes seven points authoritative:

```text
1. mode-scoped invariants:
       ASYNC_DEVELOPMENT != MATCHED_REASONING

2. deterministic DevelopmentalCut(E) for MATCHED_REASONING:
       all actor-visible developmental mutation is closed deterministically

3. H17 isolation:
       baseline H17 disables learned HGT→Hydra developmental feedback

4. concrete canonical COW storage:
       immutable chunks + bounded overlays + atomic CanonicalStateHandle swap

5. mathematically defined H18 observation/action transforms:
       permutation/topology ablation cannot leak through array position or action coordinates

6. truly matched H19 trials:
       fixed start-state/seed/horizon trial list; unused horizon is discarded

7. exact v0.7.0 falsification traceability:
       F1-F18 plus the H19 rejection condition are first-class registry entries
```

The two scientific visibility modes have distinct guarantees.

```text
ASYNC_DEVELOPMENT
    hypothesis: H17
    actor decisions may bind newer complete CanonicalStateHandles
    developmental processes publish independently
    learned HGT→Hydra lifecycle/context/validation feedback disabled in baseline
    no schedule-identical trajectory claim
    evidence = stability/plasticity distribution across perturbation replicates

MATCHED_REASONING
    hypothesis: H19
    actors bind one immutable EpochInferenceView per epoch
    every actor-visible developmental mutation is governed by DevelopmentalCut(E)
    candidate HGT training is governed by deterministic TrainingCut(E)
    fixed matched TrialManifest prevents extra starts/interactions
    schedule-independent scientific-state claim applies
```

For `MATCHED_REASONING`, epoch closure is now:

```text
EpochInferenceView(E)
→ fixed TrialManifest sampling
→ EvidenceCut(E)
→ DevelopmentalCut(E)
→ TrainingCut(E)
→ PolicyProjectionCut(E+1)
→ EpochInferenceView(E+1)
```

`DevelopmentalCut(E)` includes all canonical developmental changes that could affect the next actor-visible view:

```text
M1 support/contradiction maturation
M2 family formation
carrier/role proposal and validation
context refinement
grounding maturation
transfer validation
M4 validation
future-option updates
M5/M6/M7 maturation
lifecycle promotion/demotion/retirement/reactivation
replay-allocation metadata that affects future developmental decisions
```

No background developmental worker is allowed to mutate the next `MATCHED_REASONING` view outside this cut.

Canonical versioning is made concrete through immutable chunked roots and bounded overlays rather than assuming Python containers can be copied atomically at arbitrary scale.

H18 transforms are defined over a latent cell set and an explicit action-coordinate bijection so observation permutation never silently changes environment dynamics.

The HGT training path is unified: lifecycle/consolidation emits WAL-backed `TrainingEvidenceRecord`s or replay-priority metadata; it does not launch independent optimizer windows outside `TrainingCut`.

# 1. Design thesis

Hydra v9.7 combines two complementary forms of intelligence:

```text
explicit developmental memory
+
learned relational representation
+
recursive deliberation
```

Hydra remains the persistent cognitive substrate.

A heterogeneous GNN becomes the learned relational reasoning engine that helps Hydra decide:

```text
what memory is relevant
what structures are similar
what correspondences are plausible
what consequences are likely
what strategies are promising
how the current candidate should be refined
```

The central architecture is:

```text
experience
    ↓
Hydra M0-M7 memory
    ↓
bounded relevant subgraph
    ↓
heterogeneous GNN
    ↓
reasoning workspace Z
    ↓
candidate refinement Y
    ↓
recursive deliberation
    ↓
target-local action
    ↓
environment consequence
    ↓
causal evidence
    ↓
Hydra memory update
    ↓
GNN training evidence
```

The system therefore learns both:

```text
persistent explicit knowledge
and
learned relational computation over that knowledge
```

---

# 2. Architectural roles

## 2.1 Hydra owns persistent cognition

Hydra remains authoritative for:

```text
M0-M7 canonical memory
canonical identity
provenance
context
lineage
lifecycle
grounding
transfer validation
negative evidence
scientific evidence
snapshot/restart
developmental stages
ISF
primary valence
future-option structure
```

Hydra determines what becomes persistent knowledge.

## 2.2 GNN owns learned relational inference

The GNN learns:

```text
node representations
edge-conditioned message passing
memory relevance
candidate retrieval
structural similarity
correspondence likelihood
consequence estimation
strategy ranking
candidate refinement features
reasoning-operator selection support
```

The GNN operates on published Hydra state and ephemeral deliberation state.

## 2.3 Recursive deliberation integrates both

For one decision:

```text
X = current grounded state + published Hydra memory
Y = current candidate
Z = temporary reasoning workspace
G = bounded Hydra subgraph
```

Then:

```text
Z(t+1) = GNN(G, X, Y(t), Z(t))
Y(t+1) = refine(X, Y(t), Z(t+1))
```

Candidate refinement repeats until the deliberation budget ends.

---

# 3. Core invariants

## 3.1 Mode-independent invariants

1. Hydra canonical memory remains explicit and auditable.
2. The GNN consumes versioned published memory views.
3. GNN outputs are proposals, scores, latent representations, or candidate refinements.
4. Persistent memory changes use the normal Hydra mutation pipeline.
5. GNN training examples preserve causal provenance.
6. GNN inference and training are versioned independently from memory schema versions.
7. Environment-specific observations enter through v9 adapters and become common graph-compatible structures.
8. The GNN sees typed nodes and relations rather than environment-semantic labels.
9. Target-local action legality is resolved after deliberation against the live environment.
10. The GNN may operate at multiple memory levels simultaneously.
11. Scientific reports distinguish explicit-memory evidence from learned-model estimates.
12. Recursive deliberation cost is measured separately from environment interaction cost.
13. Per-producer causal order is preserved from actor emission through canonical commit.
14. Cross-producer scientific reductions use explicit commutative/idempotent merge semantics over stable scientific evidence identities.
15. `wal_durable_lsn`, `canonical_applied_lsn`, `snapshot_applied_lsn`, and `hgt_checkpoint_lsn` are distinct monotonic frontiers.
16. WAL durability never exposes partially applied canonical state.
17. Multi-fragment canonical transactions modify a private bounded overlay and publish one complete `CanonicalStateHandle`.
18. Training evidence is materialized atomically from durable WAL-backed records.
19. WAL and other durable objects are reclaimed only when no live recovery/scientific consumer references them.
20. `scientific_run_id`/experiment identity and process-lifetime identity are separate.
21. Every resident queue, projection, transport, WAL buffer, overlay, index cache, or derivation scheduler has an explicit bound or bounded persistent representation.
22. Derivation tasks are leased/retryable and merge through deterministic canonical identities.
23. High-volume transport payloads are serialized into bounded shared-memory slabs; intermediate routing carries descriptors.
24. `CanonicalStateHandle` atomically couples canonical graph roots, authoritative index roots, and `canonical_applied_lsn`.
25. `TrainingEvidenceRecord` segments are the sole durable HGT-training evidence authority.
26. Durable-storage governance is global and reference-aware.
27. Scientific identities derive from an immutable `ExperimentManifest`; PID, launch order, restart count, and queue timing do not create scientific identity.
28. Semantic abstention prohibits pretrained language models, pretrained lexical/multimodal embeddings, dictionaries, ontologies, synonym resources, semantic parsers, manually supplied symbol-to-world mappings, object labels, goal labels, or task-semantic categories under the core experiment.
29. Symbol preprocessing may provide stable opaque identity, order, and declared non-semantic structure only.
30. H18 structural priors are explicit experiment inputs and may be removed/permuted/withheld without changing underlying environment dynamics.
31. H16 C0/C1/C2/C3 are first-class experimental conditions with auditable exposure contracts.
32. Developmental ordering is measured rather than forced.
33. No additional trajectory-optimization hypothesis is introduced beyond v0.7.0.

## 3.2 ASYNC_DEVELOPMENT invariants

34. `ASYNC_DEVELOPMENT` is the authoritative H17 mode.
35. Developmental processes schedule independently and publish only complete `CanonicalStateHandle`s.
36. Actors/readers may bind a newer complete canonical handle between decisions; no immutable-epoch visibility guarantee is asserted.
37. Baseline H17 disables learned HGT→Hydra developmental feedback so measured stability concerns Hydra's local developmental processes.
38. H17 does not require schedule-identical trajectories or epoch manifests.
39. H17 evidence is statistical stability/plasticity across declared scheduling, threshold, hysteresis, and update-rate perturbations.
40. No global behavioral objective or hidden synchronization barrier may be introduced to stabilize the baseline.

## 3.3 MATCHED_REASONING invariants

41. `MATCHED_REASONING` is the authoritative H19 mode.
42. Every actor decision in sampling epoch E binds exactly one immutable `EpochInferenceView(E)`.
43. The next actor-visible canonical state is produced only by the deterministic `DevelopmentalCut(E)`.
44. No canonical developmental mutation outside `DevelopmentalCut(E)` may alter `EpochInferenceView(E+1)`.
45. `TrainingCut(E)` fixes the exact HGT evidence/replay/optimizer/evaluation work used to select the candidate model for E+1.
46. `PolicyProjectionCut(E+1)` builds only from the completed post-development canonical handle.
47. H19 matched conditions share the same fixed `TrialManifest`; adaptive exploration may change actions but cannot create additional starts, episodes, horizon, or environment interaction.
48. Schedule-independent `EpochInferenceView`/scientific-state reproducibility claims apply only to `MATCHED_REASONING`.

# 4. High-level architecture

The common interaction/canonical-memory path is:

```text
ENVIRONMENT
    ↓
v9 adapter / declared StructuralPriorTransform
    ↓
multimodal events
    ↓
M0 / M1 ingestion
    ↓
canonical WAL transaction
    ↓
atomic CanonicalStateHandle publication
    ↓
developmental memory M2-M7
```

The visibility regime depends on the hypothesis under test.

### ASYNC_DEVELOPMENT

```text
complete CanonicalStateHandle generations
    ↓
independent developmental workers
    ↓
new complete CanonicalStateHandles
    ↓
readers may bind a newer complete generation
```

### MATCHED_REASONING

```text
post-DevelopmentalCut CanonicalStateHandle
    ↓
bounded policy projection
    ↓
EpochInferenceView
    ↓
subgraph retrieval/indexing
    ↓
heterogeneous graph construction
    ↓
GNN reasoning engine
    ↓
candidate scores + latent reasoning state
    ↓
recursive/single-pass deliberation
    ↓
target-local legal action
```

The only authoritative HGT training path is:

```text
WAL-backed TrainingEvidenceRecords
    ↓
immutable TrainingEvidenceSegments
    ↓
deterministic TrainingCut
    ↓
fixed replay plan / optimizer work
    ↓
candidate checkpoint
    ↓
deterministic evaluation/promotion
    ↓
publication only through the next EpochInferenceView
```

Lifecycle/consolidation may add durable training evidence or replay-priority metadata. It does not launch an independent optimizer schedule in `MATCHED_REASONING`.

# 5. Graph schema for the GNN

The GNN consumes a typed heterogeneous graph.

## 5.1 Node types

Initial node types:

```text
CURRENT_STATE
M0_EVENT
M1_GROUNDED
M1_NORMALIZED
M2_FAMILY
M3_ROLE
M4_CONCEPT
M5_CONSEQUENCE
M6_OUTCOME
M7_STRATEGY
CONTEXT
LINEAGE
SYMBOL
ACTION_CANDIDATE
OUTCOME_CANDIDATE
```

Each node receives:

```text
type id
canonical identity features
bounded structural features
lifecycle/validation state
context applicability
support statistics
recency/developmental features
provenance summary
current-target relation features
```

Node features are separated into:

```text
invariant features
mutable evidence features
decision-context features
```

This separation allows the GNN to reason about both structural identity and current trust.

## 5.2 Edge types

Initial typed relations include:

```text
TEMPORALLY_PRECEDES
CAUSES
PREDICTS
SUPPORTS
CONTRADICTS
EXPLAINS
ENABLES
BLOCKS
TRANSFORMS
SIMILAR_TO
CORRESPONDS_TO
GROUNDS
DERIVED_FROM
PART_OF_CONTEXT
PART_OF_LINEAGE
TARGETS_OUTCOME
IMPLEMENTS_STRATEGY
AVAILABLE_ACTION
```

Edge features may include:

```text
support
reliability
validation state
target/context trust
temporal distance
causal watermark distance
structural score
transfer score
```

---

# 6. Subgraph retrieval

The GNN reasons over a bounded subgraph rather than the full memory graph.

## 6.1 Seed nodes

Each deliberation episode starts from:

```text
current state
current context
available actions
current candidate
recent relevant M0/M1 evidence
active target/outcome
```

## 6.2 Retrieval sources

Subgraph expansion combines:

```text
Hydra structural indices
recent causal ancestry
context/lineage relations
M2/M3 family memberships
M4/M5 consequence links
M6/M7 outcome-strategy links
previous GNN relevance scores
```

## 6.3 Budget

Scientific configuration defines:

```text
max nodes
max edges
max hops
max nodes per memory level
max candidates per relation type
max recent episodic evidence
```

The graph extractor records all omitted/selected nodes for reproducibility.

---

# 7. Heterogeneous GNN design

## 7.1 Message passing

The reference model is a relation-aware heterogeneous GNN.

Each layer performs:

```text
relation-specific message generation
→ aggregation at target node
→ gated node update
```

Conceptually:

```text
m_ij = Message(type_i, type_j, relation_ij, h_i, edge_ij)
h_j' = Update(h_j, Aggregate(m_ij))
```

The model shares parameters where relation families are structurally equivalent and uses relation-specific projections where evidence supports distinct behavior.

## 7.2 Depth

Use shallow message passing per deliberation cycle.

Recursive deliberation provides repeated computation:

```text
small GNN depth
×
multiple deliberation cycles
```

rather than forcing very deep message passing in one forward pass.

This mirrors the v9.6 recursive-refinement principle.

## 7.3 Latent dimensions

Use one common latent space for cross-environment structural reasoning.

Allow typed projections for:

```text
world-state nodes
symbol nodes
memory nodes
action candidates
outcome candidates
```

Cross-type correspondence emerges through relational training.

---


## 7.4 Concrete GNN architecture: HGT-style relational attention

v9.7.1 standardizes the learned reasoning engine as an **HGT-style heterogeneous graph transformer** operating over bounded dynamically retrieved Hydra subgraphs.

The design combines four ideas:

```text
GraphSAGE-style bounded sampling
+
R-GCN-style typed relations
+
GAT-style learned attention
+
HGT-style typed node/edge projections
```

The runtime path is:

```text
large Hydra memory graph
        ↓
structural/contextual retrieval
        ↓
bounded heterogeneous subgraph
        ↓
typed node/edge feature encoding
        ↓
HGT relational attention
        ↓
multi-head reasoning outputs
        ↓
candidate refinement
        ↓
recursive deliberation
```

### 7.4.1 Typed projections

Each node type has a learned input projection:

```text
P_M0
P_M1G
P_M1N
P_M2
P_M3
P_M4
P_M5
P_M6
P_M7
P_CONTEXT
P_SYMBOL
P_ACTION
P_OUTCOME
```

Each edge type has learned relation parameters.

Attention therefore depends on:

```text
source node type
relation type
target node type
current deliberation context
```

This allows:

```text
M3 --SUPPORTS--> M4
M3 --CORRESPONDS_TO--> M3
SYMBOL --GROUNDS--> M3
M5 --PREDICTS--> M6
M7 --TARGETS_OUTCOME--> M6
```

to carry distinct learned semantics.

### 7.4.2 Relational attention

For each typed edge:

```text
source representation
+ relation representation
+ target/query representation
→ attention score
```

Messages are weighted by learned attention and aggregated at the target node.

The GNN learns both:

```text
what information a relation carries
and
how important that relation is for the current decision
```

### 7.4.3 Bounded sampling

Subgraph construction follows a GraphSAGE-style bounded expansion policy.

The retriever begins with seed nodes from:

```text
current state
current context
current candidate
available actions
active target/outcome
recent causal evidence
```

Expansion is constrained by:

```text
max nodes
max edges
max hops
per-type quotas
per-relation quotas
recent-evidence quota
high-relevance quota
```

The HGT never requires the full Hydra graph in one forward pass.

### 7.4.4 Recursive depth

Use relatively shallow HGT depth per deliberation cycle.

Example:

```text
2-4 HGT layers
×
1-N deliberation cycles
```

The architecture therefore gains effective reasoning depth through recursive deliberation rather than one very deep graph network.

### 7.4.5 Latent outputs

The HGT returns:

```text
updated node embeddings
graph-level reasoning state
candidate-specific embedding
attention maps
head-specific predictions
```

The graph-level reasoning state feeds `Z`.

Candidate-specific outputs feed refinement of `Y`.

### 7.4.6 Attention telemetry

Persist aggregate attention evidence needed for interpretation:

```text
top attended nodes
top attended relations
attention by memory level
attention by source environment family
attention by target environment family
attention by reasoning cycle
```

Attention is treated as model behavior telemetry, while causal claims continue to depend on intervention evidence.


# 8. GNN output heads

The model produces several outputs from the same latent graph.

## 8.1 Relevance head

Scores nodes/edges for further retrieval or attention:

```text
P(relevant | X, Y, G)
```

Used to:

```text
expand the next subgraph
prioritize replay
focus deliberation
```

## 8.2 Structural similarity head

Produces learned similarity between memory structures.

Used for:

```text
M2/M3 candidate retrieval
cross-environment role matching
invariance testing
```

## 8.3 Correspondence head

Scores source→target correspondence hypotheses.

Used for:

```text
M3/M4 correspondence proposals
transfer candidate ranking
cross-modal grounding candidates
```

## 8.4 Consequence head

Predicts:

```text
next structural state features
M5 consequence activation
M6 outcome likelihood
primary-valence distribution
future-option change
```

## 8.5 Strategy head

Scores M7 strategies or action-sequence candidates against:

```text
current context
target outcome
predicted consequence
reliability
efficiency
```

## 8.6 Candidate-refinement head

Produces candidate refinement signals:

```text
retain candidate
replace candidate
change target outcome
select alternative strategy
request more retrieval
request context refinement
invoke local fallback
```

## 8.7 Reasoning-policy head

Scores reasoning operators:

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

---

# 9. Recursive GNN deliberation

The v9.7 decision cycle is:

```text
1. build X
2. generate initial Y0
3. retrieve bounded graph G0
4. run GNN -> Z1
5. evaluate/refine -> Y1
6. use relevance/operator outputs to build G1
7. run GNN -> Z2
8. evaluate/refine -> Y2
...
9. select best Y
10. resolve target-local action
11. execute
```

The graph may evolve across cycles because retrieval expands or contracts.

The published canonical memory graph remains fixed for the deliberation episode unless a new publication boundary is explicitly entered.

---

# 10. Candidate evaluation

Hydra combines explicit and learned evidence.

Evaluation vector:

```text
Hydra prediction consistency
Hydra contradiction penalty
Hydra future-option support
Hydra strategy reliability
Hydra trajectory efficiency
Hydra transfer trust
GNN consequence confidence
GNN strategy score
GNN correspondence score
GNN ambiguity
```

The evaluation function is versioned.

The best candidate may come from any cycle.

---

# 11. Training signals

The GNN learns from the same game curriculum used to teach Hydra.

## 11.1 Transition supervision

Each interaction yields:

```text
state
action
next state
boundary
primary valence
available actions
context
memory graph version
```

Training objectives:

```text
next-structure prediction
action-effect prediction
boundary prediction
primary-valence prediction
available-action prediction
```

## 11.2 Memory formation supervision

Hydra provides labels/targets from its own later consolidation:

```text
which M1 facts became stable
which M2 families formed
which M3 roles converged
which M4 concepts validated
which correspondences survived
```

This teaches the GNN which lower-level structures later proved useful.

## 11.3 Transfer supervision

Matched transfer trials provide strong labels:

```text
source structure
target structure
correspondence proposal
memory-on result
memory-off result
causal transfer delta
```

This trains correspondence and transfer-relevance heads.

## 11.4 Grounding supervision and H16 experimental controls

Grounding labels are derived from explicit H16 experiment conditions rather than inferred from one mixed stream.

The authoritative conditions are:

```text
C0 — INTERACTION_ONLY
    environmental observations, actions, consequences
    no symbolic stream

C1 — SYMBOLS_ONLY
    symbolic stream
    no aligned interaction evidence available to grounding

C2 — ALIGNED_INTERACTION_SYMBOLS
    interaction + symbols
    true temporal/structural alignment preserved

C3 — SHUFFLED_INTERACTION_SYMBOLS
    same marginal interaction exposure
    same marginal symbol exposure/frequency
    cross-modal alignment destroyed by deterministic shuffle
```

`GroundingCondition` is stored in the `ExperimentManifest`, evidence provenance, training-evidence records, and evaluation reports.

C3 shuffling uses a predeclared deterministic permutation that preserves symbol marginals and exposure counts while breaking the true interaction alignment.

C3 anti-leakage contract:

```text
preserve declared marginal symbol frequency/exposure
preserve declared interaction exposure
destroy true cross-modal pairing
remove or independently permute:
    cross-stream timestamps usable as alignment keys
    shared episode ids
    shared producer ids
    shared sequence-position keys
    common provenance ids
    adapter metadata that reveals the original pair
```

Any metadata needed only for system bookkeeping is kept outside learner-visible features and cross-modal matching.

A C3 run is invalid if the learner can recover original alignment from a retained identifier or timing side channel.

Training/evaluation measures include:

```text
symbol → interaction:
    held-out consequence prediction
    held-out action selection
    interaction prediction from prospective/descriptive symbols

interaction → symbol:
    held-out symbol interpretation/reuse
    prediction/reconstruction of symbolic combinations
    structural correspondence to held-out symbol compositions

cross-modal:
    C2 - C0
    C2 - C1
    C2 - C3
    novel-composition transfer
    symbol-removal persistence during interaction-only evaluation
```

Improved symbol prediction alone does not satisfy H16.

The developmental protocol is represented explicitly:

```text
G0 — interaction grounding
G1 — concurrent aligned symbols
G2 — descriptive-symbol linkage to interaction families/roles
G3 — prospective symbols before interaction
G4 — novel grounded composition
G5 — symbol-mediated learning about unexperienced interactions
```

Progression to a later H16 claim requires the earlier stage's declared evidence criterion to be met.

These labels train cross-modal heads only after the corresponding experiment condition and provenance are preserved.

## 11.5 Deliberation supervision

Each deliberation cycle yields:

```text
Y_t
Z_t
operator
candidate score
Y_t+1
eventual action
eventual outcome
```

Useful reasoning operations receive positive training evidence when they improve:

```text
prediction
candidate quality
outcome-conditioned efficiency when an admissible M6 comparison exists
behavioral outcome
```

## 11.6 Invariance supervision

Structure-preserving transformations generate positive pairs:

```text
rotation
reflection
translation
color permutation
symbol permutation
carrier substitution
state relabeling
```

Contradictory structures generate negative/hard-negative pairs.

---

# 11.7 Structural-prior experimental authority (H18)

H18 uses an authoritative `StructuralPriorProfile` stored in the `ExperimentManifest`.

Reference structural profiles:

```text
S0:
    stable observation-element identity
    temporal order

S1:
    S0
    + equality

S2:
    S1
    + coordinates

S3:
    S2
    + adjacency/topology

S4+:
    explicitly declared richer non-semantic structure
```

## 11.7.1 Observation-space formalization

Let one environment observation contain a finite latent element set:

```text
U = {u_1, ..., u_n}
```

with environment-internal values:

```text
v(u)
```

and optional structural relations:

```text
coord(u)
adj(u_i, u_j)
topology(U)
```

The `StructuralPriorTransform P` produces the **only** observation representation visible to Hydra/HGT:

```text
O_P = P(U, v, coord, adj, topology)
```

Raw environment coordinates/adjacency/topology are not available downstream except through `P`.

## 11.7.2 Coordinate permutation

For a coordinate-enabled environment define a bijection:

```text
π : environment coordinates → exposed coordinates
```

### ORDINARY

```text
π(c) = c
```

### FIXED_COORDINATE_PERMUTATION

One deterministic bijection `π_run` is sampled from the experiment seed and remains fixed for the complete run.

Observation exposure:

```text
exposed_coord(u) = π_run(coord_env(u))
```

Any exposed adjacency relation is computed according to the condition definition; it is never silently copied from raw environment topology when topology is ablated.

### CHANGING_COORDINATE_PERMUTATION

A deterministic schedule selects:

```text
π_t = f(ExperimentManifest, episode_id, macro_step or declared boundary)
```

The change schedule is fixed before execution.

No timestamp, original coordinate, storage offset, tensor index, or adapter metadata reveals `π_t^{-1}` to the learner.

## 11.7.3 Array/tensor anti-leak contract

An observation represented as a dense array can itself reveal topology through storage position.

Therefore a topology-reduced condition cannot simply permute coordinate feature values while leaving an ordinary 2-D tensor available.

When coordinates/topology are withheld or permuted, the adapter emits either:

```text
opaque element records:
    (opaque_element_id, value, declared allowed features)

or

a tensor whose physical indexing is transformed consistently by π
and whose original shape/index metadata is not exposed as learner structure
```

No downstream feature builder may reconstruct original neighborhood from contiguous memory layout or untransformed row/column indices.

## 11.7.4 Action-coordinate transform

If legal actions contain coordinates, spatial locations, object slots, or cell identifiers, the same structural condition applies to the action interface.

For an exposed action targeting coordinate `c_exposed`:

```text
environment_action_coordinate = π^-1(c_exposed)
```

The adapter performs this inverse mapping **outside the learner** immediately before environment execution.

Thus:

```text
underlying environment transition dynamics remain unchanged
learner sees only transformed observation/action coordinates
action legality in exposed coordinates corresponds bijectively to legality in environment coordinates
```

When topology/coordinates are withheld, action candidates use opaque target identifiers rather than leaking raw coordinates.

## 11.7.5 Topology withholding

For `TOPOLOGY_WITHHELD` / `ADJACENCY_WITHHELD`:

```text
no adjacency edges
no neighborhood helper features
no connected-component labels
no convolution/padding geometry that exposes neighborhood
no row/column distance helpers
no adapter-side structural summaries derived from the withheld topology
```

Value identity, equality, temporal order, and any other explicitly enabled structural relation remain available.

## 11.7.6 Required controls

Required spatial controls include:

```text
ordinary declared topology
fixed coordinate permutation
changing coordinate permutation where scientifically meaningful
topology withheld
adjacency withheld
```

Across these conditions keep fixed where applicable:

```text
underlying environment dynamics
action semantics after adapter inverse mapping
primary valence/reward stream
environment seed/start-state manifest
interaction TrialManifest
training/evaluation split
```

H18 reports:

```text
time/evidence to stable contingency
compression
carrier formation
role formation
validated concept formation
held-out transfer
planning/replanning
cross-modal grounding
sample efficiency
```

The experiment estimates which structural priors are necessary, replaceable by experience, or primarily sample-efficiency aids.

# 12. Training objectives

Use multi-task learning.

Reference objective:

```text
L =
w1 * transition_loss
+ w2 * consequence_loss
+ w3 * relevance_loss
+ w4 * correspondence_loss
+ w5 * similarity_loss
+ w6 * strategy_loss
+ w7 * grounding_loss
+ w8 * deliberation_improvement_loss
+ w9 * invariance_loss
```

Weights are part of `ScientificConfig`.

Curriculum stage may alter sampling proportions while retaining a versioned objective contract.

---

# 13. Continual training

The GNN learns incrementally across curriculum history from the single WAL-derived training-evidence authority.

Maintain durable/indexed strata for:

```text
recent interaction evidence
historical curriculum evidence
rare successful transfer
hard negatives
cross-family pairs
grounding controls
failed reasoning traces
successful reasoning traces
delayed validation/consolidation evidence
```

Replay indexes preserve older environment families as new stages arrive.

Track:

```text
current-stage performance
historical-stage retention
cross-family transfer
catastrophic-forgetting indicators
```

Published GNN model versions are immutable.

Scientific training configuration contains only deterministic work/content parameters:

```text
TrainingEvidence manifest selection rule
replay stratification rule/version
replay sample count
deterministic replay ordering/seed
microbatch node/edge limits
gradient accumulation
optimizer
learning rate
weight decay
gradient clipping
optimizer-step count
determinism mode
evaluation manifest
promotion thresholds
retention thresholds
```

Fields such as:

```text
examples per wall-clock trigger
optimizer steps per time window
time-based candidate-evaluation cadence
opportunistic publication timing
```

are not scientific training semantics.

Runtime scheduling may change when predetermined work executes, but not which evidence, update count, evaluation set, or candidate is used by a `MATCHED_REASONING` `TrainingCut`.

# 14. Model publication and deterministic continual training

This section defines the authoritative learned-reasoner training/publication contract for `MATCHED_REASONING`. Training may execute incrementally, but actor-visible learned inference state changes only at deterministic scientific epoch cuts.

Baseline `ASYNC_DEVELOPMENT` for H17 does not require or consume an `EpochInferenceView` and disables learned HGT→Hydra developmental feedback.

The system separates:

```text
live evidence collection
candidate training/evaluation
actor-visible publication
```

Candidate work may execute concurrently with sampling. Publication does not.

A published model records:

```text
ModelVersion
parent ModelVersion
TrainingCutId
training-evidence manifest checksum
ScientificConfigId
feature schema version
graph schema version
objective version
optimizer-plan version
determinism mode
curriculum coverage
evaluation metrics
```

## 14.1 EpochInferenceView is the MATCHED_REASONING actor-visible authority

In `MATCHED_REASONING`, every actor in sampling epoch E binds one immutable:

```text
EpochInferenceView(E)
```

containing:

```text
experiment_id
replicate_id
sampling_epoch_id
CanonicalStateHandle(E)
PolicyVersion(E)
ModelVersion(E)
stage_state(E)
normalization_state(E)
graph schema version
feature schema version
ScientificConfigId
view manifest checksum
```

The `CanonicalStateHandle` is pinned for the actor-visible epoch and contains:

```text
canonical_root
signature_index_root
grounding/index roots required for retrieval
canonical_applied_lsn
canonical schema/generation metadata
```

For every environment decision:

```text
observe live environment
    ↓
retrieve bounded subgraph from EpochInferenceView(E)
    ↓
run ModelVersion(E)
    ↓
recursive deliberation
    ↓
resolve target-local action legality against live environment
    ↓
execute action
    ↓
observe consequence
    ↓
emit causal transition/evidence
```

Live memory ingestion after the epoch view was created does not modify the graph/retrieval state used by actors in that epoch.

A deliberation trace records:

```text
DecisionUid
EpochInferenceViewId
CanonicalStateHandleId
ModelVersion
PolicyVersion
ScientificConfigId
sampling_epoch_id
```

## 14.2 One durable training-evidence authority

Every interaction, reasoning trace, delayed label, consolidation label, transfer result, grounding result, and derivation result that may train HGT is represented as a versioned `TrainingEvidenceRecord`.

Training evidence reaches durability through canonical WAL transactions.

A record contains:

```text
TrainingEvidenceId
scientific evidence identity/provenance
evidence kind
source canonical applied LSN / source epoch view
graph/feature/objective schema versions
environment family / scenario
context/candidate/action/outcome fields
reasoning trace fields when applicable
delayed-supervision ancestry when applicable
label payload
weight/quality metadata
checksum
```

Interaction-derived records are embedded directly in the corresponding canonical WAL transaction.

Evidence generated later by consolidation, derivation, transfer validation, delayed supervision, or reasoning evaluation is emitted as its own deterministic WAL-backed training-evidence transaction.

The durable HGT dataset consists of immutable WAL-derived `TrainingEvidenceSegment`s plus a manifest.

There is no independent mutable "training store" that can diverge from WAL evidence.

## 14.3 Replay is an index/cache, not an evidence source

Stratified replay remains useful, with strata such as:

```text
recent evidence
historical curriculum stages
rare successful transfer
failed transfer
hard negatives
grounding controls
successful reasoning traces
failed reasoning traces
cross-family pairs
high prediction-error cases
underrepresented memory levels
```

But replay state contains only:

```text
TrainingEvidenceId references
segment offsets
stratum indexes
sampling-plan metadata
bounded hot caches
```

Deleting or rebuilding replay indexes cannot change the underlying training evidence.

## 14.4 Deterministic TrainingCut

For each closed sampling epoch E, the system creates:

```text
TrainingCut(E)
    evidence_manifest_generation
    first_training_evidence_id
    last_training_evidence_id
    exact segment manifest checksum
    stratum/index versions
    deterministic replay plan
    RNG seeds
    objective/schema versions
    optimizer hyperparameters
    optimizer_step_count
    gradient accumulation
    microbatch/node/edge budgets
    deterministic-kernel mode
    parent ModelVersion
```

The replay plan is derived from stable evidence IDs and seeded deterministic sampling.

`MATCHED_REASONING` scientific mode does not use elapsed training time, instantaneous GPU availability, or inference queue timing to decide:

```text
which examples are trained
their order
how many optimizer updates occur
which candidate is evaluated
```

Those are functions of `TrainingCut`.

## 14.5 Asynchronous execution with deterministic work completion

Training work for `TrainingCut(E)` may begin while other non-conflicting system work proceeds.

The required optimizer work is a fixed number of deterministic work units:

```text
optimizer_step_count
× gradient_accumulation
× bounded deterministic subgraph construction
```

If some work executes asynchronously before the epoch barrier, the barrier completes the remaining predetermined work before evaluation.

Wall-clock duration may vary. Training content and update count do not.

A training failure, unsupported deterministic kernel, OOM after configured deterministic microbatch fallback, or exhausted retry budget produces an explicit `TRAINING_CUT_FAILED` result. It never silently publishes a partially completed candidate.

## 14.6 Deterministic single-GPU scheduling

Inference and candidate training may share one GPU.

Inference may retain execution priority for responsiveness, but scheduling changes wall-clock completion only.

Reference execution:

```text
run predetermined bounded training microbatch
→ yield to inference if requested
→ resume the next predetermined microbatch
```

Scientific deterministic mode requires a versioned deterministic execution contract:

```text
fixed evidence/replay order
fixed microbatch partition
fixed gradient accumulation
fixed RNG streams
fixed optimizer/state initialization
fixed precision/TF32 policy
deterministic algorithm mode
deterministic graph/subgraph construction
```

For every required GPU operation:

```text
deterministic CUDA implementation available
    → use it

otherwise deterministic replacement kernel available
    → use replacement

otherwise configured deterministic CPU implementation available
    → execute that operation/training path on CPU

otherwise
    → mark TrainingCut NONDETERMINISTIC_UNSUPPORTED
    → no scientific MATCHED_REASONING publication from that candidate
```

The runtime may provide a separate `PERFORMANCE` training mode using nondeterministic kernels, but results from that mode cannot satisfy schedule-independent H19 evidence.

Tolerance-only model equality is insufficient for deterministic scientific mode when score differences could change future actions.

A `TrainingCut` records the exact device/kernel determinism contract actually used.

## 14.7 Candidate model lifecycle

Candidate models move through:

```text
PLANNED
TRAINING
TRAINED
EVALUATING
ACCEPTED
REJECTED
PUBLISHED
SUPERSEDED
FAILED
```

Every candidate stores:

```text
TrainingCutId
parent ModelVersion
training evidence manifest checksum
optimizer-state hash
feature/graph/objective schema versions
determinism mode
curriculum coverage
evaluation metrics
candidate checkpoint checksum
publication decision
```

## 14.8 Deterministic evaluation and promotion

Candidate evaluation uses fixed versioned evaluation manifests:

```text
current curriculum evaluation
historical retention
cross-family transfer
prediction quality
correspondence quality
strategy ranking
deliberation improvement
inference latency benchmark protocol
GPU-memory benchmark protocol
```

Scientific acceptance metrics are computed from deterministic held-out sets and exact manifests.

Performance metrics such as latency may vary physically; their measurement protocol and acceptance thresholds are versioned.

Promotion is a deterministic function of:

```text
candidate checkpoint
parent model
evaluation manifest/results
promotion-rule version
ScientificConfigId
```

A candidate passing promotion is marked `ACCEPTED`, but it is not actor-visible yet.

## 14.9 Publication occurs only through EpochInferenceView

At the deterministic epoch boundary:

```text
EvidenceCut(E) closed
→ DevelopmentalCut(E) closed
→ TrainingCut(E) completed/evaluated
→ choose ModelVersion(E+1)
→ build PolicyVersion(E+1)
→ pin CanonicalStateHandle(E+1)
→ publish EpochInferenceView(E+1) atomically
```

All actors in E+1 receive the same view.

No "next decision" inside E can bind a newly trained model.

Model handles and canonical roots from older epoch views are released only after all actors/readers referencing those views have completed.

## 14.10 Delayed supervision

Hydra consolidation may produce labels after the original interaction.

Example:

```text
epoch E:
M2 candidate exists

later epoch:
candidate becomes validated M3/M4 structure
```

The later event produces a new immutable `TrainingEvidenceRecord` that references the earlier evidence IDs and the validating canonical state.

Historical records are never mutated in place.

## 14.11 Reasoning-policy learning

A deliberation episode can produce:

```text
initial candidate
operator sequence
HGT attention/relevance
candidate changes
best cycle
executed action
eventual outcome
```

The immediate trace and any later outcome/validation label are durable `TrainingEvidenceRecord`s with stable ancestry.

## 14.12 Curriculum transition

When curriculum advances:

```text
historical TrainingEvidenceSegments remain addressable
new environment-family evidence enters later manifests
deterministic replay plans rebalance strata by versioned rules
historical retention continues to be evaluated
```

The HGT therefore learns cumulatively without making replay storage a second authority.

## 14.13 Compute accounting

Track separately:

```text
actor inference GPU time
candidate training GPU time
evaluation GPU/CPU time
environment CPU time
Hydra ingestion/derivation CPU time
epoch-barrier work
```

The scientific training contract is expressed in work units. Wall-clock timing is telemetry, not a source of training semantics.

## 14.14 Initial operating mode

The reference operating mode is:

```text
one immutable EpochInferenceView per sampling epoch
one published HGT model bound to that view
one candidate training job from a deterministic TrainingCut
WAL-derived TrainingEvidenceSegments
bounded replay indexes/caches
deterministic epoch-barrier evaluation/publication
```

The same contract applies to single-GPU and dual-GPU execution.

# 15. GNN feature authority and semantic abstention

The learned relational reasoner receives typed structural memory features, provenance-derived quantities, bounded learned statistics, and current grounded state permitted by the declared observation contract.

It does not receive task-semantic shortcuts.

## 15.1 Allowed feature sources

Allowed sources include:

```text
M-level / memory type
relation type
stable opaque symbol identity
causal/provenance counts
support / contradiction / maturity
context / lineage identifiers
declared non-semantic observation structure
temporal order
equality
coordinates when enabled by StructuralPriorProfile
adjacency/topology when enabled by StructuralPriorProfile
learned interaction-derived structural descriptors
learned Hydra grounding state
bounded numeric outcome/consequence statistics
```

All supplied structural priors are versioned in the experiment manifest.

## 15.2 Prohibited semantic shortcuts

Under the core experiment the following are prohibited:

```text
pretrained language models
pretrained text/image/multimodal embeddings
pretrained lexical representations
dictionaries
ontologies
synonym tables
semantic parsers
manually supplied symbol-to-world mappings
object-category labels
goal labels
task-semantic action descriptions
task-semantic reward interpretations
hand-authored semantic concept identities
```

Text or symbolic input may be tokenized only to establish stable opaque identity, position/order, and bounded byte/token structure.

An embedding table for opaque symbols may be trained **from scratch inside the experiment**. Its parameters begin without pretrained lexical meaning and acquire value only through the experiment's own interaction/symbol evidence.

Any experiment using pretrained representations is a separate non-core condition and cannot be used as evidence for semantic-free emergence.

## 15.3 Structural-prior declaration

Every run records the exact observation structural prior budget `R_O` through a `StructuralPriorProfile`.

No feature builder may silently reconstruct or reintroduce an ablated prior from privileged adapter metadata.

H18-specific transforms occur before Hydra/GNN feature construction so downstream code sees only the declared condition.

# 16. Curriculum integration

The 14-step curriculum becomes both:

```text
Hydra developmental curriculum
and
GNN relational-learning curriculum
```

Expected GNN learning progression:

## Steps 1-2

```text
transition prediction
basic relevance
causal adjacency
future-state structure
```

## Steps 3-4

```text
functional-role embeddings
irreversibility
constraint structure
cross-instance similarity
```

## Steps 5-8

```text
cross-modal grounding
symbol/world alignment
relational composition
```

## Steps 9-11

```text
cross-representation consequence prediction
strategy ranking
ARC correspondence
```

## Step 12

```text
cross-family shared latent structure
```

## Step 13

```text
uncertainty
high-dimensional observation support
robust relevance
```

## Step 14

```text
long-horizon multimodal relational reasoning
```

---

# 17. GNN and memory lifecycle interaction

Hydra lifecycle authority remains explicit and causal.

Learned HGT scores may be used only in experiment conditions that explicitly enable `LearnedDevelopmentalFeedback`.

The baseline H17 experiment disables this coupling.

## 17.1 Lifecycle-to-training coupling

Hydra lifecycle transitions may emit WAL-backed `TrainingEvidenceRecord`s and replay-priority metadata.

Relevant lifecycle events include:

```text
PROMOTED
MERGED
SUPERSEDED
CONSOLIDATED
RETIRE_PENDING
RETIRED
REACTIVATED
```

Each event records:

```text
MemoryUid
memory level
context / lineage
supporting evidence
replacement / parent / descendant identities
CanonicalStateHandle / applied LSN
ScientificConfigId
```

These events may affect later `TrainingCut` construction through deterministic evidence/replay rules.

They do **not** directly launch optimizer work in `MATCHED_REASONING`.

### 17.1.1 Pre-retirement evidence preservation

When a substantial memory cluster enters `RETIRE_PENDING`, lifecycle code may deterministically select representative evidence before physical retirement.

Selection preserves:

```text
high explanatory reach
high transfer value
rare structural variants
contradictions / hard negatives
successful strategies
failed strategies
environment-family diversity
context diversity
```

The result is one of:

```text
new WAL-backed TrainingEvidenceRecords
or
deterministic replay-priority/index metadata
```

No `HGT consolidation training window` exists outside `TrainingCut`.

### 17.1.2 Delayed supervision from consolidation

Later developmental state may create new durable labels for earlier evidence.

Example:

```text
candidate M2 at time t
→ stable M3 role at t+k
→ validated M4 transfer ancestry at t+m
```

The later event emits a new immutable `TrainingEvidenceRecord` referencing the earlier evidence identity.

Historical evidence is never edited in place.

### 17.1.3 Reactivation

A reactivated memory may deterministically raise the replay priority of its referenced training evidence for a future `TrainingCut`.

It does not trigger immediate optimizer work.

### 17.1.4 Optional learned developmental feedback

A separate factorial condition may enable HGT-derived evidence channels such as:

```text
retrieval usefulness
explanatory usefulness
transfer usefulness
redundancy
context-split proposal score
validation proposal score
```

When enabled, HGT output remains advisory; Hydra causal rules remain authoritative.

Every such feedback use records:

```text
LearnedDevelopmentalFeedbackProfile
ModelVersion
source EpochInferenceView/CanonicalStateHandle
affected developmental decision
explicit Hydra evidence that accepted/rejected the proposal
```

This condition is **not** the baseline H17 test.

## 17.2 H17 asynchronous developmental-stability mode

H17 tests whether independent asynchronous **Hydra developmental processes** can reach stable but plastic memory regimes without one global behavioral objective.

Baseline H17 sets:

```text
ScientificVisibilityMode = ASYNC_DEVELOPMENT
LearnedDevelopmentalFeedback = DISABLED
```

Independent processes include:

```text
M1 support / contradiction update
family formation
carrier/role proposal
concept-candidate formation
transfer validation
context refinement
replay allocation
lifecycle promotion/demotion/retirement
grounding maturation
peer/reuse updates where enabled
```

Each operates over complete `CanonicalStateHandle`s and submits idempotent WAL-backed mutation proposals independently.

They are not synchronized by H19 `EvidenceCut`, `DevelopmentalCut`, `TrainingCut`, or `EpochInferenceView`.

### 17.2.1 ASYNC_DEVELOPMENT visibility

```text
independent developmental worker
→ read complete CanonicalStateHandle H
→ prepare causal proposal
→ WAL durability
→ bounded overlay application
→ atomic CanonicalStateHandle H'
```

Readers may bind H' on later decisions.

No partial mutation is visible.

No claim is made that two scheduler perturbations produce identical transition sequences.

### 17.2.2 H17 experimental factors

Vary systematically:

```text
promotion hysteresis
demotion hysteresis
retention thresholds
evidence thresholds
developmental worker/update rates
peer update rates
mutation budgets
causal-admissibility settings
```

A separate secondary factorial experiment may compare:

```text
LearnedDevelopmentalFeedback = DISABLED
LearnedDevelopmentalFeedback = ENABLED
```

but this cannot substitute for the Hydra-only H17 baseline.

### 17.2.3 H17 measures

Record continuously:

```text
promotion → demotion count
demotion → reformation count
memory state transitions
created memories
retired memories
active memories
structural persistence over Δt
prediction quality
held-out transfer quality
novel useful structure formation
memory growth
```

Compute:

```text
R_rev
R_churn
P_Δt(X)
```

H17 is supported only if a declared parameter region shows bounded reversal/churn together with useful novelty and non-degrading prediction/transfer across repeated scheduling/update-rate perturbations.

### 17.2.4 No forced developmental order

The scheduler never delays, promotes, or suppresses memory solely to make the predicted developmental sequence appear.

Milestone order is recorded from actual evidence transitions.

# 18. GNN and context formation

Context discovery receives learned support from the GNN.

The model may predict:

```text
same-context likelihood
context split likelihood
which relation caused contradiction
```

Hydra context formation still uses explicit evidence thresholds and provenance.

This allows context discovery to become learned rather than entirely descriptor-driven.

---

# 19. GNN and structural equivalence

StructuralEquivalenceSet handling becomes:

```text
Hydra structural ambiguity
+
GNN latent similarity
+
causal evidence
```

The GNN may help identify that several candidates are effectively equivalent before direct causal evidence resolves them.

Equivalence-set membership remains explicit.

---

# 20. GNN and grounding

Symbols and world structures occupy the same heterogeneous graph.

Example:

```text
SYMBOL
  ↓ temporal/cross-modal edges
M1N symbolic structure
  ↓
M2/M3 shared structure
  ↓
world role/consequence
```

The GNN learns cross-modal representations from aligned interaction evidence and grounding outcomes.

This should improve:

```text
symbol-conditioned retrieval
cross-modal correspondence
held-out composition
```

---

# 21. GNN and M4 validation

The GNN may rank M4 concept candidates for causal validation.

Priority score may use:

```text
expected information gain
cross-family novelty
correspondence confidence
uncertainty
potential transfer value
validation cost
```

This can reduce the number of expensive matched trials required to find informative candidates.

---


## 21.1 H19 strategy ranking and candidate refinement

The learned reasoner may operate over existing Hydra M6/M7 structures because H19 explicitly includes:

```text
candidate refinement
strategy selection
sample efficiency
trajectory efficiency as an evaluation measure
```

This section introduces **no additional research hypothesis**.

Hydra remains authoritative for:

```text
M6 outcome / outcome-equivalence evidence
M7 strategy evidence
strategy reliability
realized interaction cost
outcome validation
replanning provenance
```

The HGT may estimate or rank:

```text
strategy relevance
predicted outcome compatibility
predicted reliability
relative realized-cost tendency
dead-end / irreversible-risk evidence
candidate refinement value
alternative-strategy correspondence
```

Recursive deliberation may refine an action/strategy candidate using those estimates.

Any refined candidate must still execute in the environment and its result must return through ordinary Hydra causal evidence.

### 21.1.1 Outcome-conditioned efficiency only

Efficiency is not a primitive objective.

Efficiency comparison is admitted only when Hydra already provides an admissible comparison relation such as:

```text
same validated M6 outcome class
or
another explicitly justified no-worse outcome comparison
```

This preserves v0.7.0 predictions P26-P29:

```text
efficiency becomes meaningful after admissible outcome comparison exists
lower cost may improve reuse within comparable outcomes
cost cannot override worse outcome evidence
efficiency pressure must not suppress early exploration
```

The learned model does not create M6 outcome equivalence or declare a candidate successful.

### 21.1.2 Supervision

Existing H19 heads receive supervision from ordinary Hydra evidence:

```text
strategy A / strategy B
validated or comparable outcome relation
realized success/failure
realized interaction cost
replanning result
context applicability
```

No separate `L_optimization` objective or optimization-specific ontology is required.

Relevant supervision contributes to the existing:

```text
strategy-ranking loss
candidate-refinement loss
consequence-prediction loss
reasoning-policy loss
```

### 21.1.3 Evaluation

Report:

```text
strategy reliability
held-out success
same-outcome interaction cost
trajectory efficiency
replanning success
candidate-refinement improvement
reasoning cost
cross-family transfer
```

These are H19/P26-P29 measurements, not evidence for a new trajectory-optimization hypothesis.

# 22. GNN and replay

Replay can query the GNN for:

```text
high-value memories
hard contradictions
uncertain correspondences
forgotten historical regions
candidate transfer pairs
```

This makes replay selective and learned.

Replay outcomes feed back into GNN training.

---

# 23. GNN model size strategy

Start with a model large enough to learn relational structure but small enough for continual local training.

Recommended first target:

```text
30M-100M parameters
```

with:

```text
mixed precision
mini-batch subgraphs
gradient accumulation
checkpointed training
```

The architecture should support larger models later without changing the graph contract.

---


## 23.1 HGT architecture derived from Hydra structure

HGT architecture and capacity are derived from the observed structure and reasoning demands of Hydra memory rather than from a fixed parameter target.

The architecture-selection process uses four primary inputs:

```text
Hydra feature diversity
    → latent width

Hydra graph topology and useful reasoning distance
    → message-passing depth

simultaneous relation families used by decisions
    → attention-head capacity

curriculum scaling and held-out generalization
    → total model capacity
```

### 23.1.1 Latent width

A node representation must jointly encode information such as:

```text
structural identity
memory level/type
context and lineage
validation/lifecycle state
support and reliability
causal and temporal information
grounding and modality
target-relative relevance
strategy and outcome information
```

A reference capacity decomposition is:

```text
structure                 ~64 dimensions
causal / temporal         ~32
context / lineage         ~32
validation / evidence     ~32
grounding / modality      ~32
target / current state    ~32
strategy / outcome        ~32
general learned capacity  ~64
```

This motivates an initial latent width around `256–320`, with `320` as the reference baseline.

### 23.1.2 Message-passing depth

Depth is determined from measured useful graph distances.

Relevant paths can span:

```text
current state
→ M1
→ M2
→ M3
→ M4
→ M5
→ M6
→ M7
```

Recursive deliberation provides additional effective reasoning depth through repeated retrieval, graph expansion, and HGT application.

The initial reference is therefore:

```text
HGT layers = 3
```

Telemetry should measure shortest useful paths from the current state to M3, M5, M6, and M7 evidence. Observed path distributions guide later depth changes.

### 23.1.3 Attention heads

Head capacity is derived from the number of relational patterns that commonly need simultaneous consideration, including:

```text
causal evidence
structural similarity
contradiction
context relevance
strategy/outcome relevance
grounding
```

The reference architecture uses approximately `4–6` heads. The baseline uses `5` when supported by the implementation's dimensional constraints.

### 23.1.4 Feed-forward capacity

Feed-forward capacity is sized relative to the structured numerical and categorical representation used by Hydra.

Reference:

```text
d_model = 320
FFN = 768–1024
```

The baseline uses `1024`.

### 23.1.5 Reference architecture

Initial reference configuration:

```text
d_model        = 320
layers         = 3
heads          = 5
FFN            = 1024
retrieved graph = approximately 300–800 nodes
typed relations = measured from the active Hydra schema
```

Expected parameter count should be calculated from the concrete HGT implementation, including type-specific projections, relation transforms, encoders, output heads, and parameter sharing.

The expected initial envelope is approximately `10–25M` parameters.

### 23.1.6 Capacity scaling experiment

Maintain three controlled model capacities:

```text
HGT-S   ~5M
HGT-M   ~15M
HGT-L   ~40M
```

Exact configurations are generated from the implemented parameterization so that the models represent meaningful capacity steps.

Compare them under the same:

```text
training evidence
curriculum stage
interaction budget
replay policy
optimization objectives
evaluation suite
```

Measure:

```text
training loss
validation loss
prediction error
cross-family transfer
strategy-ranking accuracy
candidate-refinement success
historical retention
optimization gain
reasoning cost
```

Capacity is considered sufficient when increasing model size produces little additional held-out or cross-family improvement.

### 23.1.7 Depth and width experiments

Depth and width are evaluated independently.

Reference depth sweep:

```text
320 × 2 layers
320 × 3 layers
320 × 5 layers
```

Reference width sweep:

```text
256 × 3 layers
320 × 3 layers
512 × 3 layers
```

Recursive-deliberation depth is also varied independently so the system can determine whether additional internal reasoning cycles provide more value than additional neural depth or width.

### 23.1.8 Graph-derived architecture telemetry

Collect architecture-relevant graph statistics from real Hydra runs:

```text
median relevant subgraph size
95th-percentile relevant subgraph size
average node degree
relation-type entropy
shortest useful path: current state → M3
shortest useful path: current state → M5
shortest useful path: current state → M6
shortest useful path: current state → M7
simultaneously plausible correspondences
relevant contexts per decision
relevant lineages per decision
useful evidence reached per HGT layer
useful evidence reached per deliberation cycle
```

These statistics become inputs to architecture review at curriculum milestones.

### 23.1.9 Curriculum-based capacity gates

Model capacity is evaluated progressively.

```text
early deterministic environments
→ causality, relevance, reachability, basic prediction

Sokoban / MiniGrid
→ roles, dead ends, enablers, dependencies, optimization

BabyAI / semantic environments
→ symbol-world relations, composition, cross-modal grounding

ARC and cross-family stages
→ shared relational representation and generalization
```

A larger capacity tier is promoted when the current tier shows a reproducible capacity ceiling while Hydra evidence quality and training data continue to improve.

### 23.1.10 Architecture-selection principle

The architecture-selection rule is:

```text
Hydra graph topology
    → reasoning depth

Hydra feature and relation diversity
    → representation width and attention capacity

retrieved subgraph statistics
    → computational envelope

curriculum and held-out scaling curves
    → final parameter capacity
```

This makes HGT size an empirical property of the Hydra reasoning problem.

# 24. Inference and runtime performance

Hydra runtime performance is governed by two bounded-compute envelopes and one whole-host resource envelope:

```text
reasoning envelope
    = retrieved subgraph + HGT + deliberation

learning envelope
    = sampling + IPC + memory compilation + canonical mutation + derivation

host envelope
    = process-tree private memory + tracked shared memory + pinned snapshot generations + swap
```

All three use explicit row, byte, work, latency, and resident-memory budgets.

## 24.1 Complexity contract

Recurring runtime operations use the following complexity targets:

```text
action selection                 O(available actions + actor-local bounded policy view)
policy-generation check          O(1)
policy snapshot rebuild          O(changed actor-local projection pages)
producer-sequence reservation    O(publication batch)
producer causal admission        O(active producers with ready batches)
watermark allocation             O(publication batch)
ingestion compilation            O(admitted transition payload)
WAL group commit                 O(committed bytes)
canonical commit                 O(admitted mutation work + touched bounded indexes)
derivation scheduling            O(changed signatures + bounded persistent-index cursor)
snapshot generation cut          O(fixed root-table metadata + active builder handles)
HGT evidence advancement         O(committed WAL records)
HGT inference                    O(retrieved bounded subgraph)
```

Persistent graph size stays outside periodic policy refresh, producer admission, canonical scheduling, and snapshot-lock work. Any persistent-state traversal uses a bounded cursor, maintained aggregate, immutable chunk manifest, or bounded page cache.

## 24.2 Transactional work-and-latency-targeted canonical reduction

The canonical reducer remains the single authoritative visibility owner. A canonical transaction may require several continuation fragments, but those fragments never mutate the live root directly.

Compilation first enforces a **total transaction envelope**:

```text
transition rows
input bytes
materialized mutation bytes
write count
symbol count
derived-relation count
grounding-operation count
mutation work units
```

Initial reference limits:

```text
transaction rows                    <= 1,024
transaction materialized mutation   <= 64 MiB
transaction work units              <= configured bounded budget
continuation-fragment mutation      <= 16 MiB
fragment preferred execution        = 10-25 ms
soft fragment threshold             = 50 ms
hard corrective threshold           = 100 ms
```

A multi-row compiled batch is split into multiple independent WAL transactions before durability if adding another row would exceed the total transaction envelope. A single transition that alone exceeds the total transaction envelope is rejected/quarantined as `OVERSIZED_CANONICAL_TRANSACTION`; normal scientific payload bounds must make this condition impossible on the reference workload.

After WAL durability, one transaction executes as:

```text
base canonical root R
→ create bounded private TransactionOverlay(R, wal_tx_id)
→ apply continuation fragment 1 to overlay
→ apply continuation fragment 2 to overlay
→ ...
→ validate complete transaction
→ atomically publish overlay root R'
→ advance canonical_applied_lsn
→ retire old root when readers release it
```

Readers, actor-policy projection, snapshots, HGT publication, lifecycle, and derivation never observe an intermediate continuation fragment.

The overlay is bounded by the transaction envelope. If fragment execution fails after WAL durability, the overlay is discarded and the process enters fatal recovery; restart replays the durable WAL transaction against the last applied root.

The reducer updates EWMA estimates for:

```text
ms/work-unit
ms/mutation-byte
overlay bytes/work-unit
```

and uses them to choose fragment boundaries. A single unsplittable primitive write exceeding the hard latency threshold is recorded as `OVERSIZED_CANONICAL_PRIMITIVE`. Such events are acceptance failures unless the scientific contract explicitly permits that payload.

Cross-producer reductions use stable evidence identity:

```text
stable evidence id =
    (scientific_run_id,
     sampling_epoch_id,
     producer_id,
     producer_sequence,
     modality_subsequence)

support/evidence membership   → set union / idempotent keyed insertion
representative evidence      → stable rank/minimum identity
counts                       → exact integer sufficient statistics
floating aggregates          → fixed-point or deterministic sufficient statistics
grounding evidence           → keyed monotonic evidence union
derivation evidence          → sorted stable identities
stage/normalization state    → frozen within sampling epoch; advanced at closed epoch cut
```

Operational `ingest_sequence`, `wal_lsn`, and `canonical_generation` are transport/recovery/visibility coordinates. They do not establish scientific precedence between unrelated producers.

## 24.3 One-serialization descriptor transport and byte accounting

High-volume transition payload bytes are never carried as Python queue payloads after actor encoding.

Actors acquire bounded regions from a shared `TransportSlabPool`, serialize one `TransitionBatchEnvelope` into that region, and publish only a fixed-size descriptor:

```text
TransportSlabDescriptor
    slab_id
    offset
    length
    checksum
    scientific_run_id
    sampling_epoch_id
    producer_id
    producer_start_sequence
    producer_end_sequence
    row_count
```

Stage workers and shard workers aggregate descriptors into fixed-size `TransportBatchBundle` metadata. They do not deserialize, copy, pickle, or reserialize the encoded transition bytes.

The descriptor path uses a fixed binary control message through `multiprocessing.Connection.send_bytes` or an equivalent fixed-schema native transport. Python object queues are not used for the encoded payload itself.

The ingest worker maps the referenced slab, validates checksum/range metadata, decodes the transition rows once, and releases the transport-slab reference after compilation.

Shared-memory budgets are partitioned under one Hydra-wide ceiling:

```text
total tracked Hydra SHM ceiling     = 512 MiB
transport slab subpool              <= 128 MiB
compiled-result/derivation SHM      <= 384 MiB
```

Each transformation boundary measures only newly created output:

```text
actor transport envelope       → descriptor.length
compiled-intent SHM segment    → descriptor.size
WAL transaction frame          → framed WAL bytes
HGT dataset segment            → segment encoded bytes
snapshot chunk                 → snapshot encoded bytes
```

This makes row+byte backpressure exact while preserving the one-serialization actor transport contract.

## 24.4 Whole-system working-set governor

Memory accounting categories are mutually exclusive:

```text
A. process_private_uss
   anonymous/private pages owned by coordinator + children

B. tracked_shm_bytes
   live Hydra-owned shared-memory segments; excluded from A by USS semantics

C. snapshot_external_buffers
   Hydra-owned serialization/COW buffers not already represented in A or B
```

Immutable on-disk snapshot chunks are not added merely because they are logically pinned. File-backed cache pressure is observed through host `MemAvailable`; anonymous COW copies are counted in process USS.

Define:

```text
hydra_accounted_working_set = A + B + C
```

Sampling cadence:

```text
1 s   → process-tree RSS, tracked SHM, snapshot buffers, MemAvailable, swap
30 s  → USS reconciliation and accounting correction
on process/lease lifecycle → immediate delta update
```

Initial 64-GiB host targets remain:

```text
accounted working-set soft target = 48 GiB
accounted working-set hard target = 56 GiB
swap soft target                  = 512 MiB
minimum host MemAvailable         = 6 GiB
```

`HARD_PRESSURE_DRAIN` pauses new actor admission, scales publication intake down, prevents optional new snapshot cuts, drains produced work, and prioritizes bounded compaction.

## 24.5 Globally byte-bounded policy projection and EpochInferenceView construction

Candidate projection state may update while sampling proceeds, but actor-visible projection is frozen inside one `EpochInferenceView`.

Per-dimension limits remain safety rails; global entry and byte ceilings are authoritative.

Initial projection limits:

```text
tracked environments                    <= 128 active/recent
contexts per environment                <= 4,096
scores per context                      <= 64
action-score entries globally           <= 524,288
strategy records globally               <= 8,192
outcome records globally                <= 8,192
resident candidate projection target    <= 256 MiB
resident candidate projection hard cap  <= 512 MiB
actor-local immutable epoch view         <= 16 MiB
```

At the deterministic epoch barrier:

```text
pin CanonicalStateHandle
→ select bounded active/recent environment pages
→ build/prefetch projection pages from persistent indexes
→ enforce global byte/entry ceilings
→ generate immutable PolicyVersion
→ package PolicyVersion + canonical handle + ModelVersion
   into EpochInferenceView
```

Cold projection state never performs an unbounded graph scan on the action path.

If a required page is unexpectedly absent during active sampling:

```text
use the already-bound epoch fallback/page
record policy_view_cold_miss
queue bounded prefetch for the next EpochInferenceView
```

A current epoch is never mutated to repair a cold miss.

If dimensional and byte limits disagree, the byte ceiling wins.

## 24.6 Bounded persistent M1 signature index

Derivation support/dirty state is stored in a persistent chunked `SignatureIndexStore`, not an unbounded Python dictionary.

Each record contains:

```text
structural_signature
support
last_derived_support
derivation_dirty
priority_generation
last_evidence_id
record checksum
```

Storage architecture:

```text
immutable signature-index segments
+ bounded mutable delta page
+ bounded resident page cache
+ snapshot/WAL root metadata
```

Initial resident limits:

```text
mutable delta records        <= 65,536
resident page cache          <= 256 MiB
cursor scan/page batch       <= 8,192 records
```

Dirty refill walks persistent segments through a saved cursor without loading the entire index. Compaction merges old segments outside the canonical lock and publishes a new index root atomically.

## 24.7 Lifecycle compaction performance contract

Lifecycle compaction uses bounded persistent-index cursors for candidate discovery and latency-controlled canonical continuation fragments for mutation. Candidate scanning and signature-index segment compaction are outside the canonical lock.

Snapshot-pinned state is retired logically first. Physical reclamation waits for root/chunk reference release. Compaction has an explicit CPU duty-cycle ceiling so it cannot starve ingestion.

## 24.8 Performance, durability, and capacity telemetry

Track at minimum:

```text
environment steps/s
publication rows/s and bytes/s
wal_durable_lsn
canonical_applied_lsn
snapshot_applied_lsn
hgt_checkpoint_lsn
WAL bytes/s, group-commit p50/p95/p99
WAL retained bytes and reclaimable bytes
WAL filesystem free bytes / free fraction
canonical rows/s, work-units/s, fragment count/transaction
canonical overlay bytes and staged-root publication latency
canonical lock wait/hold p50/p95/p99
oversized canonical primitives/transactions
policy candidate generation / published policy version
model candidate version / published model version
epoch publication-cut latency
policy projection resident entries/bytes + actor-view bytes
policy cache hits/prefetch latency/cold misses
producer causal gaps/reorder bytes
transport-slab bytes/in-use/refcounts
SHM grant free bytes / worker-owned bytes / coordinator-owned bytes
signature-index cache bytes/hit rate/segment count/dirty cursor
derivation leases/expiries/retries/late duplicates/merge conflicts
snapshot cut metadata bytes, active builders, external buffers
process-tree RSS/USS, tracked SHM, snapshot external buffers, swap, MemAvailable
HGT segment commit latency, manifest generation, dataset checkpoint/lag
```

A healthy long run shows bounded resource state, monotonic valid persistence frontiers, and stable normalized per-transition latency after warm-up.

# 25. Snapshot, WAL durability, HGT materialization, and restart

v9.7.13 uses four explicit persistence frontiers:

```text
wal_durable_lsn
canonical_applied_lsn
snapshot_applied_lsn
hgt_checkpoint_lsn
```

Their invariant is:

```text
wal_durable_lsn >= canonical_applied_lsn >= snapshot_applied_lsn
hgt_checkpoint_lsn <= wal_durable_lsn
```

The frontiers intentionally differ. WAL durability is redo authority; canonical application is live visibility; snapshots are compact recovery cuts; HGT checkpointing is an independent durable consumer of WAL evidence.

## 25.1 CanonicalCommitWAL physical format and authority

WAL is a framed append-only log. One durable group has:

```text
WalGroupHeader
    magic
    format_version
    group_sequence
    first_lsn
    transaction_count
    payload_bytes
    header_crc

CanonicalCommitFrame[]
    frame_length
    wal_lsn
    wal_tx_id
    previous_lsn
    scientific_run_id
    sampling_epoch_id
    producer causal ranges
    stable evidence identities
    deterministic mutation plan
    self-contained HGT evidence
    payload/work metadata
    payload_checksum
    frame_crc

WalGroupCommitFooter
    group_sequence
    last_lsn
    group_crc
    commit_magic
```

The writer appends header, frames, and commit footer, then performs one fsync for the group. Only a complete group with valid header/footer/group CRC and valid transaction-frame checksums advances `wal_durable_lsn`.

Initial group bounds:

```text
maximum group age      <= 25 ms
maximum group bytes    <= 64 MiB
single ordered WAL writer
bounded WAL-pending bytes
```

A durable WAL transaction is committed for recovery but is not yet canonically visible.

## 25.2 Concrete immutable-chunk CanonicalStateHandle storage

Canonical atomic publication is implemented through structural sharing, not by copying the complete M0-M7 Python object graph.

The authoritative live pointer references one immutable:

```text
CanonicalStateHandle
    canonical_applied_lsn
    graph_root
    payload_root
    level_index_root
    signature_index_root
    grounding_index_root
    provenance_index_root
    lifecycle_index_root
    policy_source_index_root
    other declared authoritative roots
    schema versions
    generation
    handle checksum
```

Each root addresses immutable content-addressed or generation-addressed chunks.

### 25.2.1 Immutable base chunks

Large canonical collections are partitioned into bounded immutable chunks.

Reference shape:

```text
GraphChunk
PayloadChunk
LevelIndexChunk
SignatureIndexChunk
GroundingIndexChunk
ProvenanceChunk
LifecycleChunk
```

Each chunk has:

```text
ChunkId
schema version
entry count
encoded bytes
checksum
generation created
reference count / root reachability metadata
```

Initial target:

```text
chunk entries           <= 8,192
chunk encoded bytes     <= 64 MiB
```

A published chunk is never mutated in place.

### 25.2.2 TransactionOverlay

For durable WAL transaction L:

```text
base = current CanonicalStateHandle
overlay = TransactionOverlay(base)
```

The overlay contains bounded mutable deltas only:

```text
insert/update/delete maps keyed by canonical uid/key
new-edge delta
new-index postings
removed/tombstoned postings
lifecycle state changes
new immutable chunks being built
```

Reads during transaction execution use:

```text
overlay delta
→ otherwise base immutable chunk/index
```

The total overlay obeys the canonical transaction byte/work envelope.

### 25.2.3 Chunk finalization and root creation

At transaction completion:

```text
validate overlay
→ deterministically partition changed entries
→ encode only changed/new chunks
→ reuse all untouched base ChunkIds
→ build new immutable index/root nodes
→ construct CanonicalStateHandle'
→ verify handle checksum and root consistency
```

No full canonical-memory clone is required.

If a changed base chunk is only partially modified, the replacement chunk contains that bounded chunk's resulting entries; unrelated chunks remain shared.

### 25.2.4 Atomic visibility

The runtime owns one atomic/reference-protected pointer:

```text
current_canonical_handle
```

Commit is:

```text
atomic_store(current_canonical_handle, CanonicalStateHandle')
```

or the language/runtime-equivalent single critical-section pointer swap.

`canonical_applied_lsn` exists only inside the handle and is therefore visible atomically with graph/index roots.

Readers pin exactly one handle for the duration of a read/deliberation operation.

They never combine roots from different handles.

### 25.2.5 Reference counting and reclamation

Old handles/chunks remain reachable while referenced by:

```text
active readers
EpochInferenceViews
snapshots
transaction overlays
recovery/checkpoint manifests
declared historical scientific checkpoints
```

Retirement:

```text
logical root no longer current
→ decrement root reachability when handles release
→ enqueue unreachable chunks
→ bounded GC pass
→ physical deletion only after zero live references
```

GC itself is chunk-budgeted and runs outside the canonical visibility critical section.

### 25.2.6 Crash behavior

WAL durability occurs before overlay application.

If the process crashes:

```text
before atomic handle publication
    → old handle remains authoritative
    → replay L

after handle publication
    → new handle/LSN is authoritative
    → idempotent replay detects applied LSN
```

Incomplete newly encoded chunks not reachable from any durable handle/snapshot manifest are orphaned and reclaimed on startup.

This provides the concrete storage mechanism required for transactionally isolated multi-fragment application without unbounded Python-container copying.

## 25.3 Atomic TrainingEvidence materialization from WAL

All durable HGT supervision is represented as self-contained `TrainingEvidenceRecord`s in WAL transactions.

The dataset writer consumes contiguous durable WAL ranges and writes immutable temporary `TrainingEvidenceSegment`s:

```text
training-evidence-<first_lsn>-<last_lsn>.tmp
```

Each segment contains framed evidence records with:

```text
TrainingEvidenceId
wal_lsn
wal_tx_id
evidence kind
stable ancestry/provenance
schema versions
payload checksum
record CRC
```

Publication is:

```text
write complete temp segment
→ validate segment framing/checksum
→ fsync segment
→ rename temp to immutable segment
→ build copy-on-write TrainingEvidenceManifest
→ fsync manifest
→ atomically rename manifest
→ advance hgt_checkpoint_lsn to manifest last_lsn
```

A crash before manifest publication leaves an orphan segment that is ignored and reclaimed.

Replay buffers, strata, HGT datasets, and training samplers consume this manifest and never constitute independent evidence authorities.

The training-evidence materializer may be ahead of `snapshot_applied_lsn`, but never beyond `wal_durable_lsn`. Canonical recovery replays retained WAL before actors launch.

## 25.4 Generation-cut snapshot representation

Canonical snapshot state uses immutable chunks plus a fixed-size root table. Mutable builders rotate before reaching hard row/byte ceilings.

Initial limits:

```text
chunk rows                    <= 8,192
chunk uncompressed bytes      <= 64 MiB
active builders per family    <= 2
root-table families/indexes   <= 64
metadata sealed during cut    <= 1 MiB
```

A generation cut performs only:

```text
acquire canonical visibility lock
→ acquire current CanonicalStateHandle H
→ require H.canonical_applied_lsn = L
→ swap active builder handles with preallocated empty builders
→ capture H canonical/index root ids exactly representing state through L
→ increment root/chunk references
→ publish SnapshotRootTable(canonical_handle=H, snapshot_applied_lsn=L)
→ release lock
```

No WAL-durable-but-unapplied transaction is included or claimed by the snapshot.

Encoding, compression, checksum calculation, segment merging, and durable I/O occur outside the canonical visibility lock.

## 25.5 Copy-on-write and snapshot writer bounds

After generation G is pinned, later writes target new builders/chunks. Existing chunks reachable from G remain immutable. Logical lifecycle retirement updates the current root; physical reclamation waits for zero references.

Initial writer targets:

```text
snapshot active encode buffers       <= 4
snapshot external buffer bytes       <= 512 MiB
snapshot pinned generations          <= 2
snapshot generation-cut p95          <= 50 ms
```

Snapshot serialization does not directly backpressure actors. Pressure caused by anonymous COW or serialization buffers is handled by the whole-system governor.

## 25.6 Durable snapshot publication

A snapshot is complete only after:

```text
all referenced chunks durable
→ manifest checksums durable
→ snapshot_applied_lsn recorded
→ scientific_run_id + sampling_epoch_id recorded
→ SignatureIndex/policy-publication roots recorded
→ complete marker atomically renamed into place
```

The manifest never records `wal_durable_lsn` as though unapplied transactions were already present.

## 25.7 Global durable-storage governance and WAL retention

Durable storage is governed as one reference-aware resource, not as independent unbounded directories.

Tracked classes:

```text
CanonicalCommitWAL
canonical immutable chunks
snapshot manifests/generations
TrainingEvidenceSegments
HGT/model checkpoints
optimizer checkpoints retained for reproducible continuation
evaluation artifacts/manifests
persistent SignatureIndex segments
policy/epoch-view manifests
```

Every durable object has:

```text
object id
class
bytes
content checksum
creation generation/LSN
reference count or reference set
minimum required retention frontier
GC eligibility state
```

WAL reclamation is controlled by the slowest durable consumer:

```text
wal_reclaim_lsn =
    min(
        snapshot_applied_lsn,
        hgt_checkpoint_lsn,
        every other registered WAL consumer checkpoint
    )
```

Only complete WAL groups whose `last_lsn <= wal_reclaim_lsn` may be reclaimed.

Snapshot/chunk GC preserves every chunk reachable from:

```text
latest recovery snapshot
pinned EpochInferenceViews
active snapshot writers
required historical scientific checkpoints
```

TrainingEvidenceSegment GC preserves every segment referenced by:

```text
published/promotable ModelVersion provenance
active/future TrainingCuts
configured historical retention sets
required research evidence manifests
```

Model-checkpoint GC preserves:

```text
all published models still referenced by EpochInferenceViews
current parent model
candidate under training/evaluation
configured historical/best checkpoints
```

Reference initial limits:

```text
retained WAL soft / hard                 = 32 / 64 GiB
canonical snapshots+chunks hard          = 128 GiB
TrainingEvidenceSegments hard            = 96 GiB
models+optimizer checkpoints hard        = 24 GiB
other manifests/index/evaluation hard    = 16 GiB
global durable-storage soft / hard       = 220 / 280 GiB
minimum filesystem free fraction         = 10%
minimum filesystem free bytes            = 20 GiB
```

The limits are configurable for the host, but every run must define finite values.

Pressure response:

```text
SOFT:
    prioritize snapshot/evidence/model GC
    prioritize slow durable consumers
    suppress optional historical checkpoints

HARD:
    stop new actor admission
    stop creating new candidate TrainingCuts/checkpoints
    drain WAL/canonical/evidence consumers
    run reference-safe GC
    fail safely before filesystem exhaustion if pressure cannot clear
```

No durable object is deleted solely by age when a live scientific/reference dependency exists.

## 25.8 MATCHED_REASONING deterministic epoch closure

This section applies to `ScientificVisibilityMode.MATCHED_REASONING`.

The authoritative closure is:

```text
EvidenceCut(E)
→ DevelopmentalCut(E)
→ TrainingCut(E)
→ PolicyProjectionCut(E+1)
→ EpochInferenceView(E+1)
```

## EvidenceCut(E)

Sampling epoch E runs exactly the fixed trials declared in its `TrialManifest`.

After every producer stream for those trials closes:

```text
apply all accepted interaction transactions
→ close scientific evidence range
→ pin CanonicalStateHandle_after_interactions(E)
```

`EvidenceCut(E)` stores:

```text
experiment/replicate/sampling epoch
TrialManifest checksum
first/last scientific evidence identities
canonical evidence-tail LSN
CanonicalStateHandle_after_interactions
manifest checksum
```

## DevelopmentalCut(E)

`DevelopmentalCut(E)` deterministically defines **all** actor-visible developmental work permitted between EvidenceCut(E) and EpochInferenceView(E+1).

It contains versioned finite work plans for:

```text
M1 support/contradiction maturation
M2 family formation
carrier proposal/validation
role proposal/validation
context refinement
grounding maturation
transfer validation
M4 candidate validation
future-option contribution updates
M5 consequence structures
M6 outcome-equivalence structures
M7 strategy/replanning structures
lifecycle promotion/demotion/retirement/reactivation
persistent replay-allocation metadata
other declared canonical developmental operators
```

Each operator selects work through:

```text
eligible keys at EvidenceCut/post-prior-stage handle
→ stable deterministic priority tuple
→ stable sort
→ first N / first W work units under configured operator budget
→ exact target evidence/support versions
```

Operator ordering inside the cut is versioned in `DevelopmentalPipelineVersion` and must respect the theory's dependency/admissibility constraints without forcing predicted empirical milestones.

Work beyond a configured cut budget remains durable eligible/dirty state for later epochs according to the same stable priority rule.

Every selected work item ends in:

```text
APPLIED
NO_CHANGE
REJECTED_BY_CAUSAL_RULE
FAILED_DETERMINISTICALLY
QUARANTINED
```

Scheduler completion timing cannot decide whether an item is included.

All accepted mutations pass through the ordinary WAL → overlay → atomic `CanonicalStateHandle` path.

At completion pin:

```text
CanonicalStateHandle_after_development(E)
```

No asynchronous background developmental worker may publish an actor-visible mutation into the E+1 view outside `DevelopmentalCut(E)`.

## TrainingCut(E)

Construct the exact training plan from durable `TrainingEvidenceManifest`s visible through the configured evidence frontier after development.

It fixes:

```text
evidence manifest/range/checksum
deterministic replay plan
RNG streams
optimizer hyperparameters/state
optimizer step count
microbatch/gradient accumulation plan
kernel/device determinism contract
evaluation manifest
parent ModelVersion
```

Candidate outcome is:

```text
ACCEPTED
REJECTED
FAILED
```

Wall-clock opportunity cannot select a different candidate or partial update count.

## PolicyProjectionCut(E+1)

Using `CanonicalStateHandle_after_development(E)`:

```text
build bounded candidate projection
prefetch expected environment pages
enforce byte/entry ceilings
publish immutable PolicyVersion(E+1)
```

## EpochInferenceView(E+1)

Publish one immutable manifest:

```text
EpochInferenceView(E+1)
    ExperimentManifestId
    TrialManifest family/version
    sampling_epoch_id
    CanonicalStateHandle_after_development(E)
    DevelopmentalCutId
    TrainingCutId
    PolicyVersion(E+1)
    selected ModelVersion(E+1)
    stage_state_after_E
    normalization_state_after_E
    graph/feature/schema versions
    view checksum
```

All actors in E+1 bind exactly this view.

Snapshot serialization need not occur every epoch because WAL + immutable canonical chunks remain recovery authority.

Wall-clock time affects only completion duration.

## 25.9 Deterministic scientific identity domains

Every scientific run starts from one immutable `ExperimentManifest`.

It contains:

```text
research_contract_version
ScientificConfigId
environment-set/profile manifest
curriculum manifest
root random seed
replicate specification
graph/feature/objective schemas
adapter/environment schema versions
training determinism mode
manifest checksum
```

Derive:

```text
experiment_id =
    stable_hash(ExperimentManifest checksum)

replicate_id =
    stable_hash(experiment_id, replicate_index, replicate_seed)

producer_id =
    stable_hash(
        experiment_id,
        replicate_id,
        sampling_epoch_id,
        stable_environment_job_id,
        actor_ordinal_within_job
    )

environment_instance_id =
    stable_hash(
        experiment_id,
        replicate_id,
        stable_environment_job_id,
        environment_instance_ordinal
    )

episode_id =
    stable_hash(
        experiment_id,
        replicate_id,
        sampling_epoch_id,
        producer_id,
        episode_ordinal
    )
```

Scientific evidence identity is:

```text
experiment_id
replicate_id
sampling_epoch_id
producer_id
producer_sequence
modality_subsequence
```

These identities are independent of:

```text
PID
OS process launch order
worker slot
wall-clock time
process restart count
queue arrival order
```

Process ownership instead uses:

```text
process_run_epoch
worker_id
local_lease_counter
```

`process_run_epoch` changes after restart and is used only for shared-memory/slab namespaces, process leases, and orphan cleanup.

## 25.10 Restart

Common recovery performs:

```text
increment process_run_epoch
→ load immutable ExperimentManifest
→ derive/verify experiment_id and replicate_id
→ select newest complete recovery snapshot
→ restore snapshotted CanonicalStateHandle roots
→ restore snapshot_applied_lsn
→ scan WAL physical groups and truncate invalid/torn tail
→ establish wal_durable_lsn from last valid committed group
→ replay WAL snapshot_applied_lsn+1 ... wal_durable_lsn
   through staged CanonicalStateHandle publication
→ current CanonicalStateHandle reaches wal_durable_lsn
→ restore/rebuild bounded resident caches lazily
→ restore producer sequence state
→ restore TrainingEvidenceManifest and verify hgt_checkpoint_lsn
→ reclaim stale transport/SHM namespaces from older process_run_epoch values
```

Then recovery branches by scientific visibility mode.

### ASYNC_DEVELOPMENT

```text
verify current CanonicalStateHandle
→ restore independent developmental worker cursors/leases/dirty state
→ restore H17 parameter/perturbation manifest
→ keep LearnedDevelopmentalFeedback disabled for baseline H17
→ resume asynchronous developmental workers
→ launch/read actors against complete canonical handles
```

No `EpochInferenceView`, `DevelopmentalCut`, or `TrainingCut` is required to resume the H17 baseline.

### MATCHED_REASONING

```text
restore last complete EpochInferenceView manifest
→ verify canonical roots/model/policy/stage/normalization checksums
→ restore incomplete EvidenceCut / DevelopmentalCut / TrainingCut manifests if present
→ deterministically finish or reconstruct required cut work
→ publish the next valid EpochInferenceView only after all required cuts complete
→ launch actors against that immutable view
```

Replay is idempotent by WAL LSN, transaction identity, stable scientific evidence identity, and atomic `CanonicalStateHandle` publication.

Restart never mixes roots or state from different canonical handles or, in `MATCHED_REASONING`, from different epoch views.

# 26. Telemetry

Add GNN-specific telemetry:

```text
ModelVersion
subgraph nodes/edges
node types
edge types
retrieval recall proxy
relevance entropy
embedding drift
correspondence confidence
consequence prediction error
strategy ranking accuracy
candidate refinement rate
GNN-caused candidate changes
GNN-caused successful changes
GNN-caused harmful changes
reasoning cycles
inference latency
GPU memory
training loss by objective
historical-stage retention
cross-family transfer metrics
```

Cross-family metrics use explicit:

```text
source environment family
target environment family
curriculum step
```

---


## 26.1 Unified telemetry model

v9.7.6 consolidates runtime, memory, research, deliberation, HGT inference, HGT training, and model-publication telemetry into one hierarchy.

Telemetry is split into:

```text
primary telemetry
    → small always-visible dashboard

diagnostic telemetry
    → detailed subsystem evidence used for failure analysis
```

The primary telemetry should answer eight questions:

```text
1. Is Hydra learning?
2. Is memory becoming more abstract?
3. Is transfer emerging?
4. Does deliberation improve decisions?
5. Is HGT useful?
6. Is HGT training healthy?
7. Is Model N+1 better than Model N?
8. Is the system computationally sustainable?
```

### 26.1.1 System progress

Primary metrics:

```text
success_rate
trajectory_efficiency
environment_steps_per_second
curriculum_step
```

Diagnostic metrics may include:

```text
watermark
generation
games_seen
levels_seen
levels_solved
```

### 26.1.2 Hydra memory health

Primary metrics:

```text
M0_count
M3_count
M4_validated
memory_growth_rate
compression_ratio
```

Diagnostic metrics:

```text
M0..M7 active counts
new memories per interval
promotion rate
merge/deduplication rate
retirement rate
persistent_memory_bytes
context_count
lineage_count
higher_level_memory / total_memory
```

The higher-level-memory ratio is used to measure whether experience is consolidating upward.

### 26.1.3 Learning and prediction

Primary metric:

```text
prediction_error
```

Diagnostic metrics:

```text
prediction_error_trend
consequence_prediction_accuracy
learning_value
transfer_potential
explanatory_potential
future_option_prediction_error
```

### 26.1.4 Transfer and abstraction

Primary metrics:

```text
cross_family_transfer
false_transfer_rate
```

Diagnostic metrics:

```text
within_game_transfer
within_family_transfer
M3_role_reuse
M4_validated_concepts
structural_correspondence_precision
source→target transfer matrix
```

Transfer metrics always preserve source-environment-family and target-environment-family provenance.

### 26.1.5 Recursive deliberation

Primary metrics:

```text
reasoning_cycles
final_vs_initial_candidate_improvement
deliberation_behavior_improvement
```

Diagnostic metrics:

```text
candidate_changes
initial_candidate_score
final_candidate_score
prediction_improvement_during_deliberation
strategy_changes
reasoning_cost
reasoning_stop_reason
decisions_changed_by_deliberation_rate
changed_decisions_with_better_outcome_rate
```

### 26.1.6 HGT inference quality

Primary metrics:

```text
HGT_consequence_error
HGT_strategy_ranking_accuracy
HGT_candidate_refinement_success
```

Diagnostic metrics:

```text
subgraph_nodes
subgraph_edges
relevance_precision
correspondence_accuracy
candidate_refinement_success_rate
HGT_caused_behavior_improvement
HGT_caused_behavior_regression
inference_latency
```

The primary contribution measure is:

```text
HGT_contribution =
    outcome(Hydra + HGT)
    -
    outcome(Hydra baseline)
```

This is evaluated through matched ablations rather than inferred from internal model scores.

### 26.1.7 HGT training health

Primary metrics:

```text
HGT_training_loss
HGT_validation_loss
historical_retention
current_curriculum_gain
cross_family_validation_gain
training_step_latency
GPU_memory
```

Diagnostic metrics:

```text
training_examples_seen
effective_batch_size
loss_by_head:
    relevance
    correspondence
    consequence
    strategy
    refinement
    optimization
gradient_norm
learning_rate
training_steps
examples_per_second
GPU_utilization
train_vs_validation_gap
```

Training-health interpretation uses these patterns:

```text
underlearning
→ persistently high train and validation loss

overfitting
→ improving training loss with worsening validation gap

catastrophic forgetting
→ current-stage gain with historical-retention loss

poor generalization
→ current-stage gain with weak cross-family validation gain
```

### 26.1.8 Continual model evolution

Primary metrics:

```text
ModelVersion
historical_retention
current_curriculum_gain
cross_family_validation_gain
```

Diagnostic metrics per candidate publication:

```text
parent_ModelVersion
training_examples_since_parent
current_stage_delta
historical_retention_delta
cross_family_transfer_delta
reasoning_improvement_delta
inference_latency_delta
promotion_result
```

Every model-promotion report should summarize:

```text
Model N → Model N+1

current curriculum       delta
historical retention     delta
cross-family transfer    delta
reasoning improvement    delta
latency                   delta
promotion result
```

### 26.1.9 Primary dashboard

The always-visible dashboard should contain approximately these metrics:

```text
success_rate
trajectory_efficiency

M0_count
M3_count
M4_validated
memory_growth_rate
compression_ratio

prediction_error
cross_family_transfer
false_transfer_rate

reasoning_cycles
final_vs_initial_candidate_improvement
deliberation_behavior_improvement

HGT_consequence_error
HGT_strategy_ranking_accuracy
HGT_candidate_refinement_success

HGT_training_loss
HGT_validation_loss
historical_retention
current_curriculum_gain
cross_family_validation_gain

ModelVersion
GPU_memory
inference_latency
training_step_latency
```

The dashboard is intentionally compact enough to identify whether the current failure is primarily:

```text
environment/task progress
memory formation
abstraction/transfer
deliberation
HGT inference
HGT training
model evolution
runtime capacity
```

### 26.1.10 Diagnostic event provenance

All detailed telemetry required for causal or failure analysis should carry, where applicable:

```text
RunUid
DecisionUid
GraphGeneration
ModelVersion
ScientificConfigId
CurriculumStep
EnvironmentFamily
GameScenario
ContextUid
LineageUid
MemoryLevel
```

This allows a primary metric regression to be traced back to the relevant memory, model, curriculum, and environment state.

### 26.1.11 HGT training failure isolation

The telemetry system must support explicit diagnosis of HGT training failures.

Required failure indicators:

```text
loss not decreasing
validation divergence
gradient instability
throughput collapse
GPU memory pressure
historical retention collapse
cross-family validation collapse
candidate model regression
model-promotion rejection
```

Training failures are reported separately from:

```text
Hydra memory-quality failures
retrieval failures
deliberation failures
execution failures
```

This separation is required for scientific interpretation of the hybrid architecture.

# 26.2 Environment-adaptive exploration and viability control

Exploration remains conditioned on live action space, branching factor, context-local action coverage, learned scores, and stagnation.

For ordinary developmental runs this mechanism may adapt freely within configured budgets.

For **matched H19 comparisons**, exploration is constrained by a shared `InteractionOpportunityManifest`.

Adaptive exploration may choose different legal actions because that is part of the policy under test, but it may not:

```text
increase total steps
increase episode opportunities
add environment instances
change reset/start-state seed schedule
extend a game because one condition is performing worse
allocate extra actor jobs to one reasoning condition
```

Any condition-specific exploration state is reset/initialized according to the same comparison manifest unless the experiment explicitly declares exploration state as the factor under test.

The decision path remains:

```text
available actions
→ action-set signature / branching factor
→ context-local action coverage
→ learned + grounded action scores
→ viability / stagnation state
→ adaptive exploration pressure
→ target-local action
```

Viability telemetry is retained, but it cannot silently alter matched interaction allocation.

# 26.3 TrialManifest and heterogeneous game accounting

For ordinary developmental runs, budgets may be derived from environment horizon and desired complete-episode opportunities.

For causal H19 comparisons, the matching unit is a **fixed trial**, not merely a total step count.

A `TrialManifest` contains an ordered immutable set:

```text
TrialSpec
    trial_id
    environment/game identity
    environment configuration
    exact initial-state or reconstruction state reference when supported
    environment RNG seed
    reset seed
    actor/job identity
    fixed interaction horizon H
    timeout/truncation rule
    curriculum stage
    training/evaluation role
```

The same trial list is reused for every H19 reasoning condition.

For each trial:

```text
restore identical start state / seed
→ run at most H environment interactions
→ if terminal/success/failure occurs before H:
       end the trial
       discard the unused horizon
       do NOT convert it into a new reset/episode
→ continue with the next predeclared TrialSpec
```

Thus a condition that terminates quickly receives neither:

```text
additional reset opportunities
additional fresh start states
additional episodes
additional steps
additional actor jobs
```

The `InteractionOpportunityManifest` is the aggregate container for one or more `TrialManifest`s and also records:

```text
curriculum exposure count
training/evaluation split
trial generation procedure
environment horizon source
matched timeout semantics
manifest checksums
```

The same manifest applies to:

```text
Hydra only
Hydra + learned single-pass
Hydra + learned recursive deliberation
random/untrained relational control
frozen-model control
model-only learned policy where feasible
```

Adaptive exploration may choose different actions inside one fixed trial, because that is part of the policy under test. It cannot modify the trial list or horizon.

`game_results.log` records:

```text
ExperimentManifestId
ReasoningCondition
TrialManifestId
trial_id
start-state/seed identity
horizon
used interactions
unused discarded horizon
outcome
```

so matching can be audited directly.

# 26.4 Heterogeneous-memory evidence handling

v9.7.15 treats heterogeneous memory evidence according to memory level, relation type, environment identity, context, lineage, and evidence confidence. Environment viability produces an evidence-confidence value that is attached to applicable graph payloads and used to distinguish weak environmental evidence from structural memory failure.

The runtime therefore carries:

```text
memory identity and level
+ relation semantics
+ source environment
+ context / lineage
+ environment evidence confidence
+ lifecycle / validation state
```

through graph construction and downstream reasoning. This allows mixed M0-M7 memory to coexist while retaining the provenance required to interpret apparent absence, low support, contradiction, or transfer failure.

# 26.5 Transactionally isolated, performance-bounded sampling and canonical memory-update pipeline

v9.7.16 preserves strict producer causality, removes scheduler timing from scientific feedback, and makes WAL durability, canonical visibility, snapshot durability, and HGT materialization explicit independent frontiers.

The production path is:

```text
EpochInferenceView(E)
    ↓ binds immutable CanonicalStateHandle(E) + PolicyVersion(E) + ModelVersion(E)

actor processes
    ↓
write-once TransitionBatchEnvelope into bounded TransportSlabPool
    ↓ fixed-size TransportSlabDescriptor
fair stage/shard descriptor routing
    ↓
stable producer-affinity shard
    ↓
bounded ProducerCausalAdmission
    ↓
row+byte publication admission
    ↓
dedicated producer-sequence validation
+ operational ingest sequence allocation
    ↓
parallel decode/compile
    ↓
bounded CanonicalCommitTransaction
    ↓
compiled-result SHM credit pool
    ↓
bounded decode/WAL pending
    ↓
framed CanonicalCommitWAL group commit
    ↓ wal_durable_lsn
private staged TransactionOverlay
    ↓ bounded continuation fragments
atomic root publication
    ↓ canonical_applied_lsn
    ├────────────→ atomic HGT dataset segment materialization
    │                 ↓ hgt_checkpoint_lsn
    ↓
persistent SignatureIndexStore
    ↓
leased parallel M2/M3/M4 derivation
    ↓
deterministic merge algebra
    ↓ canonical derivation transaction through the same WAL/staged-root path

epoch closes
    ↓
EvidenceCut(E)
→ deterministic DevelopmentalCut(E)
→ deterministic TrainingCut(E)
→ bounded PolicyProjectionCut(E+1)
    ↓
EpochInferenceView(E+1)
```

## 26.5.1 Stable producer-affinity routing

Each actor receives one stable shard for its lifetime:

```text
producer_shard = stable_hash(environment_instance, actor_id) % shard_count
```

`producer_sequence` is excluded from routing. Actor batches contain contiguous producer ranges and flush on full batch, episode boundary, age limit, or actor completion.

Initial actor envelope:

```text
preferred rows   = 32
maximum rows     = 64
maximum bytes    = configured transport ceiling
```

## 26.5.2 Write-once slab transport and fair stage/shard routing

Actors acquire a bounded transport-slab region, encode the envelope once, and publish a fixed-size descriptor. Stage and shard workers carry only descriptors and range metadata.

Stage workers service destination shards round-robin. A full shard retains only its bounded local descriptor bundles while other shards continue progressing.

```text
preferred rows per shard bundle = 128
maximum rows per shard bundle   = 256
local buffered bundles          <= 2 per shard
transport slab SHM subpool      <= 128 MiB
```

Reference counts keep a slab region alive through stage/shard/coordinator forwarding. The ingest worker releases the final transport reference after successful decode/compile.

## 26.5.3 Producer causal admission

For each active producer, the coordinator tracks the next expected producer sequence and a bounded gap window.

```text
batch.start == expected[producer] → ready
batch.start >  expected[producer] → bounded hold
batch.end   <  expected[producer] → stale/duplicate protocol error
```

Ready batches from different producers are admitted by fair scheduler policy. Their relative arrival/admission order is operational only.

Initial per-producer gap bounds:

```text
held batches     <= 4
held bytes       <= 32 MiB
maximum gap age  <= 5 s
```

A missing range beyond the age limit is a transport failure.

## 26.5.4 Scientific and process identity separation

Each transition carries:

```text
ScientificEvidenceId
    scientific_run_id
    sampling_epoch_id
    producer_id
    producer_sequence
    modality_subsequence
```

and operational metadata:

```text
ingest_sequence
wal_lsn
canonical generation
process_run_epoch
```

Only the scientific identity participates in evidence identity, provenance, deduplication, abstraction input, and HGT training identity.

`process_run_epoch` is used only for process/SHM/slab ownership and changes after restart.

## 26.5.5 Dual-budget publication admission

Publication admission stops on the first reached limit:

```text
adaptive row target
publication byte target
ingest outstanding row high-water
ingest outstanding byte high-water
WAL pending-byte ceiling
retained-WAL disk pressure
memory-governor admission factor
```

Initial row targets remain 1,024 / 2,048 / 4,096 / 8,192 by backlog. Initial publication byte target is 64 MiB.

## 26.5.6 Dedicated producer-sequence synchronization

Producer sequence validation/reservation uses `_producer_sequence_lock`, separate from canonical visibility/mutation locks. Lock domains remain non-nested on the hot path.

## 26.5.7 Parallel compilation and total transaction bounds

Ingest workers decode transport slabs and compile immutable `CanonicalCommitTransaction` objects containing:

```text
stable evidence ids
deterministic mutation plan
self-contained HGT evidence
producer causal ranges
row/byte/work estimates
transaction payload checksum
```

The compiler creates a new transaction before total transaction row/byte/work ceilings would be exceeded.

A single transition exceeding the total transaction ceiling is quarantined as `OVERSIZED_CANONICAL_TRANSACTION` and fails the reference acceptance contract.

## 26.5.8 Shared-memory credit ownership states

The coordinator owns the compiled-result SHM subpool. Each worker receives a bounded local grant.

Credit and segment accounting has three disjoint states:

```text
grant_free_bytes
worker_owned_segments
coordinator_owned_segments
```

A worker allocation moves bytes:

```text
grant_free → worker_owned
```

When the coordinator receives and validates a result descriptor, ownership moves:

```text
worker_owned → coordinator_owned
```

After decode/unlink:

```text
coordinator_owned → grant_free
```

Worker death reclaims only unused grant bytes plus segments still `worker_owned`. `coordinator_owned` segments remain valid until consumed/reclaimed by the coordinator.

Initial compiled-result policy:

```text
compiled-result/derivation SHM subpool <= 384 MiB
worker credit grant                    = 48 MiB
maximum individual segment             = 32 MiB
low-credit refill threshold            = 12 MiB
```

Names use:

```text
process_run_epoch / worker_id / local_lease_counter
```

Startup scans stale prior-process epochs and reclaims orphan segments without affecting scientific identities.

## 26.5.9 Bounded decode and WAL-pending state

Decode pools, decoded transaction queues, WAL group buffers, and staged transaction overlays are all byte-bounded.

When WAL durability or disk retention becomes the bottleneck, backpressure reaches publication instead of allowing decoded/staged work to accumulate indefinitely.

## 26.5.10 Framed WAL durability then staged canonical visibility

Transactions receive contiguous WAL LSNs and enter bounded group commit.

```text
framed WAL append
→ group commit footer
→ fsync
→ wal_durable_lsn advances
```

The canonical reducer then processes the next unapplied LSN:

```text
private TransactionOverlay
→ bounded continuation fragments
→ complete validation
→ atomic canonical-root swap
→ canonical_applied_lsn advances
```

No partial fragment is visible. A post-WAL application failure is fatal and recovered by WAL replay.

## 26.5.11 Cross-producer order-independent scientific reductions

Arrival-order independence is required for all scientific cross-producer aggregation.

Canonical merge operators use:

```text
evidence sets                 = stable-id set union
support                       = cardinality / exact integer reduction
representative evidence       = deterministic stable rank
confidence sufficient stats   = fixed-point integer tuple
grounding evidence            = keyed idempotent evidence union
maturity flags                = monotonic max/OR under explicit state machine
stage evidence                = deterministic closed-epoch aggregate
normalization statistics      = frozen during epoch; recomputed/updated at epoch cut
```

No last-arrival-wins or order-dependent floating accumulation is allowed.

Tests permute unrelated producer admission while preserving each producer's causal stream and assert identical scientific canonical state.

## 26.5.12 Atomic HGT evidence materialization

HGT consumes durable WAL evidence only.

It writes complete contiguous WAL ranges into immutable HGT dataset segments. A segment becomes visible only after:

```text
segment complete
→ checksum valid
→ segment fsync
→ immutable rename
→ manifest fsync
→ manifest atomic rename
→ hgt_checkpoint_lsn advance
```

Partial segments and orphan files are never checkpoint-visible and are reclaimed on startup.

## 26.5.13 Bounded persistent signature indexing

M1 support and derivation-dirty state live in the persistent chunked `SignatureIndexStore`. Resident scheduler state contains only bounded delta pages, bounded page cache, bounded dirty-priority window, and a saved scan cursor.

No full in-memory structural-signature map is required.

## 26.5.14 Leased derivation and deterministic merge algebra

A derivation task identity is:

```text
(signature, target_support, derivation_schema_version)
```

with lease metadata:

```text
lease_epoch
attempt
lease_deadline
```

Initial scheduler bounds:

```text
pending priority entries     <= 4,096
in-flight leases             <= 64 × derivation_workers × 4
completed result rows        <= 2,048
worker batch                 <= 64
lease timeout                = initial 30 s
retry limit                  = 3 before explicit failure/quarantine
```

The resulting abstraction has a stable canonical identity derived from:

```text
derivation_schema_version
abstraction type
structural key
canonical sorted evidence identity set/digest
```

Merging independently completed derivations uses:

```text
provenance/evidence       → set union sorted by stable identity
support                   → cardinality of merged evidence set
integer/fixed-point stats → component-wise exact addition over unique evidence
maturity/validation       → monotonic state-machine max
representative payload    → deterministic structural/stable-id rank
```

Two results claiming the same canonical abstraction identity but incompatible structural payloads produce `DERIVATION_MERGE_CONFLICT`; the runtime never resolves them by completion order.

Derivation publication itself is a canonical mutation and therefore enters the same WAL → staged overlay → atomic root publication path.

There is no global derivation-result order. A straggler cannot block unrelated tasks.

## 26.5.15 Mode-specific developmental and training publication

### MATCHED_REASONING

Active actors remain bound to `EpochInferenceView(E)` while new evidence accumulates behind it.

Epoch closure is:

```text
EvidenceCut(E)
→ DevelopmentalCut(E)
→ TrainingCut(E)
→ PolicyProjectionCut(E+1)
→ EpochInferenceView(E+1)
```

`DevelopmentalCut(E)` contains every actor-visible Hydra developmental operator permitted to affect E+1 and selects finite work deterministically.

`TrainingCut(E)` contains the exact training evidence/replay/optimizer/evaluation plan.

No background developmental mutation, wall-clock policy refresh, derivation completion race, lifecycle pass, grounding maturity race, transfer-validation race, or candidate-model completion race can alter E+1 outside those cuts.

### ASYNC_DEVELOPMENT

No epoch publication cut is imposed.

Independent Hydra developmental proposals commit through WAL + atomic `CanonicalStateHandle` publication as soon as their causal/version preconditions are satisfied.

Baseline H17 disables learned HGT→Hydra developmental feedback and makes no schedule-identical trajectory claim.

## 26.5.16 Bounded immutable EpochInferenceViews and cold-cache behavior

In `MATCHED_REASONING`, an `EpochInferenceView` pins:

```text
CanonicalStateHandle
PolicyVersion
ModelVersion
stage/normalization state
schema versions
```

Expected environment projection/retrieval pages are prefetched before the epoch starts and remain within configured global/per-view byte ceilings.

If a required page is unexpectedly absent during active sampling:

```text
use the already-bound deterministic fallback/page
record epoch_view_cold_miss
queue bounded prefetch for the next epoch view
```

Action selection never reads a newer live canonical root or performs an unbounded scan to repair the current epoch view.

## 26.5.17 Snapshot interaction

Snapshot cuts pin one complete `CanonicalStateHandle`; `snapshot_applied_lsn` equals that handle's `canonical_applied_lsn`. A cut swaps only bounded builder/root metadata and never exposes a staged transaction overlay.

## 26.5.18 Whole-system memory and durable-storage governor integration

Frequent cheap sampling uses:

```text
process-tree RSS
tracked SHM
snapshot external buffers
swap
host MemAvailable
retained WAL bytes
durable bytes by class + global durable bytes
filesystem free bytes/fraction
```

USS reconciliation is less frequent.

Memory and durable-storage pressure share the NORMAL / COMPACTING / HARD_PRESSURE_DRAIN / RECOVERING control model. Hard filesystem/global-durable pressure stops new production and candidate checkpoint creation before disk exhaustion.

## 26.5.19 Backpressure matrix

Every high-volume stage has finite capacity:

```text
transport slab pool                 global byte ceiling
stage/shard descriptor bundles      row + descriptor-count ceiling
producer gap window                 per-producer batch + byte ceiling
publication                         row + byte ceiling
pending ingest                      row + byte + batch ceiling
compiled-result SHM                 grant + ownership + global byte ceiling
decode/WAL pending                  row + byte ceiling
WAL group                           age + byte ceiling
retained WAL                        disk byte + filesystem free-space ceiling
canonical transaction               total row + mutation-byte + work ceiling
transaction overlay                 byte + work ceiling
continuation fragment               mutation-byte + work + latency target
SignatureIndex resident cache       byte ceiling
derivation scheduler                entry + lease + result ceilings
HGT writer                          queued-byte + WAL-lag ceiling
policy projection                   global entry + byte ceiling
actor epoch view                    per-view byte ceiling
snapshot writer                     external-buffer + generation ceiling
resident canonical memory           lifecycle limits
whole Hydra host                    accounted working set + MemAvailable + swap
```

## 26.5.20 Completion, shutdown, and recovery proof

Epoch/shutdown completion requires:

```text
all actor final producer ranges observed
→ producer gap windows empty
→ transport slabs released
→ publication/ingest/decode drained
→ WAL pending groups durable
→ canonical_applied_lsn reaches required wal_durable_lsn boundary
→ HGT checkpoint reaches required epoch boundary
→ derivation completed or persistent dirty state records unresolved targets
→ requested snapshot complete
→ compiled-result SHM ownership sets empty/reclaimed
```

Transport equality:

```text
produced == causally_admitted == published == WAL_committed == canonically_applied
```

Persistence frontiers are checked independently rather than conflated.

## 26.5.21 End-to-end invariants

1. **Producer causality:** each producer stream is shard-affine and contiguous.
2. **Scientific identity stability:** process restart cannot change evidence identity.
3. **No fictitious cross-producer causality:** unrelated interleaving is operational metadata only.
4. **Order-independent science:** cross-producer reductions use explicit commutative/idempotent merge algebra.
5. **WAL before visibility:** `wal_durable_lsn` may lead `canonical_applied_lsn`; durability alone never exposes partial state.
6. **Transactional visibility:** continuation fragments modify private staged state and publish one atomic root.
7. **Correct snapshot frontier:** snapshots record exactly `snapshot_applied_lsn <= canonical_applied_lsn`.
8. **Atomic HGT advancement:** `hgt_checkpoint_lsn` advances only after complete dataset segment + manifest publication.
9. **Consumer-safe WAL retention:** WAL reclamation uses the minimum durable-consumer frontier.
10. **Deterministic feedback:** policy/model/stage/normalization changes publish only at closed scientific epoch cuts.
11. **Finite working set:** every resident queue/cache/slab/projection/lease pool has a row/byte/work bound or bounded persistent representation.
12. **No derivation HOL dependency:** leases and deterministic merge isolate stragglers.
13. **No memory double counting:** governor categories are mutually exclusive and reconciled against host pressure.
14. **Closed-loop learning:** durable canonical evidence, derivation, HGT learning, and bounded actor epoch views feed the next sampling epoch.

# 26.6 Research experiment contracts

Research hypotheses are represented by explicit immutable manifests rather than inferred from ad-hoc CLI combinations.

## 26.6.1 ExperimentManifest

Every research run declares:

```text
ExperimentManifestId
research_contract_version
hypotheses under test
scientific visibility mode
ScientificConfigId
StructuralPriorProfile
GroundingCondition
InteractionOpportunityManifest
environment/curriculum manifests
training/evaluation split
model/reasoner condition
random seeds
falsification thresholds/statistics
```

Only declared experimental factors may differ inside a matched comparison.

## 26.6.2 H16 GroundingExperimentManifest

Contains:

```text
C0/C1/C2/C3 condition
G0-G5 stage
symbol stream identity/exposure count
alignment/shuffle permutation
interaction opportunity manifest
symbol-removal evaluation phase
bidirectional transfer tasks/metrics
novel-composition split
```

C0-C3 matched groups preserve marginal exposure where required.

## 26.6.3 H18 StructuralPriorProfile

Contains:

```text
enabled structural relations R_O
coordinate mode
adjacency mode
topology mode
fixed/changing permutation seed and schedule
raw-observation access prohibition
```

It is applied before Hydra/GNN encoding.

## 26.6.4 H19 ReasoningCondition

Enumerated primary conditions:

```text
HYDRA_ONLY
HYDRA_HGT_SINGLE_PASS
HYDRA_HGT_RECURSIVE
```

Additional controls:

```text
RANDOM_UNTRAINED_RELATIONAL
FROZEN_EARLY_MODEL
CONTINUALLY_TRAINED_CURRENT_MODEL
EXPLICIT_SIMILARITY
LEARNED_SIMILARITY
MODEL_ONLY_POLICY   # where experimentally feasible
```

Every H19 comparison group references the same `InteractionOpportunityManifest`.

# 27. Research ablations

## 27.1 H16 — Cross-modal grounded meaning

Required conditions:

```text
C0 interaction only
C1 symbols only
C2 aligned interaction + symbols
C3 shuffled interaction + symbols
```

Required claims are tested through:

```text
symbol → interaction held-out transfer
interaction → symbol held-out transfer
novel grounded composition
symbol-removal persistence
C2 > C0
C2 > C1
C2 > C3
```

Improved symbol prediction by itself is insufficient.

## 27.2 H17 — Developmental stability without a global behavioral objective

Use `ASYNC_DEVELOPMENT`.

Vary independently:

```text
promotion/demotion hysteresis
retention/evidence thresholds
worker/update rates
peer update rates
bounded mutation settings
```

Measure:

```text
R_rev
R_churn
P_Δt
novel useful memory formation
prediction quality
transfer quality
```

Do not synchronize developmental processes through H19 epoch barriers.

## 27.3 H18 — Structural prior dependence

Required profile series:

```text
S0 identity + temporal order
S1 + equality
S2 + coordinates
S3 + adjacency
```

Required spatial controls:

```text
fixed coordinate permutation
changing coordinate permutation where meaningful
adjacency/topology withheld
```

Underlying environment dynamics and matched interaction opportunity stay fixed.

## 27.4 H19 — Learned relational reasoning

Primary matched conditions:

```text
Hydra only
Hydra + learned relational single-pass
Hydra + learned relational recursive deliberation
```

Additional controls:

```text
random/untrained relational representations
frozen early learned model
continually trained current model
explicit similarity versus learned similarity
model-only learned policy where experimentally feasible
```

All conditions share the same `InteractionOpportunityManifest`.

Primary comparisons:

```text
sample efficiency
prediction quality
held-out task success
cross-family transfer
memory growth
reasoning cost
historical-stage retention / catastrophic forgetting
trajectory efficiency
```

## 27.5 Semantic-abstention audit

Every core-experiment report verifies:

```text
no pretrained LLM
no pretrained lexical/text/image/multimodal embedding
no dictionary/ontology/synonym resource
no semantic parser
no manually supplied symbol↔world mapping
no object/goal/task-semantic labels in learner features
exact declared R_O structural priors
```

A failure of this audit invalidates a semantic-free emergence claim for that run.

# 28. Primary research questions

v9.7.15 evaluates the v0.7.0 questions without adding a new optimization hypothesis.

### Q1
Does learned relational reasoning improve retrieval and structural correspondence over explicit descriptors alone?

### Q2
Does learned consequence prediction improve candidate refinement during recursive deliberation?

### Q3
Does one shared relational model discover reusable structure across environment families without privileged semantics?

### Q4
Under matched interaction opportunities and starting states, does H19 learned relational reasoning improve held-out behavior over Hydra-only reasoning?

### Q5
Does recursive learned deliberation improve behavior relative to a single learned pass at measurable additional reasoning cost?

### Q6
Does H16 aligned cross-modal experience produce bidirectional held-out transfer beyond C0, C1, and C3?

### Q7
Can independent asynchronous developmental processes satisfy H17 stability/plasticity criteria without a global behavioral objective?

### Q8
How do H18 structural-prior removals/permutations change formation time, transfer, planning, and grounding?

### Q9
Does the observed developmental sequence and the broader v0.7.0 prediction set survive explicit falsification testing?

# 28.1 ResearchPredictionRegistry and developmental falsification traceability

The implementation maintains a versioned `ResearchPredictionRegistry`.

Each prediction entry contains:

```text
PredictionId
research-paper section
hypothesis/prediction text identifier
required observables
required controls
predeclared statistic
predeclared threshold or comparison rule
eligible experiment manifests
result: SUPPORTED / VIOLATED / INSUFFICIENT_EVIDENCE
evidence artifact references
```

The implementation never changes memory behavior to force a prediction to pass.

## 28.1.1 DevelopmentalMilestoneLedger

Record the first evidence-qualified formation event for:

```text
stable M1 contingency
M2 transformation family
persistent carrier hypothesis
stable cross-context functional role
M4 concept candidate
validated transferable M4 concept
future-option motif
effective planning/replanning
M5/M6/M7 late structure
```

Each event records:

```text
scientific evidence id / canonical LSN
sampling/developmental generation
support/evidence threshold reached
environment/context scope
provenance
validation state
```

Primary ordering tests include:

```text
P1 contingency before validated concept
P2 transformation family before carrier
P3 carrier before functional role
P4 role reuse before validated concept
P5 future-option motif before effective planning
P6 world-model structures late relative to lower levels
```

A reversal is reported as a theory violation candidate, not hidden by scheduler logic.

## 28.1.2 Exact v0.7.0 falsification registry

The registry contains first-class entries for every core falsification criterion from v0.7.0:

```text
F1  — Semantic-Prior Necessity
F2  — Object-First Emergence
F3  — Appearance-Dominated Transfer
F4  — World-Model Necessity
F5  — Failure of Prediction-Violation Allocation
F6  — Failure of Future-Option Contribution
F7  — Failure of Explanatory Reach
F8  — Failure of Context Refinement
F9  — Failure of Empirical Transfer Validation
F10 — Failure of Developmental Ordering
F11 — Failure of Outcome/Strategy Separation
F12 — Failure of Emergent Target-Like Structure
F13 — Failure of Efficiency Emergence
F14 — Failure of Conditional Compression
F15 — Persistent Architectural Thrashing
F16 — Failure of Grounded Symbolic Emergence
F17 — Failure of Developmental Stability
F18 — Failure or Mischaracterization of Structural-Prior Dependence

H19-REJECT — causally controlled held-out learned-reasoner comparison
```

Each entry preserves the paper's criterion and binds it to explicit experiments/observables.

### F1 — Semantic-Prior Necessity

Detect whether the hierarchy requires predefined object identity, semantic categories, task goals, or reward meaning in transformation representation.

Primary evidence:

```text
semantic-abstention audit
structural-prior profiles
successful/failed emergence without semantic features
```

### F2 — Object-First Emergence

Compare evidence-qualified object/carrier-like structures against transformation-family and relational-role milestones.

### F3 — Appearance-Dominated Transfer

Matched held-out transfer compares appearance similarity against:

```text
role similarity
graph similarity
future-option similarity
```

with sample size/evaluation opportunity controlled.

### F4 — World-Model Necessity

Use milestone ordering to test whether an integrated M5/world model is required before stable contingencies, roles, or concept candidates.

### F5 — Failure of Prediction-Violation Allocation

Once expectations exist, matched violating vs non-violating events test whether prediction error improves:

```text
replay allocation
context search
corrective processing
```

### F6 — Failure of Future-Option Contribution

Ablate bounded future-option structure and compare causal/predictive value for:

```text
attention
retention
transfer
planning
```

### F7 — Failure of Explanatory Reach

Controlled comparisons test promotion prediction by explanatory reach versus frequency.

### F8 — Failure of Context Refinement

Where an admissible hidden contextual partition exists, compare context refinement against immediate concept replacement on held-out prediction/transfer.

### F9 — Failure of Empirical Transfer Validation

Compare validated concepts against structural recurrence without held-out causal reuse validation.

### F10 — Failure of Developmental Ordering

Use `DevelopmentalMilestoneLedger` to detect systematic reversal of declared dependency predictions.

### F11 — Failure of Outcome/Strategy Separation

Ablate persistent outcome-equivalence identity and test:

```text
outcome recognition
alternative-strategy reuse
replanning
outcome-class stability under held-out consequences
```

### F12 — Failure of Emergent Target-Like Structure

Test whether goal-like behavior requires an externally supplied goal/reward representation and whether stable learned preference over M6 outcome-equivalence classes emerges from interaction-derived evidence.

### F13 — Failure of Efficiency Emergence

After admissible comparable outcomes exist, test whether lower-cost trajectories gain reuse/retention and whether indiscriminate shortest-path pressure harms future-option/outcome quality.

### F14 — Failure of Conditional Compression

In recurrently compressible bounded-novelty environments, test persistent-memory growth against accumulated experience while holding predictive/explanatory/transfer/planning quality.

### F15 — Persistent Architectural Thrashing

Use H17/runtime telemetry to test whether destructive oscillation persists despite declared consistency, hysteresis, publication, and bounded-mutation constraints.

### F16 — Failure of Grounded Symbolic Emergence

Use C0-C3 and G0-G5 to test:

```text
aligned vs shuffled-symbol advantage
cross-modal rather than within-symbol explanation
absence of pretrained/manual semantic mappings
held-out behavioral effect of symbols
interaction-derived constraint on novel symbol interpretation
```

### F17 — Failure of Developmental Stability

Use baseline `ASYNC_DEVELOPMENT` with learned developmental feedback disabled. Reject H17 if no admissible parameter region supports both stability and continued plasticity, or if stability requires an undeclared centralized objective.

### F18 — Failure or Mischaracterization of Structural-Prior Dependence

Use the formal H18 transforms and anti-leak audit. Reject/revise H18 if ablations fail to produce the predicted systematic differences or claimed emergence depends on undeclared structure.

### H19-REJECT

Reject/revise H19 if a causally controlled, held-out, fixed-`TrialManifest` comparison shows:

```text
no reproducible learned-reasoner advantage over Hydra-only
or
no recursive advantage over matched single-pass where multi-step refinement is required
or
gains depend on semantic shortcuts
or
gains depend on future-information leakage
or
gains bypass M0-M7 developmental validation
or
gains depend on unmatched starting states/interaction opportunities
```

Every registry result is:

```text
SUPPORTED
VIOLATED
INSUFFICIENT_EVIDENCE
```

and links directly to experiment manifests, stored statistics, and evidence artifacts.

# 29. Failure analysis

Every poor behavioral outcome should be attributable to one or more layers:

```text
observation encoding
memory formation
subgraph retrieval
GNN representation
correspondence
consequence prediction
candidate generation
candidate refinement
deliberation stopping
target-local execution
```

Telemetry must make these layers separable.

This is critical for scientific interpretation.

---

# 30. Implementation layout

Suggested GNN additions:

```text
src/v9/gnn/
    schema.py
    graph_builder.py
    features.py
    subgraph.py
    model.py
    message_passing.py
    heads.py
    inference.py
    trainer.py
    replay_buffer.py
    objectives.py
    publication.py
    checkpoint.py
    telemetry.py

src/v9/cognition/
    deliberation.py
    reasoning_workspace.py
    reasoning_policy.py
```

Performance-bounded sampling and memory update remain integrated into the authoritative runtime modules:

```text
src/v9/runtime/multiprocess.py
    producer-affinity shard assignment
    TransportSlabPool envelope writer
    fixed-size slab descriptors
    fair stage/shard descriptor routing
    scientific_run_id / sampling_epoch_id propagation
    process_run_epoch namespace isolation

src/v9/runtime/parallel_memory_coordinator.py
    ProducerCausalAdmission
    dual-budget publication admission
    WAL/disk-pressure scheduling
    scientific epoch cut coordination
    whole-system memory-governor scheduling

src/v9/runtime/memory_pipeline.py
    total transaction work/byte estimation
    canonical transaction construction
    deterministic scientific evidence ids
    derivation merge intents

src/v9/runtime/canonical_wal.py
    framed WAL groups
    CRC/checksum validation
    group commit + wal_durable_lsn
    durable-consumer checkpoint registry
    retention and disk-pressure control

src/v9/runtime/canonical_transaction.py
    private TransactionOverlay
    continuation-fragment execution
    transaction validation
    atomic CanonicalStateHandle publication

src/v9/runtime/canonical_store.py
    immutable graph/index chunks
    structural sharing
    root construction
    handle pin/release
    bounded chunk GC
    orphan-chunk recovery

src/v9/runtime/publication_throughput.py
    bounded decode/WAL pending state
    compiled-result SHM ownership accounting
    worker-local credit pools

src/v9/runtime/runtime.py
src/v9/runtime/__init__.py
    dedicated producer-sequence synchronization
    atomic CanonicalStateHandle publication/access
    deterministic stage/normalization epoch aggregates

src/v9/runtime/epoch_inference_view.py
    ExperimentManifest-derived identities
    EvidenceCut / DevelopmentalCut / TrainingCut manifests
    EpochInferenceView construction, pinning, release

src/v9/runtime/developmental_cut.py
    deterministic operator work plans
    finite actor-visible developmental closure
    operator budgets/status outcomes
    MATCHED_REASONING background-mutation gate

src/v9/runtime/scientific_modes.py
    ASYNC_DEVELOPMENT / MATCHED_REASONING visibility modes
    H17 asynchronous publication semantics

src/v9/research/experiment_manifest.py
    GroundingCondition C0-C3
    StructuralPriorProfile S0-S3 + permutation/topology controls
    InteractionOpportunityManifest
    ReasoningCondition

src/v9/research/prediction_registry.py
    ResearchPredictionRegistry
    DevelopmentalMilestoneLedger
    prediction/falsification result artifacts

src/v9/runtime/training_evidence.py
    WAL-backed TrainingEvidenceRecord schema
    immutable TrainingEvidenceSegment materialization
    deterministic replay-plan construction
    TrainingCut manifests

src/v9/runtime/policy_projection.py
    globally byte-bounded projection pages
    environment-scoped view prefetch/cache
    published policy version manifests

src/v9/runtime/signature_index.py
    persistent chunked SignatureIndexStore
    bounded resident cache/delta
    derivation dirty cursor

src/v9/runtime/derivation_merge.py
    leased derivation identity
    commutative/idempotent merge algebra
    conflict detection
    canonical derivation WAL intents

src/v9/runtime/memory_governor.py
    process-tree memory accounting
    tracked SHM/external-buffer accounting
    MemAvailable/swap pressure

src/v9/runtime/storage_governor.py
    durable bytes by object class
    WAL/snapshot/training-evidence/model retention
    reference-aware GC
    filesystem pressure integration

src/v9/runtime/chunked_snapshot.py
src/v9/runtime/snapshot_backend.py
    immutable chunk roots
    snapshot_applied_lsn
    generation pin/reference counting
    complete-manifest publication

src/v9/runtime/shared_batch_transport.py
    bounded TransportSlabPool
    fixed binary descriptors
    compiled-result SHM grants
    grant_free / worker_owned / coordinator_owned accounting
    stale process_run_epoch orphan reclamation

src/v9/runtime/lifecycle.py
    bounded persistent-index cursor
    chunk-aware logical retirement/reclamation

src/v9/hgt/epoch_dataset.py
    atomic TrainingEvidenceSegments
    COW TrainingEvidenceManifest publication
    hgt_checkpoint_lsn
    WAL transaction checksum verification

src/v9/hgt/training_cut.py
    deterministic replay plan
    fixed optimizer work plan
    deterministic RNG/kernel configuration
    candidate checkpoint/evaluation provenance
```

Existing authoritative v9 modules remain the integration points for memory, similarity, correspondence, transfer, prediction, planning, replay, grounding, and lifecycle semantics.


---

# 31. Acceptance criteria

v9.7.16 is operational when:

1. Hydra builds a typed bounded GNN subgraph from published memory.
2. The GNN runs inference on mixed M-level graphs.
3. One shared model processes multiple environment families.
4. Relevance scores affect bounded retrieval.
5. Correspondence scores affect candidate ranking.
6. Consequence predictions enter deliberation candidate evaluation.
7. Strategy scores can change the selected candidate.
8. Recursive deliberation supports multiple GNN passes.
9. GNN training examples are generated from interaction, transfer, grounding, and deliberation evidence.
10. Published model versions are immutable and reproducible.
11. Model version is recorded in every deliberation trace.
12. Historical curriculum retention is measured after later-stage training.
13. Cross-family transfer metrics identify explicit source→target pairs.
14. Hydra-only and hybrid ablations can run on the same curriculum.
15. Failure reports distinguish memory, retrieval, GNN, deliberation, and execution causes.
16. Inference and candidate training can coexist on one GPU under configured VRAM and latency budgets.
17. Every deliberation episode binds exactly one immutable ModelVersion.
18. Model candidate training may proceed during sampling, but actor-visible model publication occurs only through an `EpochInferenceView`.
19. Old model weights are released only after active decisions release their model handle.
20. Training duty cycle adapts to inference latency while preserving optimizer progress.
21. Hydra lifecycle events can drive bounded HGT consolidation training.
22. Replay-buffer compression preserves structural and environment-family diversity.
23. Major memory consolidation records HGT retention evidence.
24. Reactivated Hydra structure can receive temporary replay-priority recovery.
25. HGT can rank comparable M7 strategies for the same M6 outcome.
26. Recursive optimization can produce lower-cost outcome-preserving trajectories.
27. Optimization telemetry separates predicted and realized reliability/cost.
28. Cross-family optimization transfer is reported with explicit source→target provenance.
29. Primary telemetry is limited to the unified dashboard set.
30. HGT training failures are distinguishable from Hydra, retrieval, deliberation, and execution failures.
31. Every promoted HGT model records current-stage, retention, transfer, reasoning, latency, and promotion deltas.
32. Detailed telemetry carries sufficient provenance to trace regressions to model, graph, curriculum, and environment state.
33. Exploration adapts to branching factor, context-local action coverage, learned preference, and persisted per-game stagnation.
34. Environment viability profiles persist across restart and restore developmental exploration state.
35. Sampling budgets are derived from complete-episode opportunities and environment horizon on a per-actor basis.
36. Game-level outcomes are persisted in `game_results.log` and current-run wins are exposed in telemetry.
37. Environment evidence confidence propagates into heterogeneous graph memory payloads.
38. In `MATCHED_REASONING`, all actors within one scientific sampling epoch bind the same immutable `EpochInferenceView`; `ASYNC_DEVELOPMENT` is exempt from this epoch-freezing rule.
39. HGT batching is measurable and scalable according to GPU-memory headroom and effective-batch telemetry.
40. Actor transport, stage/shard routing, producer admission, ingestion, SHM, WAL, canonical overlays, derivation, HGT, policy projection, snapshot, and disk-retention stages each expose explicit hard capacity.
41. Every producer is assigned one stable shard for its actor lifetime, and causal admission proves contiguous producer sequence without gaps or duplicates.
42. Permuting admission order of unrelated producer streams while preserving each producer's order produces identical scientific canonical state; operational WAL/ingestion metadata may differ.
43. Cross-producer support, representative selection, grounding accumulation, confidence statistics, stage evidence, and derivation dirty state use explicit stable-identity order-independent reductions.
44. `scientific_run_id` and `sampling_epoch_id` survive process restart; `process_run_epoch` changes without changing scientific evidence IDs or canonical UIDs.
45. Producer-sequence synchronization is independent from canonical visibility/mutation synchronization.
46. Actor-policy generation checks are O(1); active sampling never performs an unbounded policy reconstruction.
47. Policy projection respects global entry/byte ceilings; each actor epoch view remains within its configured byte ceiling.
48. Expected actor views are prefetched at the epoch barrier; cold misses during sampling use the bound epoch view and cannot trigger an unbounded graph scan.
49. Stage and normalization transitions occur only on deterministic closed epoch evidence cuts.
50. Transition payload bytes are serialized once into transport slabs; stage/shard/coordinator hops pass fixed-size descriptors and do not pickle/copy the payload.
51. Total tracked Hydra SHM remains within the configured global ceiling, with transport and compiled-result subpool ceilings independently enforced.
52. Compiled-result SHM accounting distinguishes grant-free, worker-owned, and coordinator-owned bytes; worker death never reclaims coordinator-owned live segments.
53. SignatureIndexStore resident mutable delta + page cache remain bounded while dirty derivation state survives restart.
54. Publication/ingestion admission is constrained by exact carried row and byte measurements.
55. Every WAL durable group has valid physical framing, footer, group checksum, and transaction-frame checksums.
56. `wal_durable_lsn >= canonical_applied_lsn >= snapshot_applied_lsn` holds at all times.
57. Canonical mutation never begins before its WAL transaction is durable.
58. Multi-fragment canonical transactions mutate only a private staged overlay; no reader can observe partial fragments.
59. `canonical_applied_lsn` advances only after final transaction validation and atomic root publication.
60. A post-WAL/pre-publication crash or apply failure recovers by replay without double application.
61. Snapshots record exactly `snapshot_applied_lsn`, never merely the durable WAL tail.
62. HGT dataset segments become visible only after complete segment fsync and atomic manifest publication.
63. `hgt_checkpoint_lsn` advances only to the end of an atomically manifested contiguous WAL range and never beyond `wal_durable_lsn`.
64. WAL prefix reclamation uses the minimum registered durable-consumer checkpoint and never removes evidence still needed by HGT or recovery.
65. Retained WAL bytes and filesystem free space obey soft/hard disk-pressure thresholds; hard pressure stops new production before filesystem exhaustion.
66. WAL group-commit pending bytes/age remain bounded and storage slowdown backpressures publication.
67. A compiled canonical transaction obeys total row/byte/work ceilings; a single transition exceeding them is explicitly quarantined as `OVERSIZED_CANONICAL_TRANSACTION`.
68. Canonical continuation-fragment lock/execution latency meets configured targets; every primitive exception is reported.
69. Derivation tasks use leases, retries, idempotent identities, and stale-result rejection; one straggler cannot block unrelated signatures.
70. Derivation abstraction merge uses evidence-set union, exact/fixed-point sufficient statistics, monotonic maturity state, deterministic representatives, and explicit conflict failure rather than completion order.
71. Derivation publication enters the same WAL → staged-overlay → atomic-root path as interaction memory mutation.
72. Snapshot cuts swap only bounded builder/root metadata and never expose a staged transaction.
73. Memory-governor accounting categories are mutually exclusive and reconciled against host MemAvailable and swap.
74. On a 64-GiB reference host, steady-state accounted working set remains below the configured soft target and cannot remain above the hard target for more than one governor sampling interval.
75. Steady-state process-tree swap remains below the configured target and host `MemAvailable` remains above the emergency floor outside transient recovery.
76. Producer gap age remains within the configured healthy bound; exceeding it raises an explicit transport-gap failure.
77. Snapshot generation-cut p95 remains within the configured target and cut metadata stays under its fixed bound.
78. USS reconciliation averages less than 1% coordinator CPU on the reference workload.
79. Long-run soak testing spans at least 10 equivalent epochs and at least 1,000,000 admitted transitions when the environment set can supply that volume.
80. Normalized throughput in the final three soak epochs remains within 20% of the median normalized throughput of warm-up epochs 2-4 for equivalent workload and resident limits.
81. Queue occupancy, transport/compiled SHM, WAL pending/retained bytes, signature-index cache, derivation state, HGT lag, snapshot buffers, policy projection bytes, and host working set show no monotonic unbounded trend.
82. Epoch completion proves `produced == causally_admitted == published == WAL_committed == canonically_applied` for the required epoch boundary.
83. Restart from newest complete snapshot plus retained WAL reproduces canonical scientific state, producer-sequence state, SignatureIndex dirty state, policy/model publication cut, and persistence frontiers before actor launch.
84. In `MATCHED_REASONING`, re-running the same seeded scientific configuration with different process scheduling yields the same epoch-by-epoch scientific evidence identities, cut manifests, actor-visible views, and final scientific canonical state; H17 `ASYNC_DEVELOPMENT` instead uses statistical perturbation criteria.
85. Crash-injection tests cover torn WAL group, durable WAL + pre-apply crash, mid-overlay fragment crash, post-apply/pre-root-swap crash, post-root-swap/pre-snapshot crash, partial HGT segment, post-segment/pre-manifest crash, worker SHM crash, snapshot writer crash, derivation lease expiry, and disk-pressure drain.
86. In `MATCHED_REASONING`, every actor in one sampling epoch references exactly one immutable `EpochInferenceView`; in `ASYNC_DEVELOPMENT`, readers bind complete `CanonicalStateHandle`s without this epoch constraint.
87. `CanonicalStateHandle` atomically contains canonical graph roots, authoritative index roots, and `canonical_applied_lsn`; tests cannot observe a root/frontier mismatch.
88. `DevelopmentalCut(E)` is reproducible from `EvidenceCut(E)` plus versioned operator dependencies and has fixed finite per-operator work budgets; varying worker completion order produces identical E+1 actor-visible developmental state.
89. `TrainingCut(E)` fixes evidence manifest, replay order, RNG streams, optimizer-step count, microbatch/accumulation plan, objective/schema versions, and determinism mode.
90. Running the same `TrainingCut` twice in scientific deterministic mode produces the configured deterministic equivalence result for model/checkpoint outputs.
91. Candidate training wall-clock availability changes completion time only; it cannot change the number/order of optimizer updates used for the candidate evaluated at E+1.
92. Interaction, reasoning, delayed-supervision, consolidation, transfer, grounding, and derivation HGT labels all become WAL-backed `TrainingEvidenceRecord`s; no independent mutable training store exists.
93. Replay buffers and sampling strata can be deleted/rebuilt from TrainingEvidence manifests without changing the set of durable training examples.
94. `experiment_id`, `replicate_id`, `producer_id`, `environment_instance_id`, `episode_id`, and scientific evidence IDs are deterministic functions of the immutable ExperimentManifest and stable ordinals.
95. Process PID, worker slot, process launch order, process restart count, and queue arrival order do not affect any scientific identifier.
96. Global durable-storage usage obeys finite per-class and aggregate ceilings; reference-aware GC never deletes objects reachable from active EpochInferenceViews, recovery snapshots, TrainingCuts, or required model provenance.
97. Hard durable-storage pressure stops actor admission and new candidate checkpoint creation before filesystem exhaustion.
98. Epoch barrier work is bounded by deterministic derivation/training/projection work-count budgets; failed work produces explicit deterministic failure/no-promotion outcomes rather than timing-dependent partial publication.
99. Restart restores or reconstructs the exact last complete `EpochInferenceView` and cannot launch actors against an unverified mix of canonical roots, indexes, model, policy, or normalization/stage state.
100. `MATCHED_REASONING` schedule-variation testing compares epoch-by-epoch `EpochInferenceView`, scientific evidence identities, DevelopmentalCut manifests, TrainingCut manifests, selected ModelVersions, and final scientific canonical state; H17 schedule perturbations are evaluated statistically instead of requiring identical trajectories.
101. Core-experiment feature construction rejects pretrained LLMs, pretrained lexical/text/image/multimodal embeddings, dictionaries, ontologies, synonym tables, semantic parsers, manually supplied symbol-to-world mappings, and task-semantic object/goal labels.
102. Opaque symbol embeddings, when used, initialize from scratch and learn only from experiment evidence.
103. H16 C0/C1/C2/C3 are separate immutable experimental conditions with auditable matched exposure manifests.
104. C3 preserves declared marginal symbol/interactions exposure while deterministically destroying true cross-modal alignment.
105. H16 evaluation reports both symbol→interaction and interaction→symbol held-out transfer, novel composition, and symbol-removal persistence.
106. H17 runs use `ASYNC_DEVELOPMENT`; developmental workers publish independently through versioned atomic state and are not synchronized by H19 epoch barriers.
107. H17 experiments vary hysteresis, thresholds, and update/peer rates and report reversal rate, churn, structural persistence, useful novelty, prediction quality, and transfer quality.
108. H18 `StructuralPriorProfile` is applied before Hydra/GNN feature construction and downstream code cannot access ablated raw coordinates/adjacency/topology through side channels.
109. H18 includes S0-S3 plus fixed-coordinate-permutation, changing-coordinate-permutation where meaningful, and topology/adjacency-withheld controls.
110. H19 primary/control conditions in one comparison group share exactly one `InteractionOpportunityManifest`, including start/reset seeds and fixed per-job interaction ceilings.
111. Adaptive viability/exploration cannot grant additional steps, episodes, environment instances, actor jobs, or curriculum exposure to one matched H19 condition.
112. Model-only learned-policy control is implemented where experimentally feasible or explicitly reported unavailable with reason.
113. `DevelopmentalMilestoneLedger` records evidence-qualified contingency, family, carrier, role, concept-candidate, and validated-concept formation without forcing their predicted order.
114. `ResearchPredictionRegistry` maps v0.7.0 developmental/falsification predictions to concrete observables, controls, statistics, thresholds, and evidence artifacts.
115. Prediction violations are reported as `VIOLATED`; missing tests/data are `INSUFFICIENT_EVIDENCE`; the system never converts them into passing results through implementation constraints.
116. No separate general trajectory-optimization hypothesis, ontology, or optimization-specific loss is required; strategy ranking/refinement uses existing H19 heads and Hydra M6/M7 evidence.
117. Efficiency comparisons are admitted only after Hydra establishes an admissible outcome-comparison relation and are reported under v0.7.0 P26-P29/H19 metrics.
118. `MATCHED_REASONING` has no actor-visible canonical developmental mutation outside `DevelopmentalCut(E)`; context, grounding, transfer, lifecycle, family/carrier/role/concept, M5/M6/M7, and replay-allocation effects are included or explicitly deferred.
119. Baseline H17 uses `LearnedDevelopmentalFeedback=DISABLED`; enabling HGT→Hydra developmental feedback is a separate declared factorial condition.
120. No lifecycle/consolidation event launches independent HGT optimizer work in scientific mode; it emits WAL-backed training evidence or deterministic replay-priority metadata for a future `TrainingCut`.
121. Canonical storage uses immutable bounded chunks, structural sharing, bounded `TransactionOverlay`s, atomic `CanonicalStateHandle` publication, handle pinning, and reference-safe bounded GC.
122. Canonical transaction commit does not require cloning the full resident M0-M7 graph or full authoritative indexes.
123. H18 coordinate permutation transforms learner-visible observation coordinates and coordinate-bearing actions through the same bijection/inverse mapping while preserving underlying environment dynamics.
124. H18 topology-reduced conditions prevent leakage through dense-array indexing, original tensor shape/row-column metadata, adjacency helper features, or adapter side channels.
125. H19 conditions execute the same ordered fixed `TrialManifest`; early trial termination discards unused horizon instead of granting another reset/start state.
126. C3 removes or independently permutes learner-visible cross-stream timestamps, episode ids, producer ids, sequence-position keys, and provenance identifiers that could recover true alignment.
127. The `ResearchPredictionRegistry` contains explicit entries F1 through F18 plus the H19 rejection condition using the paper's criterion names.
128. Scientific deterministic HGT training uses deterministic CUDA kernels, deterministic replacements, or deterministic CPU fallback; unsupported nondeterministic work invalidates the candidate for deterministic H19 publication.

# 32. Final architecture

Hydra v9.7.16 has two explicit scientific execution regimes over one canonical memory substrate.

```text
                    common causal substrate
                              │
        interaction → WAL → immutable-chunk canonical store
                              │
                    CanonicalStateHandle
                       /               \
                      /                 \
        ASYNC_DEVELOPMENT          MATCHED_REASONING
              H17                       H19
```

For H17:

```text
independent Hydra developmental workers
→ WAL-backed causal proposals
→ bounded TransactionOverlay
→ atomic CanonicalStateHandle publication
→ readers may bind newer complete handles
→ schedule/update-rate perturbation replicates
→ stability/plasticity statistics
```

Baseline H17 excludes learned HGT→Hydra developmental feedback.

For H19:

```text
EpochInferenceView(E)
→ fixed matched TrialManifest sampling
→ EvidenceCut(E)
→ DevelopmentalCut(E)
     M1 maturation
     families
     carriers/roles
     contexts
     grounding
     transfer validation
     concepts
     M5/M6/M7
     lifecycle/replay metadata
→ deterministic TrainingCut(E)
→ PolicyProjectionCut(E+1)
→ EpochInferenceView(E+1)
```

No actor-visible developmental mutation occurs outside the declared `DevelopmentalCut`.

For H16:

```text
C0 / C1 / C2 / C3
→ anti-leak aligned/shuffled exposure contracts
→ G0-G5 staged grounding
→ bidirectional held-out transfer
```

For H18:

```text
latent environment elements/actions
→ mathematically defined StructuralPriorTransform
→ observation coordinate/topology transform
→ consistent action inverse transform
→ anti-leak structural interface
```

Canonical persistence uses:

```text
immutable bounded graph/index chunks
+ structural sharing
+ bounded TransactionOverlay
+ atomic CanonicalStateHandle pointer
+ WAL redo authority
+ reference-safe bounded chunk GC
```

HGT learning uses one path only:

```text
WAL-backed TrainingEvidenceRecords
→ immutable TrainingEvidenceSegments
→ deterministic TrainingCut
→ deterministic kernel/CPU fallback contract
→ candidate evaluation/promotion
→ publication through MATCHED_REASONING EpochInferenceView
```

Scientific traceability uses:

```text
ExperimentManifest
ScientificVisibilityMode
StructuralPriorProfile
GroundingCondition
TrialManifest / InteractionOpportunityManifest
DevelopmentalCut
TrainingCut
ResearchPredictionRegistry
DevelopmentalMilestoneLedger
```

The registry explicitly covers F1-F18 and the H19 rejection condition.

This preserves the v0.7.0 theory boundary while making the implementation semantics of asynchronous development, reproducible reasoning, structural-prior ablation, matched interaction, canonical storage, and falsification testing explicit.

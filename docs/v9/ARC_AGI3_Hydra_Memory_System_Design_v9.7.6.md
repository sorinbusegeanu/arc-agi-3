# ARC-AGI-3 Hydra Memory System Design v9.7.6

**Version:** v9.7.6  
**Status:** Hybrid target design  
**Research contract:** `Research_problem_statement_v0631.md`  
**Design predecessors:** Hydra v9.7.5, Hydra v9.7.4, v9.7.3, v9.7.2, v9.7.1, v9.7, v9.6, v9.5, v9.4  
**Purpose:** extend Hydra with an HGT-style heterogeneous graph transformer operating over dynamically retrieved Hydra subgraphs during recursive deliberation, with concurrent inference and continual training plus atomic model-version publication.

---

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

---

# 4. High-level architecture

```text
ENVIRONMENT
    ↓
v9 adapter
    ↓
multimodal events
    ↓
M0 / M1 ingestion
    ↓
developmental consolidation
    ↓
M2-M7 canonical graph
    ↓
published read view
    ↓
subgraph retrieval/indexing
    ↓
heterogeneous graph construction
    ↓
GNN reasoning engine
    ↓
candidate scores + latent reasoning state
    ↓
recursive deliberation
    ↓
best candidate
    ↓
target-local action
```

Training runs asynchronously from interaction:

```text
interaction evidence
    ↓
training examples
    ↓
replay buffer / dataset
    ↓
GNN optimization
    ↓
new GNN checkpoint/version
    ↓
published inference model
```

---

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


# 7.1 Concrete GNN architecture: HGT-style relational attention

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

## 7.1.1 Typed projections

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

## 7.1.2 Relational attention

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

## 7.1.3 Bounded sampling

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

## 7.1.4 Recursive depth

Use relatively shallow HGT depth per deliberation cycle.

Example:

```text
2-4 HGT layers
×
1-N deliberation cycles
```

The architecture therefore gains effective reasoning depth through recursive deliberation rather than one very deep graph network.

## 7.1.5 Latent outputs

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

## 7.1.6 Attention telemetry

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

## 11.4 Grounding supervision

Symbol grounding produces labels from:

```text
C0/C1/C2/C3 conditions
cross-modal prediction
causal grounding
held-out transfer
```

This trains cross-modal representations.

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
trajectory efficiency
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

The GNN is trained incrementally across the curriculum.

Maintain:

```text
recent experience buffer
stratified historical replay
rare-transfer examples
hard negatives
cross-family pairs
grounding controls
failed reasoning traces
successful reasoning traces
```

Sampling must preserve older environment families as new curriculum stages arrive.

Track:

```text
current-stage performance
historical-stage retention
cross-family transfer
catastrophic-forgetting indicators
```

GNN model versions are immutable once published.

Continuous-training configuration includes:

```text
replay capacity
replay stratification
batch size
learning rate
optimizer
weight decay
gradient clipping
examples per training trigger
optimizer steps per window
candidate evaluation cadence
publication gate
retention thresholds
```


---

# 14. Model publication

Training and inference models are separated.

```text
trainer
    ↓
candidate checkpoint
    ↓
evaluation suite
    ↓
publish ModelVersion N
    ↓
runtime inference
```

A published model records:

```text
ModelVersion
training data cut
ScientificConfigId
feature schema version
graph schema version
objective version
curriculum coverage
evaluation metrics
```

A deliberation episode uses one fixed published model version.

---


# 14.1 Continuous usage and continual training

Hydra v9.7.1 uses the GNN continuously during interaction while training it asynchronously from accumulated evidence.

The system has three concurrent loops:

```text
1. interaction/inference loop
2. evidence/training loop
3. model evaluation/publication loop
```

## 14.1.1 Interaction/inference loop

For every environment decision:

```text
observe environment
    ↓
update/publish Hydra memory view
    ↓
build bounded subgraph
    ↓
run current published HGT model
    ↓
recursive deliberation
    ↓
execute target-local action
    ↓
observe consequence
    ↓
store interaction + reasoning evidence
```

The interaction loop uses one immutable published model version at a time.

A run records:

```text
ModelVersion
GraphGeneration
ScientificConfigId
CurriculumStep
```

for every deliberation episode.

## 14.1.2 Evidence generation loop

Every interaction can generate training evidence.

Sources include:

```text
state transition
action outcome
prediction error
memory consolidation
M2/M3 formation
M4 validation
matched transfer trial
grounding trial
reasoning-cycle improvement
strategy success/failure
invariance test
negative transfer
```

Evidence is appended to a versioned training store.

Each example contains enough provenance to reconstruct:

```text
source graph cut
environment family
game/scenario
curriculum step
context
candidate
action
outcome
memory version
model version used during collection
```

## 14.1.3 Continual replay buffer

Training uses a stratified replay buffer rather than only the most recent experience.

Maintain strata for:

```text
recent examples
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

Sampling proportions are part of the training configuration.

This keeps learning cumulative across the curriculum.

## 14.1.4 Training cadence

Training proceeds asynchronously.

A practical cadence is:

```text
collect K new evidence records
→ run one or more optimizer windows
→ evaluate candidate model
→ publish if acceptance criteria are met
```

Training cadence can be expressed through:

```text
examples_per_train_trigger
optimizer_steps_per_window
max_training_seconds_per_window
max_gpu_memory
```

This allows interaction to continue while learning progresses.

## 14.1.5 Model publication boundary

The trainer never mutates the live inference model in place.

Instead:

```text
Published Model N
        ↓
interaction continues

Trainer
    ↓
Candidate Model N+1
    ↓
evaluation suite
    ↓
accept
    ↓
publish atomically as Model N+1
```

A deliberation episode completes using the model version with which it started.

The next decision may use the newly published version.

## 14.1.6 Candidate-model evaluation

Before publication, evaluate:

```text
current curriculum step
historical curriculum retention
cross-family transfer
prediction quality
correspondence quality
strategy ranking
reasoning improvement
inference latency
memory/GPU cost
```

The evaluation suite should contain fixed held-out sets plus recent-distribution validation.

## 14.1.7 Retention checks

Track forgetting after every candidate model:

```text
Step 1 retention
Step 2 retention
...
current step
```

Report:

```text
absolute performance
delta from previous model
delta from historical best
```

This makes catastrophic forgetting directly measurable.

## 14.1.8 Model promotion policy

Model promotion uses explicit metrics.

A candidate model can be published based on a configured multi-objective gate over:

```text
current-stage gain
historical retention
cross-family transfer
prediction quality
reasoning improvement
latency budget
GPU-memory budget
```

The exact gate is versioned in scientific configuration.

## 14.1.9 Training targets evolve with Hydra

Hydra's own later consolidation can create delayed labels.

Example:

```text
time t:
M2 candidate exists

time t+10000:
candidate becomes stable M3/M4 structure
```

The earlier examples can then receive stronger retrospective training labels such as:

```text
useful retrieval
useful correspondence
validated concept ancestry
successful strategy ancestry
```

Training data therefore supports delayed supervision from developmental memory.

## 14.1.10 Continuous reasoning-policy learning

Every deliberation episode produces:

```text
initial Y
operator sequence
HGT attention/relevance
candidate changes
best cycle
executed action
eventual outcome
```

These traces train the reasoning-policy head.

Over time the model learns:

```text
which memories to inspect
which relation paths matter
which reasoning operator to invoke
how much deliberation is useful
```

## 14.1.11 Curriculum transition

When the curriculum advances:

```text
old data remains in replay
new environment family enters
new examples receive higher sampling priority initially
historical retention continues to be evaluated
```

The shared HGT therefore grows one relational representation across all stages.

## 14.1.12 Online inference vs training compute

Track separately:

```text
interaction inference GPU time
training GPU time
environment CPU time
Hydra memory/replay CPU time
```

This permits explicit control of the trade-off between:

```text
playing/collecting experience
and
training the learned reasoner
```

## 14.1.13 Initial operating mode

A practical initial continuous setup is:

```text
1 published HGT inference model
1 asynchronous trainer
bounded replay buffer
periodic candidate checkpoints
atomic model publication
```

The architecture supports later scaling to multiple data-collection workers and one centralized trainer while retaining the same model-version contract.



# 14.2 Concurrent inference and training on one GPU

The default v9.7.2 operating model supports one published inference model and one candidate training model resident on the same GPU.

Reference topology:

```text
RTX 5070 Ti 16 GB
│
├── Published Model N
│      inference
│      recursive deliberation
│
└── Candidate Model N+1
       training
       evaluation preparation
```

The HGT parameter count is fixed for a configured model family. Hydra memory grows and consolidates independently.

A practical initial HGT target is:

```text
hidden dimension      512
HGT layers            4
attention heads       8
FFN width              2048
parameter count        ~40-60M
precision              BF16
```

The live inference model and candidate training model use the same architecture and feature schema.

## 14.2.1 Initial GPU budget

Use the following as an initial engineering target:

```text
published inference model + buffers     <= 2 GB
candidate training model + optimizer    <= 3 GB
training graph activations              <= 6 GB
CUDA/PyTorch/runtime reserve            >= 3 GB
remaining safety margin                 >= 2 GB
```

Actual limits are established by runtime profiling.

Training adapts its microbatch size to remain inside the configured VRAM target.

## 14.2.2 Initial training batch

Start with:

```text
microbatch                 4 reasoning subgraphs
gradient accumulation      8
effective batch            32 reasoning situations
```

Typical graph target:

```text
target nodes/subgraph      ~400
max nodes/subgraph         800
target edges/subgraph      ~1,500-3,000
max edges/subgraph         4,000
```

Batch pressure is controlled primarily through total nodes and edges rather than nominal graph count.

## 14.2.3 Inference priority

Inference has scheduling priority over training.

The training process yields GPU capacity at bounded intervals so an actor decision does not wait behind a long training kernel sequence.

The runtime records:

```text
inference queue delay
inference latency
training duty cycle
training-step latency
GPU utilization
GPU memory
```

## 14.2.4 Cooperative training cadence

Training runs in short windows.

Reference policy:

```text
collect evidence
→ start optimizer window
→ process bounded number of microbatches
→ yield to inference
→ continue optimizer window
```

Configuration includes:

```text
max_continuous_training_ms
max_optimizer_steps_before_yield
target_inference_latency_ms
training_duty_cycle
```

The scheduler may reduce microbatch size or training duty cycle when inference latency exceeds the configured target.

## 14.2.5 Model-version isolation

Each decision acquires one immutable model handle:

```text
Decision D starts
→ binds ModelVersion N
→ all recursive deliberation cycles use ModelVersion N
→ Decision D completes
```

Publication of `ModelVersion N+1` may occur concurrently.

The next decision may bind the new version.

This creates exact provenance:

```text
DecisionUid
GraphGeneration
ModelVersion
ScientificConfigId
CurriculumStep
```

## 14.2.6 Atomic model publication

Publication is pointer-based:

```text
candidate checkpoint
→ evaluation passed
→ load/warm candidate inference weights
→ atomically publish active-model handle
→ new decisions use new version
→ release old model when no decision references it
```

Model publication never interrupts Hydra memory ingestion.

## 14.2.7 Candidate model lifecycle

Candidate models move through:

```text
TRAINING
EVALUATING
ACCEPTED
PUBLISHED
SUPERSEDED
```

Every candidate stores:

```text
parent ModelVersion
training data cut
optimizer state version
feature schema version
graph schema version
curriculum coverage
evaluation metrics
publication decision
```

## 14.2.8 Training trigger

A default trigger may use:

```text
examples_per_train_trigger = 5000
```

The exact value is configuration.

One training window may perform:

```text
500-2000 optimizer steps
```

subject to time/GPU budgets.

## 14.2.9 Evaluation and promotion

A candidate model is evaluated against:

```text
current curriculum step
historical curriculum retention
cross-family transfer
prediction quality
correspondence quality
strategy ranking
deliberation improvement
inference latency
GPU-memory budget
```

Publication uses a versioned multi-objective promotion rule.

## 14.2.10 Single-GPU and dual-GPU modes

Single-GPU mode:

```text
GPU 0
├── inference Model N
└── training Model N+1
```

Dual-GPU mode:

```text
GPU 0 -> inference/deliberation
GPU 1 -> continual training/evaluation
```

Both modes use the same model-version and publication contracts.


# 15. GNN feature authority

Features come from explicit Hydra state and adapter-normalized observations.

The reference feature classes are:

```text
structural
temporal
causal
support/reliability
contextual
developmental
grounding
validation
target-relative
```

The feature schema is versioned and persisted with model checkpoints.

Cross-environment learning depends on features representing structure consistently.

---

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

The GNN can score:

```text
retrieval usefulness
explanatory usefulness
transfer usefulness
redundancy
```

These become additional evidence channels for lifecycle decisions.

Lifecycle authority remains in Hydra.

A memory node that receives low GNN relevance across many contexts may become a retirement candidate when explicit lifecycle evidence agrees.

---


# 17.1 Hydra–HGT consolidation coupling

Hydra explicit-memory compression and HGT continual learning are coordinated as one developmental consolidation process.

The system maintains three complementary forms of compression:

```text
Hydra memory:
experience
→ M0/M1
→ M2/M3
→ M4/M5/M6/M7
→ explicit higher-order structure

HGT replay:
many training examples
→ representative / high-value training evidence

HGT weights:
many relational examples
→ distributed learned relational regularities
```

The intended flow is:

```text
detailed Hydra experience
        │
        ├──→ explicit Hydra consolidation
        │        ↓
        │    higher-level memory
        │
        └──→ HGT training evidence
                 ↓
             HGT weights
```

As Hydra matures, the HGT training distribution should naturally shift from predominantly M0/M1-heavy subgraphs toward larger proportions of M2-M7 abstractions.

## 17.1.1 Lifecycle-to-training events

Hydra lifecycle transitions emit versioned consolidation events to the HGT training subsystem.

Relevant events include:

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
GraphGeneration
ScientificConfigId
```

These events become training metadata for replay prioritization and delayed supervision.

## 17.1.2 Pre-retirement consolidation pass

When a substantial memory cluster enters `RETIRE_PENDING`, the system may schedule a bounded HGT consolidation pass.

Reference sequence:

```text
memory cluster selected for consolidation
        ↓
identify representative subgraphs
        ↓
include:
    high explanatory reach
    high transfer value
    rare structural variants
    contradictions / hard negatives
    successful strategies
    failed strategies
        ↓
HGT consolidation training window
        ↓
retention evaluation
        ↓
Hydra lifecycle continues
```

The consolidation pass is prioritized according to the expected information lost by retiring the detailed cluster.

## 17.1.3 Representative replay preservation

Replay storage is compressed alongside Hydra memory.

When many examples describe the same consolidated structure:

```text
large redundant example set
        ↓
stratified representative selection
        ↓
smaller replay subset
```

Representative selection preserves:

```text
structural diversity
environment-family diversity
context diversity
positive and negative evidence
rare cases
causal transfer examples
reasoning failures
reasoning successes
```

This prevents replay growth from simply reproducing the raw experience volume that Hydra is attempting to compress.

## 17.1.4 Delayed supervision from consolidation

Hydra's later developmental state can strengthen labels for earlier GNN examples.

Example:

```text
time t:
candidate M2 structure exists

time t+k:
same lineage becomes stable M3 role

time t+m:
role contributes to validated M4 transfer
```

Earlier training examples can then receive retrospective labels such as:

```text
useful retrieval
stable structural correspondence
validated transfer ancestry
successful strategy ancestry
persistent explanatory value
```

The original causal timestamps remain preserved.

## 17.1.5 Retention verification

Before a major replay-data reduction associated with Hydra consolidation, evaluate whether the current HGT retains the relevant learned capability.

Possible checks:

```text
prediction retention
correspondence retention
strategy-ranking retention
cross-family transfer retention
reasoning-operator retention
```

Results are attached to the consolidation event.

## 17.1.6 Reactivation

When Hydra reactivates retired or superseded structure:

```text
RETIRED
→ REACTIVATED
```

the corresponding representative replay evidence receives temporary increased sampling priority.

This allows the HGT to rapidly refresh relational knowledge associated with returning environmental regimes.

## 17.1.7 Compression telemetry

Add explicit metrics:

```text
Hydra bytes retired
Hydra nodes retired
Hydra nodes replaced by higher-level abstractions
HGT replay examples before compression
HGT replay examples after compression
representative-example retention ratio
HGT retention before/after replay compression
retraining examples triggered by reactivation
explicit-memory compression ratio
replay compression ratio
```

Report these jointly with:

```text
prediction quality
transfer quality
reasoning quality
persistent-memory growth
historical-stage retention
```

## 17.1.8 Developmental interpretation

The hybrid system therefore supports two timescales of consolidation:

```text
fast:
experience
→ Hydra explicit memory
→ immediate structural reuse

slow:
accumulated graph experience
→ HGT training
→ learned relational computation
```

Hydra stores and restructures explicit developmental knowledge.

The HGT progressively distills recurrent relational regularities from the evolving memory graph.

Together they create a continual consolidation cycle:

```text
experience
→ explicit memory
→ abstraction
→ learned relational compression
→ improved reasoning
→ better interaction
→ new experience
```

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


# 21.1 General learned trajectory optimization

v9.7.4 makes solution optimization a general learned capability over Hydra outcome and strategy memory.

The optimization target is defined by Hydra's late memory structure:

```text
M6 = represented outcome / outcome-equivalence class
M7 = strategies / trajectories that may reach that outcome
```

For a selected outcome `Ω`, optimization searches for a strategy `π` that preserves the intended outcome while improving one or more declared efficiency dimensions.

Reference objective:

```text
maximize:
    probability of reaching Ω
    primary-valence consistency
    strategy reliability

minimize:
    realized trajectory cost
    unnecessary actions
    loops
    blocked actions
    repeated states
    avoidable intermediate structure
```

Efficiency comparisons remain conditioned on the same M6 outcome or another explicitly admissible no-worse outcome class.

## 21.1.1 Optimization loop

Once Hydra has at least one successful strategy:

```text
successful strategy π0
        ↓
identify achieved M6 outcome Ω
        ↓
retrieve:
    alternative M7 strategies
    useful sub-trajectories
    structurally similar successful trajectories
    failed variants / counterevidence
        ↓
HGT evaluates candidate structure
        ↓
recursive candidate refinement
        ↓
π1
        ↓
simulate / compare / refine
        ↓
π2 ... πn
        ↓
select best admissible candidate
        ↓
execute in environment
        ↓
observe actual outcome and cost
        ↓
update Hydra evidence
        ↓
add HGT optimization training evidence
```

Optimization therefore becomes a repeated interaction between:

```text
explicit outcome/strategy memory
+
learned relational reasoning
+
recursive deliberation
+
environment validation
```

## 21.1.2 HGT optimization capabilities

The HGT learns to estimate:

```text
outcome-preservation probability
strategy reliability
trajectory cost
future-option impact
dependency-chain length
dead-end / irreversible-risk likelihood
redundant trajectory fragments
promising alternative sub-trajectories
candidate improvement likelihood
```

The same shared HGT model performs these estimates across environment families.

## 21.1.3 General optimization patterns

Training across multiple games should allow the HGT to learn reusable optimization regularities such as:

```text
loop removal
repeated-state reduction
blocked-action avoidance
shorter dependency chains
earlier enabling actions
reuse of successful sub-trajectories
replacement of expensive trajectory fragments
selection of higher-reliability alternatives
replanning around unnecessary intermediate states
```

These patterns are learned from structural and outcome evidence across the curriculum.

## 21.1.4 Candidate representation

An optimization candidate may represent:

```text
complete M7 strategy
partial trajectory
sub-trajectory replacement
alternative action at a decision point
alternative dependency chain
alternative route to the same M6 outcome
```

Each candidate records:

```text
CandidateUid
TargetM6Uid
SourceM7Uid optional
CandidateType
PredictedReliability
PredictedCost
PredictedOutcomeSupport
PredictedFutureOptionEffect
Context
GraphGeneration
ModelVersion
```

## 21.1.5 Optimization supervision

Training evidence is generated whenever two or more strategies or trajectory variants are comparable under the same outcome class.

Reference pairwise supervision:

```text
πA → Ω, reliability=rA, cost=cA
πB → Ω, reliability=rB, cost=cB
```

The model learns preference relations over candidates from:

```text
outcome preservation
reliability
realized cost
trajectory efficiency
context applicability
eventual behavioral success
```

When one candidate preserves the same outcome with lower cost and comparable reliability, it supplies direct optimization supervision.

## 21.1.6 Training objective

Add a versioned optimization objective:

```text
L_optimization =
    w_outcome * outcome_preservation_loss
  + w_rank    * strategy_ranking_loss
  + w_cost    * relative_cost_loss
  + w_refine  * refinement_improvement_loss
  + w_rel     * reliability_calibration_loss
```

The weights are part of `ScientificConfig`.

This objective is part of the shared HGT multi-task loss.

## 21.1.7 Recursive optimization

Recursive deliberation provides the search process.

Example:

```text
Y0 = known successful trajectory

cycle 1:
identify redundant loop
→ Y1

cycle 2:
replace expensive dependency fragment
→ Y2

cycle 3:
predict lower reliability
→ retain Y1 as best

cycle 4:
retrieve alternative enabling strategy
→ Y3

execute best candidate Y3
```

The best candidate may originate from any deliberation cycle.

## 21.1.8 Cross-game optimization transfer

Optimization competence is evaluated across environment families.

Example transfer questions:

```text
Does loop-removal learned in FrozenLake improve ARC trajectory optimization?
Does dead-end avoidance learned in Sokoban help MiniGrid?
Does dependency-shortening learned in BabyAI improve ARC strategies?
Does alternative-route selection learned in MiniGrid improve ALFRED planning?
```

The curriculum should report source→target optimization transfer explicitly.

## 21.1.9 Optimization lifecycle

Successful optimized trajectories update M7 evidence:

```text
new M7 strategy
or
updated strategy statistics
or
new relation to existing M6 outcome
```

Repeatedly inferior trajectory variants may receive reduced reuse priority while remaining available as negative training evidence.

High-value failed optimizations remain useful for:

```text
hard-negative training
context discovery
reliability calibration
dead-end prediction
```

## 21.1.10 Optimization telemetry

Add:

```text
TargetM6Uid
SourceM7Uid
initial_solution_cost
initial_solution_reliability
optimized_solution_cost
optimized_solution_reliability
optimization_cycles
candidates_generated
candidates_refined
candidates_executed
predicted_cost
realized_cost
predicted_reliability
realized_success
outcome_preserved
relative_efficiency_gain
actions_removed
loops_removed
blocked_actions_avoided
repeated_states_removed
best_candidate_cycle
optimization_reasoning_cost
source_environment_family
target_environment_family
ModelVersion
GraphGeneration
```

Aggregate reports include:

```text
mean optimization gain
median optimization gain
success-preserving optimization rate
harmful optimization rate
prediction/calibration error
cross-family optimization transfer
optimization gain per reasoning cost
```

## 21.1.11 Research evaluation

Required comparisons:

```text
Hydra strategy reuse without learned optimization
Hydra + HGT single-pass strategy ranking
Hydra + HGT recursive trajectory optimization
```

Primary measures:

```text
same-outcome cost reduction
strategy reliability
success rate
trajectory efficiency
number of optimization attempts
reasoning cost
cross-game transfer
sample efficiency
```

The central v9.7.4 optimization hypothesis is:

> A shared learned relational reasoner operating over explicit M6/M7 outcome and strategy memory can acquire transferable trajectory-optimization competence across games, improving strategy efficiency while preserving outcome reliability.


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


# 23.1 HGT architecture derived from Hydra structure

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

## 23.1.1 Latent width

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

## 23.1.2 Message-passing depth

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

## 23.1.3 Attention heads

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

## 23.1.4 Feed-forward capacity

Feed-forward capacity is sized relative to the structured numerical and categorical representation used by Hydra.

Reference:

```text
d_model = 320
FFN = 768–1024
```

The baseline uses `1024`.

## 23.1.5 Reference architecture

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

## 23.1.6 Capacity scaling experiment

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

## 23.1.7 Depth and width experiments

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

## 23.1.8 Graph-derived architecture telemetry

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

## 23.1.9 Curriculum-based capacity gates

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

## 23.1.10 Architecture-selection principle

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

# 24. Inference performance

Runtime inference operates on bounded subgraphs.

Primary runtime controls:

```text
nodes per subgraph
edges per subgraph
GNN layers
latent dimension
deliberation cycles
candidate count
retrieval expansions
```

Track:

```text
milliseconds per GNN pass
milliseconds per deliberation episode
GPU memory
nodes/edges processed
environment steps per second
```

---

# 25. Snapshot and restart

Hydra snapshot includes:

```text
published GNN ModelVersion
graph feature schema version
deliberation configuration
GNN-dependent lifecycle evidence
reasoning-policy statistics
```

GNN model weights are stored as immutable external artifacts referenced by content hash/version.

Restart restores the same model version.

---

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


# 26.1 Unified telemetry model

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

## 26.1.1 System progress

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

## 26.1.2 Hydra memory health

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

## 26.1.3 Learning and prediction

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

## 26.1.4 Transfer and abstraction

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

## 26.1.5 Recursive deliberation

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

## 26.1.6 HGT inference quality

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

## 26.1.7 HGT training health

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

## 26.1.8 Continual model evolution

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

## 26.1.9 Primary dashboard

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

## 26.1.10 Diagnostic event provenance

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

## 26.1.11 HGT training failure isolation

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

# 27. Research ablations

Required conditions:

```text
Hydra only
Hydra + GNN single-pass
Hydra + GNN recursive deliberation
```

Additional useful controls:

```text
random GNN embeddings
frozen early GNN
current trained GNN
explicit similarity only
learned similarity
```

Primary comparisons:

```text
sample efficiency
prediction quality
cross-family transfer
held-out task success
memory growth
reasoning cost
catastrophic forgetting
trajectory efficiency
```

---

# 28. Primary research questions

v9.7 evaluates:

### Q1
Does learned relational reasoning improve retrieval and structural correspondence over explicit descriptors alone?

### Q2
Does GNN consequence prediction improve candidate refinement during recursive deliberation?

### Q3
Does one shared relational model discover reusable structure across Synthetic, Gym, Sokoban, Chess, Sudoku, MiniGrid, BabyAI, ARC, Atari, and ALFRED?

### Q4
Does explicit Hydra memory reduce catastrophic forgetting compared with model-only continual learning?

### Q5
Does recursive GNN deliberation improve behavior relative to a single GNN pass?

### Q6
Does the hybrid achieve better sample efficiency than either explicit Hydra reasoning or a model-only learned policy?

---

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

Suggested additions:

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

Existing authoritative v9 modules remain the integration points for:

```text
memory
similarity
correspondence
transfer
prediction
planning
replay
grounding
lifecycle
```

---

# 31. Acceptance criteria

v9.7 is operational when:

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
16. inference and candidate training can coexist on one GPU under configured VRAM and latency budgets.
17. every deliberation episode binds exactly one immutable ModelVersion.
18. model publication is atomic and does not interrupt memory ingestion.
19. old model weights are released only after active decisions release their model handle.
20. training duty cycle adapts to inference latency while preserving optimizer progress.
21. Hydra lifecycle events can drive bounded HGT consolidation training.
22. replay-buffer compression preserves structural and environment-family diversity.
23. major memory consolidation records HGT retention evidence.
24. reactivated Hydra structure can receive temporary replay-priority recovery.
25. HGT can rank comparable M7 strategies for the same M6 outcome.
26. recursive optimization can produce lower-cost outcome-preserving trajectories.
27. optimization telemetry separates predicted and realized reliability/cost.
28. cross-family optimization transfer is reported with explicit source→target provenance.
29. primary telemetry is limited to the unified dashboard set.
30. HGT training failures are distinguishable from Hydra, retrieval, deliberation, and execution failures.
31. every promoted HGT model records current-stage, retention, transfer, reasoning, latency, and promotion deltas.
32. detailed telemetry carries sufficient provenance to trace regressions to model, graph, curriculum, and environment state.

---

# 32. Final architecture

Hydra v9.7.6 becomes:

```text
explicit developmental memory
        +
learned heterogeneous graph reasoning
        +
recursive candidate refinement
        +
causal interaction
```

The intended developmental loop is:

```text
experience
→ explicit memory
→ learned graph reasoning
→ recursive deliberation
→ intervention
→ causal evidence
→ improved memory
→ improved GNN
```

The central hypothesis is:

> A learned relational model operating over explicit developmental memory can discover reusable cross-environment structure and improve recursive reasoning while preserving the persistence, provenance, causal validation, and continual-learning properties of Hydra.

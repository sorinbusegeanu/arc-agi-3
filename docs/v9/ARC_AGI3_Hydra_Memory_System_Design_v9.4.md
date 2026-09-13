# ARC-AGI-3 Hydra Memory System Design v9.4

**Version:** v9.4\
**Status:** Independent target design\
**Research contract:** `Research_problem_statement_v0631.md`\
**Design predecessors:** v8.55/v8.56 and Hydra v9.3\
**Purpose:** define Hydra v9 as a clean, independently runnable
implementation of the research architecture.

------------------------------------------------------------------------

## 1. Architectural boundary

Hydra v9 is a new implementation under `src/v9/`.

Its production runtime has **zero imports from `v8`**.

``` text
src/v9  -> Python/runtime dependencies
        -> environment libraries
        -> no src/v8 dependency
```

v8 is a predecessor and reference implementation. Its algorithms, tests,
measured behavior, and data formats may be studied and selectively
reimplemented in v9, but v8 code is not part of the v9 runtime
dependency graph.

The boundary applies to:

-   runtime and CLI;
-   memory model and graph;
-   proposal/mutation machinery;
-   environment contracts;
-   action selection;
-   developmental cognition;
-   lifecycle;
-   persistence;
-   research evidence and reports;
-   symbolic grounding;
-   tests.

Compatibility with v8 is isolated under `src/v9/migration/`.

------------------------------------------------------------------------

## 2. Design objective

v9 learns from heterogeneous interactive environments, passive
observations, ordered symbolic observations, interventions, structural
recurrence, and causal transfer.

The developmental hierarchy remains:

``` text
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
M4 empirically validated concepts
        ↓
M5 consequence structures/world models
        ↓
M6 outcome-equivalence abstractions
        ↓
M7 strategies/procedures
```

The research semantics of v9.3 are retained.

------------------------------------------------------------------------

## 3. Core invariants

1.  Canonical cognition is RAM-authoritative.
2.  Raw experience creates only M0/M1 evidence.
3.  M2-M7 arise from developmental operators over canonical lower
    memory.
4.  Canonical identity is independent of mutable support, lifecycle,
    lineage, provenance, and parent connectivity unless explicitly part
    of a level's invariant structural identity.
5.  Higher memory retains auditable grounded provenance.
6.  Structural similarity proposes candidates; causal held-out evidence
    validates transfer.
7.  Symbol IDs contain no supplied task semantics.
8.  Interaction remains the causal grounding authority.
9.  Native actions remain environment-local.
10. Memory, mutation, structural search, replay, and symbol processing
    remain bounded.
11. Scientific configuration is immutable within an experimental
    condition.
12. v9 runtime correctness cannot depend on v8 being installed.

------------------------------------------------------------------------

## 4. Independent system architecture

``` text
ENVIRONMENTS / DATA SOURCES
        ↓
src/v9/environments + src/v9/modalities
        ↓
Multimodal Event Timeline
        ↓
M0/M1 bounded ingestion
        ↓
Proposal queues
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
Canonical mutation
```

Scientific evidence observes this process but does not retrospectively
alter it.

------------------------------------------------------------------------

## 5. Target package organization

``` text
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

    mutation/
        proposals.py
        versions.py
        read_sets.py
        transactions.py
        lineage.py

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

------------------------------------------------------------------------

## 6. Environment and sensory boundary

Every interactive environment implements the v9-owned
`EnvironmentCognitionAdapter` contract:

``` text
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

``` text
EnvironmentFamilyId
EnvironmentTypeId
EnvironmentConfigId
EnvironmentInstanceId
EpisodeId
```

ARC, Gym/FrozenLake, Chess, Sudoku, synthetic symbolic environments,
BabyAI and later ALFRED are v9 adapters. They do not call v8 adapters at
runtime.

------------------------------------------------------------------------

## 7. Multimodal timeline and symbolic observations

The event model preserves macro actions and passive observations as
separate causal events.

``` text
world observation
symbol observations
        ↓
actor action
        ↓
passive world/symbol observations
        ↓
settled observation
```

Every event carries causal availability/order. Passive observations
never become fake actions.

Symbols retain only deterministic identity, ordering, stream/vocabulary
identity, timestamp/watermark and provenance.

No embeddings, pretrained semantics, POS tags, named entities, synonym
classes, or LLM-derived labels enter canonical cognition.

------------------------------------------------------------------------

## 8. Memory hierarchy

The M0-M7 semantics defined in v9.3 remain authoritative.

### M0

Exact grounded multimodal provenance root.

### M1G

Environment/modality-local executable or observed contingencies.

### M1N

First universal normalization boundary. Contains bounded observable
structural facts for world, symbolic and cross-modal relations.

### M2

Recurrence/transformation families formed through measurable
compression.

### M3

Functional roles defined by relational/consequence structure.

### M4

Concepts requiring empirical held-out causal transfer evidence.

### M5

Reusable consequence structures downstream of validated M4.

### M6

Learned outcome-equivalence abstractions independent of terminal labels
and native rewards.

### M7

Executable strategies linked to represented M6 outcomes and target-local
actions.

------------------------------------------------------------------------

## 9. Canonical graph and mutation authority

v9 owns its graph implementation.

Workers read immutable/versioned graph views and emit `MutationProposal`
objects.

``` text
parallel workers
    ↓
bounded read sets
    ↓
mutation proposals
    ↓
deterministic partition ownership
    ↓
validation
    ↓
atomic publication
```

Single-partition mutation is serialized by its owning reducer.
Cross-partition mutation uses deterministic atomic coordination.

Global generation advancement alone does not invalidate a proposal.
Read-set/object versions determine staleness.

------------------------------------------------------------------------

## 10. Lineage and context authority

Canonical structural identity is separated from lineage/context mutable
authority.

Use:

``` text
CanonicalNode
LineageContextOverlay
LineageAwareDependencyEdge
```

A lineage-specific mutation changes only its overlay/dependencies unless
canonical structural identity itself diverges.

Copy-on-write occurs only at the minimal canonical divergence boundary.
Downstream canonical nodes are reused when their identity remains
invariant.

------------------------------------------------------------------------

## 11. Progressive structural search

v9 implements progressive bounded structural comparison directly.

``` text
bounded candidates
    ↓
smallest radius
    ↓
scale-normalized scores
    ↓
preregistered beta_r softmax
    ↓
entropy + top-2 margin
    ↓
expand only while information improves
    ↓
candidate correspondence or StructuralEquivalenceSet
    ↓
held-out causal validation
```

Normalization is scale-stratified and uses bounded provisional and
authoritative estimator generations.

Provisional storage consists of bounded streaming summaries plus a
bounded diagnostic/sample reservoir.

------------------------------------------------------------------------

## 12. Grounding

Text is an ordered sensory stream.

Cross-modal temporal association does not establish meaning.

A `GROUNDS` relation requires interaction-relevant evidence such as
successful held-out prediction, action influence, consequence
prediction, or transfer.

Grounding maturity remains:

``` text
G0 interaction grounding
G1 concurrent symbols
G2 structural cross-modal association
G3 prospective symbol-conditioned prediction
G4 novel compositional transfer
G5 symbol-mediated learning about unexperienced interaction
```

H16 uses matched C0-C3 controls:

``` text
C0 interaction only
C1 symbols only
C2 aligned interaction + symbols
C3 shuffled interaction + symbols
```

------------------------------------------------------------------------

## 13. Persistence

v9 defines its own snapshot schema.

Snapshots persist:

-   canonical graph;
-   environment/schema registries;
-   symbol vocabularies/codecs;
-   lineage/context overlays;
-   dependency authority;
-   graph/object versions;
-   normalization estimator generations;
-   structural equivalence sets;
-   grounding state;
-   transfer trials;
-   developmental state;
-   lifecycle/replay state;
-   scientific evidence;
-   immutable scientific configuration ID.

A v9 process restores v9 snapshots without importing or executing v8.

------------------------------------------------------------------------

## 14. v8 migration boundary

v8 is supported only as an external predecessor format.

``` text
v8 snapshot/evidence
        ↓
src/v9/migration/*
        ↓
validated v9 representation
        ↓
v9 snapshot
```

Migration is offline or startup conversion. Once conversion completes,
ordinary v9 execution uses only v9 code.

Migration must reject incompatible or ambiguous state rather than
silently reinterpret it.

------------------------------------------------------------------------

## 15. CLI and execution

The independent entrypoint is:

``` bash
PYTHONPATH=src python -m v9 continuous-run ...
```

`src/v9/__main__.py` calls the v9 CLI only.

`src/v9/cli.py` constructs the v9 runtime only.

Running v9 must succeed when `src/v8/` is absent.

The v8 command remains available separately for historical experiments:

``` bash
PYTHONPATH=src python -m v8 ...
```

------------------------------------------------------------------------

## 16. Acceptance criteria for independence

v9 is architecturally independent only when all are true:

1.  `grep`/AST dependency audit finds no `v8` import outside
    `src/v9/migration/`.
2.  `src/v8` can be temporarily removed from `PYTHONPATH` and v9 imports
    successfully.
3.  `python -m v9 --help` works without v8.
4.  ARC smoke works without v8.
5.  heterogeneous environment smoke works without v8.
6.  snapshot/restart works without v8.
7.  M0-M7 developmental smoke works without v8.
8.  research evidence/report generation works without v8.
9.  H16 synthetic grounding runs without v8.
10. all v9 tests pass with v8 unavailable.

------------------------------------------------------------------------

## 17. Final target

Hydra v9 is a self-contained implementation of the v0.6.3.1 research
architecture.

It inherits **ideas and validated engineering lessons** from v8, not
runtime dependencies.

``` text
v8 = predecessor/reference
v9 = independent research implementation
```

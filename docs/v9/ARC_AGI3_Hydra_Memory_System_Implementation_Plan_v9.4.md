# ARC-AGI-3 Hydra Memory System Implementation Plan v9.4

**Status:** implementation plan\
**Target design:** `ARC_AGI3_Hydra_Memory_System_Design_v9.4.md`\
**Research contract:** `Research_problem_statement_v0631.md`\
**Reference implementation:** current v8 main\
**Target implementation:** independent `src/v9/`\
**Purpose:** build a clean v9 runtime with no production dependency on
v8 while preserving the required research semantics and selectively
reimplementing proven mechanisms.

------------------------------------------------------------------------

# 1. Implementation strategy

v9 is implemented as an independent system.

The current v8 implementation is used as:

-   behavioral reference;
-   algorithmic reference;
-   regression oracle where appropriate;
-   source format for optional migration;
-   source of lessons about performance and failure modes.

It is not a runtime library for v9.

The fundamental dependency rule is:

``` text
src/v9/* -> must not import v8
src/v9/migration/* -> may read v8 formats, preferably without executing v8
```

Do not subclass v8 runtime classes, alias v8 configuration classes, call
the v8 CLI, install v8 monkey patches, or keep v9 cognition under
`src/v8/`.

------------------------------------------------------------------------

# 2. Independence first

The first implementation milestone is the architectural boundary, before
new cognition.

Create:

``` text
src/v9/
    __init__.py
    __main__.py
    cli.py
    runtime/
    memory/
    environments/
    modalities/
    cognition/
    mutation/
    research/
    migration/
    tests/
```

Add an automated dependency test that fails on any `import v8` or
`from v8` outside the migration package.

The v9 command becomes:

``` bash
PYTHONPATH=src python -m v9 continuous-run ...
```

The v9 entrypoint must never delegate to `python -m v8`.

------------------------------------------------------------------------

# 3. Reuse policy

For each proven v8 mechanism:

``` text
inspect v8 behavior
    ↓
define v9-owned contract
    ↓
implement under src/v9
    ↓
port/adapt relevant tests
    ↓
compare behavior
    ↓
remove any temporary reference dependency
```

Prefer clean reimplementation/refactoring over copying historical patch
layers.

Do not reproduce the chronological v8 monkey-patch stack.

The target is the final effective behavior required by the research
contract.

------------------------------------------------------------------------

# 4. Scientific configuration

Implement `src/v9/runtime/config.py` first.

Create immutable `ScientificConfigId` from canonical serialized
configuration.

Include at minimum:

``` text
schema_version
research_contract_version = 0.6.3.1
design_version = 9.4
random_seeds
symbol budgets
structural radii
candidate budgets
ambiguity thresholds
beta_by_radius
normalization bootstrap thresholds
provisional sample bound/policy
probation thresholds/budgets
transfer held-out thresholds
enabled structural relations
```

Persist the ID in snapshots, evidence and reports.

------------------------------------------------------------------------

# 5. Implementation sequence

## Phase v9.00 --- Independent skeleton

Implement:

``` text
src/v9/__init__.py
src/v9/__main__.py
src/v9/cli.py
src/v9/runtime/config.py
```

Tests:

-   import v9 with v8 unavailable;
-   `python -m v9 --help`;
-   dependency audit;
-   deterministic ScientificConfigId.

Exit gate: no v8 production import exists.

------------------------------------------------------------------------

## Phase v9.01 --- Core model and identity

Implement v9-owned:

``` text
memory/model.py
memory/identity.py
memory/relations.py
environments/schemas.py
environments/registry.py
modalities/contract.py
modalities/symbols/*
```

Add:

``` text
MemoryUid
EventUid
EnvironmentFamilyId
EnvironmentTypeId
EnvironmentConfigId
EnvironmentInstanceId
EpisodeId
ModalityId
SymbolVocabularyId
SymbolStreamId
SymbolId
SymbolPosition
```

Canonical identity must exclude mutable lineage/parent connectivity.

------------------------------------------------------------------------

## Phase v9.02 --- RAM graph and publication

Implement:

``` text
runtime/rings.py
runtime/partitions.py
runtime/reducers.py
runtime/read_view.py
runtime/publication.py
mutation/proposals.py
mutation/versions.py
mutation/read_sets.py
mutation/transactions.py
```

Required properties:

-   bounded queues;
-   deterministic partition ownership;
-   immutable/versioned read views;
-   single-writer-per-partition mutation;
-   atomic cross-partition publication;
-   object-version/read-set stale validation;
-   deterministic proposal ordering.

Port behavioral tests from v8 where useful, but tests import v9 only.

------------------------------------------------------------------------

## Phase v9.03 --- M0/M1 memory foundation

Implement:

``` text
memory/m0_episode.py
memory/m1_grounded.py
memory/m1_normalized.py
memory/provenance.py
memory/residency.py
```

Build the independent path:

``` text
environment event
    ↓
M0
    ↓
M1G
    ↓
bounded M1N
```

Raw experience stops at M1.

------------------------------------------------------------------------

## Phase v9.04 --- Environment contract and ARC adapter

Implement the v9-owned environment contract and ARC adapter.

The ARC adapter may use the external ARC environment package/API, but
must not call `src/v8` adapter code.

Smoke:

``` bash
PYTHONPATH=src python -m v9 continuous-run --games <small ARC set> ...
```

Exit gate: interaction, M0/M1 formation and reporting work with `src/v8`
unavailable.

------------------------------------------------------------------------

## Phase v9.05 --- Heterogeneous adapters

Implement v9 adapters for:

``` text
FrozenLake
Chess
Sudoku
```

Then implement `mix`/research presets in v9 configuration.

Verify all adapters enter the same memory runtime.

------------------------------------------------------------------------

## Phase v9.06 --- Passive multimodal timeline

Implement a v9-owned timeline supporting:

``` text
INTERACTION
PASSIVE_WORLD
PASSIVE_SYMBOL
```

Every event carries identity, causal watermark, producer ordering,
environment/episode and modality.

Passive observations have no action ID.

Enforce symbol/event/payload budgets immediately.

------------------------------------------------------------------------

## Phase v9.07 --- M2 and M3 development

Implement independent:

``` text
cognition/compression.py
cognition/roles.py
memory/m2_family.py
memory/m3_role.py
```

M2 formation requires multiple supported M1N structures and measurable
compression.

M3 identity is relational/functional rather than raw
appearance/token/native action identity.

Mixed WORLD/SYMBOL provenance must converge when canonical structural
identity is equal.

------------------------------------------------------------------------

## Phase v9.08 --- Lineage/context authority

Implement:

``` text
mutation/lineage.py
```

Support:

``` text
LineageUid
ContextScope
LineageContextOverlay
lineage-aware dependency edges
ACTIVE/SUSPENDED/REJECTED edge authority
PROBATION
```

Implement minimal copy-on-write only when canonical structural identity
diverges.

------------------------------------------------------------------------

## Phase v9.09 --- Progressive structural statistics

Implement bounded scale-stratified normalization:

``` text
EMPTY
PROVISIONAL
AUTHORITATIVE
RECALIBRATING
```

Maintain bounded streaming summaries and bounded diagnostic samples.

Implement preregistered bootstrap criteria and estimator generations.

Cold/probation/suspended evidence cannot contaminate authoritative
statistics.

------------------------------------------------------------------------

## Phase v9.10 --- Progressive similarity and symmetry

Implement:

``` text
cognition/similarity.py
```

Flow:

``` text
bounded retrieval
→ radius r
→ normalized score
→ beta_r softmax
→ entropy/top2 margin
→ information-gain decision
→ expand or stop
→ StructuralEquivalenceSet if unresolved
```

Use stable log-softmax/log-sum-exp.

No arbitrary winner for structurally equivalent candidates.

------------------------------------------------------------------------

## Phase v9.11 --- Correspondence and transfer

Implement:

``` text
cognition/correspondence.py
cognition/transfer.py
```

Structural correspondence remains distinct from causal transfer.

Held-out matched trials provide validation.

Transfer resolves execution to target-local actions.

Support ARC→ARC, Gym→Gym and later cross-family candidates under the
same contract.

------------------------------------------------------------------------

## Phase v9.12 --- M4-M7 cognition

Implement:

``` text
memory/m4_concept.py
memory/m5_consequence.py
memory/m6_outcome.py
memory/m7_strategy.py

cognition/concepts.py
cognition/future_options.py
cognition/world_model.py
cognition/outcomes.py
cognition/strategies.py
cognition/planning.py
cognition/replay.py
```

Preserve:

-   M4 causal held-out validation;
-   validated-M4 requirement for mature M5 claims;
-   terminal-label-free M6;
-   target-local M7 execution;
-   separate primary valence, reliability and efficiency;
-   bounded replay.

------------------------------------------------------------------------

## Phase v9.13 --- Lifecycle and residency

Implement:

``` text
runtime/lifecycle.py
memory/residency.py
```

Lifecycle:

``` text
CANDIDATE
PROBATION
ACTIVE
VALIDATED
QUARANTINED
RETIRE_PENDING
RETIRED
REACTIVATED
```

Probation retirement is evidence-opportunity aware.

Separate logical provenance, hot cognitive residency and raw payload
residency.

------------------------------------------------------------------------

## Phase v9.14 --- Snapshot/restart

Implement `runtime/snapshot.py`.

Persist all state needed for restart-equivalent cognition.

Test:

``` text
learn
→ snapshot
→ terminate
→ restore in fresh v9 process
→ equivalent cognitive state/behavior
```

No v8 import is permitted during restore of native v9 snapshots.

------------------------------------------------------------------------

## Phase v9.15 --- Synthetic symbolic grounding

Implement the controlled synthetic symbolic environment directly in v9.

Use arbitrary symbols first.

Add cross-modal M1N, mixed M2/M3 and structural correspondence.

Temporal alignment alone must not create `GROUNDS`.

------------------------------------------------------------------------

## Phase v9.16 --- Empirical grounding

Implement:

``` text
cognition/grounding.py
research/grounding_h16.py
```

Grounding requires interaction-relevant evidence.

Implement maturity G0-G5 and persistent negative/context-scoped
grounding evidence.

------------------------------------------------------------------------

## Phase v9.17 --- Symbol-conditioned prediction

Allow eligible grounded higher-level symbolic lineage to influence
prediction in shadow mode first.

Measure:

``` text
baseline prediction
symbol-conditioned prediction
actual consequence
delta
```

No direct M1 symbol-to-action shortcut.

------------------------------------------------------------------------

## Phase v9.18 --- Gated action influence

Permit symbol-derived influence only after declared grounding/validation
thresholds.

Actor execution remains target-local.

Add explicit telemetry for grounded action influence.

------------------------------------------------------------------------

## Phase v9.19 --- H16 matched controls

Implement:

``` text
C0 interaction only
C1 symbols only
C2 aligned interaction + symbols
C3 shuffled interaction + symbols
```

Match seeds, environment configurations, interaction budgets and
evaluation conditions.

Required H16 evidence includes aligned-vs-controls advantage plus
held-out causal cross-modal effect.

------------------------------------------------------------------------

## Phase v9.20 --- BabyAI/MiniGrid

Implement a native v9 adapter exposing:

``` text
world observation
native actions
instruction symbol stream
boundary event
```

No parsed instruction semantics.

Run held-out compositional grounding tests.

------------------------------------------------------------------------

## Phase v9.21 --- Generalized heterogeneous transfer

Remove ARC-only assumptions from the v9 transfer experiment system.

Support:

``` text
ARC → ARC
Gym → Gym
synthetic → synthetic
BabyAI → BabyAI
admissible cross-family
cross-modal
```

Matched causal controls remain mandatory.

------------------------------------------------------------------------

## Phase v9.22 --- ALFRED

Add ALFRED only after synthetic/BabyAI grounding and bounded-memory
gates pass.

Large payloads remain outside canonical graph records.

------------------------------------------------------------------------

## Phase v9.23 --- v8 migration tooling

Implement optional conversion tools under:

``` text
src/v9/migration/
```

Supported use cases:

``` text
v8 snapshot -> v9 snapshot
v8 evidence -> v9 research archive
```

Migration code is excluded from normal v9 startup and cognition.

Ambiguous/incompatible state fails explicitly.

------------------------------------------------------------------------

# 6. Persistence rule

Every phase that introduces persistent cognitive state adds
serialization, restore, schema validation and restart tests in the same
phase.

Phase v9.14 is the full-system persistence integration gate, not the
first persistence work.

------------------------------------------------------------------------

# 7. Testing strategy

Use three layers.

### Unit

Each v9 module tested independently.

### Deterministic integration

Verify exact graph/mutation outcomes across scheduling/order variations.

### Runtime smoke

Exercise actual environments and snapshot/restart.

Add a mandatory independence job:

``` text
make src/v8 unavailable
run complete v9 test suite
run v9 CLI smoke
```

Any failure caused by missing v8 is a release blocker.

------------------------------------------------------------------------

# 8. Performance budgets

Bound:

``` text
symbols per window
passive-event queue
M1N facts per channel
payload bytes
proposal queue depth
read-set size
candidates per radius
number of radii
descriptor size
equivalence-set size
replay candidates
normalization sample reservoir
```

RAM growth should be driven by retained canonical knowledge/provenance
policy rather than historical patch-state duplication.

------------------------------------------------------------------------

# 9. Required telemetry

Track:

``` text
event counts by modality
M0-M7 counts
proposal/rejection/stale counts
read-set conflicts
cross-partition transactions
graph generation
lineage/overlay counts
probation transitions
canonical fork/reuse counts
normalization/bootstrap state
estimator generations
candidate entropy by radius
equivalence sets
transfer trials
G0-G5 grounding counts
grounding promotions/suspensions
symbol-conditioned prediction delta
grounded action influence
snapshot/restart metrics
```

Telemetry is observational.

------------------------------------------------------------------------

# 10. v8 parity policy

v9 does not need line-for-line parity with v8.

For capabilities intentionally retained from v8, define behavioral
acceptance tests and reproduce the required behavior using v9-owned
code.

Known v8 capabilities worth preserving include:

``` text
RAM-authoritative cognition
M0-M7 hierarchy
bounded normalized M1N
deterministic identity
partitioned/single-writer mutation
immutable read views
snapshot/restart
lifecycle/compaction
heterogeneous environments
macro/micro temporal evidence
scientific evidence/reporting
```

Historical v8 implementation structure is not part of the acceptance
contract.

------------------------------------------------------------------------

# 11. Definition of done

v9 is complete when:

1.  all production code is under `src/v9`;
2.  no production v9 module imports v8;
3.  v9 CLI is native;
4.  v9 runtime is native;
5.  v9 owns M0-M7 and canonical graph authority;
6.  ARC, FrozenLake, Chess and Sudoku run through v9 adapters;
7.  passive multimodal observations work;
8.  progressive structural search and equivalence sets work;
9.  lineage/context authority and read-set mutation validation work;
10. snapshot/restart is v9-native;
11. synthetic H16 C0-C3 experiments run;
12. BabyAI grounding runs;
13. generalized transfer is available;
14. v9 tests pass with `src/v8` unavailable;
15. v8 migration support, if used, is isolated under
    `src/v9/migration/`.

The final dependency model is:

``` text
v8 ----> historical/reference/migration input

v9 ----> independent executable system
```

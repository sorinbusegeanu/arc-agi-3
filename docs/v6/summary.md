# ARC-AGI v6 — Project Summary

## Overview

This is **ARC-AGI v6**, a cognitive interaction learner for the Abstraction and Reasoning Corpus (ARC). It implements a self-improving system that interacts with an environment, observes state transitions, takes actions, records everything into a hierarchical memory substrate, detects emergent patterns (contingencies, transformation families, carriers, roles, concepts), and builds a growing knowledge graph.

The project is a pure-Python package (`name = "v6"`, Python >= 3.8) with no external runtime dependencies beyond the standard library and NumPy. It persists state in SQLite databases by default (with optional Parquet migration).

## Architecture

### Core System — `V6System` (`main.py`)

The heart of v6 is the **`V6System`** class (~2500 lines), which drives every interaction:

1. **Observe** the current environment state
2. **Select an action** (random baseline, or ranked by memory query / contingency prediction)
3. **Apply** the action and observe the resulting state
4. **Extract a delta** (the change between before/after states)
5. **Record** the interaction into multiple memory tiers

The system is parameterized via `V6Config` with ~70+ tunable flags covering context depth, contingency thresholds, carrier detection, adaptive expansion, shared live memory, direct-streaming-fold storage, hypothesis-suite validation, and more.

### Memory Hierarchy (M0–M6)

Each interaction writes to a tiered memory substrate:

| Tier | Type | Purpose |
|------|------|---------|
| **M0** | Interaction / Observation / Action / Delta / Trajectory / Cost / Outcome memories | Raw per-step records and graph edges |
| **M1–M2** | Contingency / TransformationFamily memories | Learned rules: `context → action → family` |
| **M3** | Carrier memories | Higher-order signatures that span multiple families |
| **M4+** | Role, Concept, Strategy, EfficientStrategy memories | Cross-game generalizations and meta-patterns |

Each tier writes both **nodes** (facts) and **edges** (relationships), building a growing graph of discovered knowledge.

### Key Subsystems

- **`MemorySubstrate`** — Hierarchical memory with nodes + edges; supports upsert, scoring, and retention/forgetting lifecycle
- **`InteractionStore`** / **`TransformationStore`** / **`ContingencyStore`** — Per-tier SQLite-backed stores for raw interactions, deltas, and learned rules
- **`DeltaExtractor`** — Computes state changes between observations (supports cells, dx/dy shifts, color add/remove)
- **`ContextBuilder`** — Multi-scale context signatures at configurable depth
- **`ContingencyLearner`** — Learns stable `context → action → family` rules with support/confidence thresholds; tracks multi-scale contingencies
- **`TransformationClusterer`** — Clusters deltas into transformation families via centroid-based clustering; periodically reclusters to refine structure
- **`Predictor`** — Predicts which transformation family will result from a given context+action using contingency distributions
- **`GraphManager`** — Maintains an explicit graph of discovered relationships (explains, contradicts, depends_on, terminates)
- **`ContextContradictionTracker`** — Detects prediction violations; triggers adaptive context expansion when contradictions recur
- **`CarrierEmergenceTracker`** — Monitors for higher-order signatures that correlate across families and contexts
- **`MemoryLifecycleManager`** — Manages records through lifecycle: active → protected (high ISF) → replay-candidate → forgotten (low ISF); supports selective forgetting with configurable thresholds
- **`EfficiencyTracker`** / **`TrajectoryEfficiencyStore`** — Tracks action costs, repeated states/actions, terminal outcomes; computes normalized solve efficiency and trajectory bonuses
- **`FutureOptionEstimator`** — Estimates the set of reachable future states for a given observation+action pair at configurable depth
- **`MemoryQueryEngine`** / **`SnapshotMemoryQueryEngine`** — Query-driven action selection backed by learned contingencies, roles, concepts, and future options; supports cache hits/misses tracking

### Storage Backends (`storage/`)

- **SQLite backend** (default) — Full in-memory or file-based SQLite with WAL mode
- **Parquet backend** — Migration to columnar Parquet storage for large-scale runs
- **DuckDB backend** — DuckDB integration for analytical queries
- **Benchmark utilities** — Storage benchmarking, best-known solution length tracking

### Delta Extraction (`delta/`)

Extracts change signatures from between-state observations: cell changes, dx/dy shifts, color additions/removals. Supports delta-based memory and graph edges.

### Context & Contingency Learning (`context/`, `contingency/`)

- **ContextBuilder** — Builds multi-scale context representations
- **ContradictionTracker** — Tracks prediction violations; suggests context expansion when contradictions repeat
- **ContingencyLearner** — Learns stable rules of the form `(context_signature, action) → transformation_family` with support and confidence thresholds
- **ContextBuilder** (contingency submodule) — Builds contexts for contingency learning

## CLI Commands (`cli.py`)

The CLI exposes ~30 subcommands organized as a pipeline from raw interaction to high-level analysis:

### Core Pipeline

| Command | Purpose |
|---------|---------|
| `run` | Run a single game loop, collect interactions and metrics |
| `inspect` | Inspect stored families, contingencies, predictions, deltas |
| `validate` | Validate databases against reference rules |
| `milestone-1-5` | End-to-end milestone run across games/seeds/steps |

### Sampling & Folding (`interaction-sampling-v05c`)

The largest and most configurable command. Runs multi-game sampling with multiple samplers (random baseline, action balance, novelty delta, mixed, reset-aware mixed). Supports direct-streaming-fold storage for compact archival, parquet/duckdb backends, live memory injection from other workers, shared live memory mode, hypothesis-suite validation, and full postprocessing fold.

### Continuous Research (`continuous-research-run`)

Runs continuous research across games/samplers with epoch-based stepping, RAM ramping, direct-streaming-fold, and all sampling/fold options. Designed for long-running experiments that may span hours/days.

### Hypothesis Reports (H01–H11)

| Command | Purpose |
|---------|---------|
| `hypothesis-h01-report` | Evaluate contingency emergence across runs |
| `find-h01-ready-runs` | Find runs ready for H01 analysis |
| `hypothesis-h02-report` | Evaluate prediction-violation attention (H02) |
| `find-h02-ready-runs` | Find runs ready for H02 analysis |
| `hypothesis-h03-report` | Evaluate transformation-family formation (H03) |
| `find-h03-ready-runs` | Find runs ready for H03 analysis |
| `hypothesis-suite-report` | Comprehensive suite report covering all hypotheses, roles, carriers, concepts, provenance |

### Higher-Order Analysis Pipeline

These commands read from prior pipeline output and progressively build higher-order representations:

- **M2 Expand** (`m2-expand-v08c`) — Expand M2 families with additional sub-families
- **Transformation Families v07** (`transformation-families-v07`) — Classify M2 into stable transformation families
- **Context Depth Compare** (`compare-context-depth-v07`) — Compare context depths across runs
- **Role Candidates v08 / v08d** — Discover roles (cross-game pattern matchers) with discriminative fingerprinting
- **Role Transfer v09 / 09a / 09b / 09c** — Test whether roles generalize across games; progressively refine transfer accuracy
- **Concept Candidates v10 / v10fix–v10fixd / m4-role-concepts-v10e** — Discover concepts that can be transferred between role contexts; fixes for methodology, streaming, and memory safety
- **Migrate SQLite to Parquet** (`migrate-sqlite-to-parquet`) — Convert raw databases to columnar format

### Hypothesis Suite Validation

The `hypothesis-suite-report` command orchestrates the entire validation pipeline: it reads prior run outputs, validates each hypothesis (H01–H11), evaluates role transfer accuracy, concept candidate stability and transferability, and produces a comprehensive report with provenance tracking. Supports fast/full modes, incremental promotion validation, population comparability thresholds, and configurable max events per epoch.

## Data Flow

```
Environment → V6System.run_step() → DeltaExtractor → TransformationClusterer
    → ContingencyLearner (stable rules) + GraphManager (relationships)
    → MemorySubstrate (M0–M4+ tiers with nodes + edges)
    → Hypothesis Reports / Higher-Order Analysis Pipeline
```

## Key Concepts

- **Transformation Family**: A cluster of similar deltas grouped by centroid vector; represents a learned pattern of state change
- **Contingency**: A stable rule `(context, action) → family` that predicts outcomes with confidence/suppor thresholds
- **Carrier**: A higher-order signature emerging across multiple families/contexts; may indicate cross-family generalization
- **Role**: A structural pattern matcher that identifies similar contexts and actions across different games
- **Concept**: The highest-level abstraction—concepts that can be transferred between role contexts
- **ISF (Information Significance Factor)**: Composite scoring of each interaction for learning value, transfer potential, explanatory power, survival impact; drives selective forgetting
- **Delta**: The observable change between two states; the fundamental unit of transformation

## Project Structure

```
src/v6/
├── main.py                          # V6System core (~2500 lines)
├── cli.py                           # CLI with 30+ subcommands (~1940 lines)
├── __init__.py                      # Exports V6Config, V6System
├── pyproject.toml                   # Package config (name="v6", version=0.1.0)
│
├── memory/                          # Memory subsystem (~750 lines across 13 files)
│   ├── substrate.py                 # Hierarchical memory with nodes + edges
│   ├── interaction_store.py         # Raw interaction storage
│   ├── contingency_store.py         # Learned rule storage
│   ├── transformation_store.py      # Delta family storage
│   ├── compact_memory.py            # Compact/archived memory representation (~340 lines)
│   ├── compact_memory_restore.py    # Restore from compact archives
│   ├── direct_streaming_fold.py     # Direct streaming fold for compact storage
│   ├── worker_snapshot.py           # Worker-local snapshots for cross-worker sharing
│   ├── query_engine.py             # Memory query engine (action selection)
│   ├── live_memory_queue.py        # Live memory injection queue
│   ├── promotion_engine.py         # Record promotion/demotion logic
│   ├── selective_forgetting.py     # ISF-based forgetting policies
│   ├── trajectory_efficiency.py    # Trajectory-level efficiency tracking
│   └── ...                          # Additional lifecycle/utility modules
│
├── environment/                     # Environment interfaces (~50 lines)
│   ├── env_interface.py            # Protocol definition (observe, step, available_actions)
│   └── arc_adapter.py             # ARC grid adapter implementation
│
├── delta/                           # Delta extraction (~300 lines)
│   └── delta_extractor.py         # State change signature extraction
│
├── context/                         # Context building (~250 lines)
│   ├── context_builder.py          # Multi-scale context construction
│   └── contradiction_tracker.py    # Prediction violation tracking + expansion suggestion
│
├── contingency/                     # Contingency learning (~450 lines)
│   ├── contingency_learner.py      # Rule learner with support/confidence thresholds
│   └── context_builder.py          # Context-specific rule builder
│
├── storage/                         # Storage backends (~2000+ lines)
│   ├── sqlite_backend.py           # Default SQLite implementation
│   ├── parquet_backend.py         # Parquet columnar storage
│   ├── duckdb_queries.py          # DuckDB analytical queries
│   ├── migration.py               # SQLite → Parquet migration
│   ├── benchmark.py              # Storage performance benchmarks
│   └── backend.py                 # Backend abstraction layer
│
├── evaluation/                      # Validation and reporting (~3000+ lines)
│   ├── broad_game_validation.py    # Broad game validation suite
│   ├── failure_diagnostics.py     # Failure mode diagnostics across horizons/depths
│   ├── id_free_prefuture_validation.py  # ID-free future option validation
│   ├── interaction_sampling.py    # Interaction sampling configuration/reporting
│   ├── milestone_1_5.py          # End-to-end milestone runs
│   ├── prefuture_role_prediction.py     # Prefuture role prediction validation
│   ├── role_candidates.py        # Role candidate discovery validation
│   ├── role_generalization.py    # Role generalization across games
│   ├── role_validation.py        # Role validation pipeline
│   └── ...                        # Additional evaluation modules
│
├── transformation_families_v07.py  # M2 family classification pipeline
├── m2_expand_v08c.py               # M2 expansion pipeline
├── concept_candidates_v10*.py      # Concept candidate discovery and fixes (4 variants)
├── role_transfer_v09*.py           # Role transfer validation (4 versions: v09, 09a, 09b, 09c)
├── m4_role_concepts_v10e.py        # M4 role-based concepts pipeline
├── continuous_research.py          # Continuous research orchestration
├── contingency_memory.py           # Contingency memory v06 analysis
└── ...                             # Additional utility modules (~20 files)
```

## Testing (`tests/`)

~35 test files covering core system behavior, hypothesis reporting, direct streaming fold, compact memory restore, continuous research, interaction sampling, higher-order substrate, worker snapshots, and more.

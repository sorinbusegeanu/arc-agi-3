## Modules and implementation choices

### 1) Environment I/O Adapter

* **Role**: Wrap toolkit env; normalize `reset/step`; expose action schema (discrete vs coordinate).
* **Implementation**: Python wrapper; strict typed dataclasses for `Obs`, `StepResult`, `ActionSpec`.
* **Choices**:

  * Deterministic seeding and per-level episode runner
  * Action validation + canonical action encoding

---

### 2) Observation Parser

* **Role**: JSON frames → structured raw state.
* **Implementation**: Pure Python parsing + schema validators.
* **Choices**:

  * Handle 1–N frames (stacked views) consistently
  * Extract explicit UI fields (cursor, selected tool, counters, inventory)

---

### 3) State Abstraction + Canonicalization

* **Role**: Raw state → `StateAbstract` usable for hashing/search.
* **Implementation**:

  * Deterministic connected-components / entity extraction (grid-based)
  * Derived masks (walkable, blocking, interactive candidates)
  * Canonical sort of entities and normalized serialization
* **Choices**:

  * Stable `state_hash` (e.g., blake3/xxhash over compact bytes)
  * Optional symmetry canonicalization (if invariant)

---

### 4) Transition / Diff Engine

* **Role**: `(s, a, s') → delta`.
* **Implementation**: Deterministic differ:

  * tile changes, entity move/create/destroy, attribute toggles
  * inventory/counter/mode deltas
  * classify action as no-op / reversible / irreversible (heuristic)
* **Choices**:

  * Delta taxonomy (fixed enums) for learning + debugging
  * Transition compression for logging (small tokens)

---

### 5) Episodic Memory (Within-Level)

* **Role**: Store discovered dynamics and visited states.
* **Implementation**:

  * Hash tables keyed by `state_hash`
  * Transition map `(state_hash, action) → next_hash + delta + stats`
* **Choices**:

  * LRU cap for very large graphs
  * Cycle detector (k-period repeats)

---

### 6) Semantic Memory (Across Levels)

* **Role**: Reuse knowledge; set priors/options quickly.
* **Implementation**:

  * Key-value store (SQLite / LMDB / parquet) keyed by `mechanic_signature`
  * Store: mechanic templates, priors, option enablement, failure patterns
* **Choices**:

  * Retrieval via embedding or feature hashing over signature
  * Versioned schema for stability

---

### 7) Mechanic Inference (Belief Tracker)

* **Role**: Maintain `P(hypothesis)` + parameters.
* **Implementation choices**:

  * **V0**: rule-based scoring over deltas (fast to ship)
  * **V1**: small learned classifier over recent transition tokens
* **Choices**:

  * Hypothesis library = finite mechanic families + param schemas
  * Bayesian-style weight update with decay + contradiction penalties

---

### 8) Goal Detector + Progress Heuristics

* **Role**: Detect terminal success; provide shaping for planning.
* **Implementation choices**:

  * **V0**: explicit flags from env + pattern-based heuristics
  * **V1**: small learned `goal_likelihood(state)` model
* **Choices**:

  * Two signals: `terminal` and `progress_score`
  * Conservative success detection to avoid premature stop

---

### 9) World Model (Empirical)

* **Role**: Predict action outcomes when already observed.
* **Implementation**: Lookup in episodic transition map; optionally parametric predictor per hypothesis.
* **Choices**:

  * Deterministic when edge known
  * “Unknown edge” markers to drive exploration

---

### 10) Planner

* **Role**: Find minimal-action path to goal/targets.
* **Implementation choices**:

  * BFS (uniform cost) on discovered graph
  * A* with heuristic from progress/distance features
  * IDA* if memory constrained
* **Choices**:

  * Option-level planning (macro-actions) + primitive fallback
  * Replan on belief change or unexpected delta

---

### 11) Options / Macro-Action Library

* **Role**: Compress common action sequences.
* **Implementation**: Hand-coded controllers with preconditions + termination.
* **Choices**:

  * Navigate-to, interact-near, push-toward, select-tool+click
  * Each option returns: `{status, steps_used, resulting_state}`

---

### 12) Explorer

* **Role**: Choose probes that reduce uncertainty with few steps.
* **Implementation**: Deterministic scoring:

  * novelty, coverage, hypothesis discrimination, loop/no-op penalties
* **Choices**:

  * Probe sets per action schema
  * Budget split: early explore, late exploit

---

### 13) Critic

* **Role**: Audit planner/explorer proposals; flag risk/failure modes; re-rank actions.
* **Implementation choices**:

  * **V0**: rule-based checks using deltas + belief + graph stats
  * **V1**: learned risk model trained on failure trajectories
* **Checks**:

  * loop/thrash risk, irreversible risk, hypothesis mismatch, dead-end likelihood
* **Output**: penalties + tags + optional counterproposal

---

### 14) Robustness / Safety Guard

* **Role**: Enforce hard constraints and recovery.
* **Implementation**: Deterministic rules.
* **Choices**:

  * no-op caps, cycle breakers, backtracking, safe action fallback
  * action budget manager (per level)

---

### 15) Controller (Single Action Emitter)

* **Role**: Combine proposals into one env action.
* **Implementation**: Arbitration policy.
* **Choices**:

  * Candidate set = {planner action, explorer action, critic alt}
  * Score = expected success + explore value − critic penalties − safety penalties
  * Tie-break = shortest known path / least tried

---

### 16) Logger + Dataset Builder

* **Role**: Record trajectories for debugging and training.
* **Implementation**: JSONL/Parquet logs with compact tokens.
* **Choices**:

  * Store windows of transitions + outcomes + critic tags
  * Deterministic replay support

---

### 17) Learner (Offline Training)

* **Role**: Train small components from logs.
* **Implementation choices**:

  * Mechanic classifier (Transformer/MLP over tokenized transitions)
  * Action ranker (pairwise ranking or value regression)
  * Critic risk model (multi-label classification)
* **Choices**:

  * Supervised from successful vs failed episodes first
  * Optional RL fine-tune later for action efficiency

---

### 18) Evaluator / Benchmark Runner

* **Role**: Run official suite; compute score; track regressions.
* **Implementation**: Use benchmarking repo runner + your agent adapter.
* **Choices**:

  * Per-game breakdown; action counts; completion rate; failure tags summary

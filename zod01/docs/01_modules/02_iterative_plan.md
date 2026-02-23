Yes.

## Minimal staged path (module-by-module)

### Stage 0: Skeleton runner

* **1) Environment I/O Adapter**
* **18) Evaluator / Benchmark Runner**
* **16) Logger + Dataset Builder**
  Goal: deterministic replay, action validation, per-level scoring.

### Stage 1: Deterministic core state machinery

* **2) Observation Parser**
* **3) State Abstraction + Canonicalization**
* **4) Transition / Diff Engine**
  Goal: stable `state_hash`, deltas, and readable traces.

### Stage 2: Episodic graph + no-brain baseline

* **5) Episodic Memory**
* **14) Robustness / Safety Guard**
* **15) Controller** (simple: random valid action + safety rules)
  Goal: confirm graph growth, loop detection, backtracking.

### Stage 3: Empirical world model

* **9) World Model (Empirical)** (pure lookup: (s,a)→s’ if known)
  Goal: “known-edge exploitation” works; unknown edges marked.

### Stage 4: Planner-only agent

* **10) Planner** (BFS first)
  Goal: if any goal state is discovered, planner reaches it reliably.

### Stage 5: Goal/progress shaping

* **8) Goal Detector + Progress Heuristics**
  Goal: A* becomes possible; better late-episode exploitation.

### Stage 6: Explorer

* **12) Explorer**
  Goal: systematic discovery of new edges/states under budget.

### Stage 7: Options (macro-actions)

* **11) Options / Macro-Action Library**
  Goal: reduce action count; improve exploration efficiency.

### Stage 8: Mechanic inference (rule-based first)

* **7) Mechanic Inference V0**
* **6) Semantic Memory V0** (store mechanic_signature → priors)
  Goal: faster early decisions; cross-level transfer begins.

### Stage 9: Critic

* **13) Critic V0**
  Goal: reduce irreversible mistakes, dead-ends, and thrashing.

### Stage 10: Offline learner components

* **17) Learner** (start with mechanic classifier OR action ranker)
  Goal: replace heuristics with learned modules, one at a time.

## How to test at each stage (single metric per stage)

* Stage 0–1: **replay determinism rate** (same seed → same trajectory hash)
* Stage 2–3: **unique states discovered / 1k steps**
* Stage 4: **success rate given goal discovered**
* Stage 5–7: **median steps-to-solve on solved levels**
* Stage 8–9: **regression delta when enabling transfer/critic**
* Stage 10: **Ablation wins** (learned module beats heuristic on held-out levels)

## Key constraint for incremental testing

Keep **Controller (15)** stable and swap only **one scoring term/module at a time**; otherwise regressions won’t be attributabl

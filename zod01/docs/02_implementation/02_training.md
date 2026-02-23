You **don’t train the whole agent**. You train **specific learned components** (Module 17) using logs produced by the implemented agent (Module 16), then plug them back in behind flags.

## 0) Prerequisite (before training)

Your logs must contain **state changes** (e.g., `unique_states > 1`). If not, any dataset you build will be junk.

---

## 1) Data collection run

Run many episodes and save logs in a single run directory.

**Collect diversity:**

* multiple `--seed`
* multiple `--games`
* multiple swarm variants (if implemented)
* fixed `--max-actions` budget

**Log per step at minimum:**

* `episode_id, game_id, seed, variant_id, step_idx`
* `state_hash, raw_obs_hash (optional but useful)`
* `available_actions` (or canonical `ActionSpec`)
* `chosen_action` (canonical)
* `next_state_hash`
* `delta_tokens` and/or `delta.kind`
* `terminal/won`, `actions_used`
* critic tags/penalties (if present)

---

## 2) Build supervised datasets from logs

Create **one dataset per learned module**. Start with the highest-leverage one:

### A) Action Ranker dataset (recommended first)

For each step:

* input: `(state_features, action_features, belief_topk, memory_stats)`
* candidates: all `available_actions`
* labels:

  * positive: the action taken on **successful episodes**
  * negatives: (i) other available actions at that step, (ii) actions from failed episodes, (iii) actions flagged unsafe/critic

Output format: one row per `(step, action_candidate)` with `label ∈ {0,1}` or pairwise comparisons.

### B) Critic Risk dataset

For each step and chosen action:

* input: `(state_features, action_features, memory_stats, belief)`
* labels: multi-label tags such as `{loop_risk, irreversible_risk, dead_end}` + episode outcome

### C) Mechanic classifier dataset

For each step:

* input: window of last `k` `delta_tokens` (+ optional coarse state signature)
* labels: hypothesis ID (from your belief tracker once it stabilizes) OR a clustered signature ID (if labels are noisy)

---

## 3) Train one module at a time (offline)

Keep the rest of the agent fixed; swap only one module.

### Model choices (small, fast)

* Ranker: MLP or small Transformer over tokenized features
* Risk: MLP or small Transformer, multi-label BCE
* Mechanic: Transformer/MLP over `delta_tokens` sequence, CE loss

### Losses

* Ranker: pairwise ranking loss or BCE over candidates
* Risk: multi-label BCE
* Mechanic: cross-entropy

### Splits

Split by **game_id** (not random steps) to avoid leakage.

---

## 4) Integrate back into the agent

Add runtime flags:

* `use_ranker`
* `use_learned_critic`
* `use_mechanic_classifier`

Controller scoring becomes something like:

* `score = heuristic_score + w_ranker * rank_score - w_risk * risk_score - w_safety * safety_penalty`

Keep the old heuristics enabled as a fallback until you see consistent wins.

---

## 5) Evaluate with strict ablations

For each new learned module:

* Baseline (all V0 heuristics)
* * Learned module (only change)
* * Learned module with 2–3 weight settings

Report:

* win rate
* median actions on wins
* irreversible rate
* loop/thrash rate
* unique_states per 1k steps

---

## Minimal “training path” that usually works

1. Train **Action Ranker** first
2. Train **Critic Risk** second
3. Train **Mechanic Classifier** third
4. Only then consider goal/progress CNN if needed

This is the shortest path to a measurable improvement without destabilizing the pipeline.

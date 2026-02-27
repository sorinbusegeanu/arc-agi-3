##RL Training (end-to-end) for existing project

## 0) Scope

Define **how training is run** once the RL stack exists in the repo:

* modes (`collect`, `train`, `eval`)
* artifacts written
* config keys
* dataset / environment selection
* checkpoint lifecycle
* determinism requirements
  No code.

---

## 1) Entry point and CLI contract

### 1.1 New entrypoint

Add an RL entrypoint file (name per repo convention), referred to here as `run_rl`.

### 1.2 CLI interface (must be implemented exactly)

`run_rl` must support:

* `--mode` one of: `collect`, `train`, `eval`
* `--games` string selector identical to existing offline runner (e.g. `bt11-...` or comma list or file)
* `--seed` int (global seed)
* `--max-actions` int (episode horizon; must match benchmark budget semantics)
* `--episodes` int (episodes to run in collect/eval; or per-iteration in train)
* `--outdir` path (root output dir)
* `--checkpoint` optional path (load weights)
* `--config` optional path or config override mechanism consistent with repo

`run_rl` must exit non-zero on config/IO errors.

---

## 2) Config contract (single `rl` block)

All training behavior is controlled by a config block `rl.*`:

### 2.1 Model sizes

* `rl.embed_dim` (D, default 256)
* `rl.hidden_dim` (H, default 256)
* `rl.action_emb_dim` (default 32)

### 2.2 Coordinate selection

* `rl.coord_mode = "proposal"` (v1 only)
* `rl.coord_topK` (default 16)

### 2.3 Rollout

* `rl.episodes_per_iter` (default 8)
* `rl.max_steps_per_episode` (default = `--max-actions`)
* `rl.stochastic_actions_train` (default true)
* `rl.deterministic_eval` (default true)
* `rl.save_trajectory_batches` (default true)

### 2.4 Reward shaping

* `rl.reward.alpha_novel` (default 0.05)
* `rl.reward.beta_effect` (default 0.02)
* `rl.reward.delta_loop` (default 0.05)
* `rl.reward.loop_window_N` (default 25)

### 2.5 Optimizer / algorithm

* `rl.algo = "a2c"` (default) or `"ppo"`
* `rl.lr` (default 3e-4)
* `rl.gamma` (default 0.99)
* `rl.entropy_coef` (default 0.01)
* `rl.value_coef` (default 0.5)
* `rl.max_grad_norm` (default 1.0)
* `rl.updates_per_iter` (default 1)
* PPO-only (ignored if algo != ppo):

  * `rl.ppo.clip_eps` (default 0.2)
  * `rl.ppo.epochs` (default 2)
  * `rl.ppo.minibatches` (default 2)
  * `rl.ppo.target_kl` (default 0.03)

### 2.6 Checkpointing / logging

* `rl.ckpt.save_every_iters` (default 1)
* `rl.ckpt.keep_last` (default 5)
* `rl.log.write_jsonl` (default true)
* `rl.log.write_trace` (default true)

---

## 3) Training modes

## 3.1 Mode: `collect`

Purpose: generate rollouts without updating weights.

### Inputs

* games selection
* optional checkpoint

### Behavior

For each episode:

1. Reset env
2. Run up to `max_steps_per_episode`
3. At each step:

   * compute FP_Analyst outputs
   * compile TransitionEvent
   * compute shaped reward
   * store step record

### Outputs (under `outdir/collect/...`)

* `trajectories/`:

  * `batch_000001.jsonl` (or `.pt`), containing `TrajectoryBatchV1`
* `traces/`:

  * existing trace format + RL fields (see §6)
* `metrics.json` summary:

  * mean return, mean steps, novelty rate, non-noop rate, win rate

No parameter updates.

---

## 3.2 Mode: `train`

Purpose: iterative loop: collect rollouts → update model → checkpoint → periodic eval.

### Loop structure

Training runs for `rl.train.num_iters` (new config key; default 100), or until interrupted.

Per iteration `i`:

1. **Collect** `rl.episodes_per_iter` episodes using current policy.
2. **Train** for `rl.updates_per_iter` updates over the collected batch:

   * compute returns and advantages
   * update parameters
3. **Checkpoint**:

   * save if `i % save_every_iters == 0`
4. **Eval (optional)**:

   * if `rl.eval.every_iters` set (default 5), run eval on `rl.eval.episodes` episodes with deterministic policy.

### Outputs (under `outdir/train/...`)

* `checkpoints/ckpt_iter_000123.pt`
* `metrics/train_iter_000123.json` (losses + returns + win rate)
* `metrics/eval_iter_000125.json` (if eval ran)
* `traces/` for a small subset only (to limit disk), controlled by:

  * `rl.log.trace_episodes_per_iter` (default 1)

---

## 3.3 Mode: `eval`

Purpose: evaluate a checkpoint deterministically.

### Behavior

* run N episodes with `deterministic_eval=true` (argmax policy)
* no exploration bonuses for action choice
* still compute shaped reward for reporting, but do not use it to alter behavior

### Outputs

* `metrics_eval.json`
* traces for all eval episodes (or capped by config)

---

## 4) Environment selection contract

Training requires broad coverage; define selection mechanisms:

### 4.1 `--games` resolution priority

1. If it is a file path: read lines, each a game id
2. Else if it contains comma: split into explicit list
3. Else treat as single game id or existing selector supported by repo

### 4.2 Train/eval split

Config keys:

* `rl.env.train_games` (optional list or file)
* `rl.env.eval_games` (optional list or file)

If not provided, `--games` is used for both.

---

## 5) Determinism and reproducibility

### 5.1 Seeds

* Global seed from `--seed`
* Episode seed = deterministic function of `(global_seed, iter_idx, episode_idx)`
* Policy sampling uses its own RNG seeded similarly

### 5.2 Logged reproducibility bundle

Each run writes:

* `run_config.json` (full resolved config)
* `seeds.jsonl` (per-episode seeds)
* `git_commit.txt` if available

---

## 6) Trace and trajectory schema requirements (training-critical)

### 6.1 Step record (TrajectoryBatchV1 minimum)

For each step store:

* `obs_ref` (either raw observation pointer or minimal normalized snapshot)
* `available_actions_mask`
* `action_key` (including x,y if coord)
* `coord_candidates` (if ACTION6 available) and `chosen_coord_index` (if used)
* `reward_total` + `reward_terms`
* `done`
* `state_hash_before`, `state_hash_after`
* `TransitionEventV1` (embedded or referenced)

### 6.2 Optional: logits for debugging

Store compact summaries only:

* `discrete_logits_topk` (ids + logits)
* `coord_logits_topk` (indices + logits)

Do not store full tensors by default.

---

## 7) Success metrics to log each iteration

Training metrics:

* `mean_return`
* `mean_episode_len`
* `win_rate`
* `novelty_rate` (fraction steps with novel state hash)
* `effect_rate` (fraction steps non-noop)
* `loop_rate` (fraction steps repeated within window)

Optimization metrics:

* `policy_loss`
* `value_loss`
* `entropy`
* `grad_norm`
* PPO: `approx_kl`, `clip_frac`

---

## 8) Minimal training recipe (what “start training” means)

Once implemented, a correct training run is:

1. `collect` for a small game set to validate shaped reward is non-zero.
2. `train` for ~50–200 iterations on the same set.
3. `eval` deterministically on a held-out set.

The run is considered “wired correctly” if:

* metrics files are produced per iteration
* checkpoints advance
* win_rate or mean_return changes over iterations (not necessarily improves immediately)
* traces show ACTION6 coordinate proposals when ACTION6 is available



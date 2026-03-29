# `src/arc_agi_agent/rl` Implementation

This document describes the current implementation of `src/arc_agi_agent/rl` as it exists in code on 2026-03-26. It is an implementation note, not a design spec. Where the system uses hard-coded heuristics, rollout-time guards, or compatibility shims, those are described explicitly.

## Overview

`src/arc_agi_agent/rl` implements an RL-only training and evaluation stack for ARC/arcade environments. The package contains:

1. a CNN observation encoder over normalized grid observations
2. a recurrent memory core over observation embeddings plus previous action/reward/done
3. an optional hierarchical controller that selects a discrete mode
4. a policy actor with separate discrete-action and coordinate heads
5. a value head
6. an optional intrinsic RND novelty module
7. a rollout collector that runs environments, FP analysis, event compilation, and reward shaping
8. a trainer that supports both A2C and PPO
9. a multiprocessing orchestration layer for distributed rollout collection

The main executable entrypoint is [run_rl.py](/home/zodrak/zod/src/arc_agi_agent/rl/run_rl.py). The model assembly wrapper is [rl_agent.py](/home/zodrak/zod/src/arc_agi_agent/rl/rl_agent.py).

## Module Map

- [run_rl.py](/home/zodrak/zod/src/arc_agi_agent/rl/run_rl.py): CLI, config loading, rollout worker dispatch, metrics aggregation, evaluation, checkpoint policy, W&B logging, and run directory layout.
- [rl_agent.py](/home/zodrak/zod/src/arc_agi_agent/rl/rl_agent.py): constructs the RL module stack, loads defaults, moves modules to device, and exposes checkpoint/inference helpers.
- [observation_encoder.py](/home/zodrak/zod/src/arc_agi_agent/rl/observation_encoder.py): encodes normalized grid observations and scalar metadata into `z_t`.
- [recurrent_memory.py](/home/zodrak/zod/src/arc_agi_agent/rl/recurrent_memory.py): GRU/LSTM recurrent state update over `z_t`, previous action, previous reward, and previous done flag.
- [hierarchical_controller.py](/home/zodrak/zod/src/arc_agi_agent/rl/hierarchical_controller.py): optional mode policy over the recurrent hidden state.
- [policy_actor_value.py](/home/zodrak/zod/src/arc_agi_agent/rl/policy_actor_value.py): current discrete-action actor, coordinate actor, log-prob reconstruction helpers, and value head.
- [policy_value_heads.py](/home/zodrak/zod/src/arc_agi_agent/rl/policy_value_heads.py): older simpler policy/value heads; present in tree but not used by `RLAgent`.
- [coord_proposer.py](/home/zodrak/zod/src/arc_agi_agent/rl/coord_proposer.py): deterministic coordinate candidate generation from FP outputs.
- [rollout_collector.py](/home/zodrak/zod/src/arc_agi_agent/rl/rollout_collector.py): environment interaction loop and trajectory batch construction.
- [reward_shaper.py](/home/zodrak/zod/src/arc_agi_agent/rl/reward_shaper.py): step reward heuristics, flash detection, revert penalties, and POI pattern matching bonus.
- [trainer.py](/home/zodrak/zod/src/arc_agi_agent/rl/trainer.py): A2C and PPO train steps, PPO preprocessing, KL checks, LR adaptation, and trainer phase switching.
- [intrinsic_rnd.py](/home/zodrak/zod/src/arc_agi_agent/rl/intrinsic_rnd.py): predictor-target RND module and online normalization state.
- [optim.py](/home/zodrak/zod/src/arc_agi_agent/rl/optim.py): Adam optimizer creation, including optional param groups and RND predictor params.
- [obs_norm_v1.py](/home/zodrak/zod/src/arc_agi_agent/rl/obs_norm_v1.py): deterministic normalized observation schema used by the encoder and collector.
- [canonical_grid.py](/home/zodrak/zod/src/arc_agi_agent/rl/canonical_grid.py): extracts the primary 2D grid and hashes it.
- [action_key_normalize_v1.py](/home/zodrak/zod/src/arc_agi_agent/rl/action_key_normalize_v1.py): stable mapping between action ids and integer indices.
- [module_control.py](/home/zodrak/zod/src/arc_agi_agent/rl/module_control.py): RL-only module gating and logging restrictions.
- [coverage_ledger.py](/home/zodrak/zod/src/arc_agi_agent/rl/coverage_ledger.py): counts visited states, state-action pairs, and repeated no-op actions.
- [checkpoint_io.py](/home/zodrak/zod/src/arc_agi_agent/rl/checkpoint_io.py): standalone checkpoint helpers; the main run path mostly uses `RLAgent._save_checkpoint` and `_load_checkpoint`.

## Top-Level Runtime

The top-level flow is implemented in `main()` in [run_rl.py](/home/zodrak/zod/src/arc_agi_agent/rl/run_rl.py).

At startup it:

1. parses CLI arguments for `collect`, `train`, or `eval`
2. suppresses most stdout logging except selected rollout/trainer progress lines
3. prepares Python paths for `other_repos/arc-agi` and `other_repos/ARCEngine`
4. configures Torch multiprocessing shared-file transport
5. requires CUDA up front
6. resolves train and eval game sets
7. creates a timestamped run directory under `<outdir>/rl/<timestamp>`
8. loads the default config and overlays user JSON
9. applies RL-only module mode and RL-only logging guards
10. constructs `RLAgent`, module dict, and optimizer
11. optionally loads a checkpoint

In `train` mode it then iterates:

1. snapshot current policy weights to CPU
2. collect a training rollout batch, optionally through worker processes
3. aggregate train metrics and update the coverage ledger
4. train with A2C or PPO
5. write train metrics and checkpoints
6. periodically run evaluation on holdout games and optional easy games
7. update the trainer phase and summary file

In `collect` and `eval` mode it only collects one batch, writes trajectories and metrics, and exits.

## RL-Only Mode and Guards

RL-only behavior is enforced by [module_control.py](/home/zodrak/zod/src/arc_agi_agent/rl/module_control.py).

- `apply_rl_only_mode(...)` rewrites `cfg["pipeline"]["mode"]` to `rl_only` and replaces the module enable map.
- The RL stack keeps FP analyst, transition event compilation, trace writing, and RL modules enabled.
- Planner, explorer, memory-store, and other non-RL orchestration modules are forcibly disabled.
- `assert_rl_only_guards(...)` rejects configs that still expose non-RL behavior.
- `configure_rl_only_logging(...)` mutes blocked namespaces and installs a root logging filter that only allows RL/FP/Event/Trace/W&B prefixes.

This means the collector still depends on FP analysis and transition-event code, but the higher-level non-RL agent stack is intentionally cut out.

## Agent Assembly

`RLAgent` in [rl_agent.py](/home/zodrak/zod/src/arc_agi_agent/rl/rl_agent.py) is the module assembly wrapper.

Config resolution is:

1. start from `RLConfig()`
2. convert it into a plain dict with `_cfg_to_dict(...)`
3. overlay `rl_config_defaults.json` if present
4. overlay any explicit constructor config

The agent instantiates:

- `ObservationEncoder`
- `RecurrentMemory`
- `HierarchicalController` if `rl_controller` is enabled
- `PolicyActor`
- `ValueHead`
- `IntrinsicRND` only when `intrinsic.enabled` and `intrinsic.method == "rnd_grid_embed"`
- `CoordProposer`
- `RewardShaper`
- `RolloutCollector`
- `Trainer`

Device policy is strict:

- `device == "cpu"` is only used inside rollout workers
- otherwise `RLAgent` requires CUDA and raises if CUDA is unavailable

`policy_version` is tracked on the agent and used when dispatching snapshot weights to rollout workers.

## Observation and State Encoding

The observation pipeline is split across [obs_norm_v1.py](/home/zodrak/zod/src/arc_agi_agent/rl/obs_norm_v1.py), [canonical_grid.py](/home/zodrak/zod/src/arc_agi_agent/rl/canonical_grid.py), and [observation_encoder.py](/home/zodrak/zod/src/arc_agi_agent/rl/observation_encoder.py).

`normalize_obs_v1(...)` produces deterministic `OBS_NORM_V1` records:

- sorted `grids`
- scalar `meta`
- `meta_keys`
- `meta_vector`
- `step_idx`
- sorted `available_actions_sorted` when available

The encoder then:

1. builds a frame stack of canonical grids
2. optionally uses one-hot color planes instead of scalar grid values
3. optionally appends one or two frame-diff channels
4. runs a Conv/ReLU/MaxPool backbone
5. global-pools to `grid_embed`
6. runs a metadata MLP over a fixed-width metadata vector
7. concatenates `grid_embed` and `meta_embed`
8. projects to `z_t`

`ObservationEncoder.encode(...)` also returns the normalized observation and shape/debug metadata. If the collector already computed normalized grids, it can pass them through `ctx` to avoid redundant work.

## Recurrent Memory

[recurrent_memory.py](/home/zodrak/zod/src/arc_agi_agent/rl/recurrent_memory.py) maintains the recurrent hidden state.

Each memory step concatenates:

- current `z_t`
- embedding of the previous action id
- previous action coordinates if present
- clipped previous reward
- previous done flag

The recurrent core is lazily built on first use because the exact encoder output dimension is only known at runtime. Reset behavior is episode-aware: if `prev_done` is true, the recurrent state is reinitialized before the step update.

## Controller, Actor, and Value

The controller in [hierarchical_controller.py](/home/zodrak/zod/src/arc_agi_agent/rl/hierarchical_controller.py) is a simple MLP from hidden state to `mode_logits`.

- In training it can sample from `softmax(mode_logits)` when `sample_mode_train` is enabled.
- Otherwise it is greedy.

The main policy implementation is [policy_actor_value.py](/home/zodrak/zod/src/arc_agi_agent/rl/policy_actor_value.py).

Discrete action policy:

- embeds each candidate action id
- concatenates the action embedding with `h_t`
- scores each action independently through a shared MLP
- applies per-mode allow masks and additive per-mode action bias

Coordinate policy:

- constructs 8-dimensional coordinate features
- features include normalized x/y, hashed tag id, local patch occupancy/color stats, distance-to-edge, and distance to nearest FP object centroid
- scores each coordinate candidate through a second shared MLP
- applies additive per-mode coordinate bias

`compute_logp_components(...)` reconstructs mode, action, and coordinate log-probs from stored rollout records. PPO depends on this heavily for rollout-policy consistency checks and KL computation.

The value head is a separate MLP from hidden state to a scalar value.

## Coordinate Candidate Generation

[coord_proposer.py](/home/zodrak/zod/src/arc_agi_agent/rl/coord_proposer.py) deterministically generates click targets from FP outputs.

It draws candidates from:

- object centroids
- object bounding-box corners
- FP interaction points
- changed-bbox center
- fixed grid anchors

Candidates are then:

- sorted by priority, y, x, and tag
- deduplicated by coordinate
- clipped to the grid
- truncated to `coord_topK`

This keeps coordinate action `ACTION6` grounded in a bounded candidate set.

## Rollout Collection

The environment interaction loop is [rollout_collector.py](/home/zodrak/zod/src/arc_agi_agent/rl/rollout_collector.py). `RolloutCollector.collect(...)` is the authoritative source of trajectory batches.

Per episode it:

1. builds an environment from the provided factory
2. resets the environment
3. runs FP analysis on the initial observation
4. normalizes the current observation and canonical grid
5. builds the action schema from the env action space
6. maintains frame-stack history, visit counts, and optional video output
7. for each step:
   - encode observation
   - update recurrent memory
   - sample or choose a controller mode
   - optionally compute intrinsic RND error and normalized `phi`
   - propose coordinate candidates
   - run the actor
   - apply the env available-action mask before choosing a discrete action
   - optionally choose a coordinate for `ACTION6`
   - record old log-prob components and old value
   - step the environment
   - run FP analysis on the next observation
   - compile a transition event
   - compute reward shaping terms
   - write a step record
8. emit an episode record into `TRAJECTORY_BATCH_V1`

Important rollout behavior:

- `normalize_available_actions_mask(...)` accepts several legacy encodings and canonicalizes them to a boolean mask.
- If a mask becomes all-false, it falls back to all-valid rather than leaving the policy with zero actions.
- In RL-only mode, `fast_collect` defaults on unless trace/full-batch persistence is enabled.
- Optional video output writes PNG frames per episode and encodes `out.mp4` with `ffmpeg`.

Each step record carries enough data for PPO re-evaluation later, including:

- `h_t`
- `mode_id`
- `mode_logits`
- `action_ids`
- `action_index`
- `available_actions_mask`
- `coord_candidates`
- `chosen_coord_index`
- `old_logp_mode`
- `old_logp_action_discrete`
- `old_logp_coord`
- `old_value`
- reward terms, transition event, state hashes, and optional intrinsic terms

## Reward Shaping

[reward_shaper.py](/home/zodrak/zod/src/arc_agi_agent/rl/reward_shaper.py) defines the RL reward heuristics.

`effect_from_transition(...)` computes:

- raw changed-cell count
- total changed fraction
- flash-event detection using whole-grid change fraction plus color-histogram L1 distance

`compute(...)` then produces:

- `r_win`: `1.0` on win
- `r_effect`: small reward for movement-like local changes and larger reward for broader screen changes
- `r_match_poi`: bonus when a changed block appears to match a new non-overlapping object-like POI pattern
- `r_revert`: penalty for A→B→A state reversion
- `r_potential`: currently hard-coded to `0.0`
- `r_step`: growing time penalty capped by config
- `r_noop`: penalty when changed cells fall below the no-op threshold

The total reward is:

- `r_win + m_noop * (r_effect + r_match_poi + r_revert + r_potential) + r_step + r_noop`

where `m_noop` is `1` only when changed cells meet the non-noop threshold.

The most specialized heuristic is `_match_poi_blocks(...)`, which searches for duplicated or relocated object-like patterns between consecutive grids using:

- changed connected components
- perimeter-dominance box completion
- mask/shape/layout similarity
- center, size, aspect, and overlap terms

HUD masking logic has been removed; `reset_hud_cache()` remains as a no-op compatibility method.

## Intrinsic RND

[intrinsic_rnd.py](/home/zodrak/zod/src/arc_agi_agent/rl/intrinsic_rnd.py) implements a standard fixed-target / trainable-predictor novelty model over `grid_embed`.

- The target network is frozen.
- The predictor is trainable and can be added to the optimizer.
- `RNDNormState` tracks running mean/variance of scalar error.
- `compute_phi(...)` z-scores the error, floors negative values to zero, and clips to `phi_clip`.

The collector records intrinsic terms into step records. The trainer also computes a predictor loss when intrinsic RND is enabled and valid grid embeddings are present.

## Training Algorithms

The train step entrypoint is `Trainer.train_step(...)` in [trainer.py](/home/zodrak/zod/src/arc_agi_agent/rl/trainer.py).

### A2C path

`_train_step_a2c(...)`:

- recomputes controller logits, actor logits, and values step by step
- reconstructs action masks and optional coord logits
- computes returns with simple discounted backup
- uses policy-gradient losses for both controller and actor
- adds value MSE and optional controller auxiliary cross-entropy
- subtracts entropy regularization

This path is simpler and does not use clipped objectives or KL checks.

### PPO path

`_train_step_ppo(...)` is the dominant implementation and has much stricter preprocessing.

It:

1. normalizes PPO config and rejects legacy/unknown keys
2. checks that the rollout config hash matches the trainer-side hash
3. optionally recomputes missing rollout values in chunks
4. flattens all episode steps into one training table
5. computes returns and advantages with GAE or simple returns
6. stages all hidden states, old log-probs, masks, values, and coord metadata on device
7. rebuilds available-action masks and validates coord feature signatures
8. optionally performs a pre-update rollout-policy consistency / KL check
9. runs PPO epochs and minibatches
10. applies clipped objectives for mode, action, and coordinate branches
11. clips value loss if enabled
12. optionally trains intrinsic RND predictor loss
13. performs KL-based early stopping
14. adapts optimizer LR based on KL target tracking

Notable PPO implementation details:

- KL can be tracked against `mode`, `action`, or `coord` depending on `ppo.kl_metric`.
- Pre-update KL can abort the train step if the rollout policy looks stale relative to the current model.
- `phase_state["stage"]` switches from `exploration` to `win` after enough eval win-rate hits, and some entropy/target-KL settings change with phase.
- The trainer returns a large report including pre/post-update KLs, clip fractions, advantage stats, intrinsic stats, and effective PPO epoch/minibatch counts.

## Multiprocess Rollout Dispatch

Multiprocess rollout orchestration lives in [run_rl.py](/home/zodrak/zod/src/arc_agi_agent/rl/run_rl.py).

`_collect_batch(...)` works in two modes:

- local single-process collection when `workers <= 1` or `episodes <= 1`
- spawned worker-process collection otherwise

In the multiprocess case:

- current policy weights are snapshotted to CPU
- episode counts are split across workers
- each worker gets its own `seed_base` offset
- `_collect_worker(...)` forces CPU execution and low Torch thread counts
- the worker constructs a CPU `RLAgent`, loads the policy snapshot, and runs collection locally
- worker outputs are merged into one batch

Policy-version and rollout-config metadata are written onto both the batch and each episode. PPO later uses these fields to detect stale or incompatible rollouts.

## Metrics, Evaluation, and Coverage

[run_rl.py](/home/zodrak/zod/src/arc_agi_agent/rl/run_rl.py) also owns run-level metrics.

`_aggregate_metrics(...)` computes per-episode and aggregate statistics:

- win rate
- mean and median return
- average total reward
- mean episode length
- effect/novelty/loop rates
- mode entropy and optional auxiliary accuracy
- policy entropy
- coordinate selection rate
- mode/action/coord-tag usage

`CoverageLedgerV1` in [coverage_ledger.py](/home/zodrak/zod/src/arc_agi_agent/rl/coverage_ledger.py) tracks:

- unique states
- unique state-action pairs
- unique transitions
- frequently repeated states
- repeated no-op actions by state

During training, periodic eval batches are collected on holdout games and optionally an easy split. Eval results update:

- trainer phase state
- `metrics/summary.json`
- best-checkpoint decisions such as `best_total_reward.ckpt`

## Checkpoints and Persistence

The main run path uses `RLAgent._save_checkpoint(...)` and `_load_checkpoint(...)` in [rl_agent.py](/home/zodrak/zod/src/arc_agi_agent/rl/rl_agent.py).

Saved checkpoint payloads contain:

- `iter`
- `cfg`
- `encoder`
- `memory`
- `controller`
- `actor`
- `value`
- `optim`

[checkpoint_io.py](/home/zodrak/zod/src/arc_agi_agent/rl/checkpoint_io.py) provides simpler standalone helpers that save the same core module weights but do not include the controller.

## Run Artifacts

A training run writes under `<outdir>/rl/<timestamp>/`, including:

- `configs/resolved_config.json`
- `metrics/train_iter_*.json`
- `metrics/eval_iter_*.json`
- `metrics/eval_easy_iter_*.json`
- `metrics/summary.json`
- `metrics/episodes.jsonl`
- `seeds.jsonl`
- `traces/*.jsonl` for sampled eval episodes
- `trajectories/batch.json` in collect/eval mode
- `checkpoints/last.ckpt`
- `checkpoints/best_total_reward.ckpt`
- optional other checkpoints written by mode-specific logic
- `log.log`
- optional `video/episode_*/frame_*.png` plus encoded `out.mp4`

## Important Constraints and Quirks

- CUDA is mandatory for the main process; CPU is only used inside rollout workers.
- The rollout collector still depends on FP analysis and transition-event compilation even in RL-only mode.
- The current active policy/value implementation is [policy_actor_value.py](/home/zodrak/zod/src/arc_agi_agent/rl/policy_actor_value.py); [policy_value_heads.py](/home/zodrak/zod/src/arc_agi_agent/rl/policy_value_heads.py) is present but not wired into `RLAgent`.
- PPO assumes rollout records contain enough hidden-state and log-prob detail to reconstruct the old policy exactly enough for KL and clipping checks.
- Available-action masks are aggressively normalized and can fall back to all-valid when malformed or all-zero.
- `r_potential` exists in the reward schema but is currently always `0.0`.
- HUD probing hooks remain in `run_rl.py`, but the current reward shaper no longer performs HUD masking.

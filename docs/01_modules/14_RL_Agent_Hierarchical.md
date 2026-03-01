rl_agent (Hierarchical RL)

1. Purpose

This document defines the authoritative contract for running, training, evaluating, and checkpointing the hierarchical RL agent.

When --agent rl_agent is selected:

The RL stack is the sole decision-maker for actions.

Swarm/Planner/Proposer modules are not used for action selection.

FP_Analyst and TransitionEvent remain valid infrastructure components.

Memory is optional and read-only unless explicitly enabled.

This document supersedes swarm decision logic for RL runs.

2. Authoritative Action Source

For rl_agent:

At each step:

Observation
→ Observation_Encoder
→ Recurrent_Memory
→ Hierarchical_Controller (mode_t)
→ Policy_Actor (action_t)
→ Environment.step(action_t)

No Planner, Proposer, Hypothesis Engine, or Explorer may override action_t.

3. Required Artifacts per Run

All RL runs must produce:

outdir/rl/<run_id>/
    configs/
        resolved_config.json
    checkpoints/
        latest_XXXX.ckpt
        best_exploration.ckpt (optional)
        best_win.ckpt (optional)
    metrics/
        train_iter_XXXX.json
        eval_iter_XXXX.json
        summary.json
    traces/
        trace_train_*.jsonl
        trace_eval_*.jsonl
    trajectories/
        batch_XXXX.jsonl
    seeds.jsonl

No artifact → run is invalid.

4. Training Modes
4.1 collect

No weight updates.

Stochastic policy.

Writes trajectories + metrics.

Used for dataset generation or debugging.

4.2 train

Iterative loop:

collect → train_step → checkpoint → optional eval

All NN modules updated jointly.

Deterministic eval every eval_every_iters.

4.3 eval

Deterministic policy:

mode = argmax

action = argmax

No parameter updates.

Writes metrics and traces.

5. Determinism Rules

Eval must be fully deterministic:

Fixed seeds derived from:

global_seed, eval_iter, episode_idx

No stochastic sampling.

Fixed action masking and mode biases.

Training:

Stochastic sampling allowed.

Seeds logged in seeds.jsonl.

6. Metrics Contract

All runs must compute:

Primary Metrics

WinRate

MeanReturn

MedianReturn

Transition Metrics

EffectRate

NoveltyRate

LoopRate

Controller Metrics

ModeUsage[probe]

ModeUsage[exploit]

ModeUsage[escape_loop]

ModeUsage[focus_click]

ModeEntropy

Policy Metrics

PolicyEntropy

CoordSelectRate

ActionUsage[*]

Optimization Metrics (train only)

PolicyLoss_actor

PolicyLoss_controller

ValueLoss

GradNorm_total

PPO: ApproxKL, ClipFrac (if enabled)

Metrics must be stored per-iteration in JSON format.

7. Checkpoint Selection Policy
Phase A (Exploration phase)

Active until EvalWinRate >= win_switch_threshold for K consecutive evals.

Best checkpoint:

Maximize EvalMeanReturn

Tie-break:

higher EvalEffectRate

lower EvalLoopRate

earlier iteration

Saved as:

best_exploration.ckpt
Phase B (Win phase)

Active after threshold condition met.

Best checkpoint:

Maximize EvalWinRate

Tie-break:

higher EvalMeanReturn

lower EvalMeanEpisodeLen

lower EvalLoopRate

Saved as:

best_win.ckpt

Never use training metrics for best selection.

8. Parallelism Contract

Rollout workers may be parallel.

Training update must be single synchronized update step.

Metrics aggregated across all workers before logging.

9. Model Update Scope

During train:

Updated every iteration:

Observation_Encoder

Recurrent_Memory

Hierarchical_Controller

Policy_Actor

Value_Head

Never updated:

CoordProposer

Reward_Shaper

FP_Analyst

TransitionEvent compiler

10. Evaluation Sets

Must define:

train_games

eval_holdout_games

Holdout set:

Never used in training.

Fixed across runs for regression comparison.

Eval runs must output:

metrics_eval.json
per_episode_eval.jsonl
11. Failure Conditions

Run is invalid if:

No checkpoints written.

Any NaNs in loss.

Missing required metrics.

Deterministic eval produces different results on identical seed.

Action selection overridden by non-RL module.

12. Swarm / Planner Interaction

When rl_agent is active:

Planner, Proposer, Hypothesis Engine are bypassed.

Swarm orchestrator may still exist for logging or benchmarking, but must not influence action selection.

Hybrid mode requires separate contract.

13. Success Definition

An RL run is considered valid and progressing if:

EvalWinRate increases over time on held-out set.

EvalLoopRate decreases during exploration phase.

ModeUsage stabilizes (probe early, exploit later).

No collapse to single mode.

No entropy collapse before wins appear.

This document defines the authoritative behavior of RL runs in the project.

# Training Metrics Schema

## Always logged
- `losses.total`
- `losses.actor_policy`
- `losses.controller_policy`
- `losses.value`
- `losses.entropy_actor`
- `losses.entropy_controller`
- `losses.aux_mode_ce`
- `losses.grad_norm_total`
- `mean_return`
- `mean_episode_len`
- `win_rate`

## PPO-only metrics (`algo=ppo`)
- `losses.approx_kl`
- `losses.clipfrac_mode`
- `losses.clipfrac_action`
- `losses.clipfrac_coord`
- `losses.ppo_epochs_ran`
- `losses.adv_mean`
- `losses.adv_std`

## W&B keys (when enabled)
- `train/approx_kl`
- `train/clipfrac_mode`
- `train/clipfrac_action`
- `train/clipfrac_coord`
- `train/ppo_epochs_ran`
- `train/adv_mean`
- `train/adv_std`

These PPO metrics are for optimization-health monitoring and early-stopping diagnostics.

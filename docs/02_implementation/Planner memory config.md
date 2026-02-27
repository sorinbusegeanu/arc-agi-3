PLanner config
memory_weights:
  enabled: true

  # Global action efficacy (from memory.action_stats[action_key])
  action_effect_bonus: 1.0        # bonus for high effect
  action_noop_penalty: 1.5        # penalty for high no-op rate
  action_diversity_bonus: 0.5     # bonus for under-tried actions

  # Per-state action memory (from memory.state_action_stats[(state_hash, action_key)])
  state_noop_penalty: 2.0         # penalize repeating no-ops in same state
  repeat_self_loop_penalty: 3.0   # penalize repeated self-loop transitions

  # Coord priors (if coord actions exist)
  coord_effect_bonus: 1.0
  coord_noop_penalty: 2.0

  # Rule/template memory (optional)
  template_success_bonus: 0.5
  template_failure_penalty: 0.5

  # Safety clamp on total memory adjustment applied to base_score
  max_memory_adjustment_abs: 5.0

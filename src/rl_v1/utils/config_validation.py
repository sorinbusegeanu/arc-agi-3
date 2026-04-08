from __future__ import annotations

from rl_v1.configs.schema import V1Config


def validate_v1_config(
    cfg: V1Config,
    *,
    mode: str | None = None,
    require_checkpoint: bool = False,
) -> None:
    if cfg.env.execution_mode not in {"sequential_multi_env", "parallel_workers"}:
        raise ValueError("cfg.env.execution_mode must be one of: sequential_multi_env, parallel_workers")
    if cfg.env.mp_start_method not in {"spawn", "fork", "forkserver"}:
        raise ValueError("cfg.env.mp_start_method must be one of: spawn, fork, forkserver")
    if not isinstance(cfg.env.game_episode_multipliers, dict):
        raise ValueError("cfg.env.game_episode_multipliers must be a dict")
    game_id_set = {str(game_id) for game_id in cfg.env.game_ids}
    for key, value in cfg.env.game_episode_multipliers.items():
        if not isinstance(key, str) or not key:
            raise ValueError("cfg.env.game_episode_multipliers keys must be non-empty strings")
        if key not in game_id_set:
            raise ValueError("cfg.env.game_episode_multipliers keys must be in cfg.env.game_ids")
        if not isinstance(value, int):
            raise ValueError("cfg.env.game_episode_multipliers values must be integers")
        if value < 1:
            raise ValueError("cfg.env.game_episode_multipliers values must be >= 1")
    if not isinstance(cfg.reward.zero_steps_penalty, int) or cfg.reward.zero_steps_penalty < 0:
        raise ValueError("cfg.reward.zero_steps_penalty must be an integer >= 0")
    if cfg.reward.step_penalty > 0.0:
        raise ValueError("cfg.reward.step_penalty must be <= 0.0")
    if cfg.reward.level_complete_bonus < 0.0:
        raise ValueError("cfg.reward.level_complete_bonus must be >= 0.0")
    if cfg.reward.game_win_bonus < 0.0:
        raise ValueError("cfg.reward.game_win_bonus must be >= 0.0")
    if not isinstance(cfg.reward.repeat_state_penalty_enabled, bool):
        raise ValueError("cfg.reward.repeat_state_penalty_enabled must be a boolean")
    if not isinstance(cfg.reward.repeat_state_penalty, (int, float)):
        raise ValueError("cfg.reward.repeat_state_penalty must be a numeric value")
    if cfg.reward.repeat_state_penalty > 0.0:
        raise ValueError("cfg.reward.repeat_state_penalty must be <= 0.0")
    if not isinstance(cfg.reward.revisit_window_size, int) or cfg.reward.revisit_window_size < 1:
        raise ValueError("cfg.reward.revisit_window_size must be an integer >= 1")
    if cfg.reward.revisit_penalty_mode not in {"binary", "count", "decay"}:
        raise ValueError("cfg.reward.revisit_penalty_mode must be one of: binary, count, decay")
    if not isinstance(cfg.reward.revisit_penalty_decay, (int, float)) or cfg.reward.revisit_penalty_decay <= 0.0:
        raise ValueError("cfg.reward.revisit_penalty_decay must be > 0.0")
    if not isinstance(cfg.reward.same_action_streak_threshold, int) or cfg.reward.same_action_streak_threshold < 1:
        raise ValueError("cfg.reward.same_action_streak_threshold must be an integer >= 1")
    if not isinstance(cfg.reward.same_action_streak_penalty, (int, float)):
        raise ValueError("cfg.reward.same_action_streak_penalty must be a numeric value")
    if cfg.reward.same_action_streak_penalty > 0.0:
        raise ValueError("cfg.reward.same_action_streak_penalty must be <= 0.0")
    if cfg.acting.mode not in {"policy_only", "planner_eval_only", "planner_act"}:
        raise ValueError("cfg.acting.mode must be one of: policy_only, planner_eval_only, planner_act")
    if not isinstance(cfg.planner.allow_click_action_in_planner, bool):
        raise ValueError("cfg.planner.allow_click_action_in_planner must be a boolean")
    if cfg.model.variant not in {"v1_full", "recurrent_baseline"}:
        raise ValueError("cfg.model.variant must be one of: v1_full, recurrent_baseline")
    if not isinstance(cfg.wandb.enabled, bool):
        raise ValueError("cfg.wandb.enabled must be a boolean")
    if not isinstance(cfg.wandb.project, str) or not cfg.wandb.project.strip():
        raise ValueError("cfg.wandb.project must be a non-empty string")
    if cfg.wandb.entity is not None and not isinstance(cfg.wandb.entity, str):
        raise ValueError("cfg.wandb.entity must be null or a string")
    if cfg.wandb.run_name is not None and not isinstance(cfg.wandb.run_name, str):
        raise ValueError("cfg.wandb.run_name must be null or a string")
    if not isinstance(cfg.wandb.tags, list):
        raise ValueError("cfg.wandb.tags must be a list")
    if cfg.wandb.mode not in {"online", "offline", "disabled"}:
        raise ValueError("cfg.wandb.mode must be one of: online, offline, disabled")
    if not isinstance(cfg.wandb.log_every_updates, int) or cfg.wandb.log_every_updates < 1:
        raise ValueError("cfg.wandb.log_every_updates must be an integer >= 1")
    if cfg.runtime.accelerator not in {"auto", "cpu", "cuda", "gpu"}:
        raise ValueError("cfg.runtime.accelerator must be one of: auto, cpu, cuda, gpu")
    if str(getattr(cfg.runtime, "inference_device", "gpu")).lower() not in {"cpu", "gpu", "cuda"}:
        raise ValueError("cfg.runtime.inference_device must be one of: cpu, gpu, cuda")
    if not isinstance(cfg.runtime.devices, int) or cfg.runtime.devices < 1:
        raise ValueError("cfg.runtime.devices must be an integer >= 1")
    if not isinstance(cfg.runtime.precision, str) or not cfg.runtime.precision:
        raise ValueError("cfg.runtime.precision must be a non-empty string")
    if not isinstance(cfg.runtime.rollout_processes, int) or cfg.runtime.rollout_processes < 1:
        raise ValueError("cfg.runtime.rollout_processes must be an integer >= 1")
    if cfg.env.execution_mode == "parallel_workers" and cfg.runtime.rollout_processes < 1:
        raise ValueError("cfg.runtime.rollout_processes must be >= 1 when cfg.env.execution_mode=parallel_workers")
    if not isinstance(cfg.runtime.training_seed, int):
        raise ValueError("cfg.runtime.training_seed must be an integer")
    if not isinstance(cfg.runtime.evaluation_seed, int):
        raise ValueError("cfg.runtime.evaluation_seed must be an integer")
    if not isinstance(cfg.runtime.world_pretrain_seed, int):
        raise ValueError("cfg.runtime.world_pretrain_seed must be an integer")
    if not isinstance(cfg.runtime.deterministic_torch, bool):
        raise ValueError("cfg.runtime.deterministic_torch must be boolean")
    if not isinstance(cfg.runtime.cudnn_deterministic, bool):
        raise ValueError("cfg.runtime.cudnn_deterministic must be boolean")
    if not isinstance(cfg.runtime.cudnn_benchmark, bool):
        raise ValueError("cfg.runtime.cudnn_benchmark must be boolean")
    if not isinstance(cfg.model.use_world_model_pretraining, bool):
        raise ValueError("cfg.model.use_world_model_pretraining must be boolean")
    if not isinstance(cfg.model.predict_next_frame, bool):
        raise ValueError("cfg.model.predict_next_frame must be boolean")
    if not isinstance(cfg.model.predict_change_mask, bool):
        raise ValueError("cfg.model.predict_change_mask must be boolean")
    if not isinstance(cfg.model.predict_reward, bool):
        raise ValueError("cfg.model.predict_reward must be boolean")
    if not isinstance(cfg.model.predict_done, bool):
        raise ValueError("cfg.model.predict_done must be boolean")
    for name, value in {
        "model.metadata_embed_dim": cfg.model.metadata_embed_dim,
        "model.action_condition_dim": cfg.model.action_condition_dim,
        "model.step_count_embed_dim": cfg.model.step_count_embed_dim,
        "model.max_step_count": cfg.model.max_step_count,
        "world_pretrain.updates": cfg.world_pretrain.updates,
        "world_pretrain.batch_size": cfg.world_pretrain.batch_size,
        "world_pretrain.unroll_length": cfg.world_pretrain.unroll_length,
        "world_pretrain.learning_rate": cfg.world_pretrain.learning_rate,
        "world_pretrain.weight_decay": cfg.world_pretrain.weight_decay,
        "world_pretrain.grad_clip_norm": cfg.world_pretrain.grad_clip_norm,
        "world_pretrain.eval_every_updates": cfg.world_pretrain.eval_every_updates,
        "world_pretrain.save_every_updates": cfg.world_pretrain.save_every_updates,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    if cfg.loss_weights.change_mask_coef < 0:
        raise ValueError("loss_weights.change_mask_coef must be >= 0")
    if cfg.loss_weights.next_frame_coef < 0:
        raise ValueError("loss_weights.next_frame_coef must be >= 0")
    for name, value in {
        "model.encoder_dim": cfg.model.encoder_dim,
        "model.num_slots": cfg.model.num_slots,
        "model.slot_dim": cfg.model.slot_dim,
        "model.slot_iters": cfg.model.slot_iters,
        "model.slot_transformer_layers": cfg.model.slot_transformer_layers,
        "model.slot_transformer_heads": cfg.model.slot_transformer_heads,
        "model.gru_hidden_size": cfg.model.gru_hidden_size,
        "model.action_embed_dim": cfg.model.action_embed_dim,
        "model.latent_dim": cfg.model.latent_dim,
        "rollout.unroll_length": cfg.rollout.unroll_length,
        "rollout.num_episodes_per_collect": cfg.rollout.num_episodes_per_collect,
        "rollout.max_steps_per_level": cfg.rollout.max_steps_per_level,
        "optimization.learning_rate": cfg.optimization.learning_rate,
        "optimization.grad_clip_norm": cfg.optimization.grad_clip_norm,
        "optimization.batch_size": cfg.optimization.batch_size,
        "optimization.ppo_epochs": cfg.optimization.ppo_epochs,
        "optimization.gamma": cfg.optimization.gamma,
        "optimization.gae_lambda": cfg.optimization.gae_lambda,
        "optimization.clip_eps": cfg.optimization.clip_eps,
        "optimization.entropy_coef": cfg.optimization.entropy_coef,
        "optimization.value_coef": cfg.optimization.value_coef,
        "loss_weights.transition_coef": cfg.loss_weights.transition_coef,
        "loss_weights.reward_coef": cfg.loss_weights.reward_coef,
        "loss_weights.done_coef": cfg.loss_weights.done_coef,
        "logging.log_every_updates": cfg.logging.log_every_updates,
        "logging.eval_every_updates": cfg.logging.eval_every_updates,
        "checkpoint.save_every_updates": cfg.checkpoint.save_every_updates,
        "checkpoint.keep_last_n": cfg.checkpoint.keep_last_n,
        "planner.beam_width": cfg.planner.beam_width,
        "planner.search_depth": cfg.planner.search_depth,
        "planner.action_topk": cfg.planner.action_topk,
        "planner.discount": cfg.planner.discount,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    if mode is not None:
        normalized = str(mode)
        if normalized not in {"pretrain_world", "train_rl", "eval_policy", "train", "eval"}:
            raise ValueError(f"unsupported mode for validation: {mode}")
        if normalized in {"pretrain_world"} and not hasattr(cfg, "world_pretrain"):
            raise ValueError("world-pretrain mode requires cfg.world_pretrain section")
        if normalized in {"train_rl", "train"} and cfg.model.variant not in {"v1_full", "recurrent_baseline"}:
            raise ValueError("RL mode requires a gameplay model variant")
        if normalized in {"eval_policy", "eval"} and not hasattr(cfg, "evaluation"):
            raise ValueError("eval mode requires cfg.evaluation section")
    if require_checkpoint and not cfg.checkpoint.restore_path:
        raise ValueError("checkpoint restore path is required but missing")

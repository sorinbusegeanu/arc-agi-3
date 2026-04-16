from __future__ import annotations

from rl_v1.metrics.metric_keys import GAME_WIN_RATE, LEVEL_COMPLETION_RATE, MEAN_LEVELS_REACHED, MEAN_STEPS_PER_COMPLETED_LEVEL, TRAINING_LOSS


def with_metric_fields(summary: dict, metrics: dict) -> dict:
    output = dict(summary)
    for key in (
        GAME_WIN_RATE,
        LEVEL_COMPLETION_RATE,
        MEAN_LEVELS_REACHED,
        MEAN_STEPS_PER_COMPLETED_LEVEL,
        TRAINING_LOSS,
        "world_total_loss",
        "change_mask_loss",
        "next_frame_loss",
        "reward_prediction_loss",
        "done_prediction_loss",
        "transition_loss",
        "change_mask_precision",
        "change_mask_recall",
        "change_mask_f1",
        "reward_prediction_mae",
        "done_accuracy",
        "diagnostic_game_win_rate",
        "diagnostic_level_completion_rate",
        "diagnostic_mean_levels_reached",
        "diagnostic_mean_steps_per_completed_level",
    ):
        if key in metrics:
            output[key] = metrics[key]
    return output


def build_run_summary(cfg, metrics: dict, *, mode: str | None = None) -> dict:
    resolved_mode = str(mode or metrics.get("mode") or "train_rl")
    summary = with_metric_fields({}, metrics)
    summary["training_mode"] = resolved_mode
    summary["model_variant"] = cfg.model.variant
    summary["acting_mode"] = cfg.acting.mode
    summary["env_num_workers"] = cfg.runtime.rollout_processes
    summary["env_execution_mode"] = cfg.env.execution_mode
    summary["runtime_rollout_processes"] = cfg.runtime.rollout_processes
    summary["worker_inference_device"] = _resolve_worker_inference_device(cfg)
    summary["env_game_episode_multipliers"] = dict(cfg.env.game_episode_multipliers)
    summary["reward_zero_steps_penalty"] = cfg.reward.zero_steps_penalty
    summary["reward_step_penalty"] = cfg.reward.step_penalty
    summary["reward_level_complete_bonus"] = cfg.reward.level_complete_bonus
    summary["reward_game_win_bonus"] = cfg.reward.game_win_bonus
    summary["planner_enabled"] = cfg.planner.enabled
    summary["planner_beam_width"] = cfg.planner.beam_width
    summary["planner_search_depth"] = cfg.planner.search_depth
    summary["planner_action_topk"] = cfg.planner.action_topk
    summary["planner_discount"] = cfg.planner.discount
    summary["disable_slots"] = cfg.ablations.disable_slots
    summary["disable_recurrent_memory"] = cfg.ablations.disable_recurrent_memory
    summary["disable_planner"] = cfg.ablations.disable_planner
    summary["disable_transition_loss"] = cfg.ablations.disable_transition_loss
    summary["disable_click_head"] = cfg.ablations.disable_click_head
    summary["wandb_enabled"] = cfg.wandb.enabled
    summary["wandb_project"] = cfg.wandb.project
    summary["wandb_entity"] = cfg.wandb.entity
    summary["wandb_run_name"] = cfg.wandb.run_name
    summary["wandb_mode"] = cfg.wandb.mode
    summary["mode"] = metrics.get("mode", resolved_mode)
    summary["checkpoint_restore_path"] = cfg.checkpoint.restore_path
    summary["effective_game_ids"] = list(cfg.env.game_ids)
    summary["evaluation_episodes"] = cfg.evaluation.episodes
    summary["evaluation_deterministic"] = cfg.evaluation.deterministic
    summary["compare_policy_vs_configured"] = bool(getattr(cfg.evaluation, "compare_policy_vs_configured", False))
    summary["training_seed"] = cfg.runtime.training_seed
    summary["evaluation_seed"] = cfg.runtime.evaluation_seed
    summary["world_pretrain_seed"] = cfg.runtime.world_pretrain_seed
    if resolved_mode == "pretrain_world":
        for gameplay_key in (
            GAME_WIN_RATE,
            LEVEL_COMPLETION_RATE,
            MEAN_LEVELS_REACHED,
            MEAN_STEPS_PER_COMPLETED_LEVEL,
        ):
            if gameplay_key in summary and f"diagnostic_{gameplay_key}" not in summary:
                summary[f"diagnostic_{gameplay_key}"] = summary.pop(gameplay_key)
    return summary


def _resolve_worker_inference_device(cfg) -> str:
    inference_device = str(getattr(cfg.runtime, "inference_device", "")).lower()
    if inference_device in {"gpu", "cuda"}:
        return "cuda"
    if inference_device == "cpu":
        return "cpu"
    accelerator = str(getattr(cfg.runtime, "accelerator", "auto")).lower()
    if accelerator == "gpu":
        accelerator = "cuda"
    if accelerator in {"auto", "cuda"}:
        return "cuda"
    return "cpu"

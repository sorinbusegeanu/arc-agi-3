from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from rl_v1.configs.load import load_config, normalize_game_episode_multipliers
from rl_v1.eval.eval_main import eval_main
from rl_v1.model.model_factory import build_model
from rl_v1.training.trainer import Trainer
from rl_v1.utils.artifact_writer import ArtifactWriter
from rl_v1.utils.checkpoint_manager import CheckpointManager
from rl_v1.utils.config_validation import validate_v1_config
from rl_v1.utils.runtime import set_global_seeds
from rl_v1.utils.wandb_logger import WandbLogger


def train_main(
    config_path: str | None = None,
    preset: str = "v2",
    mode: str = "train_rl",
    updates: int = 1,
    game_ids: list[str] | None = None,
    num_workers: int | None = None,
    rollout_processes: int | None = None,
    accelerator: str | None = None,
    devices: int | None = None,
    execution_mode: str | None = None,
    render_mode: str | None = None,
    video: bool = False,
    debug_log_path: str | None = None,
    checkpoint_path: str | None = None,
    eval_kind: str | None = None,
    per_game: bool = False,
    world_metrics_only: bool = False,
    eval_episodes: int | None = None,
    acting_mode_override: str | None = None,
    deterministic_eval_override: bool | None = None,
    compare_policy_vs_configured_override: bool | None = None,
    world_updates: int | None = None,
    world_batch_size: int | None = None,
    world_unroll_length: int | None = None,
    world_eval_every: int | None = None,
    world_save_every: int | None = None,
    freeze_encoder: bool = False,
    freeze_recurrent: bool = False,
    init_from_checkpoint: str | None = None,
    log_gameplay_metrics_during_pretrain: bool = False,
    dry_run: bool = False,
    smoke_test: bool = False,
):
    cfg = _load_and_apply_overrides(
        config_path=config_path,
        preset=preset,
        game_ids=game_ids,
        num_workers=num_workers,
        rollout_processes=rollout_processes,
        accelerator=accelerator,
        devices=devices,
        execution_mode=execution_mode,
        render_mode=render_mode,
        video=video,
        debug_log_path=debug_log_path,
        checkpoint_path=checkpoint_path,
        eval_episodes=eval_episodes,
        acting_mode_override=acting_mode_override,
        deterministic_eval_override=deterministic_eval_override,
        compare_policy_vs_configured_override=compare_policy_vs_configured_override,
        world_updates=world_updates,
        world_batch_size=world_batch_size,
        world_unroll_length=world_unroll_length,
        world_eval_every=world_eval_every,
        world_save_every=world_save_every,
        freeze_encoder=freeze_encoder,
        freeze_recurrent=freeze_recurrent,
        init_from_checkpoint=init_from_checkpoint,
        log_gameplay_metrics_during_pretrain=log_gameplay_metrics_during_pretrain,
    )
    cfg.logging.output_dir = "runs_rl_v1"
    cfg.logging.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    validate_v1_config(cfg, mode=mode, require_checkpoint=False)
    run_root = Path(cfg.logging.output_dir) / cfg.logging.run_name
    artifacts = ArtifactWriter(run_root / "artifacts")
    artifacts.write_effective_config(
        {
            "mode": mode,
            "checkpoint_restore_path": cfg.checkpoint.restore_path,
            "config": cfg.to_dict(),
        }
    )
    preflight = _preflight_validate(cfg, mode=mode)
    artifacts.write_preflight_report(preflight)
    _print_resolved_line(cfg, mode=mode)
    model = build_model(cfg)
    if dry_run:
        return {"status": "dry_run_ok", "mode": mode, "preflight": preflight}
    if smoke_test:
        smoke = _run_smoke_test(cfg, model, mode=mode)
        artifacts.write_smoke_test_report(smoke)
        return {"status": "smoke_test_ok", "mode": mode, "smoke": smoke}
    if mode == "pretrain_world":
        return run_world_pretrain(
            cfg,
            model,
            artifacts=artifacts,
            log_gameplay_metrics_during_pretrain=log_gameplay_metrics_during_pretrain,
        )
    if mode == "train_rl":
        return run_rl_train(cfg, model, updates=updates, artifacts=artifacts)
    if mode == "eval_policy":
        return run_eval(
            config_path=config_path,
            preset=preset,
            game_ids=game_ids,
            num_workers=num_workers,
            rollout_processes=rollout_processes,
            accelerator=accelerator,
            devices=devices,
            execution_mode=execution_mode,
            render_mode=render_mode,
            video=video,
            debug_log_path=debug_log_path,
            checkpoint_path=cfg.checkpoint.restore_path,
            eval_kind=eval_kind,
            per_game=per_game,
            world_metrics_only=world_metrics_only,
            eval_episodes=eval_episodes,
            acting_mode_override=acting_mode_override,
            deterministic_eval_override=deterministic_eval_override,
            compare_policy_vs_configured_override=compare_policy_vs_configured_override,
            dry_run=dry_run,
            smoke_test=smoke_test,
        )
    raise ValueError(f"unsupported mode: {mode}")


def run_world_pretrain(cfg, model, *, artifacts: ArtifactWriter, log_gameplay_metrics_during_pretrain: bool = False):
    set_global_seeds(
        cfg.runtime.world_pretrain_seed,
        deterministic_torch=cfg.runtime.deterministic_torch,
        cudnn_deterministic=cfg.runtime.cudnn_deterministic,
        cudnn_benchmark=cfg.runtime.cudnn_benchmark,
    )
    if cfg.world_pretrain.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    if cfg.world_pretrain.freeze_recurrent:
        for parameter in model.recurrent.parameters():
            parameter.requires_grad = False
    cfg.optimization.learning_rate = cfg.world_pretrain.learning_rate
    cfg.optimization.weight_decay = cfg.world_pretrain.weight_decay
    cfg.optimization.grad_clip_norm = cfg.world_pretrain.grad_clip_norm
    cfg.optimization.batch_size = cfg.world_pretrain.batch_size
    cfg.rollout.unroll_length = cfg.world_pretrain.unroll_length
    wandb_logger = WandbLogger(cfg)
    wandb_logger.log_config(cfg.to_dict())
    print(
        "pretrain metrics: world_total_loss transition_loss change_mask_loss "
        "change_mask_f1 reward_prediction_loss done_prediction_loss "
        + ("next_frame_loss" if bool(cfg.model.predict_next_frame) else "")
    )
    trainer = Trainer(
        cfg,
        model,
        wandb_logger=wandb_logger,
        training_mode="pretrain_world",
        log_gameplay_metrics_during_pretrain=log_gameplay_metrics_during_pretrain,
    )
    try:
        summary = trainer.train(updates=int(cfg.world_pretrain.updates), mode="world_pretrain")
        artifacts.write_world_pretrain_summary(build_world_pretrain_summary(cfg, summary))
        _print_final_world_summary(summary)
    finally:
        wandb_logger.finish()
    return {"mode": "pretrain_world", "summary": summary}


def run_rl_train(cfg, model, *, updates: int, artifacts: ArtifactWriter):
    set_global_seeds(
        cfg.runtime.training_seed,
        deterministic_torch=cfg.runtime.deterministic_torch,
        cudnn_deterministic=cfg.runtime.cudnn_deterministic,
        cudnn_benchmark=cfg.runtime.cudnn_benchmark,
    )
    if cfg.world_pretrain.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    if cfg.world_pretrain.freeze_recurrent:
        for parameter in model.recurrent.parameters():
            parameter.requires_grad = False
    wandb_logger = WandbLogger(cfg)
    wandb_logger.log_config(cfg.to_dict())
    trainer = Trainer(cfg, model, wandb_logger=wandb_logger, training_mode="train_rl")
    resume_state = "fresh"
    try:
        if cfg.checkpoint.restore_path:
            metadata = CheckpointManager().read_metadata(cfg.checkpoint.restore_path)
            if metadata.get("training_mode") == "train_rl":
                resume_state = "resumed"
            else:
                resume_state = "warm_started"
        summary = trainer.train(updates=updates, mode="train_rl")
    finally:
        wandb_logger.finish()
    return {"mode": "train_rl", "resume_state": resume_state, "summary": summary}


def run_eval(**kwargs):
    return eval_main(**kwargs)


def _load_and_apply_overrides(
    *,
    config_path,
    preset,
    game_ids,
    num_workers,
    rollout_processes,
    accelerator,
    devices,
    execution_mode,
    render_mode,
    video,
    debug_log_path,
    checkpoint_path,
    eval_episodes,
    acting_mode_override,
    deterministic_eval_override,
    compare_policy_vs_configured_override,
    world_updates,
    world_batch_size,
    world_unroll_length,
    world_eval_every,
    world_save_every,
    freeze_encoder,
    freeze_recurrent,
    init_from_checkpoint,
    log_gameplay_metrics_during_pretrain,
):
    load_kwargs = {"preset": preset}
    if rollout_processes is not None:
        load_kwargs["runtime_rollout_processes"] = rollout_processes
    if accelerator is not None:
        load_kwargs["runtime_accelerator"] = accelerator
    if devices is not None:
        load_kwargs["runtime_devices"] = devices
    if execution_mode is not None:
        load_kwargs["env_execution_mode"] = execution_mode
    cfg = load_config(config_path, **load_kwargs)
    if game_ids is not None:
        cfg.env.game_ids = list(game_ids)
    if num_workers is not None:
        cfg.runtime.rollout_processes = int(num_workers)
    if render_mode is not None:
        cfg.env.render_mode = render_mode
    if video:
        cfg.env.save_recording = True
    if debug_log_path is not None:
        cfg.env.debug_log_path = str(debug_log_path)
    if checkpoint_path is not None:
        cfg.checkpoint.restore_path = checkpoint_path
    if init_from_checkpoint is not None:
        cfg.checkpoint.restore_path = str(init_from_checkpoint)
    if eval_episodes is not None:
        cfg.evaluation.episodes = int(eval_episodes)
    if acting_mode_override is not None:
        cfg.acting.mode = str(acting_mode_override)
    if deterministic_eval_override is not None:
        cfg.evaluation.deterministic = bool(deterministic_eval_override)
    if compare_policy_vs_configured_override is not None:
        cfg.evaluation.compare_policy_vs_configured = bool(compare_policy_vs_configured_override)
    if world_updates is not None:
        cfg.world_pretrain.updates = int(world_updates)
    if world_batch_size is not None:
        cfg.world_pretrain.batch_size = int(world_batch_size)
    if world_unroll_length is not None:
        cfg.world_pretrain.unroll_length = int(world_unroll_length)
    if world_eval_every is not None:
        cfg.world_pretrain.eval_every_updates = int(world_eval_every)
    if world_save_every is not None:
        cfg.world_pretrain.save_every_updates = int(world_save_every)
    if freeze_encoder:
        cfg.world_pretrain.freeze_encoder = True
    if freeze_recurrent:
        cfg.world_pretrain.freeze_recurrent = True
    normalize_game_episode_multipliers(cfg)
    return cfg


def _preflight_validate(cfg, *, mode: str) -> dict:
    report = {
        "mode": mode,
        "checkpoint_restore_path": cfg.checkpoint.restore_path,
        "game_ids": list(cfg.env.game_ids),
        "ok": True,
    }
    if cfg.checkpoint.restore_path:
        checkpoint_path = Path(cfg.checkpoint.restore_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint restore path does not exist: {cfg.checkpoint.restore_path}")
        metadata = CheckpointManager().read_metadata(checkpoint_path)
        report["checkpoint_metadata"] = metadata
        if mode == "eval_policy":
            variant = metadata.get("model_variant")
            if variant is not None and variant != cfg.model.variant:
                raise ValueError(
                    "checkpoint/model variant mismatch: "
                    f"checkpoint={variant}, requested={cfg.model.variant}"
                )
    if mode == "train_rl":
        if not hasattr(build_model(cfg), "policy_head"):
            raise ValueError("RL mode requires policy/value heads")
    return report


def _run_smoke_test(cfg, model, *, mode: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    hidden = model.initial_hidden(1, device)
    finite = torch.isfinite(hidden).all().item()
    return {
        "mode": mode,
        "device": str(device),
        "hidden_shape": tuple(hidden.shape),
        "hidden_finite": bool(finite),
    }


def _print_resolved_line(cfg, *, mode: str) -> None:
    line = (
        f"mode={mode} acting={cfg.acting.mode} checkpoint={cfg.checkpoint.restore_path} "
        f"games={cfg.env.game_ids} rollout_processes={cfg.runtime.rollout_processes} "
        f"eval_episodes={cfg.evaluation.episodes} deterministic={cfg.evaluation.deterministic} "
        f"seed(train/eval/world)={cfg.runtime.training_seed}/{cfg.runtime.evaluation_seed}/{cfg.runtime.world_pretrain_seed}"
    )
    print(line)


def build_world_pretrain_summary(cfg, summary: dict) -> dict:
    from rl_v1.utils.run_summary import build_run_summary

    return build_run_summary(cfg, summary | {"mode": "pretrain_world"}, mode="pretrain_world")


def _print_final_world_summary(summary: dict) -> None:
    parts = []
    for key, label in (
        ("world_total_loss", "loss"),
        ("transition_loss", "tr"),
        ("change_mask_loss", "cm"),
        ("change_mask_f1", "cm_f1"),
        ("reward_prediction_loss", "rw"),
        ("done_prediction_loss", "done"),
        ("next_frame_loss", "nf"),
    ):
        if key in summary and isinstance(summary[key], (int, float)):
            parts.append(f"{label}={summary[key]:.4f}")
    print("world_final " + " ".join(parts))

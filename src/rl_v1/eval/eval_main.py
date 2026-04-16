from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rl_v1.configs.load import load_config, normalize_game_episode_multipliers
from rl_v1.eval.evaluator import Evaluator
from rl_v1.model.model_factory import build_model
from rl_v1.utils.artifact_writer import ArtifactWriter
from rl_v1.utils.checkpoint_manager import CheckpointManager
from rl_v1.utils.config_validation import validate_v1_config
from rl_v1.utils.runtime import set_global_seeds
from rl_v1.utils.wandb_logger import WandbLogger


def eval_main(
    config_path: str | None = None,
    preset: str = "v2",
    mode: str = "eval_policy",
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
    dry_run: bool = False,
    smoke_test: bool = False,
    **_unused,
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
    cfg.logging.output_dir = "runs_rl_v1"
    cfg.logging.run_name = datetime.now().strftime("%Y%m%d_%H%M%S_eval")
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
    if eval_episodes is not None:
        cfg.evaluation.episodes = int(eval_episodes)
    if acting_mode_override is not None:
        cfg.acting.mode = str(acting_mode_override)
    if deterministic_eval_override is not None:
        cfg.evaluation.deterministic = bool(deterministic_eval_override)
    if compare_policy_vs_configured_override is not None:
        cfg.evaluation.compare_policy_vs_configured = bool(compare_policy_vs_configured_override)
    normalize_game_episode_multipliers(cfg)
    validate_v1_config(cfg, mode=mode, require_checkpoint=False)
    run_root = Path(cfg.logging.output_dir) / cfg.logging.run_name
    artifacts = ArtifactWriter(run_root / "artifacts")
    artifacts.write_effective_config(
        {
            "mode": mode,
            "checkpoint_restore_path": cfg.checkpoint.restore_path,
            "eval_kind": eval_kind or "policy",
            "per_game": bool(per_game),
            "world_metrics_only": bool(world_metrics_only),
            "config": cfg.to_dict(),
        }
    )
    preflight = _preflight_eval(cfg)
    artifacts.write_preflight_report(preflight)
    _print_eval_resolved_line(cfg, mode=mode)
    if dry_run:
        return {"status": "dry_run_ok", "mode": mode, "preflight": preflight}
    set_global_seeds(
        cfg.runtime.evaluation_seed,
        deterministic_torch=cfg.runtime.deterministic_torch,
        cudnn_deterministic=cfg.runtime.cudnn_deterministic,
        cudnn_benchmark=cfg.runtime.cudnn_benchmark,
    )
    wandb_logger = WandbLogger(cfg)
    wandb_logger.log_config(cfg.to_dict())
    model = build_model(cfg)
    if cfg.checkpoint.restore_path:
        CheckpointManager().load(cfg.checkpoint.restore_path, model=model, cfg=cfg)
    evaluator = Evaluator(
        cfg,
        model,
        wandb_logger=wandb_logger,
        training_mode=mode,
        eval_kind=eval_kind or "policy",
    )
    try:
        if eval_kind == "world":
            summary = evaluator.evaluate_world_model(metrics_only=True, per_game=bool(per_game))
        else:
            summary = evaluator.evaluate_policy(per_game=bool(per_game))
        if smoke_test:
            artifacts.write_smoke_test_report({"mode": mode, "summary_keys": sorted(summary.keys())})
        return summary
    finally:
        wandb_logger.finish()


def _preflight_eval(cfg) -> dict:
    report = {
        "checkpoint_restore_path": cfg.checkpoint.restore_path,
        "acting_mode": cfg.acting.mode,
        "eval_episodes": cfg.evaluation.episodes,
        "deterministic": cfg.evaluation.deterministic,
    }
    if cfg.checkpoint.restore_path:
        path = Path(cfg.checkpoint.restore_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint restore path does not exist: {cfg.checkpoint.restore_path}")
        report["checkpoint_metadata"] = CheckpointManager().read_metadata(path)
    return report


def _print_eval_resolved_line(cfg, *, mode: str) -> None:
    print(
        f"mode={mode} acting={cfg.acting.mode} checkpoint={cfg.checkpoint.restore_path} "
        f"games={cfg.env.game_ids} rollout_processes={cfg.runtime.rollout_processes} "
        f"eval_episodes={cfg.evaluation.episodes} deterministic={cfg.evaluation.deterministic} "
        f"seed={cfg.runtime.evaluation_seed}"
    )

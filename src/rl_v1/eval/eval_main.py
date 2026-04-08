from __future__ import annotations

from rl_v1.configs.load import load_config, normalize_game_episode_multipliers
from rl_v1.eval.evaluator import Evaluator
from rl_v1.model.model_factory import build_model
from rl_v1.utils.checkpoint_manager import CheckpointManager
from rl_v1.utils.config_validation import validate_v1_config
from rl_v1.utils.wandb_logger import WandbLogger


def eval_main(
    config_path: str | None = None,
    preset: str = "v2",
    game_ids: list[str] | None = None,
    num_workers: int | None = None,
    rollout_processes: int | None = None,
    accelerator: str | None = None,
    devices: int | None = None,
    execution_mode: str | None = None,
    render_mode: str | None = None,
    checkpoint_path: str | None = None,
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
    cfg.wandb.enabled = False
    cfg.wandb.mode = "disabled"
    if game_ids is not None:
        cfg.env.game_ids = list(game_ids)
    if num_workers is not None:
        cfg.runtime.rollout_processes = int(num_workers)
    if render_mode is not None:
        cfg.env.render_mode = render_mode
    if checkpoint_path is not None:
        cfg.checkpoint.restore_path = checkpoint_path
    # CLI eval runs one episode by default unless config is explicitly changed elsewhere.
    cfg.evaluation.episodes = 1
    # Eval mode default: use planner in evaluation while keeping training policy-only semantics.
    cfg.acting.mode = "planner_eval_only"
    normalize_game_episode_multipliers(cfg)
    validate_v1_config(cfg)
    wandb_logger = WandbLogger(cfg)
    wandb_logger.log_config(cfg.to_dict())
    model = build_model(cfg)
    if cfg.checkpoint.restore_path:
        CheckpointManager().load(cfg.checkpoint.restore_path, model=model, cfg=cfg)
    try:
        return Evaluator(cfg, model, wandb_logger=wandb_logger).evaluate()
    finally:
        wandb_logger.finish()

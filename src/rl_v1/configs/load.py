from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rl_v1.configs.schema import (
    AblationsConfig,
    ActingConfig,
    CheckpointConfig,
    EnvConfig,
    EvalConfig,
    LoggingConfig,
    LossWeightsConfig,
    ModelConfig,
    OptimizationConfig,
    PlannerConfig,
    RewardConfig,
    RolloutConfig,
    RuntimeConfig,
    V1Config,
    WandbConfig,
    WorldPretrainConfig,
)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def _preset_path(name: str) -> Path:
    mapping = {
        "default_v1": "default.yaml",
        "debug_v1": "debug.yaml",
        "v2": "v2.yaml",
        "default_v2": "v2.yaml",
        "default": "default.yaml",
        "debug": "debug.yaml",
    }
    return _config_dir() / mapping.get(name, name)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config at {path} must be a mapping")
    return payload


def _from_dict(payload: dict[str, Any]) -> V1Config:
    rollout_payload = dict(payload.get("rollout", {}))
    if "max_steps_per_level" not in rollout_payload and "max_steps_per_episode" in rollout_payload:
        rollout_payload["max_steps_per_level"] = rollout_payload["max_steps_per_episode"]
    cfg = V1Config(
        env=EnvConfig(**payload.get("env", {})),
        reward=RewardConfig(**payload.get("reward", {})),
        acting=ActingConfig(**payload.get("acting", {})),
        planner=PlannerConfig(**payload.get("planner", {})),
        ablations=AblationsConfig(**payload.get("ablations", {})),
        model=ModelConfig(**payload.get("model", {})),
        rollout=RolloutConfig(**rollout_payload),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
        optimization=OptimizationConfig(**payload.get("optimization", {})),
        loss_weights=LossWeightsConfig(**payload.get("loss_weights", {})),
        logging=LoggingConfig(**payload.get("logging", {})),
        wandb=WandbConfig(**payload.get("wandb", {})),
        checkpoint=CheckpointConfig(**payload.get("checkpoint", {})),
        evaluation=EvalConfig(**payload.get("evaluation", {})),
        world_pretrain=WorldPretrainConfig(**payload.get("world_pretrain", {})),
    )
    cfg.model.slot_count = cfg.model.num_slots
    cfg.model.relation_layers = cfg.model.slot_transformer_layers
    cfg.model.relation_heads = cfg.model.slot_transformer_heads
    cfg.model.hidden_dim = cfg.model.latent_dim
    cfg.model.use_slots = not cfg.ablations.disable_slots
    cfg.model.use_recurrent_memory = not cfg.ablations.disable_recurrent_memory
    cfg.model.use_click_branch = not cfg.ablations.disable_click_head
    cfg.model.use_transition_model = True
    cfg.model.baseline = cfg.model.variant == "recurrent_baseline"
    cfg.rollout.episodes_per_collect = cfg.rollout.num_episodes_per_collect
    cfg.rollout.max_steps = cfg.rollout.max_steps_per_level
    cfg.optimization.lr = cfg.optimization.learning_rate
    cfg.optimization.max_grad_norm = cfg.optimization.grad_clip_norm
    normalize_game_episode_multipliers(cfg)
    return cfg


def load_config(
    path: str | None = None,
    preset: str = "v2",
    runtime_rollout_processes: int | None = None,
    runtime_accelerator: str | None = None,
    runtime_devices: int | None = None,
    env_execution_mode: str | None = None,
) -> V1Config:
    payload = _load_yaml(_preset_path(preset))
    if path is not None:
        payload = _merge(payload, _load_yaml(Path(path)))
    if runtime_rollout_processes is not None:
        payload = _merge(payload, {"runtime": {"rollout_processes": int(runtime_rollout_processes)}})
    if runtime_accelerator is not None:
        payload = _merge(payload, {"runtime": {"accelerator": str(runtime_accelerator)}})
    if runtime_devices is not None:
        payload = _merge(payload, {"runtime": {"devices": int(runtime_devices)}})
    if env_execution_mode is not None:
        payload = _merge(payload, {"env": {"execution_mode": str(env_execution_mode)}})
    return _from_dict(payload)


def normalize_game_episode_multipliers(cfg: V1Config) -> None:
    configured_games = [str(game_id) for game_id in cfg.env.game_ids]
    raw_multipliers = cfg.env.game_episode_multipliers or {}
    cfg.env.game_episode_multipliers = {
        game_id: raw_multipliers.get(game_id, 1)
        for game_id in configured_games
    }

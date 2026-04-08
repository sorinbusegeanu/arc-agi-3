from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    game_ids: list[str] = field(default_factory=lambda: ["ez01"])
    game_episode_multipliers: dict[str, int] = field(default_factory=dict)
    operation_mode: str = "offline"
    environments_dir: str | None = None
    seed: int = 0
    save_recording: bool = False
    render_mode: str | None = None
    debug_log_path: str | None = None
    execution_mode: str = "sequential_multi_env"
    mp_start_method: str = "spawn"


@dataclass
class RewardConfig:
    zero_steps_penalty: int = 10
    step_penalty: float = -0.01
    level_complete_bonus: float = 1.0
    game_win_bonus: float = 5.0
    repeat_state_penalty_enabled: bool = False
    repeat_state_penalty: float = 0.0
    revisit_window_size: int = 8
    revisit_penalty_mode: str = "binary"
    revisit_penalty_decay: float = 1.0
    same_action_streak_threshold: int = 4
    same_action_streak_penalty: float = -0.05


@dataclass
class ActingConfig:
    mode: str = "policy_only"


@dataclass
class PlannerConfig:
    enabled: bool = True
    beam_width: int = 8
    search_depth: int = 5
    action_topk: int = 6
    discount: float = 0.99
    allow_click_action_in_planner: bool = False


@dataclass
class AblationsConfig:
    disable_slots: bool = False
    disable_recurrent_memory: bool = False
    disable_planner: bool = False
    disable_transition_loss: bool = False
    disable_click_head: bool = False


@dataclass
class ModelConfig:
    variant: str = "v1_full"
    canvas_height: int = 64
    canvas_width: int = 64
    encoder_channels: tuple[int, int, int] = (32, 64, 128)
    encoder_dim: int = 256
    num_slots: int = 6
    slot_dim: int = 256
    slot_iters: int = 3
    slot_transformer_layers: int = 2
    slot_transformer_heads: int = 4
    gru_hidden_size: int = 256
    action_embed_dim: int = 64
    latent_dim: int = 256
    max_game_ids: int = 64
    game_embed_dim: int = 32
    max_level_index: int = 256
    use_world_model_pretraining: bool = True
    predict_next_frame: bool = False
    predict_change_mask: bool = True
    predict_reward: bool = True
    predict_done: bool = True
    metadata_embed_dim: int = 64
    action_condition_dim: int = 64
    step_count_embed_dim: int = 32
    max_step_count: int = 256
    pooling: str = "attention"
    # Compatibility aliases for already-implemented code paths.
    slot_count: int = 6
    relation_layers: int = 2
    relation_heads: int = 4
    hidden_dim: int = 256
    use_slots: bool = True
    use_recurrent_memory: bool = True
    use_click_branch: bool = True
    use_transition_model: bool = True
    baseline: bool = False


@dataclass
class RolloutConfig:
    unroll_length: int = 16
    num_episodes_per_collect: int = 2
    max_steps_per_level: int = 32
    deterministic_eval: bool = True
    # Compatibility aliases.
    episodes_per_collect: int = 2
    max_steps: int = 32


@dataclass
class OptimizationConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    batch_size: int = 4
    ppo_epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    precision: str = "32-true"
    # Compatibility aliases.
    lr: float = 3e-4
    max_grad_norm: float = 1.0


@dataclass
class RuntimeConfig:
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    rollout_processes: int = 1
    inference_device: str = "gpu"
    training_seed: int = 0
    evaluation_seed: int = 0
    world_pretrain_seed: int = 0
    deterministic_torch: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True


@dataclass
class LossWeightsConfig:
    transition_coef: float = 1.0
    change_mask_coef: float = 1.0
    next_frame_coef: float = 1.0
    reward_coef: float = 0.5
    done_coef: float = 0.25
    # Compatibility aliases.
    policy_weight: float = 1.0
    value_weight: float = 0.5
    entropy_weight: float = 0.01


@dataclass
class LoggingConfig:
    run_name: str = "rl_v1_v1"
    output_dir: str = "runs/rl_v1"
    planner_trace_sample_rate: float = 0.25
    log_every_updates: int = 1
    eval_every_updates: int = 1


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "arc_rl_v1"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    log_every_updates: int = 1


@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_every_updates: int = 10
    keep_last_n: int = 3
    restore_path: str | None = None


@dataclass
class EvalConfig:
    episodes: int = 2
    deterministic: bool = True
    compare_policy_vs_configured: bool = False


@dataclass
class WorldPretrainConfig:
    enabled: bool = False
    updates: int = 1000
    batch_size: int = 32
    unroll_length: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    eval_every_updates: int = 10
    save_every_updates: int = 50
    freeze_encoder: bool = False
    freeze_recurrent: bool = False
    teacher_forcing: bool = True


@dataclass
class V1Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    acting: ActingConfig = field(default_factory=ActingConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    ablations: AblationsConfig = field(default_factory=AblationsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    world_pretrain: WorldPretrainConfig = field(default_factory=WorldPretrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

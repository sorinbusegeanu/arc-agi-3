from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class FPAnalystConfig:
    connectivity: int = 4
    min_area: int = 1
    max_objects: int = 512
    enable_tracking: bool = True
    enable_symmetry: bool = True
    enable_periodicity: bool = True
    max_period: int = 16
    bg_detection_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "frequency": 0.5,
            "border": 0.3,
            "connectedness": 0.2,
        }
    )
    iou_threshold: float = 0.25
    centroid_distance_threshold: float = 10.0
    iou_soft_threshold: float = 0.10
    overlays: List[str] = field(
        default_factory=lambda: [
            "bbox_overlay",
            "component_id_overlay",
            "diff_mask",
            "object_motion_overlay",
        ]
    )
    save_images: bool = False
    output_dir: str = "runs"


@dataclass(frozen=True)
class RLConfig:
    pipeline: Dict[str, Any] = field(default_factory=lambda: {"mode": "rl_only"})
    modules: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": {
                "fp_analyst": True,
                "transition_event": True,
                "trace_writer": True,
                "rl_encoder": True,
                "rl_memory": True,
                "rl_controller": True,
                "rl_actor": True,
                "rl_value": True,
                "rl_coord_proposer": True,
                "rl_reward_shaper": True,
                "rl_rollout_collector": True,
                "rl_trainer": True,
                "swarm_orchestrator": False,
                "planner": False,
                "simple_explorer": False,
                "full_explorer": False,
                "rule_proposer": False,
                "mechanic_classifier": False,
                "hypothesis_engine": False,
                "discriminating_test_selector": False,
                "mechanic_synthesizer": False,
                "memory_store": False,
            }
        }
    )
    embed_dim: int = 256
    hidden_dim: int = 256
    action_emb_dim: int = 32
    controller_hidden: int = 128
    num_modes: int = 4
    modes: List[str] = field(default_factory=lambda: ["probe", "exploit", "escape_loop", "focus_click"])
    controller: Dict[str, Any] = field(
        default_factory=lambda: {
            "sample_mode_train": True,
            "temperature": 1.0,
        }
    )
    mode_action_allow: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
        }
    )
    mode_action_bias: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "0": {},
            "1": {},
            "2": {"ACTION6": 0.25},
            "3": {"ACTION6": 0.5},
        }
    )
    mode_coord_bias: Dict[str, float] = field(
        default_factory=lambda: {
            "0": 0.0,
            "1": 0.0,
            "2": 0.25,
            "3": 0.5,
        }
    )
    coord_mode: str = "proposal"
    coord_topK: int = 16
    rollout_batch_episodes: int = 8
    rollout_max_steps: int = 40
    frame_stack: int = 4
    hud_probe_enabled: bool = False
    hud_probe_steps: int = 30
    hud_cache_dir: str = "runs/cache"
    hud_detect_window: int = 30
    hud_change_rate_threshold: float = 0.8
    hud_min_changed_cells_per_step: int = 1
    hud_edge_margin: int = 4
    hud_min_component_area: int = 20
    hud_dilate_px: int = 1
    episodes_per_iter: int = 8
    max_steps_per_episode: int = 40
    stochastic_actions_train: bool = True
    deterministic_eval: bool = True
    save_trajectory_batches: bool = True
    reward: Dict[str, float] = field(
        default_factory=lambda: {
            "alpha_novel": 0.05,
            "beta_effect": 0.02,
            "match_poi": 0.5,
            "negative_step": 0.5,
            "delta_loop": 0.05,
            "loop_window_N": 25,
        }
    )
    algo: str = "a2c"
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    controller_coef: float = 1.0
    actor_coef: float = 1.0
    aux_mode_ce_coef: float = 0.2
    max_grad_norm: float = 1.0
    updates_per_iter: int = 1
    optimizers: Dict[str, Any] = field(default_factory=lambda: {"mode": "single"})
    ppo: Dict[str, float] = field(
        default_factory=lambda: {
            "clip_eps": 0.2,
            "epochs": 2,
            "minibatches": 2,
            "target_kl": 0.03,
        }
    )
    ckpt: Dict[str, int] = field(
        default_factory=lambda: {
            "save_every_iters": 1,
            "keep_last": 5,
        }
    )
    log: Dict[str, Any] = field(
        default_factory=lambda: {
            "write_jsonl": True,
            "write_trace": True,
            "trace_episodes_per_iter": 1,
        }
    )
    aux: Dict[str, float] = field(
        default_factory=lambda: {
            "controller_aux_weight": 0.2,
        }
    )
    train: Dict[str, int] = field(default_factory=lambda: {"num_iters": 100})
    eval: Dict[str, int] = field(
        default_factory=lambda: {
            "every_iters": 5,
            "episodes": 4,
            "trace_eval_episodes": 10,
            "win_switch_threshold": 0.05,
            "win_switch_consecutive_k": 3,
        }
    )

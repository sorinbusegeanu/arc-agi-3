from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionSequence:
    sequence_id: str
    actions: list[str]
    source: str
    parent_sequence_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    episode_id: str
    sequence_id: str
    actions: list[str]
    frame_dir: str
    video_path: str
    step_count: int
    done: bool
    truncated: bool
    output_dir: str
    total_reward: float = 0.0
    action_log: list[dict[str, Any]] = field(default_factory=list)
    frame_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelAnalysisResult:
    sprite_summary: str
    selected_poi_summary: str
    pattern_match_summary: str
    screen_change_summary: str
    ui_summary: str
    proposed_actions_start: list[str] = field(default_factory=list)
    proposed_actions_update: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprite_summary": self.sprite_summary,
            "selected_poi_summary": self.selected_poi_summary,
            "pattern_match_summary": self.pattern_match_summary,
            "screen_change_summary": self.screen_change_summary,
            "ui_summary": self.ui_summary,
            "proposed_actions_start": list(self.proposed_actions_start),
            "proposed_actions_update": list(self.proposed_actions_update),
            "raw_text": self.raw_text,
        }


@dataclass(frozen=True)
class LoopConfig:
    output_root: str = "runs/vlm_loop"
    fps: int = 2
    max_steps: int = 12
    max_iterations: int = 2
    max_sequences_per_iter: int = 16
    max_returned_sequences_per_video: int = 4
    action_length_cap: int = 6
    timeout_sec: float = 120.0
    agents_per_iteration: int = 4
    retry_count: int = 2
    seed: int = 0
    task_prompt: str = ""
    llm_backend: str = "vllm"
    ollama_url: str = "http://192.168.0.51:11434"
    ollama_model: str = "qwen3-vl:8b"
    ollama_num_ctx: int = 16384
    vllm_url: str = "http://192.168.0.51:8000"
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    disable_thinking: bool = True
    greedy: bool = False
    top_p: float = 0.8
    top_k: int = 20
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    presence_penalty: float = 1.5
    out_seq_length: int = 16384
    prompt_config_path: str = "src/vlm_loop/prompt_config.json"
    max_prompt_frames: int = 2
    initial_bootstrap_enabled: bool = True
    initial_bootstrap_episode_index: int = 0
    initial_bootstrap_num_actions: int = 6
    initial_bootstrap_actions: list[str] = field(default_factory=list)
    render_terminal: bool = False
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VLMV2Config:
    output_root: str
    env_factory_path: str
    env_id: str
    env_root: str
    seed: int
    render_terminal: bool
    fps: int
    max_actions_budget: int
    bootstrap_actions: list[str]
    prompt_config_path: str
    system_prompt: str
    start_level_prompt: str
    get_list_of_objects_actions_prompt: str
    in_loop_prompt: str
    ollama_url: str
    ollama_model: str
    ollama_num_ctx: int
    timeout_sec: float
    retry_count: int
    max_prompt_frames: int
    max_parallel_branches: int
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectDescriptor:
    name: str
    color: str
    position: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StartLevelAnalysis:
    layout: str
    player: str
    reasoning: str
    objects: list[ObjectDescriptor]
    hud: str
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "player": self.player,
            "reasoning": self.reasoning,
            "objects": [item.to_dict() for item in self.objects],
            "hud": self.hud,
            "raw_text": self.raw_text,
        }


@dataclass(frozen=True)
class ObjectActionProposal:
    object_name: str
    sequence_letters: str
    actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchPlan:
    branch_id: str
    object_name: str
    current_level_actions: list[str]
    source_prompt_stage: str
    parent_branch_id: str | None = None
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchRunResult:
    branch: BranchPlan
    output_dir: str
    frame_dir: str
    video_path: str
    action_log: list[dict[str, Any]]
    executed_actions: list[str]
    executed_count: int
    total_reward: float
    done: bool
    truncated: bool
    won: bool
    levels_completed_before: int
    levels_completed_after: int
    start_info: dict[str, Any] = field(default_factory=dict)
    end_info: dict[str, Any] = field(default_factory=dict)

    @property
    def level_advanced(self) -> bool:
        return int(self.levels_completed_after) > int(self.levels_completed_before)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch.to_dict(),
            "output_dir": self.output_dir,
            "frame_dir": self.frame_dir,
            "video_path": self.video_path,
            "action_log": list(self.action_log),
            "executed_actions": list(self.executed_actions),
            "executed_count": int(self.executed_count),
            "total_reward": float(self.total_reward),
            "done": bool(self.done),
            "truncated": bool(self.truncated),
            "won": bool(self.won),
            "levels_completed_before": int(self.levels_completed_before),
            "levels_completed_after": int(self.levels_completed_after),
            "level_advanced": bool(self.level_advanced),
            "start_info": dict(self.start_info),
            "end_info": dict(self.end_info),
        }

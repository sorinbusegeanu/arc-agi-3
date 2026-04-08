from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .action_schema import parse_action_letters
from .models import VLMV2Config


DEFAULT_ENV_ROOT = "/home/zodrak/zod/other_repos/arc-interactive/environment_files"
DEFAULT_ENV_FACTORY_PATH = "arc_agi_agent.envs.arc_env_factory:create_env"


def default_session_id() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def load_prompt_config(path: str) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("prompt config must be a JSON object")
    for key in ("start_level_prompt", "get_list_of_objects_actions_prompt", "in_loop_prompt"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            raise ValueError(f"prompt config missing non-empty string: {key}")
    return payload


def build_config(
    *,
    output_root: str,
    env_factory_path: str,
    env_id: str,
    env_root: str,
    seed: int,
    render_terminal: bool,
    fps: int,
    max_actions_budget: int,
    action_list: str,
    prompt_config_path: str,
    timeout_sec: float,
    retry_count: int,
    max_prompt_frames: int,
    max_parallel_branches: int,
    debug: bool,
) -> VLMV2Config:
    prompt_cfg = load_prompt_config(prompt_config_path)
    return VLMV2Config(
        output_root=output_root,
        env_factory_path=env_factory_path,
        env_id=env_id,
        env_root=env_root,
        seed=int(seed),
        render_terminal=bool(render_terminal),
        fps=int(fps),
        max_actions_budget=int(max_actions_budget),
        bootstrap_actions=parse_action_letters(action_list),
        prompt_config_path=prompt_config_path,
        system_prompt=str(prompt_cfg.get("system_prompt") or "Return plain text only. No markdown fences."),
        start_level_prompt=str(prompt_cfg["start_level_prompt"]),
        get_list_of_objects_actions_prompt=str(prompt_cfg["get_list_of_objects_actions_prompt"]),
        in_loop_prompt=str(prompt_cfg["in_loop_prompt"]),
        ollama_url=str(prompt_cfg.get("ollama_url") or "http://127.0.0.1:11434"),
        ollama_model=str(prompt_cfg.get("ollama_model") or "qwen3-vl:8b"),
        ollama_num_ctx=int(prompt_cfg.get("ollama_num_ctx") or 16384),
        timeout_sec=float(timeout_sec if timeout_sec > 0 else prompt_cfg.get("timeout_sec") or 120.0),
        retry_count=int(retry_count if retry_count >= 0 else prompt_cfg.get("retry_count") or 2),
        max_prompt_frames=int(max_prompt_frames if max_prompt_frames > 0 else prompt_cfg.get("max_prompt_frames") or 6),
        max_parallel_branches=int(max_parallel_branches if max_parallel_branches > 0 else prompt_cfg.get("max_parallel_branches") or 8),
        debug=bool(debug),
    )

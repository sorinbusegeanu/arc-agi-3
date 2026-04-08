from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BootstrapConfig:
    primary_sequence: tuple[str, ...]
    fallback_sequences: tuple[tuple[str, ...], ...]
    stop_after_unique_avatar_found: bool
    capture_raw_observations: bool
    export_pngs: bool
    export_video: bool
    png_scale_factor: int
    video_fps: int
    enable_llm_hud_analysis: bool
    enable_vlm_hud_analysis: bool
    enable_llm_poi_analysis: bool
    enable_vlm_poi_analysis: bool
    hud_text_prompt_path: str
    hud_video_prompt_path: str
    poi_text_prompt_path: str
    poi_video_prompt_path: str
    toon_text_endpoint_name: str
    toon_video_endpoint_name: str
    toon_poi_text_endpoint_name: str
    toon_poi_video_endpoint_name: str
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3-vl:8b"
    ollama_num_ctx: int = 24000
    timeout_sec: float = 120.0
    retry_count: int = 2


def load_bootstrap_config(config_path: str | None = None) -> BootstrapConfig:
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parent / "agents_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    bootstrap = payload["bootstrap"]
    return BootstrapConfig(
        primary_sequence=tuple(bootstrap["primary_sequence"]),
        fallback_sequences=(tuple(bootstrap["fallback_sequences"]),),
        stop_after_unique_avatar_found=bool(bootstrap["stop_after_unique_avatar_found"]),
        capture_raw_observations=bool(bootstrap["capture_raw_observations"]),
        export_pngs=bool(bootstrap["export_pngs"]),
        export_video=bool(bootstrap["export_video"]),
        png_scale_factor=int(bootstrap["png_scale_factor"]),
        video_fps=int(bootstrap["video_fps"]),
        enable_llm_hud_analysis=bool(bootstrap["enable_llm_hud_analysis"]),
        enable_vlm_hud_analysis=bool(bootstrap["enable_vlm_hud_analysis"]),
        enable_llm_poi_analysis=bool(bootstrap["enable_llm_poi_analysis"]),
        enable_vlm_poi_analysis=bool(bootstrap["enable_vlm_poi_analysis"]),
        hud_text_prompt_path=str(Path(__file__).resolve().parents[1] / bootstrap["prompts"]["hud_text_prompt_path"]),
        hud_video_prompt_path=str(Path(__file__).resolve().parents[1] / bootstrap["prompts"]["hud_video_prompt_path"]),
        poi_text_prompt_path=str(Path(__file__).resolve().parents[1] / bootstrap["prompts"]["poi_text_prompt_path"]),
        poi_video_prompt_path=str(Path(__file__).resolve().parents[1] / bootstrap["prompts"]["poi_video_prompt_path"]),
        toon_text_endpoint_name=str(bootstrap["toon"]["text_endpoint_name"]),
        toon_video_endpoint_name=str(bootstrap["toon"]["video_endpoint_name"]),
        toon_poi_text_endpoint_name=str(bootstrap["toon"].get("poi_text_endpoint_name", bootstrap["toon"]["text_endpoint_name"])),
        toon_poi_video_endpoint_name=str(bootstrap["toon"].get("poi_video_endpoint_name", bootstrap["toon"]["video_endpoint_name"])),
        ollama_url=str(bootstrap["toon"]["ollama_url"]),
        ollama_model=str(bootstrap["toon"]["ollama_model"]),
        ollama_num_ctx=int(bootstrap["toon"]["ollama_num_ctx"]),
        timeout_sec=float(bootstrap["toon"]["timeout_sec"]),
        retry_count=int(bootstrap["toon"]["retry_count"]),
    )

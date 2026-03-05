from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from .lsa_bootstrap_explorer import run_bootstrap_explorer
from .lsa_controller import run_episode as run_controller
from .lsa_visual_describer import run_visual_describer


@dataclass
class AgentCatalogEntry:
    name: str
    run: Callable[..., Any]
    description: str


def build_agent_catalog() -> Dict[str, AgentCatalogEntry]:
    return {
        "bootstrap_explorer": AgentCatalogEntry(
            name="bootstrap_explorer",
            run=run_bootstrap_explorer,
            description="Deterministic bootstrap probe trace collector.",
        ),
        "visual_describer": AgentCatalogEntry(
            name="visual_describer",
            run=run_visual_describer,
            description="LLM-based visual describer.",
        ),
        "poi_router": AgentCatalogEntry(
            name="poi_router",
            run=_not_implemented("poi_router"),
            description="POI router (not implemented).",
        ),
        "poi_explorer": AgentCatalogEntry(
            name="poi_explorer",
            run=_not_implemented("poi_explorer"),
            description="POI explorer (not implemented).",
        ),
        "change_point_detector": AgentCatalogEntry(
            name="change_point_detector",
            run=_not_implemented("change_point_detector"),
            description="Change-point detector (not implemented).",
        ),
        "segment_memory": AgentCatalogEntry(
            name="segment_memory",
            run=_not_implemented("segment_memory"),
            description="Segment memory agent (not implemented).",
        ),
        "controller": AgentCatalogEntry(
            name="controller",
            run=run_controller,
            description="Controller agent.",
        ),
        "failure_analyser": AgentCatalogEntry(
            name="failure_analyser",
            run=_not_implemented("failure_analyser"),
            description="Failure analyser agent (not implemented).",
        ),
    }


def default_call_order() -> List[str]:
    return [
        "bootstrap_explorer",
        "visual_describer",
        "poi_router",
        "poi_explorer",
        "change_point_detector",
        "segment_memory",
        "controller",
        "failure_analyser",
    ]


def _not_implemented(agent_name: str) -> Callable[..., Any]:
    def _placeholder(*_args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "schema_version": "NotImplementedV1",
            "agent_name": agent_name,
            "episode_id": kwargs.get("episode_id"),
            "game_id": kwargs.get("game_id"),
            "seed": kwargs.get("seed"),
            "trace_id": kwargs.get("trace_id"),
            "timestamp_step": kwargs.get("timestamp_step"),
            "errors": ["not_implemented"],
        }

    return _placeholder

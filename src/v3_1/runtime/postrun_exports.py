from __future__ import annotations

import ray

from v3_1.visualization.exports import export_postrun_artifact
from v3_1.visualization.heatmaps import build_visit_heatmap
from v3_1.visualization.summaries import build_run_summary


def _persist(storage_agent, **kwargs) -> str:
    persist = getattr(storage_agent, "persist", None)
    if persist is not None and hasattr(persist, "remote"):
        return ray.get(persist.remote(**kwargs))
    return storage_agent.persist(**kwargs)


def export_postrun(storage_agent, *, session_id: str, round_id: int, episodes: list[dict], won: bool, blackboard_version: str, memory_version: str, width: int, height: int) -> dict:
    summary = build_run_summary(
        rounds_completed=round_id,
        won=won,
        latest_blackboard_version=blackboard_version,
        latest_memory_version=memory_version,
    )
    heatmap = build_visit_heatmap(episodes, width=width, height=height)
    summary_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="report", name="summary.json", payload=summary)
    heatmap_path = _persist(storage_agent, session_id=session_id, round_id=round_id, kind="heatmap", name="visit_heatmap.json", payload=heatmap)
    return {"summary_path": summary_path, "heatmap_path": heatmap_path}

from __future__ import annotations

import ray

from v3_1.analysis.episode_analysis import analyze_episode


@ray.remote
class AnalysisWorker:
    def analyze(
        self,
        raw_episode,
        analysis_mode: str,
        blackboard_snapshot: dict | None = None,
        mechanic_graph_snapshot: dict | None = None,
        hypothesis_config: object | None = None,
        llm_adapter: object | None = None,
        hypothesis_registry_snapshot: dict | None = None,
    ):
        normalized = str(analysis_mode or "").strip().lower()
        if normalized not in {"probe", "directed_outcome"}:
            raise ValueError(f"analysis_mode must be 'probe' or 'directed_outcome', got {analysis_mode!r}")
        return analyze_episode(raw_episode, normalized, blackboard_snapshot, mechanic_graph_snapshot, hypothesis_config, llm_adapter, hypothesis_registry_snapshot)


@ray.remote
def analyze_episode_task(
    raw_episode,
    analysis_mode: str,
    blackboard_snapshot: dict | None = None,
    mechanic_graph_snapshot: dict | None = None,
    hypothesis_config: object | None = None,
    llm_adapter: object | None = None,
    hypothesis_registry_snapshot: dict | None = None,
):
    normalized = str(analysis_mode or "").strip().lower()
    if normalized not in {"probe", "directed_outcome"}:
        raise ValueError(f"analysis_mode must be 'probe' or 'directed_outcome', got {analysis_mode!r}")
    return analyze_episode(raw_episode, normalized, blackboard_snapshot, mechanic_graph_snapshot, hypothesis_config, llm_adapter, hypothesis_registry_snapshot)

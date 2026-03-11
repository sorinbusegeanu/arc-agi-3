from __future__ import annotations

import ray

from v3_1.analysis.episode_analysis import analyze_episode


@ray.remote
class AnalysisWorker:
    def analyze(self, raw_episode):
        return analyze_episode(raw_episode)

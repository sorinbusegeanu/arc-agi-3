from __future__ import annotations

from typing import Dict, List, Optional

from codex_baseline_v2.shared.config import TrajectoryAnalysisConfigV2
from codex_baseline_v2.shared.schemas import BlackboardStateV2, TrajectoryEpisodeV2
from codex_baseline_v2.trajectory_analysis.analyzer import analyze_trajectories

from .messages import AnalyzedEpisode, BlackboardMergeRequest, BlackboardMergeResult
from .versions import new_blackboard_version


class BlackboardActor:
    def __init__(self, trajectory_analysis_cfg: Optional[Dict[str, object]] = None) -> None:
        self.cfg = TrajectoryAnalysisConfigV2(**(trajectory_analysis_cfg or {}))
        self.blackboard: Optional[BlackboardStateV2] = None
        self.snapshot_registry: Dict[str, Dict[str, object]] = {}

    def merge(self, request: BlackboardMergeRequest) -> BlackboardMergeResult:
        episodes = [TrajectoryEpisodeV2.from_dict(row.analyzed_episode) for row in request.analyzed_episodes]
        prior = BlackboardStateV2.from_dict(request.prior_blackboard) if request.prior_blackboard is not None else self.blackboard
        self.blackboard = analyze_trajectories(episodes, self.cfg, round_id=request.round_id, prior_blackboard=prior)
        version = new_blackboard_version(request.game_id, request.round_id)
        snapshot_ref = f"bb_snapshot:{version}"
        payload = self.blackboard.to_dict()
        self.snapshot_registry[snapshot_ref] = payload
        return BlackboardMergeResult(
            game_id=request.game_id,
            round_id=request.round_id,
            blackboard_version=version,
            snapshot_ref=snapshot_ref,
            blackboard=payload,
            merge_stats={"episode_count": len(episodes)},
        )

    def get_snapshot(self, snapshot_ref: str) -> Optional[Dict[str, object]]:
        return self.snapshot_registry.get(snapshot_ref)

    def latest_snapshot(self) -> Optional[Dict[str, object]]:
        return self.blackboard.to_dict() if self.blackboard is not None else None

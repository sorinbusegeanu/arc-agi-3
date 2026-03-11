from __future__ import annotations

from typing import Iterable, List

from codex_baseline_v2.analyst.analyst import analyze_episodes
from codex_baseline_v2.shared.config import AnalystConfigV2
from codex_baseline_v2.shared.schemas import TrajectoryEpisodeV2

from .messages import AnalyzedEpisode, RawEpisode


def _raw_to_episode(raw: RawEpisode) -> TrajectoryEpisodeV2:
    return TrajectoryEpisodeV2.from_dict(
        {
            "schema_version": "V2",
            "game_id": raw.game_id,
            "episode_id": raw.episode_id,
            "steps": [
                {
                    "schema_version": "V2",
                    "game_id": raw.game_id,
                    "episode_id": raw.episode_id,
                    "step_idx": step.step_idx,
                    "action": step.action,
                    "reward": step.reward,
                    "done": step.done,
                    "observation": step.observation,
                    "observation_summary": None,
                    "info": step.info,
                }
                for step in raw.steps
            ],
            "done": raw.done,
            "win": raw.win,
            "seed": raw.seed,
            "metadata": raw.metadata,
        }
    )


class EpisodeAnalyzerWorker:
    def __init__(self, worker_id: str, analyst_cfg: dict | None = None) -> None:
        self.worker_id = worker_id
        self.cfg = AnalystConfigV2(**(analyst_cfg or {}))

    def analyze(self, raw_episode: RawEpisode) -> AnalyzedEpisode:
        analyzed = analyze_episodes([_raw_to_episode(raw_episode)], self.cfg)[0]
        return AnalyzedEpisode(
            game_id=raw_episode.game_id,
            episode_id=raw_episode.episode_id,
            round_id=raw_episode.round_id,
            analyzed_episode=analyzed.to_dict(),
            summary={"step_count": len(analyzed.steps)},
        )


def dispatch_analysis(worker_pool: List[object], raw_episodes: Iterable[RawEpisode]):
    return [worker.analyze.remote(raw_episode) for worker, raw_episode in zip(worker_pool, raw_episodes)]


def collect_analysis_results(ray_module, refs) -> List[AnalyzedEpisode]:
    return list(ray_module.get(refs))

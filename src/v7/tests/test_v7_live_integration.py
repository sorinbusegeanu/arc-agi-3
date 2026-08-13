from __future__ import annotations

import json

import numpy as np

from v7.derivation.online import OnlineHierarchyBuilder
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import EpisodeEvidence
from v7.evaluation import collect_evidence, write_evidence_report
from v7.experiment import V7ExperimentConfig, parse_games, run_experiment
from v7.memory.ids import MemoryLevel
from v7.memory.writer import CanonicalMemoryWriter


def test_online_hierarchy_derives_m2_m4_once() -> None:
    writer = CanonicalMemoryWriter()
    pipeline = MemoryLearningPipeline(writer)
    pipeline.observe_episode(EpisodeEvidence(10, 2, 100, True))
    pipeline.observe_episode(EpisodeEvidence(11, 2, 100, True))
    builder = OnlineHierarchyBuilder(writer, pipeline)
    first = builder.derive()
    assert first.families == 1
    assert first.roles == 2
    assert first.concepts == 1
    count = len(getattr(writer, "_nodes"))
    second = builder.derive()
    assert second.total == 0
    assert len(getattr(writer, "_nodes")) == count
    levels = [node.level for node in getattr(writer, "_nodes").values()]
    assert MemoryLevel.M2 in levels
    assert MemoryLevel.M3 in levels
    assert MemoryLevel.M4 in levels


class _ExperimentAdapter:
    def __init__(self, **_kwargs) -> None:
        self.grid = np.array([[0, 0], [0, 0]], dtype=np.int64)
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        self.last_levels_completed = 0
        self.level_completed_event = False
        self.reset_count = 0
        self.step_count = 0

    def observe(self):
        return self.grid.copy()

    def available_actions(self):
        return [1, 2]

    def action_data(self, _action, rng=None):
        return {}

    def step(self, action, data=None):
        self.step_count += 1
        self.grid = np.array([[0, int(action)], [self.step_count % 2, 0]], dtype=np.int64)
        self.level_completed_event = False
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        return self.grid.copy()


def test_multi_game_experiment_and_evidence_report(tmp_path) -> None:
    result = run_experiment(
        tmp_path,
        V7ExperimentConfig(games=("game-a", "game-b"), steps_per_game=4, epochs=1, commit_every=2, epsilon=0.0),
        env_factory=_ExperimentAdapter,
    )
    assert result.games == 2
    assert result.total_steps == 8
    assert result.final_generation > 0
    assert (tmp_path / "experiment_summary.json").exists()
    summary = collect_evidence(tmp_path)
    assert summary.total_memories > 0
    report = write_evidence_report(tmp_path)
    assert report["schema"] == "v7-experiment-evidence-v1"
    stored = json.loads((tmp_path / "reports" / "v7_evidence.json").read_text(encoding="utf-8"))
    assert stored["summary"]["generation"] == summary.generation


def test_parse_games_accepts_lists_and_commas() -> None:
    assert parse_games(["a,b", "b", "c"]) == ("a", "b", "c")

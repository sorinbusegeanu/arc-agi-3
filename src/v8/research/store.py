from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import ExperimentSpec, RuntimeHypothesis


_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runtime_hypothesis (
    hypothesis_id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    target_chain_edge TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence TEXT NOT NULL,
    failure_level TEXT NOT NULL,
    category TEXT NOT NULL,
    parent_hypothesis_id TEXT,
    scope_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment (
    experiment_id TEXT PRIMARY KEY,
    purpose TEXT NOT NULL,
    code_revision TEXT NOT NULL,
    snapshot_id TEXT,
    games_json TEXT NOT NULL,
    seeds_json TEXT NOT NULL,
    interaction_budget INTEGER NOT NULL,
    conditions_json TEXT NOT NULL,
    frozen INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS experiment_prediction (
    experiment_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    hypothesis_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    expected_direction TEXT NOT NULL,
    expected_min_effect REAL NOT NULL,
    flat_tolerance REAL NOT NULL,
    falsifier TEXT NOT NULL,
    PRIMARY KEY (experiment_id, ordinal),
    FOREIGN KEY(experiment_id) REFERENCES experiment(experiment_id)
);
CREATE TABLE IF NOT EXISTS experiment_result (
    experiment_id TEXT NOT NULL,
    condition_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    mean REAL NOT NULL,
    variance REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, condition_id, metric),
    FOREIGN KEY(experiment_id) REFERENCES experiment(experiment_id)
);
"""


class ResearchStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResearchStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def upsert_hypothesis(self, hypothesis: RuntimeHypothesis) -> None:
        self._conn.execute(
            """INSERT INTO runtime_hypothesis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(hypothesis_id) DO UPDATE SET
               claim=excluded.claim,
               target_chain_edge=excluded.target_chain_edge,
               status=excluded.status,
               confidence=excluded.confidence,
               failure_level=excluded.failure_level,
               category=excluded.category,
               parent_hypothesis_id=excluded.parent_hypothesis_id,
               scope_json=excluded.scope_json""",
            (
                hypothesis.hypothesis_id,
                hypothesis.claim,
                hypothesis.target_chain_edge,
                hypothesis.status,
                hypothesis.confidence.value,
                hypothesis.failure_level.value,
                hypothesis.category.value,
                hypothesis.parent_hypothesis_id,
                json.dumps(dict(hypothesis.scope), sort_keys=True),
            ),
        )
        self._conn.commit()

    def create_experiment(self, spec: ExperimentSpec) -> None:
        spec.validate()
        self._conn.execute(
            """INSERT INTO experiment(
               experiment_id,purpose,code_revision,snapshot_id,games_json,seeds_json,
               interaction_budget,conditions_json,frozen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                spec.experiment_id,
                spec.purpose.value,
                spec.code_revision,
                spec.snapshot_id,
                json.dumps(spec.games),
                json.dumps(spec.seeds),
                spec.interaction_budget,
                json.dumps(spec.conditions, sort_keys=True),
            ),
        )
        self._conn.executemany(
            """INSERT INTO experiment_prediction(
               experiment_id,ordinal,hypothesis_id,metric,expected_direction,
               expected_min_effect,flat_tolerance,falsifier)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    spec.experiment_id,
                    idx,
                    prediction.hypothesis_id,
                    prediction.metric,
                    prediction.direction.value,
                    prediction.expected_min_effect,
                    prediction.flat_tolerance,
                    prediction.falsifier,
                )
                for idx, prediction in enumerate(spec.predictions)
            ],
        )
        self._conn.commit()

    def freeze_experiment(self, experiment_id: str) -> None:
        cur = self._conn.execute(
            "UPDATE experiment SET frozen=1 WHERE experiment_id=?",
            (experiment_id,),
        )
        if cur.rowcount != 1:
            raise KeyError(experiment_id)
        self._conn.commit()

    def is_frozen(self, experiment_id: str) -> bool:
        row = self._conn.execute(
            "SELECT frozen FROM experiment WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return bool(row[0])

    def record_result(
        self,
        experiment_id: str,
        condition_id: str,
        metric: str,
        mean: float,
        variance: float,
        sample_count: int,
    ) -> None:
        if not self.is_frozen(experiment_id):
            raise RuntimeError("experiment predictions must be frozen before results")
        self._conn.execute(
            """INSERT INTO experiment_result(
               experiment_id,condition_id,metric,mean,variance,sample_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (experiment_id, condition_id, metric, mean, variance, sample_count),
        )
        self._conn.commit()

    def result_count(self, experiment_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM experiment_result WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        return int(row[0])

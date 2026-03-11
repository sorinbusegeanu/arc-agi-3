from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List, Optional

from codex_baseline_v2.shared.schemas import TrajectoryEpisodeV2


class SQLiteIntermediateStoreV2:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episode_metadata (
                    game_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    episode_id TEXT NOT NULL,
                    step_count INTEGER NOT NULL,
                    done INTEGER NOT NULL,
                    win INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (game_id, round_id, episode_id)
                );
                CREATE TABLE IF NOT EXISTS step_metadata (
                    game_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    episode_id TEXT NOT NULL,
                    step_idx INTEGER NOT NULL,
                    area_id TEXT,
                    pre_state_hash TEXT,
                    post_state_hash TEXT,
                    action_context_key TEXT,
                    PRIMARY KEY (game_id, round_id, episode_id, step_idx)
                );
                CREATE TABLE IF NOT EXISTS observation_summary_metadata (
                    game_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    episode_id TEXT NOT NULL,
                    step_idx INTEGER NOT NULL,
                    area_id TEXT,
                    state_signature_id TEXT,
                    avatar_candidate_count INTEGER NOT NULL,
                    poi_count INTEGER NOT NULL,
                    navigation_context_key TEXT,
                    PRIMARY KEY (game_id, round_id, episode_id, step_idx)
                );
                CREATE TABLE IF NOT EXISTS area_assignment_results (
                    game_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    episode_id TEXT NOT NULL,
                    step_idx INTEGER NOT NULL,
                    area_id TEXT,
                    PRIMARY KEY (game_id, round_id, episode_id, step_idx)
                );
                CREATE TABLE IF NOT EXISTS analysis_stage_stats (
                    game_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    PRIMARY KEY (game_id, round_id, stage)
                );
                CREATE INDEX IF NOT EXISTS idx_episode_metadata_game_id ON episode_metadata(game_id);
                CREATE INDEX IF NOT EXISTS idx_step_metadata_episode ON step_metadata(game_id, episode_id, step_idx);
                CREATE INDEX IF NOT EXISTS idx_step_metadata_area_id ON step_metadata(area_id);
                CREATE INDEX IF NOT EXISTS idx_obs_summary_area_id ON observation_summary_metadata(area_id);
                CREATE INDEX IF NOT EXISTS idx_area_assignment_area_id ON area_assignment_results(area_id);
                """
            )

    def write_episode_batch(self, game_id: str, round_id: int, episodes: List[TrajectoryEpisodeV2]) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("BEGIN")
            cur.executemany(
                """
                INSERT OR REPLACE INTO episode_metadata
                (game_id, round_id, episode_id, step_count, done, win, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        game_id,
                        round_id,
                        episode.episode_id,
                        len(episode.steps),
                        int(bool(episode.done)),
                        int(bool(episode.win)),
                        json.dumps(episode.metadata, sort_keys=True),
                    )
                    for episode in episodes
                ],
            )
            step_rows = []
            summary_rows = []
            area_rows = []
            for episode in episodes:
                for step in episode.steps:
                    summary = step.observation_summary
                    step_rows.append(
                        (
                            game_id,
                            round_id,
                            episode.episode_id,
                            step.step_idx,
                            step.area_id,
                            step.pre_state_hash,
                            step.post_state_hash,
                            step.action_context_key,
                        )
                    )
                    summary_rows.append(
                        (
                            game_id,
                            round_id,
                            episode.episode_id,
                            step.step_idx,
                            summary.area_id if summary is not None else None,
                            summary.state_signature_id if summary is not None else None,
                            len(summary.avatar_candidates) if summary is not None else 0,
                            len(summary.candidate_pois) if summary is not None else 0,
                            summary.navigation_context_key if summary is not None else None,
                        )
                    )
                    area_rows.append(
                        (
                            game_id,
                            round_id,
                            episode.episode_id,
                            step.step_idx,
                            step.area_id,
                        )
                    )
            cur.executemany(
                """
                INSERT OR REPLACE INTO step_metadata
                (game_id, round_id, episode_id, step_idx, area_id, pre_state_hash, post_state_hash, action_context_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                step_rows,
            )
            cur.executemany(
                """
                INSERT OR REPLACE INTO observation_summary_metadata
                (game_id, round_id, episode_id, step_idx, area_id, state_signature_id, avatar_candidate_count, poi_count, navigation_context_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                summary_rows,
            )
            cur.executemany(
                """
                INSERT OR REPLACE INTO area_assignment_results
                (game_id, round_id, episode_id, step_idx, area_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                area_rows,
            )
            conn.commit()

    def write_stage_stats(self, game_id: str, round_id: int, stage: str, stats: Dict[str, object]) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_stage_stats
                (game_id, round_id, stage, stats_json)
                VALUES (?, ?, ?, ?)
                """,
                (game_id, round_id, stage, json.dumps(stats, sort_keys=True)),
            )
            conn.commit()


def sqlite_db_path_for_round(root_dir: str, game_id: str, round_id: int) -> str:
    return os.path.join(root_dir, f"game_{game_id}", f"round_{round_id:03d}", "analyst_outputs", "analysis_intermediates.sqlite")

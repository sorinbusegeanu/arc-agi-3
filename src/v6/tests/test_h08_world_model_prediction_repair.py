from __future__ import annotations

import sqlite3

from v6.h08_world_model_prediction_repair import (
    _issue_world_model_prediction,
    _match_world_model_predictions,
    _world_model_prediction_metrics,
)


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE future_option_events (
            event_id TEXT PRIMARY KEY,
            owner_type TEXT,
            owner_key TEXT,
            source_family_id TEXT,
            first_seen_global_step INTEGER,
            last_seen_global_step INTEGER,
            context_key TEXT,
            game TEXT,
            motif_type TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE world_model_prediction_events (
            prediction_event_id TEXT PRIMARY KEY,
            component_signature TEXT NOT NULL,
            prediction_global_step INTEGER NOT NULL,
            predicted_family TEXT,
            predicted_effect TEXT,
            predicted_outcome TEXT,
            game_key TEXT,
            context_key TEXT,
            action_key TEXT,
            baseline_prediction_score REAL,
            component_prediction_score REAL,
            observed_event_id TEXT,
            observed_global_step INTEGER,
            observed_family TEXT,
            observed_effect TEXT,
            prediction_correct INTEGER,
            provenance_status TEXT
        )
        """
    )
    return conn


def test_matches_recurrence_after_prediction_using_last_seen_step() -> None:
    conn = _connection()
    conn.execute(
        """
        INSERT INTO future_option_events (
            event_id, owner_type, owner_key, source_family_id,
            first_seen_global_step, last_seen_global_step,
            context_key, game, motif_type
        ) VALUES ('fo-1', 'family', 'fam-a', 'fam-a', 100, 250, 'ctx-a', 'game-a', 'motif-a')
        """
    )
    conn.execute(
        """
        INSERT INTO world_model_prediction_events (
            prediction_event_id, component_signature, prediction_global_step,
            predicted_family, game_key, context_key, baseline_prediction_score,
            provenance_status
        ) VALUES ('pred-1', 'wm:test', 200, 'fam-a', 'game-a', 'ctx-a', 0.5, 'prospective')
        """
    )

    _match_world_model_predictions(conn, 'wm:test')

    row = conn.execute(
        """
        SELECT observed_event_id, observed_global_step, observed_family,
               prediction_correct, provenance_status
        FROM world_model_prediction_events WHERE prediction_event_id='pred-1'
        """
    ).fetchone()
    assert row == ('fo-1', 250, 'fam-a', 1, 'verified')
    metrics = _world_model_prediction_metrics(conn, 'wm:test')
    assert metrics['matched'] == 1
    assert metrics['unmatched'] == 0
    assert metrics['stale'] == 0


def test_stale_prediction_does_not_block_fresh_prediction() -> None:
    conn = _connection()
    conn.execute(
        """
        INSERT INTO future_option_events (
            event_id, owner_type, owner_key, source_family_id,
            first_seen_global_step, last_seen_global_step,
            context_key, game, motif_type
        ) VALUES ('fo-2', 'family', 'fam-a', 'fam-a', 250, 300, 'ctx-new', 'game-a', 'motif-a')
        """
    )
    conn.execute(
        """
        INSERT INTO world_model_prediction_events (
            prediction_event_id, component_signature, prediction_global_step,
            predicted_family, game_key, context_key, baseline_prediction_score,
            provenance_status
        ) VALUES ('pred-old', 'wm:test', 200, 'fam-a', 'game-a', 'ctx-old', 0.5, 'prospective')
        """
    )

    _match_world_model_predictions(conn, 'wm:test')
    old_status = conn.execute(
        "SELECT provenance_status FROM world_model_prediction_events WHERE prediction_event_id='pred-old'"
    ).fetchone()[0]
    assert old_status == 'stale'

    _issue_world_model_prediction(
        conn,
        signature='wm:test',
        prediction_step=300,
        families=['fam-a'],
        contexts=['ctx-new'],
        games=['game-a'],
    )

    rows = conn.execute(
        """
        SELECT prediction_global_step, provenance_status
        FROM world_model_prediction_events
        WHERE component_signature='wm:test'
        ORDER BY prediction_global_step
        """
    ).fetchall()
    assert rows == [(200, 'stale'), (300, 'prospective')]
    metrics = _world_model_prediction_metrics(conn, 'wm:test')
    assert metrics['matched'] == 0
    assert metrics['unmatched'] == 1
    assert metrics['stale'] == 1


def test_live_unmatched_prediction_still_blocks_duplicate_issue() -> None:
    conn = _connection()
    conn.execute(
        """
        INSERT INTO world_model_prediction_events (
            prediction_event_id, component_signature, prediction_global_step,
            predicted_family, baseline_prediction_score, provenance_status
        ) VALUES ('pred-live', 'wm:test', 300, 'fam-a', 0.5, 'prospective')
        """
    )

    _issue_world_model_prediction(
        conn,
        signature='wm:test',
        prediction_step=300,
        families=['fam-a'],
        contexts=[],
        games=[],
    )

    count = conn.execute(
        "SELECT COUNT(*) FROM world_model_prediction_events WHERE component_signature='wm:test'"
    ).fetchone()[0]
    assert count == 1

from __future__ import annotations

import json
import sqlite3
from hashlib import sha1
from typing import Any

_INSTALLED = False


def install_h08_world_model_prediction_repair() -> None:
    """Install prospective prediction matching fixes for v6.3 world models."""
    global _INSTALLED
    if _INSTALLED:
        return

    from v6 import v63_higher_order_semantics as semantics

    semantics._issue_world_model_prediction = _issue_world_model_prediction
    semantics._match_world_model_predictions = _match_world_model_predictions
    semantics._world_model_prediction_metrics = _world_model_prediction_metrics
    _INSTALLED = True


def _future_event_family_expr(conn: sqlite3.Connection) -> str:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(future_option_events)").fetchall()
    }
    if "source_family_id" in columns:
        return "COALESCE(source_family_id, CASE WHEN owner_type='family' THEN owner_key END)"
    return "CASE WHEN owner_type='family' THEN owner_key END"


def _current_evidence_step(conn: sqlite3.Connection) -> int | None:
    values: list[int] = []
    for table, column in (
        ("future_option_events", "last_seen_global_step"),
        ("stable_contingencies", "last_seen_global_step"),
        ("concept_candidates", "last_seen_global_step"),
    ):
        try:
            row = conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
        except sqlite3.Error:
            continue
        if row is not None and row[0] is not None:
            values.append(int(row[0]))
    return max(values) if values else None


def _clean_scope(values: list[str], *, reject_surrogate: bool = False) -> list[str]:
    cleaned = sorted({str(value) for value in values if value is not None and str(value)})
    if reject_surrogate:
        cleaned = [value for value in cleaned if not value.startswith("surrogate_game:")]
    return cleaned


def _prediction_scope_payload(
    *, predicted_family: str, contexts: list[str], games: list[str]
) -> str:
    return json.dumps(
        {
            "kind": "next_family",
            "family": str(predicted_family),
            "contexts": _clean_scope(contexts),
            "games": _clean_scope(games, reject_surrogate=True),
            "scope_version": "component_scope_v2",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_prediction_scope(
    payload: Any,
    *,
    fallback_context: Any = None,
    fallback_game: Any = None,
) -> tuple[list[str], list[str]]:
    contexts: list[str] = []
    games: list[str] = []
    if payload:
        try:
            parsed = json.loads(str(payload))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("scope_version") == "component_scope_v2":
            contexts = _clean_scope(list(parsed.get("contexts") or []))
            games = _clean_scope(list(parsed.get("games") or []), reject_surrogate=True)
    if not contexts and fallback_context:
        contexts = [str(fallback_context)]
    if not games and fallback_game:
        games = [str(fallback_game)]
    return contexts, games


def _issue_world_model_prediction(
    conn: sqlite3.Connection,
    *,
    signature: str,
    prediction_step: int,
    families: list[str],
    contexts: list[str],
    games: list[str],
) -> None:
    families = _clean_scope(families)
    contexts = _clean_scope(contexts)
    games = _clean_scope(games, reject_surrogate=True)
    if not families:
        return

    # Only a live prospective prediction blocks a new one. Predictions marked
    # stale are historical evidence and must not permanently suppress renewal.
    existing = conn.execute(
        "SELECT 1 FROM world_model_prediction_events "
        "WHERE component_signature=? AND observed_event_id IS NULL "
        "AND COALESCE(provenance_status, 'prospective')='prospective'",
        (signature,),
    ).fetchone()
    if existing is not None:
        return

    family_expr = _future_event_family_expr(conn)
    params: list[Any] = [int(prediction_step)]
    where = ["first_seen_global_step <= ?"]
    if contexts:
        placeholders = ",".join("?" for _ in contexts)
        where.append(f"context_key IN ({placeholders})")
        params.extend(contexts)
    if games:
        placeholders = ",".join("?" for _ in games)
        where.append(f"game IN ({placeholders})")
        params.extend(games)
    placeholders = ",".join("?" for _ in families)
    where.append(f"{family_expr} IN ({placeholders})")
    params.extend(families)

    try:
        rows = conn.execute(
            f"SELECT {family_expr} AS family, COUNT(*) AS n "
            "FROM future_option_events WHERE " + " AND ".join(where)
            + " GROUP BY family ORDER BY n DESC, family ASC",
            params,
        ).fetchall()
    except sqlite3.Error:
        rows = []

    predicted_family = families[0]
    baseline = 0.0
    if rows:
        total = sum(int(row[1] or 0) for row in rows)
        predicted_family = str(rows[0][0])
        baseline = float(int(rows[0][1] or 0) / max(1, total))

    scope_payload = _prediction_scope_payload(
        predicted_family=predicted_family,
        contexts=contexts,
        games=games,
    )
    # Keep legacy scalar scope columns only when the component itself has a
    # single context/game. Multi-context models are validated over the full
    # persisted component scope encoded in predicted_outcome.
    exact_context = contexts[0] if len(contexts) == 1 else None
    exact_game = games[0] if len(games) == 1 else None
    event_id = "wm-pred:" + sha1(
        json.dumps(
            [signature, int(prediction_step), scope_payload, predicted_family],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    conn.execute(
        """
        INSERT OR IGNORE INTO world_model_prediction_events (
            prediction_event_id, component_signature, prediction_global_step,
            predicted_family, predicted_effect, predicted_outcome,
            game_key, context_key, action_key, baseline_prediction_score,
            component_prediction_score, provenance_status
        ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, NULL, 'prospective')
        """,
        (
            event_id,
            signature,
            int(prediction_step),
            predicted_family,
            scope_payload,
            exact_game,
            exact_context,
            baseline,
        ),
    )


def _match_world_model_predictions(conn: sqlite3.Connection, signature: str) -> None:
    family_expr = _future_event_family_expr(conn)
    rows = conn.execute(
        """
        SELECT prediction_event_id, prediction_global_step, predicted_family,
               game_key, context_key, predicted_outcome
        FROM world_model_prediction_events
        WHERE component_signature=? AND observed_event_id IS NULL
          AND COALESCE(provenance_status, 'prospective')='prospective'
        ORDER BY prediction_global_step ASC, prediction_event_id ASC
        """,
        (signature,),
    ).fetchall()

    current_step = _current_evidence_step(conn)
    for row in rows:
        prediction_step = int(row[1])
        contexts, games = _parse_prediction_scope(
            row[5], fallback_context=row[4], fallback_game=row[3]
        )
        where = ["last_seen_global_step > ?"]
        params: list[Any] = [prediction_step]
        if contexts:
            placeholders = ",".join("?" for _ in contexts)
            where.append(f"context_key IN ({placeholders})")
            params.extend(contexts)
        if games:
            placeholders = ",".join("?" for _ in games)
            where.append(f"game IN ({placeholders})")
            params.extend(games)
        try:
            observed = conn.execute(
                f"SELECT event_id, last_seen_global_step, {family_expr} AS family, motif_type "
                "FROM future_option_events WHERE " + " AND ".join(where)
                + " ORDER BY last_seen_global_step ASC, event_id ASC LIMIT 1",
                params,
            ).fetchone()
        except sqlite3.Error:
            observed = None

        if observed is None:
            # Once newer compact evidence exists outside the eligible component
            # scope, keep the prediction live. It becomes stale only after the
            # evidence frontier advances and there is still no eligible event.
            if current_step is not None and int(current_step) > prediction_step:
                conn.execute(
                    "UPDATE world_model_prediction_events "
                    "SET provenance_status='stale' WHERE prediction_event_id=?",
                    (str(row[0]),),
                )
            continue

        observed_family = None if observed[2] is None else str(observed[2])
        correct = int(observed_family == str(row[2]))
        conn.execute(
            """
            UPDATE world_model_prediction_events
            SET observed_event_id=?, observed_global_step=?, observed_family=?,
                observed_effect=?, prediction_correct=?, component_prediction_score=?,
                provenance_status='verified'
            WHERE prediction_event_id=?
            """,
            (
                str(observed[0]),
                int(observed[1]),
                observed_family,
                None if observed[3] is None else str(observed[3]),
                correct,
                float(correct),
                str(row[0]),
            ),
        )


def _world_model_prediction_metrics(
    conn: sqlite3.Connection,
    signature: str,
) -> dict[str, float | int | None]:
    rows = conn.execute(
        """
        SELECT prediction_correct, baseline_prediction_score,
               component_prediction_score, observed_event_id,
               COALESCE(provenance_status, 'prospective')
        FROM world_model_prediction_events
        WHERE component_signature=?
        """,
        (signature,),
    ).fetchall()
    matched_rows = [row for row in rows if row[3] is not None and str(row[4]) == "verified"]
    live_unmatched_rows = [
        row for row in rows if row[3] is None and str(row[4]) == "prospective"
    ]
    stale_rows = [row for row in rows if row[3] is None and str(row[4]) == "stale"]
    correct = sum(int(row[0] or 0) for row in matched_rows)
    matched = len(matched_rows)
    accuracy = float(correct / matched) if matched else 0.0
    baselines = [float(row[1]) for row in matched_rows if row[1] is not None]
    components = [float(row[2]) for row in matched_rows if row[2] is not None]
    baseline = sum(baselines) / len(baselines) if baselines else None
    component = sum(components) / len(components) if components else None
    gain = (
        float(component - baseline)
        if component is not None and baseline is not None
        else 0.0
    )
    return {
        "matched": matched,
        "unmatched": len(live_unmatched_rows),
        "stale": len(stale_rows),
        "correct": correct,
        "accuracy": accuracy,
        "baseline": baseline,
        "component": component,
        "gain": gain,
    }

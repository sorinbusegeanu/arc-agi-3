from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


_INSTALLED = False
_ORIGINAL_FOLD_SINGLE_DB: Any = None
_ORIGINAL_IMPORT_CANDIDATE: Any = None
_ORIGINAL_TEMPORAL_MILESTONES_FOR_DB: Any = None

DEFAULT_STABLE_TRANSFORMATION_FAMILY_SUPPORT = 5


def install_v63_temporal_semantics_completion() -> None:
    """Install end-to-end threshold-crossing timing semantics for H03/H04/H05."""
    global _INSTALLED
    global _ORIGINAL_FOLD_SINGLE_DB
    global _ORIGINAL_IMPORT_CANDIDATE
    global _ORIGINAL_TEMPORAL_MILESTONES_FOR_DB
    if _INSTALLED:
        return

    from v6.carrier_emergence import CarrierEmergenceTracker
    from v6.memory import compact_memory as compact
    from v6.evaluation import interaction_sampling as sampling

    _ORIGINAL_FOLD_SINGLE_DB = compact._fold_single_db
    _ORIGINAL_IMPORT_CANDIDATE = CarrierEmergenceTracker.import_candidate
    _ORIGINAL_TEMPORAL_MILESTONES_FOR_DB = sampling._temporal_milestones_for_db

    compact._fold_single_db = _fold_single_db_with_threshold_timing
    CarrierEmergenceTracker.import_candidate = _import_candidate_preserving_emergence_step
    sampling._temporal_milestones_for_db = _temporal_milestones_for_db_with_carrier_threshold
    _INSTALLED = True


def _import_candidate_preserving_emergence_step(self: Any, **kwargs: Any) -> Any:
    """Keep the persisted emergence threshold when restoring an emergent carrier.

    Compact v6.3 carrier rows intentionally store the threshold-crossing step in
    first_seen_global_step once a carrier is emergent. The legacy restore path
    reconstructs synthetic observations and can otherwise lose that threshold.
    """
    result = _ORIGINAL_IMPORT_CANDIDATE(self, **kwargs)
    carrier_signature = str(kwargs.get("carrier_signature") or "")
    carrier_source = str(kwargs.get("carrier_source") or "unknown")
    first_emergent_step = kwargs.get("first_seen_global_step")
    if (
        carrier_signature
        and bool(kwargs.get("is_emergent"))
        and carrier_source != "context_action_fallback"
        and first_emergent_step is not None
    ):
        first_steps = getattr(self, "_v63_first_emergent_steps", None)
        if first_steps is None:
            first_steps = {}
            self._v63_first_emergent_steps = first_steps
        existing = first_steps.get(carrier_signature)
        step = int(first_emergent_step)
        first_steps[carrier_signature] = step if existing is None else min(int(existing), step)
    return result


def _temporal_milestones_for_db_with_carrier_threshold(
    path: Path,
    *,
    game: str,
    sampler_name: str,
    seed: int,
) -> dict[str, object]:
    result = dict(
        _ORIGINAL_TEMPORAL_MILESTONES_FOR_DB(
            path,
            game=game,
            sampler_name=sampler_name,
            seed=seed,
        )
    )
    emergence_step = _carrier_emergence_step_from_sidecar(Path(path))
    if emergence_step is not None:
        result["first_emergent_carrier_step"] = int(emergence_step)
    return result


def _fold_single_db_with_threshold_timing(
    *,
    db_path: Path,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    replay_conn: sqlite3.Connection,
    fold_config: Any,
    totals: dict[str, Any],
    busy_timeout_ms: int = 10000,
) -> None:
    _ORIGINAL_FOLD_SINGLE_DB(
        db_path=db_path,
        state_conn=state_conn,
        graph_conn=graph_conn,
        replay_conn=replay_conn,
        fold_config=fold_config,
        totals=totals,
        busy_timeout_ms=busy_timeout_ms,
    )
    _repair_fold_threshold_timing(
        db_path=Path(db_path),
        state_conn=state_conn,
    )


def _repair_fold_threshold_timing(
    *,
    db_path: Path,
    state_conn: sqlite3.Connection,
) -> dict[str, int | None]:
    """Repair compact milestones from prospective threshold-crossing evidence."""
    stable_family_step = _stable_transformation_family_threshold_step(db_path)
    emergent_carrier_step = _carrier_emergence_step_from_sidecar(db_path)

    from v6.memory import compact_memory as compact

    game = compact._path_segment(db_path, -4)
    sampler = compact._path_segment(db_path, -3)
    seed = compact._seed_from_db_path(db_path)

    state_conn.execute(
        """
        INSERT INTO temporal_milestones (
            game, sampler, seed,
            first_stable_transformation_family_step,
            first_emergent_carrier_step
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(game, sampler, seed) DO UPDATE SET
            first_stable_transformation_family_step = CASE
                WHEN temporal_milestones.first_stable_transformation_family_step IS NULL
                    THEN excluded.first_stable_transformation_family_step
                WHEN excluded.first_stable_transformation_family_step IS NULL
                    THEN temporal_milestones.first_stable_transformation_family_step
                ELSE MIN(
                    temporal_milestones.first_stable_transformation_family_step,
                    excluded.first_stable_transformation_family_step
                )
            END,
            first_emergent_carrier_step = CASE
                WHEN temporal_milestones.first_emergent_carrier_step IS NULL
                    THEN excluded.first_emergent_carrier_step
                WHEN excluded.first_emergent_carrier_step IS NULL
                    THEN temporal_milestones.first_emergent_carrier_step
                ELSE MIN(
                    temporal_milestones.first_emergent_carrier_step,
                    excluded.first_emergent_carrier_step
                )
            END
        """,
        (
            game,
            sampler,
            int(seed),
            stable_family_step,
            emergent_carrier_step,
        ),
    )

    # The compact schema predates an explicit first_emergent column and uses
    # first_seen_global_step as the H04 emergence timestamp for rows that are
    # already emergent. Explicit sidecar threshold evidence is authoritative;
    # do not minimize it against a stale first-observation timestamp.
    for carrier_signature, threshold_step in _carrier_thresholds_from_sidecar(db_path).items():
        state_conn.execute(
            """
            UPDATE carrier_candidates
            SET first_seen_global_step = ?,
                carrier_timing_source = 'real_evidence'
            WHERE carrier_signature = ?
              AND COALESCE(is_emergent, 0) = 1
              AND carrier_source != 'context_action_fallback'
            """,
            (int(threshold_step), carrier_signature),
        )

    return {
        "first_stable_transformation_family_step": stable_family_step,
        "first_emergent_carrier_step": emergent_carrier_step,
    }


def _stable_transformation_family_threshold_step(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "prediction_results" not in tables:
                return None
            support_threshold = int(
                _sampling_metadata_value(
                    connection,
                    "transformation_family_stable_support",
                    DEFAULT_STABLE_TRANSFORMATION_FAMILY_SUPPORT,
                )
            )
            offset = int(
                _sampling_metadata_value(connection, "global_step_offset", 0)
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(prediction_results)"
                ).fetchall()
            }
            step_expr = (
                "COALESCE(global_step, interaction_id + ?)"
                if "global_step" in columns
                else "interaction_id + ?"
            )
            rows = connection.execute(
                f"""
                SELECT {step_expr}, actual_family
                FROM prediction_results
                WHERE actual_family IS NOT NULL
                ORDER BY {step_expr} ASC, interaction_id ASC
                """,
                (offset, offset),
            ).fetchall()
    except sqlite3.Error:
        return None

    counts: Counter[str] = Counter()
    for global_step, actual_family in rows:
        family = str(actual_family)
        counts[family] += 1
        if counts[family] >= max(1, support_threshold):
            return int(global_step)
    return None


def _sampling_metadata_value(
    connection: sqlite3.Connection,
    key: str,
    fallback: int,
) -> int:
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sampling_metadata'"
        ).fetchone()
        if exists is None:
            return int(fallback)
        row = connection.execute(
            "SELECT value FROM sampling_metadata WHERE key = ?",
            (str(key),),
        ).fetchone()
    except sqlite3.Error:
        return int(fallback)
    if row is None or row[0] is None:
        return int(fallback)
    try:
        return int(json.loads(str(row[0])))
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return int(fallback)


def _carrier_thresholds_from_sidecar(db_path: Path) -> dict[str, int]:
    sidecar = db_path.with_name("carrier_candidates.json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}

    thresholds: dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        carrier_signature = str(
            item.get("carrier_signature") or item.get("carrier_id") or ""
        )
        if not carrier_signature:
            continue
        if str(item.get("carrier_source") or "unknown") == "context_action_fallback":
            continue
        if str(item.get("status") or "") != "emergent_carrier":
            continue
        threshold = item.get("first_emergent_global_step")
        if threshold is None:
            # Missing threshold evidence is not equivalent to first observation.
            continue
        try:
            step = int(threshold)
        except (TypeError, ValueError):
            continue
        existing = thresholds.get(carrier_signature)
        thresholds[carrier_signature] = step if existing is None else min(existing, step)
    return thresholds


def _carrier_emergence_step_from_sidecar(db_path: Path) -> int | None:
    thresholds = _carrier_thresholds_from_sidecar(db_path)
    return min(thresholds.values()) if thresholds else None

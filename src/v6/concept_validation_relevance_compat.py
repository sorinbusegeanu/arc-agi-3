from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINAL_VALIDATE: Any = None


def _repair_population_accounting(memory_dir: Path, diagnostic_epoch_id: Any) -> None:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return
    epoch_key = "" if diagnostic_epoch_id is None else str(diagnostic_epoch_id)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            """
            SELECT rowid, payload_json
            FROM concept_promotion_validation_diagnostics
            WHERE diagnostic_epoch_id = ?
            """,
            (epoch_key,),
        ).fetchall()
        changed = False
        for rowid, payload_json in rows:
            try:
                payload = json.loads(str(payload_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            total = int(payload.get("total_later_event_count", 0) or 0)
            relevant = int(payload.get("relevant_heldout_event_count", 0) or 0)
            invalid = int(payload.get("invalid_explanation_event_count", 0) or 0)
            unrelated = max(0, total - relevant - invalid)
            if unrelated <= int(payload.get("unrelated_event_count", 0) or 0):
                continue
            payload["unrelated_event_count"] = unrelated
            conn.execute(
                "UPDATE concept_promotion_validation_diagnostics SET payload_json=? WHERE rowid=?",
                (json.dumps(payload, sort_keys=True), int(rowid)),
            )
            changed = True
        if changed:
            conn.commit()


def _validate_with_population_accounting(*args: Any, **kwargs: Any):
    result = _ORIGINAL_VALIDATE(*args, **kwargs)
    validate_roles_and_concepts = bool(kwargs.get("validate_roles_and_concepts", False))
    if not validate_roles_and_concepts:
        return result
    memory_dir_raw = kwargs.get("memory_dir") if "memory_dir" in kwargs else (args[0] if args else None)
    if memory_dir_raw is None:
        return result
    _repair_population_accounting(
        Path(memory_dir_raw),
        kwargs.get("diagnostic_epoch_id"),
    )
    return result


def install_concept_validation_relevance_compat() -> None:
    global _INSTALLED, _ORIGINAL_VALIDATE
    if _INSTALLED:
        return

    from v6 import concept_validation_fastpath_compat as compat
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    # Existing integration contracts expose the installed future-option helper
    # through the compatibility module. Keep the identity while the helper
    # itself remains relevance-pruned.
    compat._future_option_motif_explanation_events = substrate._future_option_motif_explanation_events

    _ORIGINAL_VALIDATE = substrate.validate_incremental_promotions_only
    substrate.validate_incremental_promotions_only = _validate_with_population_accounting
    suite.validate_incremental_promotions_only = _validate_with_population_accounting
    _INSTALLED = True

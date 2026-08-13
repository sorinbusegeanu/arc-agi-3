from __future__ import annotations

import sqlite3
from hashlib import sha256
from pathlib import Path

_INSTALLED = False


def _state_fingerprint(memory_dir: Path) -> str:
    """Hash only structural role/concept state, across supported schema versions."""
    db = Path(memory_dir) / "current_state.sqlite"
    digest = sha256()
    volatile_tokens = (
        "promotion", "validation", "demotion", "probation", "reactivation",
        "currently_promoted", "historically_promoted", "is_promoted", "concept_status",
    )
    with sqlite3.connect(db) as conn:
        for table in ("role_candidates", "role_links", "concept_candidates", "concept_links"):
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists is None:
                continue
            columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            stable = [
                name for name in columns
                if not any(token in name.lower() for token in volatile_tokens)
            ]
            if not stable:
                continue
            digest.update(table.encode("utf-8"))
            digest.update("|".join(stable).encode("utf-8"))
            selected = ", ".join('"' + name.replace('"', '""') + '"' for name in stable)
            for row in conn.execute(f"SELECT {selected} FROM {table} ORDER BY 1").fetchall():
                digest.update(repr(tuple(row)).encode("utf-8"))
    return digest.hexdigest()


def install_concept_validation_fastpath_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v6 import concept_validation_fastpath_compat as compat

    compat._state_fingerprint = _state_fingerprint
    _INSTALLED = True

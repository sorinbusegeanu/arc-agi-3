from __future__ import annotations

import sqlite3


_PATCHED = False


def install_v63_report_repairs_compat() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import v6.future_options as module

    original = module._concept_validation_records
    if getattr(original, "_v63_row_factory_safe", False):
        return

    def row_factory_safe(state_conn: sqlite3.Connection):
        previous = state_conn.row_factory
        state_conn.row_factory = sqlite3.Row
        try:
            return original(state_conn)
        finally:
            state_conn.row_factory = previous

    row_factory_safe._v63_row_factory_safe = True  # type: ignore[attr-defined]
    module._concept_validation_records = row_factory_safe

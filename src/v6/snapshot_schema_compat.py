from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def apply_patch() -> None:
    import v6.memory.v621_runtime as module

    cls = module.V621SnapshotMemoryQueryEngine
    original = cls._load_extension_indexes

    def _load_extension_indexes(self: Any) -> None:
        source_dir = getattr(self.snapshot, "source_memory_dir", None)
        if source_dir:
            path = Path(source_dir) / "current_state.sqlite"
            if path.exists():
                connection = sqlite3.connect(
                    f"file:{path.resolve()}?mode=ro",
                    uri=True,
                    timeout=10.0,
                )
                try:
                    columns = {
                        str(row[1])
                        for row in connection.execute("PRAGMA table_info(memory_nodes)").fetchall()
                    }
                finally:
                    connection.close()
                if columns and "status" not in columns:
                    # Legacy compact-memory snapshots predate the v6.1 node-status
                    # extension. Their base snapshot indexes remain valid; there is
                    # no v6.2.1 extension state to preload from this schema.
                    return
        original(self)

    cls._load_extension_indexes = _load_extension_indexes

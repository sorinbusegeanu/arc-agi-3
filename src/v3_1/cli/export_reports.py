from __future__ import annotations

import argparse
import ray

from v3_1.agents.storage_agent import StorageAgent
from v3_1.storage.paths import get_persistent_memory_db_path
from v3_1.storage.persistent_memory import PersistentMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--include-persistent-memory", action="store_true")
    args = parser.parse_args()
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    storage = StorageAgent.remote(root_dir=args.root, sqlite_path=None)
    payload = {"status": "ok"}
    if args.include_persistent_memory:
        db_path = str(get_persistent_memory_db_path(args.root))
        payload["persistent_memory_db_path"] = db_path
        try:
            store = PersistentMemoryStore(db_path)
            with store._connect() as conn:
                payload["persistent_memory_tables"] = {
                    table: int(conn.execute(f"select count(*) from {table}").fetchone()[0])
                    for table in ("sessions", "memory_snapshots", "skill_stats", "candidate_outcomes", "failure_patterns", "recovery_patterns")
                }
        except Exception as exc:
            payload["persistent_memory_error"] = str(exc)
    path = ray.get(storage.persist.remote(session_id=args.session_id, round_id=args.round_id, kind="report", name="manual_report.json", payload=payload))
    print(path)


if __name__ == "__main__":
    main()

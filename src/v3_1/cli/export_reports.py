from __future__ import annotations

import argparse
import ray

from v3_1.agents.storage_agent import StorageAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--round-id", type=int, default=0)
    args = parser.parse_args()
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    storage = StorageAgent.remote(root_dir=args.root, sqlite_path=None)
    path = ray.get(storage.persist.remote(session_id=args.session_id, round_id=args.round_id, kind="report", name="manual_report.json", payload={"status": "ok"}))
    print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from v3_1.eval.avatar_tracking_metrics import build_avatar_tracking_metrics


def main() -> None:
    base = Path("runs_v3_1")
    traces = sorted(base.glob("session_*/avatar_tracking_trace.json"))
    rows = []
    for trace in traces:
        payload = json.loads(trace.read_text())
        events = list(payload.get("events", []) or [])
        rows.append(
            {
                "session_id": trace.parent.name,
                "mode": "new_live_motion_tracker",
                **build_avatar_tracking_metrics([dict(row.get("payload", {}) or {}) | {"termination_reason": dict(row.get("payload", {}) or {}).get("termination_reason")} for row in events]),
            }
        )
    print(json.dumps({"sessions": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

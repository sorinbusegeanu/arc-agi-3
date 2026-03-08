from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def log_event(log_dir: str, event: str, payload: Dict[str, Any]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    entry = {
        "event": event,
        "timestamp": time.time(),
        **payload,
    }
    path = os.path.join(log_dir, "events.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

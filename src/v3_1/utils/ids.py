from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_digest(payload: Any, *, length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:length]


def make_handle(prefix: str, payload: Any) -> str:
    return f"{prefix}:{stable_digest(payload)}"


from __future__ import annotations

from typing import Dict, List, Optional

from .trace import TraceReader


def read_trace(path: Optional[str]) -> List[Dict]:
    if not path:
        return []
    return list(TraceReader(path))

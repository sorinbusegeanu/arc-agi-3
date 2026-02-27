from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional


class TraceWriter:
    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, payload: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            json.dump(payload, f)
            f.write("\n")


class TraceReader:
    def __init__(self, path: str) -> None:
        self.path = path

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)

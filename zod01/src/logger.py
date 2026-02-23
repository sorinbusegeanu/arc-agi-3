from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any


class EpisodeLogger:
    def __init__(self, out_dir: str = "zod01/logs") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._hasher = hashlib.blake2b(digest_size=16)
        self._fp = None
        self._episode_id = ""

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def start(
        self,
        game_id: str,
        seed: int,
        variant_id: str,
        episode_id: str | None = None,
    ) -> Path:
        self._episode_id = episode_id or f"{game_id}.s{seed}.{variant_id}.{int(time.time())}.{uuid.uuid4().hex[:8]}"
        safe_id = self._episode_id.replace("/", "_")
        path = self.out_dir / f"{safe_id}.jsonl"
        self._fp = path.open("w", encoding="utf-8")
        return path

    def log(self, event: dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("logger not started")
        line = json.dumps(event, sort_keys=True)
        self._fp.write(line + "\n")
        self._hasher.update(line.encode("utf-8"))

    def close(self) -> str:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        return self._hasher.hexdigest()

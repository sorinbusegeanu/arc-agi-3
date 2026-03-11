from __future__ import annotations

from pathlib import Path

from v3_1.storage.paths import round_root
from v3_1.storage.serialization import dumps


class ArtifactStore:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def write_json(self, *, session_id: str, round_id: int, artifact_name: str, payload) -> str:
        root = round_root(self.root_dir, session_id, round_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / artifact_name
        encoded = dumps(payload)
        path.write_text(encoded, encoding="utf-8")
        return str(path)


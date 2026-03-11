from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionStore:
    manifests: list[dict] = field(default_factory=list)

    def record(self, entry: dict) -> None:
        self.manifests.append(entry)


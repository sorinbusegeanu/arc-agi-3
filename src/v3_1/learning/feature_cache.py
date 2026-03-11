from __future__ import annotations


class FeatureCache:
    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        return self._rows.get(key)

    def put(self, key: str, value: dict) -> None:
        self._rows[key] = value


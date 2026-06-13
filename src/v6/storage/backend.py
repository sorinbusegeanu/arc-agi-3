from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def write_interactions(self, records: list[dict]) -> None: ...

    @abstractmethod
    def write_deltas(self, records: list[dict]) -> None: ...

    @abstractmethod
    def write_transformation_families(self, records: list[dict]) -> None: ...

    @abstractmethod
    def write_contingencies(self, records: list[dict]) -> None: ...

    @abstractmethod
    def write_future_effects(self, records: list[dict]) -> None: ...

    @abstractmethod
    def write_role_candidates(self, records: list[dict]) -> None: ...

    @abstractmethod
    def write_run_summary(self, record: dict) -> None: ...

    @abstractmethod
    def finalize(self) -> None: ...

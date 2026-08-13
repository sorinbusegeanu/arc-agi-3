from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from v7.memory.generation import GenerationId
from v7.memory.read_view import MemoryReadView


@dataclass(frozen=True, slots=True)
class ReadViewHandle:
    """Opaque handle identifying one immutable published read view."""

    generation_id: GenerationId
    transport_key: str


class ReadViewTransport(Protocol):
    """Transport boundary for immutable generation read views."""

    def publish(self, view: MemoryReadView) -> ReadViewHandle:
        ...

    def attach(self, handle: ReadViewHandle) -> MemoryReadView:
        ...

    def release(self, handle: ReadViewHandle) -> None:
        ...

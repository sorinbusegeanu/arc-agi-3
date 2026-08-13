from __future__ import annotations

from v7.memory.read_view import MemoryReadView
from v7.memory.transport.base import ReadViewHandle


class LocalReadViewTransport:
    """Direct-reference transport for single-process use and transport-contract tests.

    This is intentionally not the final multi-process transport. It establishes the
    publication/attachment contract without serializing read views or coupling the
    publisher to a storage format. v7.1 can replace it with mmap/shared-memory
    transport while preserving the same handle semantics.
    """

    def __init__(self) -> None:
        self._views: dict[str, MemoryReadView] = {}

    def publish(self, view: MemoryReadView) -> ReadViewHandle:
        key = f"generation-{int(view.generation_id)}"
        existing = self._views.get(key)
        if existing is not None and existing is not view:
            raise ValueError(f"generation already published: {int(view.generation_id)}")
        self._views[key] = view
        return ReadViewHandle(generation_id=view.generation_id, transport_key=key)

    def attach(self, handle: ReadViewHandle) -> MemoryReadView:
        view = self._views.get(handle.transport_key)
        if view is None:
            raise KeyError(handle.transport_key)
        if view.generation_id != handle.generation_id:
            raise ValueError("read-view handle generation mismatch")
        return view

    def release(self, handle: ReadViewHandle) -> None:
        self._views.pop(handle.transport_key, None)

    @property
    def retained_generations(self) -> tuple[int, ...]:
        return tuple(sorted(int(view.generation_id) for view in self._views.values()))

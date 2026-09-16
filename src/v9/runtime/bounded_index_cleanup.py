from __future__ import annotations

from typing import Any


def install_bounded_index_cleanup(graph_cls: type) -> None:
    if getattr(graph_cls, "_bounded_index_cleanup_installed", False):
        return
    original_delete = graph_cls.delete_low_level_nodes_batch

    def delete_low_level_nodes_batch(self: Any, plans: tuple[Any, ...]):
        deleted = tuple(original_delete(self, plans))
        adjacency = getattr(self, "_bounded_edges_by_uid", None)
        if isinstance(adjacency, dict):
            for uid in deleted:
                adjacency.pop(uid, None)
        return deleted

    graph_cls.delete_low_level_nodes_batch = delete_low_level_nodes_batch
    graph_cls._bounded_index_cleanup_installed = True

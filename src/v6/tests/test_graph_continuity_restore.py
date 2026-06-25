from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.main import V6Config, V6System
from v6.memory.compact_memory import ensure_memory_layout


class _Env:
    def __init__(self) -> None:
        import numpy as np

        self.grid = np.zeros((2, 2), dtype=int)

    def observe(self):
        return self.grid.copy()

    def step(self, action: int):
        return self.grid.copy()

    def available_actions(self) -> list[int]:
        return [0]


def test_graph_is_restored_before_sampling(tmp_path: Path) -> None:
    memory = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(memory.graph) as connection:
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("context:ctx-a", "context", "ctx-a", 1, 10, 1))
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("contingency:c1", "contingency", "c1", 1, 10, 1))
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("family:f1", "family", "f1", 1, 10, 1))
        connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)", ("carrier:car1", "carrier", "car1", 1, 10, 1))
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("context:ctx-a->supports->contingency:c1", "context:ctx-a", "contingency:c1", "supports", 1, 10, 1, 1.0))
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("contingency:c1->member_of->family:f1", "contingency:c1", "family:f1", "member_of", 1, 10, 1, 1.0))
        connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("carrier:car1->explains->family:f1", "carrier:car1", "family:f1", "explains", 1, 10, 1, 1.0))
        connection.commit()
    memory.summary_json.write_text(json.dumps({"fold_summary": {"graph_node_count": 4}}, indent=2), encoding="utf-8")

    system = V6System(env=_Env(), config=V6Config(database_path=":memory:", memory_input_dir=str(memory.root), restore_compact_memory=True))
    try:
        graph = system.graph.graph
        assert graph.has_node("context:ctx-a")
        assert graph.has_node("contingency:c1")
        assert graph.has_node("family:f1")
        assert graph.has_node("carrier:car1")
        assert graph.has_edge("context:ctx-a", "contingency:c1")
        assert graph.has_edge("contingency:c1", "family:f1")
        assert graph.has_edge("carrier:car1", "family:f1")
    finally:
        system.close()

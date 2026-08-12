from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from v6.memory.causal_evidence import _causal_value, _previous_stage, _weights_for_stage


def test_stage_weights_and_lag_snapshot() -> None:
    assert _weights_for_stage("survival")["prediction_error"] == 0.0
    assert _weights_for_stage("concept_transfer")["transfer_potential"] == 0.30
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memory_development_state(key TEXT PRIMARY KEY, value_json TEXT)")
    conn.execute("INSERT INTO memory_development_state VALUES ('current', '{\"stage\":\"survival\",\"next_stage\":\"graph_expansion\"}')")
    assert _previous_stage(SimpleNamespace(connection=conn)) == "graph_expansion"


def test_realized_evidence_is_not_visible_before_its_step() -> None:
    attrs = {"learning_value": 0.4, "learning_value_realized": 0.9, "learning_value_realized_step": 11}
    value, kind = _causal_value(attrs, prospective_key="learning_value", realized_key="learning_value_realized", score_step=10)
    assert value == 0.4
    assert kind == "prospective"
    value, kind = _causal_value(attrs, prospective_key="learning_value", realized_key="learning_value_realized", score_step=11)
    assert value == 0.9
    assert kind == "realized"

from __future__ import annotations

import sqlite3
from pathlib import Path

from v6.future_options import (
    FutureOptionDevelopmentStage,
    _build_future_option_event,
    derive_future_option_events,
    is_complete_context_key,
    resolve_future_option_development_stage,
)
from v6.memory.compact_memory import ensure_memory_layout


def _event(**overrides: object) -> dict:
    values: dict[str, object] = {
        "owner_type": "interaction",
        "owner_key": "i1",
        "source_kind": "stable_contingency",
        "game": "g1",
        "sampler": "s1",
        "context_key": "ctx",
        "action_key": "a1",
        "text_fragments": ["unstructured"],
        "support_count": 1,
        "polarity": None,
        "first_seen": 1,
        "last_seen": 1,
        "mean_prediction_error": 0.0,
        "mean_replay_priority": 0.0,
        "stability_score": 0.0,
        "event_id_seed": "development",
        "evidence_json": {},
    }
    values.update(overrides)
    return _build_future_option_event(**values)  # type: ignore[arg-type]


def test_auto_development_stage_follows_structural_maturity(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        assert resolve_future_option_development_stage(conn, requested_stage="auto") is FutureOptionDevelopmentStage.SURVIVAL
        conn.execute("INSERT INTO stable_contingencies (canonical_key) VALUES ('stable')")
        assert resolve_future_option_development_stage(conn, requested_stage="auto") is FutureOptionDevelopmentStage.MOVEMENT_FREEDOM
        conn.execute("INSERT INTO transformation_families (canonical_signature) VALUES ('family')")
        assert resolve_future_option_development_stage(conn, requested_stage="auto") is FutureOptionDevelopmentStage.ENVIRONMENTAL_INFLUENCE
        conn.execute("INSERT INTO carrier_candidates (carrier_signature, is_emergent) VALUES ('carrier', 1)")
        assert resolve_future_option_development_stage(conn, requested_stage="auto") is FutureOptionDevelopmentStage.GRAPH_EXPANSION
        conn.execute("INSERT INTO role_candidates (role_signature, is_emergent) VALUES ('role', 1)")
        assert resolve_future_option_development_stage(conn, requested_stage="auto") is FutureOptionDevelopmentStage.ROLE_DISCOVERY
        conn.execute("INSERT INTO concept_candidates (concept_signature, is_promoted) VALUES ('concept', 1)")
        conn.execute("INSERT INTO role_transfer_attempts (attempt_id, reuse_success) VALUES ('transfer', 1)")
        assert resolve_future_option_development_stage(conn, requested_stage="auto") is FutureOptionDevelopmentStage.CONCEPT_TRANSFER


def test_structural_effects_precede_text_fallback_and_store_component_vector() -> None:
    enabled = _event(live_option_delta=2.0, text_fragments=["block"])
    blocked = _event(live_option_delta=-2.0, event_id_seed="blocked")
    terminated = _event(effect_type="terminal_transition", event_id_seed="terminated")
    transformed = _event(effect_type="positive_change", event_id_seed="transformed")
    neutral = _event(live_option_delta=0.0, event_id_seed="neutral")
    reversible = _event(text_fragments=["reverse"], event_id_seed="reversible")
    unknown = _event(text_fragments=["opaque"], event_id_seed="unknown")

    assert (enabled["motif_type"], enabled["motif_classification_reason"]) == ("enable", "structural_option_delta")
    assert blocked["motif_type"] == "block"
    assert terminated["motif_type"] == "terminate"
    assert transformed["motif_type"] == "transform"
    assert neutral["motif_type"] == "neutral"
    assert reversible["motif_type"] == "unknown"
    assert reversible["motif_classification_reason"] == "unverified_fallback"
    assert unknown["motif_type"] == "unknown"
    assert enabled["movement_freedom_delta"] == 2.0
    assert terminated["survival_delta"] == -1.0
    assert transformed["environmental_influence_delta"] == 1.0


def test_event_derivation_persists_auto_stage_and_components(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as state_conn, sqlite3.connect(memory_dir / "graph.sqlite") as graph_conn:
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        state_conn.execute(
            """INSERT INTO stable_contingencies (
                canonical_key, action, effect_signature, support_count, stability_score
            ) VALUES ('ctx|a1', 1, 'change', 5, 0.8)"""
        )
        stage = resolve_future_option_development_stage(state_conn, requested_stage="auto")
        derive_future_option_events(state_conn, graph_conn, max_events=10, development_stage=stage)
        row = state_conn.execute(
            """SELECT future_option_development_stage, movement_freedom_delta,
                      developmental_option_value, motif_classification_reason
               FROM future_option_events"""
        ).fetchone()
        assert tuple(row)[0] == "movement_freedom"
        assert tuple(row)[1] is not None
        assert tuple(row)[2] == tuple(row)[1]
        assert str(tuple(row)[3]).startswith(("structural", "unverified_fallback"))


def test_classification_provenance_and_incomplete_context_are_explicit() -> None:
    event = _event(
        context_key="[null,null,1]",
        effect_type="positive_change",
        source_family_ids={"family-1"},
    )
    assert event["classification_source"] == "structural_effect"
    assert event["classification_provenance_status"] == "verified"
    assert event["source_context_is_surrogate"] == 1
    assert event["context_resolution_source"] == "surrogate"
    assert str(event["source_context_key"]).startswith("surrogate_context:")
    assert not is_complete_context_key("[null,null,1]")


def test_future_option_edge_event_keeps_concrete_edge_provenance(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as state_conn, sqlite3.connect(memory_dir / "graph.sqlite") as graph_conn:
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        state_conn.execute(
            """INSERT INTO memory_edges (
                source_node_id, target_node_id, edge_type, support_count, evidence_json
            ) VALUES ('M0:interaction:11', 'M0:interaction:12', 'expands_future_options', 3, '{\"kind\":\"test\"}')"""
        )
        derive_future_option_events(state_conn, graph_conn, max_events=10)
        row = state_conn.execute(
            """SELECT classification_source, source_interaction_id, target_interaction_id,
                      classification_provenance_status
               FROM future_option_events WHERE source_kind = 'future_option_edge'"""
        ).fetchone()
    assert tuple(row) == ("future_option_edge", "11", "12", "verified")

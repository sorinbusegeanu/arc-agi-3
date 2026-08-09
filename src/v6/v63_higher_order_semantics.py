from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import replace
from hashlib import sha1
from itertools import combinations
from typing import Any


_INSTALLED = False
_ORIGINAL_CARRIER_RECORD: Any = None
_ORIGINAL_CARRIER_BUILD: Any = None
_ORIGINAL_DERIVE_ROLES: Any = None
_ORIGINAL_DERIVE_WORLD_MODELS: Any = None
_ORIGINAL_H06_PROVENANCE_ERROR: Any = None
_ORIGINAL_H08_COMPONENT_GATE: Any = None

MAX_RELATIONAL_COMPONENTS = 64


def install_v63_higher_order_semantics() -> None:
    """Install v6.3 evidence semantics before higher-order derivation/reporting."""
    global _INSTALLED
    global _ORIGINAL_CARRIER_RECORD
    global _ORIGINAL_CARRIER_BUILD
    global _ORIGINAL_DERIVE_ROLES
    global _ORIGINAL_DERIVE_WORLD_MODELS
    global _ORIGINAL_H06_PROVENANCE_ERROR
    global _ORIGINAL_H08_COMPONENT_GATE
    if _INSTALLED:
        return

    from v6.carrier_emergence import CarrierEmergenceTracker
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_h06_report as h06
    from v6 import hypothesis_h08_report as h08

    _ORIGINAL_CARRIER_RECORD = CarrierEmergenceTracker.record_interaction
    _ORIGINAL_CARRIER_BUILD = CarrierEmergenceTracker._build_candidate
    _ORIGINAL_DERIVE_ROLES = substrate.derive_role_candidates
    _ORIGINAL_DERIVE_WORLD_MODELS = substrate.derive_world_model_components
    _ORIGINAL_H06_PROVENANCE_ERROR = h06._transfer_provenance_error
    _ORIGINAL_H08_COMPONENT_GATE = h08._component_passes_h08_validity

    CarrierEmergenceTracker.record_interaction = _carrier_record_interaction
    CarrierEmergenceTracker._build_candidate = _carrier_build_candidate
    substrate.derive_role_candidates = _derive_role_candidates
    substrate.derive_world_model_components = _derive_world_model_components
    h06._transfer_provenance_error = _h06_transfer_provenance_error
    h08._component_passes_h08_validity = _h08_component_passes
    _INSTALLED = True


def _carrier_record_interaction(self: Any, **kwargs: Any) -> Any:
    event = _ORIGINAL_CARRIER_RECORD(self, **kwargs)
    if event is None:
        return None
    carrier = str(event.carrier_signature)
    first_steps = getattr(self, "_v63_first_emergent_steps", None)
    if first_steps is None:
        first_steps = {}
        self._v63_first_emergent_steps = first_steps
    if carrier not in first_steps:
        candidate = _ORIGINAL_CARRIER_BUILD(self, carrier)
        if str(candidate.status) == "emergent_carrier":
            step = kwargs.get("global_step")
            if step is not None:
                first_steps[carrier] = int(step)
    return event


def _carrier_build_candidate(self: Any, carrier_signature: str) -> Any:
    candidate = _ORIGINAL_CARRIER_BUILD(self, carrier_signature)
    first_steps = getattr(self, "_v63_first_emergent_steps", {})
    first_emergent = first_steps.get(str(carrier_signature))
    if str(candidate.status) != "emergent_carrier":
        first_emergent = None
    return replace(candidate, first_emergent_global_step=first_emergent)


def _derive_role_candidates(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_carriers: int,
    max_roles: int,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    summary = _ORIGINAL_DERIVE_ROLES(
        state_conn,
        graph_conn,
        max_carriers,
        max_roles,
        progress_factory,
    )
    _repair_role_emergence_steps(state_conn, graph_conn)
    row = state_conn.execute(
        "SELECT MIN(first_emergent_global_step) FROM role_candidates "
        "WHERE COALESCE(is_emergent, 0)=1"
    ).fetchone()
    first_role = None if row is None or row[0] is None else int(row[0])
    from v6 import higher_order_substrate as substrate

    substrate._write_milestone(
        state_conn, "first_emergent_role_step", first_role, None
    )
    state_conn.commit()
    result = dict(summary)
    result["first_emergent_role_step"] = first_role
    result["role_emergence_timing_version"] = "v63_threshold_crossing_v1"
    return result


def _repair_role_emergence_steps(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
) -> None:
    columns = {
        str(row[1])
        for row in state_conn.execute("PRAGMA table_info(role_candidates)").fetchall()
    }
    if "first_emergent_global_step" not in columns:
        state_conn.execute(
            "ALTER TABLE role_candidates ADD COLUMN first_emergent_global_step INTEGER"
        )

    role_carriers: dict[str, list[str]] = defaultdict(list)
    for row in state_conn.execute(
        "SELECT role_signature, linked_key FROM role_links "
        "WHERE linked_type='carrier' ORDER BY role_signature, linked_key"
    ).fetchall():
        role_carriers[str(row[0])].append(str(row[1]))

    carrier_meta = {
        str(row[0]): {
            "first": None if row[1] is None else int(row[1]),
            "last": None if row[2] is None else int(row[2]),
        }
        for row in state_conn.execute(
            "SELECT carrier_signature, first_seen_global_step, last_seen_global_step "
            "FROM carrier_candidates"
        ).fetchall()
    }
    from v6 import higher_order_substrate as substrate

    carrier_links = substrate._carrier_links_by_carrier(state_conn)
    contexts = {
        f"context:{context}"
        for links in carrier_links.values()
        for context in links.get("context", set())
    }
    context_games = substrate._context_games_for_context_nodes(
        graph_conn, contexts
    ) if contexts else {}

    emergent_roles = {
        str(row[0])
        for row in state_conn.execute(
            "SELECT role_signature FROM role_candidates WHERE COALESCE(is_emergent,0)=1"
        ).fetchall()
    }
    for role, carriers in role_carriers.items():
        if role not in emergent_roles:
            state_conn.execute(
                "UPDATE role_candidates SET first_emergent_global_step=NULL "
                "WHERE role_signature=?",
                (role,),
            )
            continue
        ordered = sorted(
            (
                carrier_meta.get(carrier, {}).get("first"),
                carrier,
            )
            for carrier in carriers
            if carrier_meta.get(carrier, {}).get("first") is not None
        )
        seen_carriers: set[str] = set()
        families: set[str] = set()
        role_contexts: set[str] = set()
        games: set[str] = set()
        emergence_step: int | None = None
        index = 0
        while index < len(ordered):
            step = int(ordered[index][0])
            batch: list[str] = []
            while index < len(ordered) and int(ordered[index][0]) == step:
                batch.append(str(ordered[index][1]))
                index += 1
            for carrier in batch:
                seen_carriers.add(carrier)
                links = carrier_links.get(carrier, {})
                families.update(str(x) for x in links.get("family", set()))
                new_contexts = {str(x) for x in links.get("context", set())}
                role_contexts.update(new_contexts)
                for context in new_contexts:
                    games.update(
                        str(x)
                        for x in context_games.get(f"context:{context}", set())
                    )
            stability = (
                0.25 * min(1.0, len(seen_carriers) / 3.0)
                + 0.25 * min(1.0, len(families) / 3.0)
                + 0.25 * min(1.0, len(role_contexts) / 3.0)
                + 0.25 * min(1.0, len(games) / 2.0)
            )
            if (
                len(seen_carriers) >= 2
                and len(families) >= 1
                and (len(role_contexts) >= 2 or len(games) >= 2)
                and stability >= 0.50
            ):
                emergence_step = step
                break
        state_conn.execute(
            "UPDATE role_candidates SET first_emergent_global_step=? "
            "WHERE role_signature=?",
            (emergence_step, role),
        )


def _h06_transfer_provenance_error(row: sqlite3.Row) -> str | None:
    kind = str(row["transfer_kind"] or "")
    if kind == "cross_game" and (
        int(row["source_game_is_surrogate"] or 0)
        or int(row["target_game_is_surrogate"] or 0)
    ):
        return "surrogate_game_provenance"
    # A row with unresolved game identity may remain useful as diagnostic or
    # cross-context evidence, but it must not enter verified game aggregates.
    if (
        int(row["source_game_is_surrogate"] or 0)
        or int(row["target_game_is_surrogate"] or 0)
    ):
        return "surrogate_game_provenance"
    return _ORIGINAL_H06_PROVENANCE_ERROR(row)


def _h08_component_passes(record: dict[str, Any]) -> bool:
    if int(record.get("concept_link_count") or 0) < 2:
        return False
    return bool(_ORIGINAL_H08_COMPONENT_GATE(record))


def _derive_world_model_components(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    progress_factory: Any | None = None,
    max_world_model_family_links: int = 50,
) -> dict[str, Any]:
    # Run the legacy derivation first so all established source tables and
    # persistent state are refreshed, then replace single-concept M5 rows with
    # canonical relational components.
    _ORIGINAL_DERIVE_WORLD_MODELS(
        state_conn,
        graph_conn,
        progress_factory,
        max_world_model_family_links,
    )
    return _build_relational_world_models(
        state_conn,
        max_world_model_family_links=max_world_model_family_links,
    )


def _build_relational_world_models(
    state_conn: sqlite3.Connection,
    *,
    max_world_model_family_links: int = 50,
) -> dict[str, Any]:
    state_conn.row_factory = sqlite3.Row
    concept_rows = [
        dict(row)
        for row in state_conn.execute(
            """
            SELECT c.concept_signature, c.promotion_score,
                   c.first_seen_global_step, c.last_seen_global_step,
                   COALESCE(s.currently_promoted, c.is_promoted, 0) AS promoted
            FROM concept_candidates AS c
            LEFT JOIN concept_promotion_state AS s
              ON s.concept_signature=c.concept_signature
            WHERE COALESCE(s.currently_promoted, c.is_promoted, 0)=1
            ORDER BY COALESCE(c.promotion_score,0) DESC, c.concept_signature
            """
        ).fetchall()
    ]
    from v6 import higher_order_substrate as substrate

    concept_links = substrate._links_by_signature(
        state_conn, "concept_links", "concept_signature"
    )
    candidates: list[tuple[tuple[int, int, float, str, str], dict[str, Any]]] = []
    for left, right in combinations(concept_rows, 2):
        left_id = str(left["concept_signature"])
        right_id = str(right["concept_signature"])
        ll = concept_links.get(left_id, {})
        rr = concept_links.get(right_id, {})
        shared_roles = set(ll.get("role", set())) & set(rr.get("role", set()))
        shared_families = set(ll.get("family", set())) & set(rr.get("family", set()))
        shared_contexts = set(ll.get("context", set())) & set(rr.get("context", set()))
        relation_strength = (
            3 * len(shared_roles)
            + 2 * len(shared_families)
            + len(shared_contexts)
        )
        if relation_strength <= 0:
            continue
        unions = {
            kind: set(ll.get(kind, set())) | set(rr.get(kind, set()))
            for kind in ("role", "carrier", "family", "context", "game")
        }
        if len(unions["family"]) < 2 or len(unions["carrier"]) < 2:
            continue
        score = float(left.get("promotion_score") or 0.0) + float(
            right.get("promotion_score") or 0.0
        )
        candidates.append(
            (
                (
                    relation_strength,
                    len(unions["context"]) + len(unions["game"]),
                    score,
                    left_id,
                    right_id,
                ),
                {"left": left, "right": right, "links": unions},
            )
        )
    candidates.sort(
        key=lambda item: (
            -item[0][0], -item[0][1], -item[0][2], item[0][3], item[0][4]
        )
    )
    selected = [item[1] for item in candidates[:MAX_RELATIONAL_COMPONENTS]]

    # Single-concept rows are not canonical M5 and must not survive the v6.3
    # derivation. Prediction events are intentionally preserved across epochs.
    state_conn.execute("DELETE FROM world_model_family_links")
    state_conn.execute("DELETE FROM world_model_links")
    state_conn.execute("DELETE FROM world_model_components")

    current_step = _current_evidence_step(state_conn)
    component_count = 0
    coherent_count = 0
    for spec in selected:
        concepts = sorted(
            [str(spec["left"]["concept_signature"]), str(spec["right"]["concept_signature"])]
        )
        links = spec["links"]
        signature = "wm:rel:" + sha1(
            json.dumps(concepts, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        _match_world_model_predictions(state_conn, signature)
        prediction = _world_model_prediction_metrics(state_conn, signature)
        if current_step is not None:
            _issue_world_model_prediction(
                state_conn,
                signature=signature,
                prediction_step=current_step,
                families=sorted(str(x) for x in links["family"]),
                contexts=sorted(str(x) for x in links["context"]),
                games=sorted(str(x) for x in links["game"]),
            )
            prediction = _world_model_prediction_metrics(state_conn, signature)

        role_count = len(links["role"])
        family_count = len(links["family"])
        carrier_count = len(links["carrier"])
        context_count = len(links["context"])
        game_count = len(links["game"])
        node_count = 2 + role_count + family_count + carrier_count + context_count + game_count
        explanatory = float(family_count + carrier_count + role_count) / max(1, node_count)
        scope = min(1.0, float(context_count + game_count) / 5.0)
        structural = min(
            1.0,
            0.4
            + 0.1 * min(2, role_count)
            + 0.1 * min(2, family_count)
            + 0.1 * min(2, carrier_count),
        )
        functional = max(
            0.0,
            min(
                1.0,
                0.45 * float(prediction["accuracy"])
                + 0.35 * max(0.0, float(prediction["gain"]))
                + 0.20 * scope,
            ),
        )
        coherence = max(0.0, min(1.0, 0.4 * structural + 0.6 * functional))
        is_coherent = int(
            prediction["matched"] > 0
            and role_count >= 1
            and family_count >= 2
            and carrier_count >= 2
            and (context_count >= 3 or game_count >= 2)
            and prediction["gain"] > 0.0
            and coherence >= 0.55
        )
        first_seen_values = [
            value
            for value in (
                spec["left"].get("first_seen_global_step"),
                spec["right"].get("first_seen_global_step"),
            )
            if value is not None
        ]
        last_seen_values = [
            value
            for value in (
                spec["left"].get("last_seen_global_step"),
                spec["right"].get("last_seen_global_step"),
            )
            if value is not None
        ]
        first_seen = max(int(x) for x in first_seen_values) if first_seen_values else None
        last_seen = max(int(x) for x in last_seen_values) if last_seen_values else None
        state_conn.execute(
            """
            INSERT INTO world_model_components (
                component_signature, component_type, node_count, edge_count,
                linked_concept_count, linked_role_count, linked_family_count,
                linked_carrier_count, cross_context_count, cross_game_count,
                explanatory_coverage, prediction_support_count,
                contradiction_coverage_count, coherence_score, candidate_only,
                predicted_outcome_count, predicted_outcome_count_is_proxy,
                first_seen_global_step, last_seen_global_step, is_coherent,
                structural_prediction_support_count, observed_outcome_count,
                correct_prediction_count, prediction_error_count,
                prediction_evidence_status, baseline_prediction_score,
                component_prediction_score, heldout_prediction_gain,
                matched_prediction_event_count, unmatched_prediction_event_count,
                structural_coherence_score, functional_coherence_score,
                combined_coherence_score, candidate_family_link_count,
                retained_family_link_count, dropped_family_link_count,
                family_links_dropped_low_support, family_links_dropped_limit
            ) VALUES (?, 'relational_concept_component', ?, ?, 2, ?, ?, ?, ?, ?, ?,
                      0, 0, ?, 0, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, 0, 0, 0)
            """,
            (
                signature,
                node_count,
                role_count + family_count + carrier_count + context_count + game_count + 1,
                role_count,
                family_count,
                carrier_count,
                context_count,
                game_count,
                explanatory,
                coherence,
                prediction["matched"],
                first_seen,
                last_seen,
                is_coherent,
                family_count + role_count,
                prediction["matched"],
                prediction["correct"],
                max(0, prediction["matched"] - prediction["correct"]),
                "verified" if prediction["matched"] else ("proxy" if prediction["unmatched"] else "missing"),
                prediction["baseline"],
                prediction["component"],
                prediction["gain"],
                prediction["matched"],
                prediction["unmatched"],
                structural,
                functional,
                coherence,
                family_count,
                min(family_count, int(max_world_model_family_links)),
            ),
        )
        for concept in concepts:
            substrate._insert_link(
                state_conn, "world_model_links", "component_signature",
                signature, "concept", concept, 1, first_seen, last_seen,
            )
        for kind in ("role", "carrier", "family", "context", "game"):
            for value in sorted(str(x) for x in links[kind]):
                substrate._insert_link(
                    state_conn, "world_model_links", "component_signature",
                    signature, kind, value, 1, first_seen, last_seen,
                )
        for family in sorted(str(x) for x in links["family"])[: int(max_world_model_family_links)]:
            support_row = state_conn.execute(
                "SELECT COALESCE(SUM(support_count),0) FROM family_members WHERE family_signature=?",
                (family,),
            ).fetchone()
            state_conn.execute(
                """
                INSERT OR REPLACE INTO world_model_family_links (
                    component_signature, family_signature, family_link_support_count,
                    family_link_role_count, family_link_event_count,
                    family_link_prediction_gain, family_link_provenance_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signature,
                    family,
                    int((support_row or [0])[0] or 0),
                    sum(
                        1 for role in links["role"]
                        if family in substrate._links_by_signature(
                            state_conn, "role_links", "role_signature"
                        ).get(str(role), {}).get("family", set())
                    ),
                    _future_event_count_for_family(state_conn, family),
                    0.0,
                    "verified" if _future_event_count_for_family(state_conn, family) else "proxy",
                ),
            )
        state_conn.execute(
            """
            INSERT INTO world_model_component_state (
                component_signature, historically_coherent, currently_coherent,
                first_coherent_global_step, last_validated_global_step,
                consecutive_validation_failures, validation_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(component_signature) DO UPDATE SET
                historically_coherent=MAX(world_model_component_state.historically_coherent, excluded.historically_coherent),
                currently_coherent=excluded.currently_coherent,
                first_coherent_global_step=COALESCE(world_model_component_state.first_coherent_global_step, excluded.first_coherent_global_step),
                last_validated_global_step=excluded.last_validated_global_step,
                consecutive_validation_failures=CASE WHEN excluded.currently_coherent=1 THEN 0 ELSE world_model_component_state.consecutive_validation_failures+1 END,
                validation_status=excluded.validation_status,
                updated_at=excluded.updated_at
            """,
            (
                signature,
                is_coherent,
                is_coherent,
                first_seen if is_coherent else None,
                last_seen,
                0 if is_coherent else 1,
                "passed" if is_coherent else "awaiting_heldout_prediction" if not prediction["matched"] else "failed",
            ),
        )
        component_count += 1
        coherent_count += is_coherent

    first_component = state_conn.execute(
        "SELECT MIN(first_seen_global_step) FROM world_model_components"
    ).fetchone()[0]
    first_coherent = state_conn.execute(
        "SELECT MIN(first_seen_global_step) FROM world_model_components WHERE COALESCE(is_coherent,0)=1"
    ).fetchone()[0]
    substrate._write_milestone(state_conn, "first_world_model_component_step", first_component, None)
    substrate._write_milestone(state_conn, "first_coherent_world_model_step", first_coherent, None)
    state_conn.commit()
    return {
        "world_model_component_count": component_count,
        "coherent_world_model_component_count": coherent_count,
        "candidate_only_world_model_component_count": component_count - coherent_count,
        "world_model_semantics_version": "v63_relational_multiconcept_v1",
    }


def _current_evidence_step(conn: sqlite3.Connection) -> int | None:
    values: list[int] = []
    for table, column in (
        ("future_option_events", "last_seen_global_step"),
        ("stable_contingencies", "last_seen_global_step"),
        ("concept_candidates", "last_seen_global_step"),
    ):
        try:
            row = conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
        except sqlite3.Error:
            continue
        if row is not None and row[0] is not None:
            values.append(int(row[0]))
    return max(values) if values else None


def _future_event_columns(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(future_option_events)").fetchall()
    }


def _future_event_family_expr(conn: sqlite3.Connection) -> str:
    columns = _future_event_columns(conn)
    if "source_family_id" in columns:
        return "COALESCE(source_family_id, CASE WHEN owner_type='family' THEN owner_key END)"
    return "CASE WHEN owner_type='family' THEN owner_key END"


def _future_event_count_for_family(conn: sqlite3.Connection, family: str) -> int:
    expr = _future_event_family_expr(conn)
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM future_option_events WHERE {expr}=?",
            (family,),
        ).fetchone()
        return int((row or [0])[0] or 0)
    except sqlite3.Error:
        return 0


def _issue_world_model_prediction(
    conn: sqlite3.Connection,
    *,
    signature: str,
    prediction_step: int,
    families: list[str],
    contexts: list[str],
    games: list[str],
) -> None:
    if not families:
        return
    existing = conn.execute(
        "SELECT 1 FROM world_model_prediction_events "
        "WHERE component_signature=? AND observed_event_id IS NULL",
        (signature,),
    ).fetchone()
    if existing is not None:
        return
    family_expr = _future_event_family_expr(conn)
    best_context: str | None = None
    predicted_family = families[0]
    baseline = 0.0
    best_count = -1
    for context in contexts or [None]:
        params: list[Any] = [int(prediction_step)]
        where = "first_seen_global_step <= ?"
        if context is not None:
            where += " AND context_key=?"
            params.append(context)
        placeholders = ",".join("?" for _ in families)
        where += f" AND {family_expr} IN ({placeholders})"
        params.extend(families)
        try:
            rows = conn.execute(
                f"SELECT {family_expr} AS family, COUNT(*) AS n "
                f"FROM future_option_events WHERE {where} "
                "GROUP BY family ORDER BY n DESC, family ASC",
                params,
            ).fetchall()
        except sqlite3.Error:
            rows = []
        total = sum(int(row[1] or 0) for row in rows)
        if rows and int(rows[0][1] or 0) > best_count:
            best_count = int(rows[0][1] or 0)
            predicted_family = str(rows[0][0])
            best_context = context
            baseline = float(best_count / max(1, total))
    event_id = "wm-pred:" + sha1(
        json.dumps(
            [signature, int(prediction_step), best_context, predicted_family],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    conn.execute(
        """
        INSERT OR IGNORE INTO world_model_prediction_events (
            prediction_event_id, component_signature, prediction_global_step,
            predicted_family, predicted_effect, predicted_outcome,
            game_key, context_key, action_key, baseline_prediction_score,
            component_prediction_score, provenance_status
        ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, NULL, 'prospective')
        """,
        (
            event_id,
            signature,
            int(prediction_step),
            predicted_family,
            f"next_family:{predicted_family}",
            games[0] if len(games) == 1 and not games[0].startswith("surrogate_game:") else None,
            best_context,
            baseline,
        ),
    )


def _match_world_model_predictions(conn: sqlite3.Connection, signature: str) -> None:
    family_expr = _future_event_family_expr(conn)
    rows = conn.execute(
        """
        SELECT prediction_event_id, prediction_global_step, predicted_family,
               game_key, context_key
        FROM world_model_prediction_events
        WHERE component_signature=? AND observed_event_id IS NULL
        ORDER BY prediction_global_step ASC, prediction_event_id ASC
        """,
        (signature,),
    ).fetchall()
    for row in rows:
        where = ["first_seen_global_step > ?"]
        params: list[Any] = [int(row[1])]
        if row[4]:
            where.append("context_key=?")
            params.append(str(row[4]))
        if row[3]:
            where.append("game=?")
            params.append(str(row[3]))
        try:
            observed = conn.execute(
                f"SELECT event_id, first_seen_global_step, {family_expr} AS family, motif_type "
                "FROM future_option_events WHERE " + " AND ".join(where)
                + " ORDER BY first_seen_global_step ASC, event_id ASC LIMIT 1",
                params,
            ).fetchone()
        except sqlite3.Error:
            observed = None
        if observed is None:
            continue
        observed_family = None if observed[2] is None else str(observed[2])
        correct = int(observed_family == str(row[2]))
        conn.execute(
            """
            UPDATE world_model_prediction_events
            SET observed_event_id=?, observed_global_step=?, observed_family=?,
                observed_effect=?, prediction_correct=?, component_prediction_score=?,
                provenance_status='verified'
            WHERE prediction_event_id=?
            """,
            (
                str(observed[0]),
                int(observed[1]),
                observed_family,
                None if observed[3] is None else str(observed[3]),
                correct,
                float(correct),
                str(row[0]),
            ),
        )


def _world_model_prediction_metrics(
    conn: sqlite3.Connection, signature: str
) -> dict[str, float | int | None]:
    rows = conn.execute(
        """
        SELECT prediction_correct, baseline_prediction_score,
               component_prediction_score, observed_event_id
        FROM world_model_prediction_events
        WHERE component_signature=?
        """,
        (signature,),
    ).fetchall()
    matched_rows = [row for row in rows if row[3] is not None]
    unmatched = len(rows) - len(matched_rows)
    correct = sum(int(row[0] or 0) for row in matched_rows)
    matched = len(matched_rows)
    accuracy = float(correct / matched) if matched else 0.0
    baselines = [float(row[1]) for row in matched_rows if row[1] is not None]
    components = [float(row[2]) for row in matched_rows if row[2] is not None]
    baseline = sum(baselines) / len(baselines) if baselines else None
    component = sum(components) / len(components) if components else None
    gain = (
        float(component - baseline)
        if component is not None and baseline is not None
        else 0.0
    )
    return {
        "matched": matched,
        "unmatched": unmatched,
        "correct": correct,
        "accuracy": accuracy,
        "baseline": baseline,
        "component": component,
        "gain": gain,
    }

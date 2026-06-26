from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from v6.memory.compact_memory import ensure_memory_layout


ROLE_CLEAR_TABLES = (
    "role_neighborhood_signatures",
    "role_candidates",
    "role_links",
    "role_transfer_attempts",
    "concept_candidates",
    "concept_links",
    "world_model_components",
    "world_model_links",
    "higher_order_milestones",
)


@dataclass(frozen=True)
class CarrierNeighborhood:
    carrier_signature: str
    carrier_source: str
    support_count: int
    stability_score: float
    is_emergent: bool
    first_seen_global_step: int | None
    last_seen_global_step: int | None
    role_signature: str
    role_type: str
    tokens: tuple[str, ...]
    families: tuple[str, ...]
    contexts: tuple[str, ...]
    contingencies: tuple[str, ...]
    games: tuple[str, ...]


def derive_higher_order_memory(
    *,
    memory_dir: Path,
    run_dir: Path | None = None,
    max_carriers: int = 100_000,
    max_roles: int = 50_000,
    max_transfer_attempts: int = 250_000,
) -> dict[str, Any]:
    del run_dir
    paths = ensure_memory_layout(memory_dir)
    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
    ):
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        for table in ROLE_CLEAR_TABLES:
            state_conn.execute(f"DELETE FROM {table}")
        role_summary = derive_role_candidates(state_conn, graph_conn, max_carriers=max_carriers, max_roles=max_roles)
        transfer_summary = derive_role_transfer_attempts(state_conn, graph_conn, max_transfer_attempts=max_transfer_attempts)
        concept_summary = derive_concept_candidates(state_conn)
        world_summary = derive_world_model_components(state_conn, graph_conn)
        summary = {
            **role_summary,
            **transfer_summary,
            **concept_summary,
            **world_summary,
        }
        state_conn.commit()
        graph_conn.commit()
        return summary


def derive_role_candidates(state_conn: sqlite3.Connection, graph_conn: sqlite3.Connection, max_carriers: int, max_roles: int) -> dict[str, Any]:
    family_rows = state_conn.execute(
        """
        SELECT canonical_signature, effect_type, action_group, polarity, support_count, member_count
        FROM transformation_families
        ORDER BY canonical_signature ASC
        """
    ).fetchall()
    family_meta = {str(row["canonical_signature"]): dict(row) for row in family_rows}
    carrier_rows = state_conn.execute(
        """
        SELECT carrier_signature, carrier_source, support_count, stability_score, is_emergent,
               first_seen_global_step, last_seen_global_step
        FROM carrier_candidates
        WHERE carrier_source != 'context_action_fallback'
          AND (COALESCE(is_emergent, 0) = 1 OR COALESCE(support_count, 0) >= 3)
        ORDER BY carrier_signature ASC
        LIMIT ?
        """,
        (int(max_carriers),),
    ).fetchall()
    if not carrier_rows:
        _write_milestone(state_conn, "first_role_candidate_step", None, None)
        _write_milestone(state_conn, "first_emergent_role_step", None, None)
        return {
            "role_candidate_count": 0,
            "emergent_role_count": 0,
            "stable_role_count": 0,
            "role_neighborhood_count": 0,
        }

    link_rows = state_conn.execute(
        """
        SELECT carrier_signature, linked_type, linked_key
        FROM carrier_links
        ORDER BY carrier_signature ASC, linked_type ASC, linked_key ASC
        """
    ).fetchall()
    links_by_carrier: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in link_rows:
        carrier = str(row["carrier_signature"])
        linked_type = str(row["linked_type"])
        linked_key = row["linked_key"]
        if linked_key not in (None, ""):
            links_by_carrier[carrier][linked_type].add(str(linked_key))

    graph_nodes = {
        str(row["node_id"]): str(row["node_type"] or "")
        for row in graph_conn.execute("SELECT node_id, node_type FROM graph_nodes").fetchall()
    }
    edge_out_rows = graph_conn.execute(
        "SELECT source_node_id, target_node_id, edge_type FROM graph_edges ORDER BY source_node_id ASC, edge_type ASC, target_node_id ASC"
    ).fetchall()
    edge_in_rows = graph_conn.execute(
        "SELECT target_node_id, source_node_id, edge_type FROM graph_edges ORDER BY target_node_id ASC, edge_type ASC, source_node_id ASC"
    ).fetchall()
    edges_out_by_source: dict[str, list[sqlite3.Row]] = defaultdict(list)
    edges_in_by_target: dict[str, list[sqlite3.Row]] = defaultdict(list)
    context_games: dict[str, set[str]] = defaultdict(set)
    for row in edge_out_rows:
        source = str(row["source_node_id"])
        target = str(row["target_node_id"])
        edges_out_by_source[source].append(row)
        if source.startswith("game:") and str(row["edge_type"]) == "observed_in" and target.startswith("context:"):
            context_games[target[len("context:"):]].add(source[len("game:"):])
    for row in edge_in_rows:
        edges_in_by_target[str(row["target_node_id"])].append(row)

    neighborhoods: list[CarrierNeighborhood] = []
    for row in carrier_rows:
        carrier_signature = str(row["carrier_signature"])
        carrier_links = links_by_carrier.get(carrier_signature, {})
        families = tuple(sorted(carrier_links.get("family", set())))
        contexts = tuple(sorted(carrier_links.get("context", set())))
        contingencies = tuple(sorted(carrier_links.get("contingency", set())))
        games = tuple(sorted({game for context_key in contexts for game in context_games.get(context_key, set())}))
        carrier_node_id = f"carrier:{carrier_signature}"
        out_edges = list(edges_out_by_source.get(carrier_node_id, []))
        in_edges = list(edges_in_by_target.get(carrier_node_id, []))
        tokens = sorted(
            _build_role_tokens(
                families=families,
                contexts=contexts,
                games=games,
                out_edges=out_edges,
                in_edges=in_edges,
                graph_nodes=graph_nodes,
                family_meta=family_meta,
            )
        )
        role_signature = _role_signature(tokens)
        role_type = _role_type(tokens, len(families), len(contexts), len(games))
        state_conn.execute(
            """
            INSERT INTO role_neighborhood_signatures (
                carrier_signature, role_signature, role_type, token_json, linked_family_count, linked_context_count,
                linked_game_count, in_edge_count, out_edge_count, first_seen_global_step, last_seen_global_step, stability_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                carrier_signature,
                role_signature,
                role_type,
                json.dumps(tokens, separators=(",", ":"), sort_keys=True),
                len(families),
                len(contexts),
                len(games),
                len(in_edges),
                len(out_edges),
                row["first_seen_global_step"],
                row["last_seen_global_step"],
                float(row["stability_score"] or 0.0),
            ),
        )
        neighborhoods.append(
            CarrierNeighborhood(
                carrier_signature=carrier_signature,
                carrier_source=str(row["carrier_source"] or ""),
                support_count=int(row["support_count"] or 0),
                stability_score=float(row["stability_score"] or 0.0),
                is_emergent=bool(int(row["is_emergent"] or 0)),
                first_seen_global_step=None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"]),
                last_seen_global_step=None if row["last_seen_global_step"] is None else int(row["last_seen_global_step"]),
                role_signature=role_signature,
                role_type=role_type,
                tokens=tuple(tokens),
                families=families,
                contexts=contexts,
                contingencies=contingencies,
                games=games,
            )
        )

    grouped: dict[str, list[CarrierNeighborhood]] = defaultdict(list)
    for entry in neighborhoods:
        grouped[entry.role_signature].append(entry)
    selected_role_signatures = sorted(grouped)[: int(max_roles)]
    emergent_role_count = 0
    stable_role_count = 0
    first_role_candidate_step: int | None = None
    first_emergent_role_step: int | None = None
    for role_signature in selected_role_signatures:
        items = sorted(grouped[role_signature], key=lambda item: item.carrier_signature)
        families = sorted({family for item in items for family in item.families})
        contexts = sorted({context for item in items for context in item.contexts})
        games = sorted({game for item in items for game in item.games})
        support_count = sum(item.support_count for item in items)
        linked_carrier_count = len(items)
        linked_family_count = len(families)
        linked_context_count = len(contexts)
        cross_game_count = len(games)
        cross_context_count = len(contexts)
        first_seen = min((item.first_seen_global_step for item in items if item.first_seen_global_step is not None), default=None)
        last_seen = max((item.last_seen_global_step for item in items if item.last_seen_global_step is not None), default=None)
        role_stability_score = (
            0.25 * min(1.0, linked_carrier_count / 3.0)
            + 0.25 * min(1.0, linked_family_count / 3.0)
            + 0.25 * min(1.0, cross_context_count / 3.0)
            + 0.25 * min(1.0, cross_game_count / 2.0)
        )
        is_emergent = int(
            linked_carrier_count >= 2
            and linked_family_count >= 1
            and (cross_context_count >= 2 or cross_game_count >= 2)
            and role_stability_score >= 0.50
        )
        role_type = items[0].role_type
        state_conn.execute(
            """
            INSERT INTO role_candidates (
                role_signature, role_type, support_count, linked_carrier_count, linked_family_count,
                linked_context_count, cross_game_count, cross_context_count, first_seen_global_step,
                last_seen_global_step, role_stability_score, is_emergent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                role_signature,
                role_type,
                support_count,
                linked_carrier_count,
                linked_family_count,
                linked_context_count,
                cross_game_count,
                cross_context_count,
                first_seen,
                last_seen,
                role_stability_score,
                is_emergent,
            ),
        )
        if role_stability_score >= 0.50:
            stable_role_count += 1
        if is_emergent:
            emergent_role_count += 1
        if first_seen is not None:
            first_role_candidate_step = first_seen if first_role_candidate_step is None else min(first_role_candidate_step, first_seen)
            if is_emergent:
                first_emergent_role_step = first_seen if first_emergent_role_step is None else min(first_emergent_role_step, first_seen)
        for item in items:
            _insert_link(
                state_conn,
                table="role_links",
                signature_column="role_signature",
                signature=role_signature,
                linked_type="carrier",
                linked_key=item.carrier_signature,
                support_count=1,
                first_seen=first_seen,
                last_seen=last_seen,
            )
        for family in families:
            _insert_link(state_conn, "role_links", "role_signature", role_signature, "family", family, 1, first_seen, last_seen)
        for context in contexts:
            _insert_link(state_conn, "role_links", "role_signature", role_signature, "context", context, 1, first_seen, last_seen)
        for game in games:
            _insert_link(state_conn, "role_links", "role_signature", role_signature, "game", game, 1, first_seen, last_seen)

    _write_milestone(state_conn, "first_role_candidate_step", first_role_candidate_step, None)
    _write_milestone(state_conn, "first_emergent_role_step", first_emergent_role_step, None)
    return {
        "role_candidate_count": len(selected_role_signatures),
        "emergent_role_count": emergent_role_count,
        "stable_role_count": stable_role_count,
        "role_neighborhood_count": len(neighborhoods),
    }


def derive_role_transfer_attempts(state_conn: sqlite3.Connection, graph_conn: sqlite3.Connection, max_transfer_attempts: int) -> dict[str, Any]:
    role_rows = state_conn.execute(
        """
        SELECT carrier_signature, role_signature, token_json, first_seen_global_step, last_seen_global_step
        FROM role_neighborhood_signatures
        ORDER BY role_signature ASC, carrier_signature ASC
        """
    ).fetchall()
    if not role_rows:
        _write_milestone(state_conn, "first_role_transfer_attempt_step", None, None)
        _write_milestone(state_conn, "first_role_transfer_success_step", None, None)
        return {
            "transfer_attempt_count": 0,
            "successful_transfer_count": 0,
            "successful_role_count": 0,
        }
    links = state_conn.execute(
        """
        SELECT role_signature, linked_type, linked_key
        FROM role_links
        ORDER BY role_signature ASC, linked_type ASC, linked_key ASC
        """
    ).fetchall()
    role_scopes: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in links:
        if row["linked_key"] not in (None, ""):
            role_scopes[str(row["role_signature"])][str(row["linked_type"])].add(str(row["linked_key"]))
    carrier_to_role: dict[str, sqlite3.Row] = {str(row["carrier_signature"]): row for row in role_rows}
    del carrier_to_role
    carriers_by_role: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in role_rows:
        carriers_by_role[str(row["role_signature"])].append(row)
    carrier_link_rows = state_conn.execute(
        """
        SELECT carrier_signature, linked_type, linked_key
        FROM carrier_links
        ORDER BY carrier_signature ASC, linked_type ASC, linked_key ASC
        """
    ).fetchall()
    carrier_contexts: dict[str, set[str]] = defaultdict(set)
    for row in carrier_link_rows:
        if str(row["linked_type"]) == "context" and row["linked_key"] not in (None, ""):
            carrier_contexts[str(row["carrier_signature"])].add(str(row["linked_key"]))
    graph_edge_rows = graph_conn.execute(
        """
        SELECT source_node_id, target_node_id, edge_type
        FROM graph_edges
        WHERE edge_type = 'observed_in'
        ORDER BY source_node_id ASC, target_node_id ASC
        """
    ).fetchall()
    context_games: dict[str, set[str]] = defaultdict(set)
    for row in graph_edge_rows:
        source = str(row["source_node_id"])
        target = str(row["target_node_id"])
        if source.startswith("game:") and target.startswith("context:"):
            context_games[target[len("context:"):]].add(source[len("game:"):])
    attempts: list[dict[str, Any]] = []
    for role_signature in sorted(carriers_by_role):
        carrier_rows = sorted(carriers_by_role[role_signature], key=lambda row: str(row["carrier_signature"]))
        role_games = sorted(role_scopes.get(role_signature, {}).get("game", set()))
        del role_games
        role_contexts = sorted(role_scopes.get(role_signature, {}).get("context", set()))
        del role_contexts
        for carrier_row in carrier_rows:
            target_carrier_signature = str(carrier_row["carrier_signature"])
            target_tokens = set(_load_token_json(carrier_row["token_json"]))
            target_context_values = sorted(carrier_contexts.get(target_carrier_signature, set()))
            carrier_games = sorted({game for context in target_context_values for game in context_games.get(context, set())})
            first_seen = None if carrier_row["first_seen_global_step"] is None else int(carrier_row["first_seen_global_step"])
            last_seen = None if carrier_row["last_seen_global_step"] is None else int(carrier_row["last_seen_global_step"])
            for target_game in carrier_games:
                source_rows = [
                    row for row in carrier_rows
                    if target_game not in {game for context in carrier_contexts.get(str(row["carrier_signature"]), set()) for game in context_games.get(context, set())}
                ]
                source_rows = [row for row in source_rows if str(row["carrier_signature"]) != target_carrier_signature]
                if source_rows:
                    attempts.append(
                        _build_transfer_attempt(
                            role_signature=role_signature,
                            transfer_kind="cross_game",
                            source_scope_type="not_game",
                            source_scope_key=target_game,
                            target_scope_type="game",
                            target_scope_key=target_game,
                            target_carrier_signature=target_carrier_signature,
                            target_tokens=target_tokens,
                            source_rows=source_rows,
                            observed_role_signature=role_signature,
                            first_seen=first_seen,
                            last_seen=last_seen,
                        )
                    )
            for target_context in target_context_values:
                source_rows = [
                    row for row in carrier_rows
                    if target_context not in carrier_contexts.get(str(row["carrier_signature"]), set())
                ]
                source_rows = [row for row in source_rows if str(row["carrier_signature"]) != target_carrier_signature]
                if source_rows:
                    attempts.append(
                        _build_transfer_attempt(
                            role_signature=role_signature,
                            transfer_kind="cross_context",
                            source_scope_type="not_context",
                            source_scope_key=target_context,
                            target_scope_type="context",
                            target_scope_key=target_context,
                            target_carrier_signature=target_carrier_signature,
                            target_tokens=target_tokens,
                            source_rows=source_rows,
                            observed_role_signature=role_signature,
                            first_seen=first_seen,
                            last_seen=last_seen,
                        )
                    )
    attempts = sorted(
        attempts,
        key=lambda item: (
            str(item["role_signature"]),
            str(item["target_scope_key"]),
            str(item["target_carrier_signature"]),
            str(item["transfer_kind"]),
        ),
    )[: int(max_transfer_attempts)]
    successful_roles: set[str] = set()
    first_attempt_step: int | None = None
    first_success_step: int | None = None
    success_count = 0
    for item in attempts:
        state_conn.execute(
            """
            INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature,
                observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason,
                first_seen_global_step, last_seen_global_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["attempt_id"],
                item["role_signature"],
                item["transfer_kind"],
                item["source_scope_type"],
                item["source_scope_key"],
                item["target_scope_type"],
                item["target_scope_key"],
                item["target_carrier_signature"],
                item["predicted_role_signature"],
                item["observed_role_signature"],
                item["similarity_score"],
                item["transfer_score"],
                item["reuse_success"],
                item["failure_reason"],
                item["first_seen_global_step"],
                item["last_seen_global_step"],
            ),
        )
        if item["first_seen_global_step"] is not None:
            first_attempt_step = item["first_seen_global_step"] if first_attempt_step is None else min(first_attempt_step, int(item["first_seen_global_step"]))
        if int(item["reuse_success"]) == 1:
            success_count += 1
            successful_roles.add(str(item["role_signature"]))
            if item["first_seen_global_step"] is not None:
                first_success_step = item["first_seen_global_step"] if first_success_step is None else min(first_success_step, int(item["first_seen_global_step"]))
    _write_milestone(state_conn, "first_role_transfer_attempt_step", first_attempt_step, None)
    _write_milestone(state_conn, "first_role_transfer_success_step", first_success_step, None)
    return {
        "transfer_attempt_count": len(attempts),
        "successful_transfer_count": success_count,
        "successful_role_count": len(successful_roles),
    }


def derive_concept_candidates(state_conn: sqlite3.Connection) -> dict[str, Any]:
    role_rows = state_conn.execute(
        """
        SELECT role_signature, role_type, support_count, linked_carrier_count, linked_family_count,
               linked_context_count, cross_game_count, cross_context_count, first_seen_global_step,
               last_seen_global_step, role_stability_score, is_emergent
        FROM role_candidates
        WHERE COALESCE(is_emergent, 0) = 1 OR COALESCE(role_stability_score, 0.0) >= 0.50
        ORDER BY role_signature ASC
        """
    ).fetchall()
    if not role_rows:
        _write_milestone(state_conn, "first_concept_candidate_step", None, None)
        _write_milestone(state_conn, "first_promoted_concept_step", None, None)
        return {"concept_candidate_count": 0, "promoted_concept_count": 0}
    role_link_rows = state_conn.execute(
        "SELECT role_signature, linked_type, linked_key FROM role_links ORDER BY role_signature ASC, linked_type ASC, linked_key ASC"
    ).fetchall()
    role_links: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in role_link_rows:
        if row["linked_key"] not in (None, ""):
            role_links[str(row["role_signature"])][str(row["linked_type"])].add(str(row["linked_key"]))
    neighborhood_rows = state_conn.execute(
        "SELECT role_signature, token_json FROM role_neighborhood_signatures ORDER BY role_signature ASC, carrier_signature ASC"
    ).fetchall()
    tokens_by_role: dict[str, set[str]] = defaultdict(set)
    for row in neighborhood_rows:
        tokens_by_role[str(row["role_signature"])].update(_load_token_json(row["token_json"]))
    transfer_rows = state_conn.execute(
        """
        SELECT role_signature, reuse_success
        FROM role_transfer_attempts
        ORDER BY role_signature ASC, attempt_id ASC
        """
    ).fetchall()
    success_by_role: dict[str, int] = defaultdict(int)
    for row in transfer_rows:
        if int(row["reuse_success"] or 0) == 1:
            success_by_role[str(row["role_signature"])] += 1
    concept_groups: dict[str, dict[str, Any]] = {}
    for row in role_rows:
        role_signature = str(row["role_signature"])
        links = role_links.get(role_signature, {})
        if not links.get("carrier") or not links.get("family"):
            continue
        concept_tokens = _concept_tokens(str(row["role_type"] or "unknown"), sorted(tokens_by_role.get(role_signature, set())))
        concept_signature = _concept_signature(concept_tokens)
        group = concept_groups.setdefault(
            concept_signature,
            {
                "concept_signature": concept_signature,
                "concept_type": str(row["role_type"] or "unknown"),
                "roles": set(),
                "carriers": set(),
                "families": set(),
                "contexts": set(),
                "games": set(),
                "support_count": 0,
                "transfer_success_count": 0,
                "first_seen": None,
                "last_seen": None,
            },
        )
        group["roles"].add(role_signature)
        group["carriers"].update(links.get("carrier", set()))
        group["families"].update(links.get("family", set()))
        group["contexts"].update(links.get("context", set()))
        group["games"].update(links.get("game", set()))
        group["support_count"] += int(row["support_count"] or 0)
        group["transfer_success_count"] += int(success_by_role.get(role_signature, 0))
        first_seen = None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"])
        last_seen = None if row["last_seen_global_step"] is None else int(row["last_seen_global_step"])
        group["first_seen"] = first_seen if group["first_seen"] is None else min(group["first_seen"], first_seen or group["first_seen"])
        group["last_seen"] = last_seen if group["last_seen"] is None else max(group["last_seen"], last_seen or group["last_seen"])
    promoted_concepts = 0
    first_concept_candidate_step: int | None = None
    first_promoted_concept_step: int | None = None
    for concept_signature in sorted(concept_groups):
        group = concept_groups[concept_signature]
        linked_role_count = len(group["roles"])
        linked_carrier_count = len(group["carriers"])
        linked_family_count = len(group["families"])
        cross_context_count = len(group["contexts"])
        cross_game_count = len(group["games"])
        transfer_success_count = int(group["transfer_success_count"])
        compression_gain = float(linked_carrier_count / max(1, linked_role_count))
        explanatory_reach = float(linked_family_count + cross_context_count + (2 * cross_game_count) + transfer_success_count)
        promotion_score = (
            0.25 * min(1.0, compression_gain / 2.0)
            + 0.30 * min(1.0, transfer_success_count / 3.0)
            + 0.20 * min(1.0, cross_context_count / 3.0)
            + 0.15 * min(1.0, cross_game_count / 2.0)
            + 0.10 * min(1.0, linked_family_count / 3.0)
        )
        is_promoted = int(
            transfer_success_count >= 2
            and compression_gain >= 1.50
            and (cross_context_count >= 2 or cross_game_count >= 1)
            and promotion_score >= 0.55
        )
        if is_promoted:
            promoted_concepts += 1
        first_seen = group["first_seen"]
        last_seen = group["last_seen"]
        if first_seen is not None:
            first_concept_candidate_step = first_seen if first_concept_candidate_step is None else min(first_concept_candidate_step, first_seen)
            if is_promoted:
                first_promoted_concept_step = first_seen if first_promoted_concept_step is None else min(first_promoted_concept_step, first_seen)
        state_conn.execute(
            """
            INSERT INTO concept_candidates (
                concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count,
                linked_family_count, transfer_success_count, cross_game_count, cross_context_count,
                compression_gain, explanatory_reach, promotion_score, first_seen_global_step,
                last_seen_global_step, is_promoted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept_signature,
                group["concept_type"],
                int(group["support_count"]),
                linked_role_count,
                linked_carrier_count,
                linked_family_count,
                transfer_success_count,
                cross_game_count,
                cross_context_count,
                compression_gain,
                explanatory_reach,
                promotion_score,
                first_seen,
                last_seen,
                is_promoted,
            ),
        )
        for role_signature in sorted(group["roles"]):
            _insert_link(state_conn, "concept_links", "concept_signature", concept_signature, "role", role_signature, 1, first_seen, last_seen)
        for carrier_signature in sorted(group["carriers"]):
            _insert_link(state_conn, "concept_links", "concept_signature", concept_signature, "carrier", carrier_signature, 1, first_seen, last_seen)
        for family_signature in sorted(group["families"]):
            _insert_link(state_conn, "concept_links", "concept_signature", concept_signature, "family", family_signature, 1, first_seen, last_seen)
        for context_signature in sorted(group["contexts"]):
            _insert_link(state_conn, "concept_links", "concept_signature", concept_signature, "context", context_signature, 1, first_seen, last_seen)
        for game in sorted(group["games"]):
            _insert_link(state_conn, "concept_links", "concept_signature", concept_signature, "game", game, 1, first_seen, last_seen)
    _write_milestone(state_conn, "first_concept_candidate_step", first_concept_candidate_step, None)
    _write_milestone(state_conn, "first_promoted_concept_step", first_promoted_concept_step, None)
    return {
        "concept_candidate_count": len(concept_groups),
        "promoted_concept_count": promoted_concepts,
    }


def derive_world_model_components(state_conn: sqlite3.Connection, graph_conn: sqlite3.Connection) -> dict[str, Any]:
    promoted_rows = state_conn.execute(
        """
        SELECT concept_signature, concept_type, linked_role_count, linked_carrier_count, linked_family_count,
               cross_context_count, cross_game_count, promotion_score, first_seen_global_step, last_seen_global_step,
               is_promoted
        FROM concept_candidates
        ORDER BY COALESCE(is_promoted, 0) DESC, COALESCE(promotion_score, 0.0) DESC, concept_signature ASC
        """
    ).fetchall()
    if not promoted_rows:
        _write_milestone(state_conn, "first_world_model_component_step", None, None)
        _write_milestone(state_conn, "first_coherent_world_model_step", None, None)
        return {"world_model_component_count": 0, "coherent_world_model_component_count": 0}
    concept_link_rows = state_conn.execute(
        "SELECT concept_signature, linked_type, linked_key FROM concept_links ORDER BY concept_signature ASC, linked_type ASC, linked_key ASC"
    ).fetchall()
    concept_links: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in concept_link_rows:
        if row["linked_key"] not in (None, ""):
            concept_links[str(row["concept_signature"])][str(row["linked_type"])].add(str(row["linked_key"]))
    role_link_rows = state_conn.execute(
        "SELECT role_signature, linked_type, linked_key FROM role_links ORDER BY role_signature ASC, linked_type ASC, linked_key ASC"
    ).fetchall()
    role_links: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in role_link_rows:
        if row["linked_key"] not in (None, ""):
            role_links[str(row["role_signature"])][str(row["linked_type"])].add(str(row["linked_key"]))
    successful_transfer_by_role: dict[str, int] = defaultdict(int)
    for row in state_conn.execute("SELECT role_signature, reuse_success FROM role_transfer_attempts ORDER BY role_signature ASC, attempt_id ASC").fetchall():
        if int(row["reuse_success"] or 0) == 1:
            successful_transfer_by_role[str(row["role_signature"])] += 1
    graph_edge_rows = graph_conn.execute(
        "SELECT source_node_id, target_node_id, edge_type FROM graph_edges ORDER BY source_node_id ASC, edge_type ASC, target_node_id ASC"
    ).fetchall()
    contradiction_rows = state_conn.execute("SELECT canonical_key FROM contradiction_clusters ORDER BY canonical_key ASC").fetchall()
    contradiction_keys = {str(row["canonical_key"]) for row in contradiction_rows}
    concept_rows = [dict(row) for row in promoted_rows if int(row["is_promoted"] or 0) == 1]
    if not concept_rows:
        concept_rows = [dict(row) for row in promoted_rows[:20]]
    coherent_count = 0
    first_component_step: int | None = None
    first_coherent_step: int | None = None
    for row in concept_rows:
        concept_signature = str(row["concept_signature"])
        links = concept_links.get(concept_signature, {})
        roles = sorted(links.get("role", set()))
        carriers = sorted(links.get("carrier", set()))
        families = sorted(links.get("family", set()))
        contexts = sorted(links.get("context", set()))
        games = sorted(links.get("game", set()))
        nodes = {f"concept:{concept_signature}"}
        nodes.update(f"role:{role}" for role in roles)
        nodes.update(f"carrier:{carrier}" for carrier in carriers)
        nodes.update(f"family:{family}" for family in families)
        nodes.update(f"context:{context}" for context in contexts)
        nodes.update(f"game:{game}" for game in games)
        graph_edge_count = sum(
            1
            for edge in graph_edge_rows
            if str(edge["source_node_id"]) in nodes or str(edge["target_node_id"]) in nodes
        )
        implicit_edge_count = len(roles) + len(carriers) + len(families) + len(contexts) + len(games)
        prediction_support_count = len(families) + sum(successful_transfer_by_role.get(role, 0) for role in roles)
        contradiction_coverage_count = 0
        family_nodes = {f"family:{family}" for family in families}
        for edge in graph_edge_rows:
            source = str(edge["source_node_id"])
            target = str(edge["target_node_id"])
            if source in family_nodes and target.startswith("contradiction:"):
                if target[len("contradiction:"):] in contradiction_keys:
                    contradiction_coverage_count += 1
            elif target in family_nodes and source.startswith("contradiction:"):
                if source[len("contradiction:"):] in contradiction_keys:
                    contradiction_coverage_count += 1
        node_count = len(nodes)
        edge_count = graph_edge_count + implicit_edge_count
        linked_role_count = len(roles)
        linked_family_count = len(families)
        linked_carrier_count = len(carriers)
        cross_context_count = len(contexts)
        cross_game_count = len(games)
        explanatory_coverage = (
            linked_family_count + linked_carrier_count + prediction_support_count + contradiction_coverage_count
        ) / max(1, node_count)
        coherence_score = (
            0.25 * min(1.0, edge_count / max(1.0, node_count * 2.0))
            + 0.25 * min(1.0, explanatory_coverage)
            + 0.20 * min(1.0, linked_role_count / 2.0)
            + 0.15 * min(1.0, cross_context_count / 3.0)
            + 0.15 * min(1.0, cross_game_count / 2.0)
        )
        is_coherent = int(
            len({concept_signature}) >= 1
            and linked_role_count >= 1
            and linked_family_count >= 2
            and linked_carrier_count >= 2
            and coherence_score >= 0.45
            and int(row.get("is_promoted", 0) or 0) == 1
        )
        if is_coherent:
            coherent_count += 1
        first_seen = None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"])
        last_seen = None if row["last_seen_global_step"] is None else int(row["last_seen_global_step"])
        if first_seen is not None:
            first_component_step = first_seen if first_component_step is None else min(first_component_step, first_seen)
            if is_coherent:
                first_coherent_step = first_seen if first_coherent_step is None else min(first_coherent_step, first_seen)
        component_signature = f"wm:{sha1(concept_signature.encode('utf-8')).hexdigest()[:20]}"
        component_type = "promoted_concept_component" if int(row.get("is_promoted", 0) or 0) == 1 else "candidate_concept_component"
        state_conn.execute(
            """
            INSERT INTO world_model_components (
                component_signature, component_type, node_count, edge_count, linked_concept_count,
                linked_role_count, linked_family_count, linked_carrier_count, cross_context_count,
                cross_game_count, explanatory_coverage, prediction_support_count, contradiction_coverage_count,
                coherence_score, first_seen_global_step, last_seen_global_step, is_coherent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                component_signature,
                component_type,
                node_count,
                edge_count,
                1,
                linked_role_count,
                linked_family_count,
                linked_carrier_count,
                cross_context_count,
                cross_game_count,
                explanatory_coverage,
                prediction_support_count,
                contradiction_coverage_count,
                coherence_score,
                first_seen,
                last_seen,
                is_coherent,
            ),
        )
        _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "concept", concept_signature, 1, first_seen, last_seen)
        for role in roles:
            _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "role", role, 1, first_seen, last_seen)
        for carrier in carriers:
            _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "carrier", carrier, 1, first_seen, last_seen)
        for family in families:
            _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "family", family, 1, first_seen, last_seen)
        for context in contexts:
            _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "context", context, 1, first_seen, last_seen)
        for game in games:
            _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "game", game, 1, first_seen, last_seen)
    _write_milestone(state_conn, "first_world_model_component_step", first_component_step, None)
    _write_milestone(state_conn, "first_coherent_world_model_step", first_coherent_step, None)
    return {
        "world_model_component_count": len(concept_rows),
        "coherent_world_model_component_count": coherent_count,
    }


def _build_role_tokens(
    *,
    families: tuple[str, ...],
    contexts: tuple[str, ...],
    games: tuple[str, ...],
    out_edges: list[sqlite3.Row],
    in_edges: list[sqlite3.Row],
    graph_nodes: dict[str, str],
    family_meta: dict[str, dict[str, Any]],
) -> set[str]:
    tokens: set[str] = set()
    tokens.add(f"family_count:{_bucket(len(families))}")
    tokens.add(f"context_count:{_bucket(len(contexts))}")
    tokens.add(f"game_count:{_bucket(len(games))}")
    motifs: set[str] = set()
    for family_signature in families:
        meta = family_meta.get(family_signature, {})
        effect_type = str(meta.get("effect_type") or "unknown")
        action_group = str(meta.get("action_group") or "unknown")
        polarity = str(meta.get("polarity") or "unknown")
        tokens.add(f"fam_effect:{effect_type}")
        tokens.add(f"fam_action:{action_group}")
        tokens.add(f"fam_polarity:{polarity}")
        if effect_type in {"unknown", "", "None"} or action_group in {"unknown", "", "None"} or polarity in {"unknown", "", "None"}:
            tokens.add(f"family_cluster:{sha1(family_signature.encode('utf-8')).hexdigest()[:8]}")
        motifs.update(_motifs_for_text(" ".join([effect_type, action_group, polarity, family_signature]).lower()))
    out_type_counts: dict[str, int] = defaultdict(int)
    in_type_counts: dict[str, int] = defaultdict(int)
    out_neighbor_types: dict[str, int] = defaultdict(int)
    in_neighbor_types: dict[str, int] = defaultdict(int)
    for row in out_edges:
        edge_type = str(row["edge_type"] or "related_to")
        out_type_counts[edge_type] += 1
        target_type = graph_nodes.get(str(row["target_node_id"]), "unknown")
        out_neighbor_types[target_type] += 1
    for row in in_edges:
        edge_type = str(row["edge_type"] or "related_to")
        in_type_counts[edge_type] += 1
        source_type = graph_nodes.get(str(row["source_node_id"]), "unknown")
        in_neighbor_types[source_type] += 1
    for edge_type, count in sorted(out_type_counts.items()):
        tokens.add(f"edge_out:{edge_type}:{_bucket(count)}")
    for edge_type, count in sorted(in_type_counts.items()):
        tokens.add(f"edge_in:{edge_type}:{_bucket(count)}")
    for node_type, count in sorted(out_neighbor_types.items()):
        tokens.add(f"neighbor_out_type:{node_type}:{_bucket(count)}")
    for node_type, count in sorted(in_neighbor_types.items()):
        tokens.add(f"neighbor_in_type:{node_type}:{_bucket(count)}")
    if not motifs:
        motifs.add("unknown")
    for motif in sorted(motifs):
        tokens.add(f"future_motif:{motif}")
    return tokens


def _motifs_for_text(text: str) -> set[str]:
    motif_map = {
        "enable": ("enable", "open", "unlock", "add", "create", "reveal", "allow", "access"),
        "block": ("block", "close", "obstacle", "wall", "prevent", "forbid", "stop"),
        "terminate": ("remove", "delete", "clear", "destroy", "consume", "vanish", "end"),
        "reversible": ("reverse", "restore", "toggle", "swap", "move_back", "undo"),
        "transform": ("transform", "recolor", "change", "convert", "shift"),
    }
    motifs = {label for label, keywords in motif_map.items() if any(keyword in text for keyword in keywords)}
    return motifs


def _bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if 3 <= value <= 5:
        return "3_5"
    if 6 <= value <= 10:
        return "6_10"
    return "11_plus"


def _role_signature(tokens: list[str]) -> str:
    return "role:" + sha1(json.dumps(sorted(set(tokens)), separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _role_type(tokens: list[str], linked_family_count: int, linked_context_count: int, linked_game_count: int) -> str:
    token_set = set(tokens)
    if "future_motif:enable" in token_set:
        return "enabler"
    if "future_motif:block" in token_set:
        return "blocker"
    if "future_motif:terminate" in token_set:
        return "terminator"
    if "future_motif:reversible" in token_set:
        return "reversible_operator"
    if "future_motif:transform" in token_set:
        return "transformer"
    if linked_family_count >= 2 and linked_context_count >= 2:
        return "graph_bridge"
    if linked_game_count >= 2:
        return "graph_bridge"
    return "contextual_anchor"


def _build_transfer_attempt(
    *,
    role_signature: str,
    transfer_kind: str,
    source_scope_type: str,
    source_scope_key: str,
    target_scope_type: str,
    target_scope_key: str,
    target_carrier_signature: str,
    target_tokens: set[str],
    source_rows: list[sqlite3.Row],
    observed_role_signature: str,
    first_seen: int | None,
    last_seen: int | None,
) -> dict[str, Any]:
    source_tokens = sorted({token for row in source_rows for token in _load_token_json(row["token_json"])})
    similarity_score = _jaccard(set(source_tokens), target_tokens)
    source_carrier_count = len(source_rows)
    reuse_success = int(similarity_score >= 0.60 and role_signature == observed_role_signature)
    transfer_score = similarity_score * min(1.0, source_carrier_count / 2.0)
    failure_reason = "success" if reuse_success else ("low_similarity" if similarity_score < 0.60 else "role_mismatch")
    attempt_seed = "|".join(
        (
            transfer_kind,
            role_signature,
            source_scope_key,
            target_scope_key,
            target_carrier_signature,
        )
    )
    return {
        "attempt_id": sha1(attempt_seed.encode("utf-8")).hexdigest(),
        "role_signature": role_signature,
        "transfer_kind": transfer_kind,
        "source_scope_type": source_scope_type,
        "source_scope_key": source_scope_key,
        "target_scope_type": target_scope_type,
        "target_scope_key": target_scope_key,
        "target_carrier_signature": target_carrier_signature,
        "predicted_role_signature": role_signature,
        "observed_role_signature": observed_role_signature,
        "similarity_score": similarity_score,
        "transfer_score": transfer_score,
        "reuse_success": reuse_success,
        "failure_reason": failure_reason,
        "first_seen_global_step": first_seen,
        "last_seen_global_step": last_seen,
    }


def _concept_tokens(role_type: str, tokens: list[str]) -> list[str]:
    concept_tokens = {f"concept_type:{role_type}"}
    for token in tokens:
        if token.startswith("future_motif:") or token.startswith("fam_effect:") or token.startswith("fam_action:"):
            concept_tokens.add(token)
        elif token.startswith("edge_out:"):
            concept_tokens.add(":".join(token.split(":")[:2]))
        elif token.startswith("edge_in:"):
            concept_tokens.add(":".join(token.split(":")[:2]))
    return sorted(concept_tokens)


def _concept_signature(tokens: list[str]) -> str:
    return "concept:" + sha1(json.dumps(sorted(set(tokens)), separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _insert_link(
    connection: sqlite3.Connection,
    table: str,
    signature_column: str,
    signature: str,
    linked_type: str,
    linked_key: str | None,
    support_count: int,
    first_seen: int | None,
    last_seen: int | None,
) -> None:
    if linked_key in (None, ""):
        return
    connection.execute(
        f"""
        INSERT INTO {table} (
            {signature_column}, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT({signature_column}, linked_type, linked_key) DO UPDATE SET
            support_count = {table}.support_count + excluded.support_count,
            first_seen_global_step = MIN({table}.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX({table}.last_seen_global_step, excluded.last_seen_global_step)
        """,
        (signature, linked_type, linked_key, int(support_count), first_seen, last_seen),
    )


def _write_milestone(connection: sqlite3.Connection, milestone_name: str, first_global_step: int | None, evidence_key: str | None) -> None:
    connection.execute(
        """
        INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key)
        VALUES (?, ?, ?)
        """,
        (milestone_name, first_global_step, evidence_key),
    )


def _load_token_json(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))

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
    signature_tokens: tuple[str, ...]
    diagnostic_tokens: tuple[str, ...]
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
        summary = {**role_summary, **transfer_summary, **concept_summary, **world_summary}
        state_conn.commit()
        graph_conn.commit()
        return summary


def derive_role_candidates(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_carriers: int,
    max_roles: int,
) -> dict[str, Any]:
    family_meta = {
        str(row["canonical_signature"]): dict(row)
        for row in state_conn.execute(
            """
            SELECT canonical_signature, effect_type, action_group, polarity, support_count, member_count
            FROM transformation_families
            ORDER BY canonical_signature ASC
            """
        ).fetchall()
    }
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

    links_by_carrier = _carrier_links_by_carrier(state_conn)
    carrier_signatures = [str(row["carrier_signature"]) for row in carrier_rows]
    relevant_node_ids: set[str] = set()
    context_node_ids: set[str] = set()
    for carrier_signature in carrier_signatures:
        relevant_node_ids.add(f"carrier:{carrier_signature}")
        carrier_links = links_by_carrier.get(carrier_signature, {})
        for family in carrier_links.get("family", set()):
            relevant_node_ids.add(f"family:{family}")
        for context in carrier_links.get("context", set()):
            context_node_ids.add(f"context:{context}")
            relevant_node_ids.add(f"context:{context}")
    edge_rows = _fetch_edges_for_nodes(graph_conn, relevant_node_ids)
    graph_nodes = _graph_nodes_for_ids(graph_conn, {node_id for row in edge_rows for node_id in (str(row["source_node_id"]), str(row["target_node_id"]))} | relevant_node_ids)
    context_games = _context_games_for_context_nodes(graph_conn, context_node_ids)
    edges_out_by_source: dict[str, list[sqlite3.Row]] = defaultdict(list)
    edges_in_by_target: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in edge_rows:
        edges_out_by_source[str(row["source_node_id"])].append(row)
        edges_in_by_target[str(row["target_node_id"])].append(row)

    neighborhoods: list[CarrierNeighborhood] = []
    for row in carrier_rows:
        carrier_signature = str(row["carrier_signature"])
        carrier_links = links_by_carrier.get(carrier_signature, {})
        families = tuple(sorted(carrier_links.get("family", set())))
        contexts = tuple(sorted(carrier_links.get("context", set())))
        contingencies = tuple(sorted(carrier_links.get("contingency", set())))
        games = tuple(sorted({game for context in contexts for game in context_games.get(f"context:{context}", set())}))
        carrier_node_id = f"carrier:{carrier_signature}"
        out_edges = list(edges_out_by_source.get(carrier_node_id, []))
        in_edges = list(edges_in_by_target.get(carrier_node_id, []))
        signature_tokens, diagnostic_tokens = _build_role_tokens(
            families=families,
            contexts=contexts,
            games=games,
            out_edges=out_edges,
            in_edges=in_edges,
            graph_nodes=graph_nodes,
            family_meta=family_meta,
        )
        role_signature = _role_signature(signature_tokens)
        role_type = _role_type(signature_tokens, len(families), len(contexts), len(games))
        state_conn.execute(
            """
            INSERT INTO role_neighborhood_signatures (
                carrier_signature, role_signature, role_type, token_json, diagnostic_token_json,
                linked_family_count, linked_context_count, linked_game_count, in_edge_count, out_edge_count,
                first_seen_global_step, last_seen_global_step, stability_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                carrier_signature,
                role_signature,
                role_type,
                json.dumps(signature_tokens, separators=(",", ":"), sort_keys=True),
                json.dumps(diagnostic_tokens, separators=(",", ":"), sort_keys=True),
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
                signature_tokens=tuple(signature_tokens),
                diagnostic_tokens=tuple(diagnostic_tokens),
                families=families,
                contexts=contexts,
                contingencies=contingencies,
                games=games,
            )
        )

    grouped: dict[str, list[CarrierNeighborhood]] = defaultdict(list)
    for item in neighborhoods:
        grouped[item.role_signature].append(item)
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
        signature_token_set = sorted({token for item in items for token in item.signature_tokens})
        diagnostic_token_set = sorted({token for item in items for token in item.diagnostic_tokens})
        state_conn.execute(
            """
            INSERT INTO role_candidates (
                role_signature, role_type, support_count, linked_carrier_count, linked_family_count,
                linked_context_count, cross_game_count, cross_context_count, first_seen_global_step,
                last_seen_global_step, role_stability_score, is_emergent, role_signature_token_count,
                diagnostic_token_count, exact_family_token_count, exact_identity_token_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                len(signature_token_set),
                len(diagnostic_token_set),
                sum(1 for token in diagnostic_token_set if token.startswith("family_cluster:")),
                sum(
                    1
                    for token in signature_token_set
                    if token.startswith(("carrier:", "context:", "game:"))
                ),
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
            _insert_link(state_conn, "role_links", "role_signature", role_signature, "carrier", item.carrier_signature, 1, first_seen, last_seen)
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


def derive_role_transfer_attempts(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_transfer_attempts: int,
) -> dict[str, Any]:
    rows = state_conn.execute(
        """
        SELECT carrier_signature, role_signature, token_json, first_seen_global_step, last_seen_global_step
        FROM role_neighborhood_signatures
        ORDER BY carrier_signature ASC
        """
    ).fetchall()
    if not rows:
        _write_milestone(state_conn, "first_role_transfer_attempt_step", None, None)
        _write_milestone(state_conn, "first_role_transfer_success_step", None, None)
        return _empty_transfer_summary()

    carrier_contexts, carrier_games = _carrier_scope_maps(state_conn, graph_conn)
    role_rows = {
        str(row["carrier_signature"]): {
            "carrier_signature": str(row["carrier_signature"]),
            "role_signature": str(row["role_signature"]),
            "tokens": tuple(_load_token_json(row["token_json"])),
            "first_seen_global_step": None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"]),
            "last_seen_global_step": None if row["last_seen_global_step"] is None else int(row["last_seen_global_step"]),
        }
        for row in rows
    }
    carrier_signatures = sorted(role_rows)
    target_attempt_specs: list[tuple[str, str, str]] = []
    unique_scopes: set[tuple[str, str]] = set()
    for carrier_signature in carrier_signatures:
        for scope_key in sorted(carrier_games.get(carrier_signature, set())):
            target_attempt_specs.append((carrier_signature, "cross_game", scope_key))
            unique_scopes.add(("cross_game", scope_key))
        for scope_key in sorted(carrier_contexts.get(carrier_signature, set())):
            target_attempt_specs.append((carrier_signature, "cross_context", scope_key))
            unique_scopes.add(("cross_context", scope_key))
    profile_cache = _build_transfer_profile_cache(
        role_rows=role_rows,
        carrier_contexts=carrier_contexts,
        carrier_games=carrier_games,
        target_scopes=sorted(unique_scopes, key=lambda item: (0 if item[0] == "cross_game" else 1, item[1])),
    )
    inserted = 0
    success_count = 0
    successful_roles: set[str] = set()
    first_attempt_step: int | None = None
    first_success_step: int | None = None
    role_mismatch_count = 0
    low_similarity_count = 0
    insufficient_source_support_count = 0
    no_source_profile_count = 0
    transfer_score_sum = 0.0
    best_margin_sum = 0.0
    best_margin_count = 0
    source_carrier_count_sum = 0
    candidate_role_count_sum = 0
    cross_game_attempt_count = 0
    cross_game_success_count = 0
    cross_context_attempt_count = 0
    cross_context_success_count = 0

    for carrier_signature, transfer_kind, target_scope_key in target_attempt_specs:
        if inserted >= int(max_transfer_attempts):
            break
        attempt = _predict_transfer_attempt(
            profile_cache=profile_cache,
            role_rows=role_rows,
            target_carrier_signature=carrier_signature,
            transfer_kind=transfer_kind,
            target_scope_key=target_scope_key,
        )
        state_conn.execute(
            """
            INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature,
                observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason,
                best_margin, source_carrier_count, candidate_role_count, first_seen_global_step, last_seen_global_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt["attempt_id"],
                attempt["role_signature"],
                attempt["transfer_kind"],
                attempt["source_scope_type"],
                attempt["source_scope_key"],
                attempt["target_scope_type"],
                attempt["target_scope_key"],
                attempt["target_carrier_signature"],
                attempt["predicted_role_signature"],
                attempt["observed_role_signature"],
                attempt["similarity_score"],
                attempt["transfer_score"],
                attempt["reuse_success"],
                attempt["failure_reason"],
                attempt["best_margin"],
                attempt["source_carrier_count"],
                attempt["candidate_role_count"],
                attempt["first_seen_global_step"],
                attempt["last_seen_global_step"],
            ),
        )
        inserted += 1
        transfer_score_sum += float(attempt["transfer_score"] or 0.0)
        source_carrier_count_sum += int(attempt["source_carrier_count"] or 0)
        candidate_role_count_sum += int(attempt["candidate_role_count"] or 0)
        if attempt["best_margin"] is not None:
            best_margin_sum += float(attempt["best_margin"])
            best_margin_count += 1
        if attempt["first_seen_global_step"] is not None:
            first_attempt_step = (
                attempt["first_seen_global_step"]
                if first_attempt_step is None
                else min(first_attempt_step, int(attempt["first_seen_global_step"]))
            )
        if attempt["transfer_kind"] == "cross_game":
            cross_game_attempt_count += 1
        else:
            cross_context_attempt_count += 1
        failure_reason = str(attempt["failure_reason"])
        if failure_reason == "role_mismatch":
            role_mismatch_count += 1
        elif failure_reason == "low_similarity":
            low_similarity_count += 1
        elif failure_reason == "insufficient_source_support":
            insufficient_source_support_count += 1
        elif failure_reason == "no_source_profile":
            no_source_profile_count += 1
        if int(attempt["reuse_success"]) == 1:
            success_count += 1
            successful_roles.add(str(attempt["observed_role_signature"]))
            if attempt["transfer_kind"] == "cross_game":
                cross_game_success_count += 1
            else:
                cross_context_success_count += 1
            if attempt["first_seen_global_step"] is not None:
                first_success_step = (
                    attempt["first_seen_global_step"]
                    if first_success_step is None
                    else min(first_success_step, int(attempt["first_seen_global_step"]))
                )

    _write_milestone(state_conn, "first_role_transfer_attempt_step", first_attempt_step, None)
    _write_milestone(state_conn, "first_role_transfer_success_step", first_success_step, None)
    return {
        "transfer_attempt_count": inserted,
        "successful_transfer_count": success_count,
        "successful_role_count": len(successful_roles),
        "role_mismatch_count": role_mismatch_count,
        "low_similarity_count": low_similarity_count,
        "insufficient_source_support_count": insufficient_source_support_count,
        "no_source_profile_count": no_source_profile_count,
        "cross_game_attempt_count": cross_game_attempt_count,
        "cross_game_success_count": cross_game_success_count,
        "cross_context_attempt_count": cross_context_attempt_count,
        "cross_context_success_count": cross_context_success_count,
        "mean_transfer_score": (transfer_score_sum / inserted) if inserted else None,
        "max_transfer_score": _safe_scalar_float(
            state_conn,
            "SELECT MAX(COALESCE(transfer_score, 0.0)) FROM role_transfer_attempts",
        ),
        "mean_best_margin": (best_margin_sum / best_margin_count) if best_margin_count else None,
        "mean_source_carrier_count": (source_carrier_count_sum / inserted) if inserted else None,
        "candidate_role_count_mean": (candidate_role_count_sum / inserted) if inserted else None,
        "transfer_profile_cache_scope_count": len(profile_cache),
        "transfer_profile_cache_profile_count": sum(len(items) for items in profile_cache.values()),
    }


def derive_concept_candidates(state_conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        state_conn.execute("ALTER TABLE concept_candidates ADD COLUMN transfer_success_concentration REAL")
    except sqlite3.DatabaseError:
        pass
    try:
        state_conn.execute("ALTER TABLE concept_candidates ADD COLUMN is_overconcentrated INTEGER DEFAULT 0")
    except sqlite3.DatabaseError:
        pass
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
        return {
            "concept_candidate_count": 0,
            "promoted_concept_count": 0,
            "concept_strong_transfer_success_count": 0,
        }
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    neighborhood_rows = state_conn.execute(
        """
        SELECT role_signature, token_json
        FROM role_neighborhood_signatures
        ORDER BY role_signature ASC, carrier_signature ASC
        """
    ).fetchall()
    tokens_by_role: dict[str, set[str]] = defaultdict(set)
    for row in neighborhood_rows:
        tokens_by_role[str(row["role_signature"])].update(_load_token_json(row["token_json"]))
    transfer_rows = state_conn.execute(
        """
        SELECT role_signature, reuse_success, similarity_score, best_margin, source_carrier_count, candidate_role_count
        FROM role_transfer_attempts
        ORDER BY role_signature ASC, attempt_id ASC
        """
    ).fetchall()
    success_by_role: dict[str, int] = defaultdict(int)
    strong_success_by_role: dict[str, int] = defaultdict(int)
    for row in transfer_rows:
        role_signature = str(row["role_signature"])
        if int(row["reuse_success"] or 0) == 1:
            success_by_role[role_signature] += 1
            if (
                int(row["source_carrier_count"] or 0) >= 2
                and int(row["candidate_role_count"] or 0) >= 2
                and float(row["similarity_score"] or 0.0) >= 0.60
                and float(row["best_margin"] or 0.0) >= 0.10
            ):
                strong_success_by_role[role_signature] += 1

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
                "strong_transfer_success_count": 0,
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
        group["strong_transfer_success_count"] += int(strong_success_by_role.get(role_signature, 0))
        first_seen = None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"])
        last_seen = None if row["last_seen_global_step"] is None else int(row["last_seen_global_step"])
        if first_seen is not None:
            group["first_seen"] = first_seen if group["first_seen"] is None else min(group["first_seen"], first_seen)
        if last_seen is not None:
            group["last_seen"] = last_seen if group["last_seen"] is None else max(group["last_seen"], last_seen)

    promoted_concepts = 0
    overconcentrated_concepts = 0
    promoted_overconcentrated_concepts = 0
    first_concept_candidate_step: int | None = None
    first_promoted_concept_step: int | None = None
    strong_transfer_total = 0
    concept_strong_counts: list[int] = []
    for concept_signature in sorted(concept_groups):
        group = concept_groups[concept_signature]
        linked_role_count = len(group["roles"])
        linked_carrier_count = len(group["carriers"])
        linked_family_count = len(group["families"])
        cross_context_count = len(group["contexts"])
        cross_game_count = len(group["games"])
        transfer_success_count = int(group["transfer_success_count"])
        strong_transfer_success_count = int(group["strong_transfer_success_count"])
        strong_transfer_total += strong_transfer_success_count
        concept_strong_counts.append(strong_transfer_success_count)
    for concept_signature in sorted(concept_groups):
        group = concept_groups[concept_signature]
        linked_role_count = len(group["roles"])
        linked_carrier_count = len(group["carriers"])
        linked_family_count = len(group["families"])
        cross_context_count = len(group["contexts"])
        cross_game_count = len(group["games"])
        transfer_success_count = int(group["transfer_success_count"])
        strong_transfer_success_count = int(group["strong_transfer_success_count"])
        transfer_success_concentration = (
            float(strong_transfer_success_count) / float(strong_transfer_total)
            if strong_transfer_total > 0
            else None
        )
        overconcentrated = bool(
            transfer_success_concentration is not None
            and transfer_success_concentration > 0.80
        )
        compression_gain = float(linked_carrier_count / max(1, linked_role_count))
        explanatory_reach = float(linked_family_count + cross_context_count + (2 * cross_game_count) + strong_transfer_success_count)
        promotion_score = (
            0.25 * min(1.0, compression_gain / 2.0)
            + 0.30 * min(1.0, strong_transfer_success_count / 3.0)
            + 0.20 * min(1.0, cross_context_count / 3.0)
            + 0.15 * min(1.0, cross_game_count / 2.0)
            + 0.10 * min(1.0, linked_family_count / 3.0)
        )
        is_promoted = int(
            not overconcentrated
            and strong_transfer_success_count >= 2
            and linked_role_count >= 2
            and linked_carrier_count >= 2
            and linked_family_count >= 2
            and compression_gain >= 1.50
            and (cross_game_count >= 2 or cross_context_count >= 3)
            and promotion_score >= 0.55
        )
        if overconcentrated:
            overconcentrated_concepts += 1
        if is_promoted:
            promoted_concepts += 1
            if overconcentrated:
                promoted_overconcentrated_concepts += 1
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
                   linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count,
                   cross_context_count, compression_gain, explanatory_reach, promotion_score,
                   transfer_success_concentration, is_overconcentrated,
                   first_seen_global_step, last_seen_global_step, is_promoted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept_signature,
                group["concept_type"],
                int(group["support_count"]),
                linked_role_count,
                linked_carrier_count,
                linked_family_count,
                transfer_success_count,
                strong_transfer_success_count,
                cross_game_count,
                cross_context_count,
                compression_gain,
                explanatory_reach,
                promotion_score,
                transfer_success_concentration,
                int(overconcentrated),
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
        "concept_strong_transfer_success_count": strong_transfer_total,
        "overconcentrated_concept_count": overconcentrated_concepts,
        "promoted_overconcentrated_concept_count": promoted_overconcentrated_concepts,
        "concept_transfer_success_concentration": (
            max(concept_strong_counts) / strong_transfer_total
            if concept_strong_counts and strong_transfer_total > 0
            else None
        ),
    }


def derive_world_model_components(state_conn: sqlite3.Connection, graph_conn: sqlite3.Connection) -> dict[str, Any]:
    concept_rows = [dict(row) for row in state_conn.execute(
        """
        SELECT concept_signature, concept_type, linked_role_count, linked_carrier_count, linked_family_count,
               cross_context_count, cross_game_count, promotion_score, first_seen_global_step, last_seen_global_step,
               is_promoted
        FROM concept_candidates
        ORDER BY COALESCE(is_promoted, 0) DESC, COALESCE(promotion_score, 0.0) DESC, concept_signature ASC
        """
    ).fetchall()]
    if not concept_rows:
        _write_milestone(state_conn, "first_world_model_component_step", None, None)
        _write_milestone(state_conn, "first_coherent_world_model_step", None, None)
        return {
            "world_model_component_count": 0,
            "coherent_world_model_component_count": 0,
            "candidate_only_world_model_component_count": 0,
        }
    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    successful_transfer_by_role: dict[str, int] = defaultdict(int)
    for row in state_conn.execute(
        """
        SELECT role_signature, reuse_success, source_carrier_count, candidate_role_count, similarity_score, best_margin
        FROM role_transfer_attempts
        ORDER BY role_signature ASC, attempt_id ASC
        """
    ).fetchall():
        if (
            int(row["reuse_success"] or 0) == 1
            and int(row["source_carrier_count"] or 0) >= 2
            and int(row["candidate_role_count"] or 0) >= 2
            and float(row["similarity_score"] or 0.0) >= 0.60
            and float(row["best_margin"] or 0.0) >= 0.10
        ):
            successful_transfer_by_role[str(row["role_signature"])] += 1

    contradiction_keys = {
        str(row["canonical_key"])
        for row in state_conn.execute("SELECT canonical_key FROM contradiction_clusters ORDER BY canonical_key ASC").fetchall()
    }
    promoted_rows = [row for row in concept_rows if int(row.get("is_promoted", 0) or 0) == 1]
    candidate_rows = promoted_rows if promoted_rows else concept_rows[:20]
    coherent_count = 0
    candidate_only_count = 0
    first_component_step: int | None = None
    first_coherent_step: int | None = None
    for row in candidate_rows:
        concept_signature = str(row["concept_signature"])
        links = concept_links.get(concept_signature, {})
        roles = sorted(links.get("role", set()))
        carriers = sorted(links.get("carrier", set()))
        families = sorted(links.get("family", set()))
        contexts = sorted(links.get("context", set()))
        games = sorted(links.get("game", set()))
        node_ids = {f"concept:{concept_signature}"}
        node_ids.update(f"role:{role}" for role in roles)
        node_ids.update(f"carrier:{carrier}" for carrier in carriers)
        node_ids.update(f"family:{family}" for family in families)
        node_ids.update(f"context:{context}" for context in contexts)
        node_ids.update(f"game:{game}" for game in games)
        edge_rows = _fetch_edges_for_nodes(graph_conn, node_ids)
        graph_edge_count = sum(
            1
            for edge in edge_rows
            if str(edge["source_node_id"]) in node_ids or str(edge["target_node_id"]) in node_ids
        )
        implicit_edge_count = len(roles) + len(carriers) + len(families) + len(contexts) + len(games)
        prediction_support_count = len(families) + sum(successful_transfer_by_role.get(role, 0) for role in roles)
        contradiction_coverage_count = 0
        family_nodes = {f"family:{family}" for family in families}
        for edge in edge_rows:
            source = str(edge["source_node_id"])
            target = str(edge["target_node_id"])
            if source in family_nodes and target.startswith("contradiction:") and target[len("contradiction:"):] in contradiction_keys:
                contradiction_coverage_count += 1
            elif target in family_nodes and source.startswith("contradiction:") and source[len("contradiction:"):] in contradiction_keys:
                contradiction_coverage_count += 1
        node_count = len(node_ids)
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
        is_promoted = int(row.get("is_promoted", 0) or 0) == 1
        candidate_only = 0 if is_promoted else 1
        predicted_outcome_count = prediction_support_count
        predicted_outcome_count_is_proxy = 1
        is_coherent = int(
            is_promoted
            and linked_role_count >= 2
            and linked_family_count >= 2
            and linked_carrier_count >= 2
            and (cross_context_count >= 3 or cross_game_count >= 2)
            and prediction_support_count > 0
            and coherence_score >= 0.55
        )
        if candidate_only:
            candidate_only_count += 1
        if is_coherent:
            coherent_count += 1
        first_seen = None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"])
        last_seen = None if row["last_seen_global_step"] is None else int(row["last_seen_global_step"])
        if first_seen is not None:
            first_component_step = first_seen if first_component_step is None else min(first_component_step, first_seen)
            if is_coherent:
                first_coherent_step = first_seen if first_coherent_step is None else min(first_coherent_step, first_seen)
        component_signature = f"wm:{sha1(concept_signature.encode('utf-8')).hexdigest()[:20]}"
        component_type = "promoted_concept_component" if is_promoted else "candidate_concept_component"
        state_conn.execute(
            """
            INSERT INTO world_model_components (
                component_signature, component_type, node_count, edge_count, linked_concept_count,
                linked_role_count, linked_family_count, linked_carrier_count, cross_context_count,
                cross_game_count, explanatory_coverage, prediction_support_count, contradiction_coverage_count,
                coherence_score, candidate_only, predicted_outcome_count, predicted_outcome_count_is_proxy,
                first_seen_global_step, last_seen_global_step, is_coherent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                candidate_only,
                predicted_outcome_count,
                predicted_outcome_count_is_proxy,
                first_seen,
                last_seen,
                is_coherent,
            ),
        )
        _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "concept", concept_signature, 1, first_seen, last_seen)
        for role in roles:
            _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "role", role, 1, first_seen, last_seen)
            for carrier in sorted(role_links.get(role, {}).get("carrier", set())):
                _insert_link(state_conn, "world_model_links", "component_signature", component_signature, "carrier", carrier, 1, first_seen, last_seen)
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
        "world_model_component_count": len(candidate_rows),
        "coherent_world_model_component_count": coherent_count,
        "candidate_only_world_model_component_count": candidate_only_count,
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
) -> tuple[list[str], list[str]]:
    signature_tokens: set[str] = set()
    diagnostic_tokens: set[str] = set()
    signature_tokens.add(f"family_count:{_bucket(len(families))}")
    signature_tokens.add(f"context_count:{_bucket(len(contexts))}")
    signature_tokens.add(f"game_count:{_bucket(len(games))}")
    motifs: set[str] = set()
    for family_signature in families:
        meta = family_meta.get(family_signature, {})
        effect_type = str(meta.get("effect_type") or "unknown")
        action_group = str(meta.get("action_group") or "unknown")
        polarity = str(meta.get("polarity") or "unknown")
        signature_tokens.add(f"fam_effect:{effect_type}")
        signature_tokens.add(f"fam_action:{action_group}")
        signature_tokens.add(f"fam_polarity:{polarity}")
        if effect_type in {"unknown", "", "None"} or action_group in {"unknown", "", "None"} or polarity in {"unknown", "", "None"}:
            diagnostic_tokens.add(f"family_cluster:{sha1(family_signature.encode('utf-8')).hexdigest()[:8]}")
        motifs.update(_motifs_for_text(" ".join([effect_type, action_group, polarity, family_signature]).lower()))
    out_type_counts: dict[str, int] = defaultdict(int)
    in_type_counts: dict[str, int] = defaultdict(int)
    out_neighbor_types: dict[str, int] = defaultdict(int)
    in_neighbor_types: dict[str, int] = defaultdict(int)
    for row in out_edges:
        edge_type = str(row["edge_type"] or "related_to")
        out_type_counts[edge_type] += 1
        out_neighbor_types[graph_nodes.get(str(row["target_node_id"]), "unknown")] += 1
    for row in in_edges:
        edge_type = str(row["edge_type"] or "related_to")
        in_type_counts[edge_type] += 1
        in_neighbor_types[graph_nodes.get(str(row["source_node_id"]), "unknown")] += 1
    for edge_type, count in sorted(out_type_counts.items()):
        signature_tokens.add(f"edge_out:{edge_type}:{_bucket(count)}")
    for edge_type, count in sorted(in_type_counts.items()):
        signature_tokens.add(f"edge_in:{edge_type}:{_bucket(count)}")
    for node_type, count in sorted(out_neighbor_types.items()):
        signature_tokens.add(f"neighbor_out_type:{node_type}:{_bucket(count)}")
    for node_type, count in sorted(in_neighbor_types.items()):
        signature_tokens.add(f"neighbor_in_type:{node_type}:{_bucket(count)}")
    if not motifs:
        motifs.add("unknown")
    for motif in sorted(motifs):
        signature_tokens.add(f"future_motif:{motif}")
    return sorted(signature_tokens), sorted(diagnostic_tokens)


def _predict_transfer_attempt(
    *,
    profile_cache: dict[tuple[str, str], list[dict[str, Any]]],
    role_rows: dict[str, dict[str, Any]],
    target_carrier_signature: str,
    transfer_kind: str,
    target_scope_key: str,
) -> dict[str, Any]:
    target = role_rows[target_carrier_signature]
    target_tokens = set(target["tokens"])
    profiles = [
        dict(profile)
        for profile in profile_cache.get((transfer_kind, target_scope_key), [])
        if not (
            int(profile.get("source_carrier_count") or 0) == 1
            and str(profile.get("role_signature") or "") == str(target["role_signature"])
        )
    ]
    if not profiles:
        return _no_source_profile_attempt(target, transfer_kind, target_scope_key)
    scored_profiles: list[dict[str, Any]] = []
    for profile in profiles:
        scored_profiles.append(
            {
                **profile,
                "similarity_score": _jaccard(set(profile["profile_tokens"]), target_tokens),
            }
        )
    scored_profiles.sort(
        key=lambda item: (
            -float(item["similarity_score"]),
            -int(item["source_carrier_count"]),
            str(item["role_signature"]),
        )
    )
    best = scored_profiles[0]
    second = scored_profiles[1] if len(scored_profiles) > 1 else None
    best_margin = None if second is None else float(best["similarity_score"]) - float(second["similarity_score"])
    predicted_role_signature = str(best["role_signature"])
    observed_role_signature = str(target["role_signature"])
    source_carrier_count = int(best["source_carrier_count"])
    candidate_role_count = len(scored_profiles)
    similarity_score = float(best["similarity_score"])
    reuse_success = int(
        predicted_role_signature == observed_role_signature
        and similarity_score >= 0.60
        and source_carrier_count >= 2
    )
    if reuse_success:
        failure_reason = "success"
    elif source_carrier_count < 2:
        failure_reason = "insufficient_source_support"
    elif similarity_score < 0.60:
        failure_reason = "low_similarity"
    elif predicted_role_signature != observed_role_signature:
        failure_reason = "role_mismatch"
    else:
        failure_reason = "no_source_profile"
    source_scope_type = "not_game" if transfer_kind == "cross_game" else "not_context"
    target_scope_type = "game" if transfer_kind == "cross_game" else "context"
    transfer_score = similarity_score * min(1.0, source_carrier_count / 2.0)
    attempt_seed = "|".join((transfer_kind, target_scope_key, target_carrier_signature))
    return {
        "attempt_id": sha1(attempt_seed.encode("utf-8")).hexdigest(),
        "role_signature": observed_role_signature,
        "transfer_kind": transfer_kind,
        "source_scope_type": source_scope_type,
        "source_scope_key": target_scope_key,
        "target_scope_type": target_scope_type,
        "target_scope_key": target_scope_key,
        "target_carrier_signature": target_carrier_signature,
        "predicted_role_signature": predicted_role_signature,
        "observed_role_signature": observed_role_signature,
        "similarity_score": similarity_score,
        "transfer_score": transfer_score,
        "reuse_success": reuse_success,
        "failure_reason": failure_reason,
        "best_margin": best_margin,
        "source_carrier_count": source_carrier_count,
        "candidate_role_count": candidate_role_count,
        "first_seen_global_step": target["first_seen_global_step"],
        "last_seen_global_step": target["last_seen_global_step"],
    }


def _build_transfer_profile_cache(
    *,
    role_rows: dict[str, dict[str, Any]],
    carrier_contexts: dict[str, set[str]],
    carrier_games: dict[str, set[str]],
    target_scopes: list[tuple[str, str]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    carrier_items = sorted(role_rows.items(), key=lambda item: item[0])
    for transfer_kind, target_scope_key in target_scopes:
        grouped_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate_signature, candidate in carrier_items:
            candidate_scopes = carrier_games.get(candidate_signature, set()) if transfer_kind == "cross_game" else carrier_contexts.get(candidate_signature, set())
            if target_scope_key in candidate_scopes:
                continue
            grouped_sources[str(candidate["role_signature"])].append(candidate)
        profiles: list[dict[str, Any]] = []
        for role_signature in sorted(grouped_sources):
            source_candidates = grouped_sources[role_signature]
            profiles.append(
                {
                    "role_signature": role_signature,
                    "profile_tokens": sorted({token for candidate in source_candidates for token in candidate["tokens"]}),
                    "source_carrier_count": len(source_candidates),
                    "source_context_count": len(
                        {
                            context
                            for candidate in source_candidates
                            for context in carrier_contexts.get(str(candidate["carrier_signature"]), set())
                        }
                    ),
                    "source_game_count": len(
                        {
                            game
                            for candidate in source_candidates
                            for game in carrier_games.get(str(candidate["carrier_signature"]), set())
                        }
                    ),
                }
            )
        cache[(transfer_kind, target_scope_key)] = profiles
    return cache


def _no_source_profile_attempt(target: dict[str, Any], transfer_kind: str, target_scope_key: str) -> dict[str, Any]:
    source_scope_type = "not_game" if transfer_kind == "cross_game" else "not_context"
    target_scope_type = "game" if transfer_kind == "cross_game" else "context"
    attempt_seed = "|".join((transfer_kind, target_scope_key, str(target["carrier_signature"]), "no_source_profile"))
    return {
        "attempt_id": sha1(attempt_seed.encode("utf-8")).hexdigest(),
        "role_signature": str(target["role_signature"]),
        "transfer_kind": transfer_kind,
        "source_scope_type": source_scope_type,
        "source_scope_key": target_scope_key,
        "target_scope_type": target_scope_type,
        "target_scope_key": target_scope_key,
        "target_carrier_signature": str(target["carrier_signature"]),
        "predicted_role_signature": None,
        "observed_role_signature": str(target["role_signature"]),
        "similarity_score": 0.0,
        "transfer_score": 0.0,
        "reuse_success": 0,
        "failure_reason": "no_source_profile",
        "best_margin": None,
        "source_carrier_count": 0,
        "candidate_role_count": 0,
        "first_seen_global_step": target["first_seen_global_step"],
        "last_seen_global_step": target["last_seen_global_step"],
    }


def _carrier_links_by_carrier(state_conn: sqlite3.Connection) -> dict[str, dict[str, set[str]]]:
    links: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in state_conn.execute(
        """
        SELECT carrier_signature, linked_type, linked_key
        FROM carrier_links
        ORDER BY carrier_signature ASC, linked_type ASC, linked_key ASC
        """
    ).fetchall():
        if row["linked_key"] not in (None, ""):
            links[str(row["carrier_signature"])][str(row["linked_type"])].add(str(row["linked_key"]))
    return links


def _carrier_scope_maps(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    links_by_carrier = _carrier_links_by_carrier(state_conn)
    carrier_contexts = {carrier: set(link_types.get("context", set())) for carrier, link_types in links_by_carrier.items()}
    carrier_games: dict[str, set[str]] = defaultdict(set)
    context_node_ids = {f"context:{context}" for contexts in carrier_contexts.values() for context in contexts}
    context_games = _context_games_for_context_nodes(graph_conn, context_node_ids)
    for carrier_signature, contexts in carrier_contexts.items():
        for context in contexts:
            carrier_games[carrier_signature].update(context_games.get(f"context:{context}", set()))
    return carrier_contexts, carrier_games


def _links_by_signature(
    state_conn: sqlite3.Connection,
    table: str,
    signature_column: str,
) -> dict[str, dict[str, set[str]]]:
    results: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in state_conn.execute(
        f"""
        SELECT {signature_column} AS signature_value, linked_type, linked_key
        FROM {table}
        ORDER BY {signature_column} ASC, linked_type ASC, linked_key ASC
        """
    ).fetchall():
        if row["linked_key"] not in (None, ""):
            results[str(row["signature_value"])][str(row["linked_type"])].add(str(row["linked_key"]))
    return results


def _fetch_edges_for_nodes(graph_conn: sqlite3.Connection, node_ids: set[str], batch_size: int = 500) -> list[sqlite3.Row]:
    if not node_ids:
        return []
    ordered = sorted(node_ids)
    rows: list[sqlite3.Row] = []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            graph_conn.execute(
                f"""
                SELECT source_node_id, target_node_id, edge_type
                FROM graph_edges
                WHERE source_node_id IN ({placeholders})
                   OR target_node_id IN ({placeholders})
                ORDER BY source_node_id ASC, edge_type ASC, target_node_id ASC
                """,
                tuple(batch) + tuple(batch),
            ).fetchall()
        )
    seen: set[tuple[str, str, str]] = set()
    deduped: list[sqlite3.Row] = []
    for row in rows:
        key = (str(row["source_node_id"]), str(row["target_node_id"]), str(row["edge_type"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _graph_nodes_for_ids(graph_conn: sqlite3.Connection, node_ids: set[str]) -> dict[str, str]:
    if not node_ids:
        return {}
    ordered = sorted(node_ids)
    results: dict[str, str] = {}
    for start in range(0, len(ordered), 500):
        batch = ordered[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        for row in graph_conn.execute(
            f"""
            SELECT node_id, node_type
            FROM graph_nodes
            WHERE node_id IN ({placeholders})
            ORDER BY node_id ASC
            """,
            tuple(batch),
        ).fetchall():
            results[str(row["node_id"])] = str(row["node_type"] or "unknown")
    return results


def _context_games_for_context_nodes(graph_conn: sqlite3.Connection, context_node_ids: set[str]) -> dict[str, set[str]]:
    context_games: dict[str, set[str]] = defaultdict(set)
    if not context_node_ids:
        return context_games
    ordered = sorted(context_node_ids)
    for start in range(0, len(ordered), 500):
        batch = ordered[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        for row in graph_conn.execute(
            f"""
            SELECT source_node_id, target_node_id
            FROM graph_edges
            WHERE edge_type = 'observed_in'
              AND target_node_id IN ({placeholders})
            ORDER BY source_node_id ASC, target_node_id ASC
            """,
            tuple(batch),
        ).fetchall():
            source = str(row["source_node_id"])
            target = str(row["target_node_id"])
            if source.startswith("game:") and target.startswith("context:"):
                context_games[target].add(source[len("game:"):])
    return context_games


def _motifs_for_text(text: str) -> set[str]:
    motif_map = {
        "enable": ("enable", "open", "unlock", "add", "create", "reveal", "allow", "access"),
        "block": ("block", "close", "obstacle", "wall", "prevent", "forbid", "stop"),
        "terminate": ("remove", "delete", "clear", "destroy", "consume", "vanish", "end"),
        "reversible": ("reverse", "restore", "toggle", "swap", "move_back", "undo"),
        "transform": ("transform", "recolor", "change", "convert", "shift"),
    }
    return {label for label, keywords in motif_map.items() if any(keyword in text for keyword in keywords)}


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


def _concept_tokens(role_type: str, tokens: list[str]) -> list[str]:
    concept_tokens = {f"concept_type:{role_type}"}
    for token in tokens:
        if token.startswith(("future_motif:", "fam_effect:", "fam_action:")):
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
        ON CONFLICT({signature_column}, linked_type, linked_key)
        DO UPDATE SET
            support_count = COALESCE({table}.support_count, 0) + excluded.support_count,
            first_seen_global_step = CASE
                WHEN {table}.first_seen_global_step IS NULL THEN excluded.first_seen_global_step
                WHEN excluded.first_seen_global_step IS NULL THEN {table}.first_seen_global_step
                ELSE MIN({table}.first_seen_global_step, excluded.first_seen_global_step)
            END,
            last_seen_global_step = CASE
                WHEN {table}.last_seen_global_step IS NULL THEN excluded.last_seen_global_step
                WHEN excluded.last_seen_global_step IS NULL THEN {table}.last_seen_global_step
                ELSE MAX({table}.last_seen_global_step, excluded.last_seen_global_step)
            END
        """,
        (signature, linked_type, linked_key, support_count, first_seen, last_seen),
    )


def _write_milestone(connection: sqlite3.Connection, name: str, first_global_step: int | None, evidence_key: str | None) -> None:
    connection.execute(
        """
        INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key)
        VALUES (?, ?, ?)
        ON CONFLICT(milestone_name)
        DO UPDATE SET first_global_step = excluded.first_global_step, evidence_key = excluded.evidence_key
        """,
        (name, first_global_step, evidence_key),
    )


def _load_token_json(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _safe_scalar_float(connection: sqlite3.Connection, query: str) -> float | None:
    row = connection.execute(query).fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def _empty_transfer_summary() -> dict[str, Any]:
    return {
        "transfer_attempt_count": 0,
        "successful_transfer_count": 0,
        "successful_role_count": 0,
        "role_mismatch_count": 0,
        "low_similarity_count": 0,
        "insufficient_source_support_count": 0,
        "no_source_profile_count": 0,
        "cross_game_attempt_count": 0,
        "cross_game_success_count": 0,
        "cross_context_attempt_count": 0,
        "cross_context_success_count": 0,
        "mean_transfer_score": None,
        "max_transfer_score": None,
        "mean_best_margin": None,
        "mean_source_carrier_count": None,
        "candidate_role_count_mean": None,
    }

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
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
    "concept_promotion_validation_diagnostics",
    "world_model_components",
    "world_model_links",
    "higher_order_milestones",
)

ROLE_ONLY_CLEAR_TABLES = (
    "role_neighborhood_signatures",
    "role_candidates",
    "role_links",
    "role_transfer_attempts",
    "concept_candidates",
    "concept_links",
    "concept_promotion_validation_diagnostics",
    "world_model_components",
    "world_model_links",
    "higher_order_milestones",
)

ROLE_TRANSFER_ONLY_CLEAR_TABLES = (
    "role_transfer_attempts",
    "concept_candidates",
    "concept_links",
    "concept_promotion_validation_diagnostics",
    "world_model_components",
    "world_model_links",
)

CONCEPT_ONLY_CLEAR_TABLES = (
    "concept_candidates",
    "concept_links",
    "concept_promotion_validation_diagnostics",
    "world_model_components",
    "world_model_links",
)

WORLD_MODEL_ONLY_CLEAR_TABLES = (
    "world_model_components",
    "world_model_links",
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


@dataclass(frozen=True)
class IncrementalPromotionValidationConfig:
    """Thresholds for optional held-out promotion validation.

    This layer is deliberately separate from the existing role, concept, and
    world-model scoring formulas.  It is enabled only by the Phase 3 feature
    flag and therefore leaves legacy promotion behaviour unchanged by default.
    """

    enabled: bool = False
    min_incremental_coverage: float = 0.05
    min_cross_context_or_game_evidence: int = 2
    min_behavioral_or_predictive_lift: float = 0.01
    demotion_failure_limit: int = 2


def derive_higher_order_memory(
    *,
    memory_dir: Path,
    run_dir: Path | None = None,
    max_carriers: int = 25_000,
    max_roles: int = 10_000,
    max_transfer_attempts: int = 25_000,
    workers: int = 1,
    chunk_size: int = 5_000,
    progress_factory: Any | None = None,
    incremental_promotion_validation: IncrementalPromotionValidationConfig | None = None,
) -> dict[str, Any]:
    role_summary = derive_role_candidates_only(
        memory_dir=memory_dir,
        run_dir=run_dir,
        max_carriers=max_carriers,
        max_roles=max_roles,
        progress_factory=progress_factory,
    )
    transfer_summary = derive_role_transfer_attempts_only(
        memory_dir=memory_dir,
        run_dir=run_dir,
        max_transfer_attempts=max_transfer_attempts,
        workers=workers,
        chunk_size=chunk_size,
        progress_factory=progress_factory,
    )
    concept_summary = derive_concept_candidates_only(memory_dir=memory_dir, run_dir=run_dir, progress_factory=progress_factory)
    validation_config = incremental_promotion_validation or IncrementalPromotionValidationConfig()
    promotion_summary: dict[str, Any] = {}
    if validation_config.enabled:
        promotion_summary = validate_incremental_promotions_only(
            memory_dir=memory_dir,
            config=validation_config,
            validate_roles_and_concepts=True,
            validate_world_models=False,
        )
    world_summary = derive_world_model_components_only(memory_dir=memory_dir, run_dir=run_dir, progress_factory=progress_factory)
    if validation_config.enabled:
        world_promotion_summary = validate_incremental_promotions_only(
            memory_dir=memory_dir,
            config=validation_config,
            validate_roles_and_concepts=False,
            validate_world_models=True,
        )
        promotion_summary["world_model_components_demoted"] = int(
            world_promotion_summary.get("world_model_components_demoted", 0) or 0
        )
    return {**role_summary, **transfer_summary, **concept_summary, **world_summary, **promotion_summary}


def derive_role_candidates_only(
    *,
    memory_dir: Path,
    run_dir: Path | None = None,
    max_carriers: int = 25_000,
    max_roles: int = 10_000,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    del run_dir
    return _run_higher_order_stage(
        memory_dir=memory_dir,
        clear_tables=ROLE_ONLY_CLEAR_TABLES,
        runner=lambda state_conn, graph_conn: derive_role_candidates(
            state_conn,
            graph_conn,
            max_carriers=max_carriers,
            max_roles=max_roles,
            progress_factory=progress_factory,
        ),
    )


def derive_role_transfer_attempts_only(
    *,
    memory_dir: Path,
    run_dir: Path | None = None,
    max_transfer_attempts: int = 25_000,
    workers: int = 1,
    chunk_size: int = 5_000,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    del run_dir
    paths = ensure_memory_layout(memory_dir)
    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
    ):
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        for table in ROLE_TRANSFER_ONLY_CLEAR_TABLES:
            state_conn.execute(f"DELETE FROM {table}")
        summary = derive_role_transfer_attempts_parallel(
            state_conn,
            graph_conn,
            max_transfer_attempts=max_transfer_attempts,
            workers=workers,
            chunk_size=chunk_size,
            progress_factory=progress_factory,
        )
        state_conn.execute(
            """
            INSERT INTO memory_summary (key, value_json)
            VALUES ('higher_order_transfer_summary', ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (json.dumps(summary, sort_keys=True),),
        )
        state_conn.commit()
        graph_conn.commit()
        return summary


def derive_concept_candidates_only(
    *,
    memory_dir: Path,
    run_dir: Path | None = None,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    del run_dir
    return _run_higher_order_stage(
        memory_dir=memory_dir,
        clear_tables=CONCEPT_ONLY_CLEAR_TABLES,
        runner=lambda state_conn, graph_conn: derive_concept_candidates(state_conn, progress_factory=progress_factory),
    )


def derive_world_model_components_only(
    *,
    memory_dir: Path,
    run_dir: Path | None = None,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    del run_dir
    return _run_higher_order_stage(
        memory_dir=memory_dir,
        clear_tables=WORLD_MODEL_ONLY_CLEAR_TABLES,
        runner=lambda state_conn, graph_conn: derive_world_model_components(state_conn, graph_conn, progress_factory=progress_factory),
    )


def derive_role_candidates(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_carriers: int,
    max_roles: int,
    progress_factory: Any | None = None,
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
    edge_rows = _fetch_edges_for_nodes(graph_conn, relevant_node_ids, progress_factory=progress_factory)
    graph_nodes = _graph_nodes_for_ids(
        graph_conn,
        {node_id for row in edge_rows for node_id in (str(row["source_node_id"]), str(row["target_node_id"]))} | relevant_node_ids,
        progress_factory=progress_factory,
    )
    context_games = _context_games_for_context_nodes(graph_conn, context_node_ids, progress_factory=progress_factory)
    edges_out_by_source: dict[str, list[sqlite3.Row]] = defaultdict(list)
    edges_in_by_target: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in edge_rows:
        edges_out_by_source[str(row["source_node_id"])].append(row)
        edges_in_by_target[str(row["target_node_id"])].append(row)

    neighborhoods: list[CarrierNeighborhood] = []
    carrier_tracker = progress_factory("derive_role_candidates carriers", len(carrier_rows), "carrier", False) if progress_factory else None
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
        if carrier_tracker is not None:
            carrier_tracker.update(1)
    _close_progress_tracker(carrier_tracker)

    grouped: dict[str, list[CarrierNeighborhood]] = defaultdict(list)
    for item in neighborhoods:
        grouped[item.role_signature].append(item)
    selected_role_signatures = sorted(grouped)[: int(max_roles)]

    emergent_role_count = 0
    stable_role_count = 0
    first_role_candidate_step: int | None = None
    first_emergent_role_step: int | None = None
    role_tracker = progress_factory("derive_role_candidates roles", len(selected_role_signatures), "role", False) if progress_factory else None
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
        if role_tracker is not None:
            role_tracker.update(1)
    _close_progress_tracker(role_tracker)

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
    return derive_role_transfer_attempts_parallel(
        state_conn,
        graph_conn,
        max_transfer_attempts=max_transfer_attempts,
        workers=1,
        chunk_size=max(1, int(max_transfer_attempts or 1)),
    )


def derive_role_transfer_attempts_parallel(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_transfer_attempts: int,
    *,
    workers: int,
    chunk_size: int,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    max_attempts_per_role = 1000
    max_attempts_per_target_scope = 1000
    cross_game_quota_ratio = 0.5
    cross_context_quota_ratio = 0.5
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
    target_tracker = progress_factory("derive_role_transfer_attempt_specs", len(carrier_signatures), "carrier", False) if progress_factory else None
    for carrier_signature in carrier_signatures:
        for scope_key in sorted(carrier_games.get(carrier_signature, set())):
            target_attempt_specs.append((carrier_signature, "cross_game", scope_key))
            unique_scopes.add(("cross_game", scope_key))
        for scope_key in sorted(carrier_contexts.get(carrier_signature, set())):
            target_attempt_specs.append((carrier_signature, "cross_context", scope_key))
            unique_scopes.add(("cross_context", scope_key))
        if target_tracker is not None:
            target_tracker.update(1, extra={"target_attempt_specs": len(target_attempt_specs)})
    _close_progress_tracker(target_tracker, extra={"target_attempt_specs": len(target_attempt_specs)})
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

    total_possible_transfer_attempts = len(target_attempt_specs)
    target_attempt_specs = _sample_transfer_attempt_specs(
        target_attempt_specs=target_attempt_specs,
        role_rows=role_rows,
        max_transfer_attempts=max_transfer_attempts,
        max_attempts_per_role=max_attempts_per_role,
        max_attempts_per_target_scope=max_attempts_per_target_scope,
        cross_game_quota_ratio=cross_game_quota_ratio,
        cross_context_quota_ratio=cross_context_quota_ratio,
    )
    sampled_cross_game_attempt_count = sum(1 for _, kind, _ in target_attempt_specs if kind == "cross_game")
    sampled_cross_context_attempt_count = sum(1 for _, kind, _ in target_attempt_specs if kind == "cross_context")
    chunks = [
        target_attempt_specs[index:index + max(1, int(chunk_size))]
        for index in range(0, len(target_attempt_specs), max(1, int(chunk_size)))
    ]
    attempt_rows: list[tuple[Any, ...]] = []
    if int(workers or 1) <= 1 or len(chunks) <= 1:
        chunk_tracker = progress_factory("derive_role_transfer chunks", len(chunks), "chunk", False) if progress_factory else None
        for chunk in chunks:
            attempt_rows.extend(
                _derive_role_transfer_attempts_chunk(
                    chunk=chunk,
                    role_rows=role_rows,
                    profile_cache=profile_cache,
                )
            )
            if chunk_tracker is not None:
                chunk_tracker.update(1)
        _close_progress_tracker(chunk_tracker)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [
                executor.submit(
                    _derive_role_transfer_attempts_chunk,
                    chunk=chunk,
                    role_rows=role_rows,
                    profile_cache=profile_cache,
                )
                for chunk in chunks
            ]
            chunk_tracker = progress_factory("derive_role_transfer chunks", len(futures), "chunk", False) if progress_factory else None
            for future in futures:
                attempt_rows.extend(list(future.result()))
                if chunk_tracker is not None:
                    chunk_tracker.update(1)
            _close_progress_tracker(chunk_tracker)
    state_conn.executemany(
        """
        INSERT INTO role_transfer_attempts (
            attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
            target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature,
            observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason,
            best_margin, source_carrier_count, candidate_role_count, first_seen_global_step, last_seen_global_step
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        attempt_rows,
    )
    inserted = len(attempt_rows)
    for row in attempt_rows:
        transfer_kind = str(row[2])
        transfer_score = float(row[11] or 0.0)
        reuse_success = int(row[12] or 0)
        failure_reason = str(row[13] or "")
        best_margin = row[14]
        source_carrier_count = int(row[15] or 0)
        candidate_role_count = int(row[16] or 0)
        first_seen_global_step = row[17]
        role_signature = str(row[1])
        transfer_score_sum += transfer_score
        source_carrier_count_sum += source_carrier_count
        candidate_role_count_sum += candidate_role_count
        if best_margin is not None:
            best_margin_sum += float(best_margin)
            best_margin_count += 1
        if first_seen_global_step is not None:
            first_attempt_step = (
                first_seen_global_step
                if first_attempt_step is None
                else min(first_attempt_step, int(first_seen_global_step))
            )
        if transfer_kind == "cross_game":
            cross_game_attempt_count += 1
        else:
            cross_context_attempt_count += 1
        if failure_reason == "role_mismatch":
            role_mismatch_count += 1
        elif failure_reason == "low_similarity":
            low_similarity_count += 1
        elif failure_reason == "insufficient_source_support":
            insufficient_source_support_count += 1
        elif failure_reason == "no_source_profile":
            no_source_profile_count += 1
        if reuse_success == 1:
            success_count += 1
            successful_roles.add(str(row[9]))
            if transfer_kind == "cross_game":
                cross_game_success_count += 1
            else:
                cross_context_success_count += 1
            if first_seen_global_step is not None:
                first_success_step = (
                    first_seen_global_step
                    if first_success_step is None
                    else min(first_success_step, int(first_seen_global_step))
                )

    _write_milestone(state_conn, "first_role_transfer_attempt_step", first_attempt_step, None)
    _write_milestone(state_conn, "first_role_transfer_success_step", first_success_step, None)
    return {
        "transfer_attempt_count": inserted,
        "total_possible_transfer_attempts": total_possible_transfer_attempts,
        "sampled_transfer_attempts": len(target_attempt_specs),
        "skipped_by_cap_count": max(0, total_possible_transfer_attempts - len(target_attempt_specs)),
        "sampled_cross_game_attempt_count": sampled_cross_game_attempt_count,
        "sampled_cross_context_attempt_count": sampled_cross_context_attempt_count,
        "transfer_sampling_strategy": "stratified_balanced",
        "max_attempts_per_role": max_attempts_per_role,
        "max_attempts_per_target_scope": max_attempts_per_target_scope,
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


def _sample_transfer_attempt_specs(
    *,
    target_attempt_specs: list[tuple[str, str, str]],
    role_rows: dict[str, dict[str, Any]],
    max_transfer_attempts: int,
    max_attempts_per_role: int,
    max_attempts_per_target_scope: int,
    cross_game_quota_ratio: float,
    cross_context_quota_ratio: float,
) -> list[tuple[str, str, str]]:
    limit = max(0, int(max_transfer_attempts or 0))
    if limit <= 0 or len(target_attempt_specs) <= limit:
        return list(target_attempt_specs)

    buckets = {
        "cross_game": [spec for spec in target_attempt_specs if spec[1] == "cross_game"],
        "cross_context": [spec for spec in target_attempt_specs if spec[1] == "cross_context"],
    }
    quotas = {
        "cross_game": min(len(buckets["cross_game"]), int(round(limit * float(cross_game_quota_ratio)))),
        "cross_context": min(len(buckets["cross_context"]), int(round(limit * float(cross_context_quota_ratio)))),
    }
    assigned = quotas["cross_game"] + quotas["cross_context"]
    if assigned < limit:
        for kind in ("cross_game", "cross_context"):
            available = len(buckets[kind]) - quotas[kind]
            take = min(available, limit - assigned)
            quotas[kind] += take
            assigned += take
            if assigned >= limit:
                break

    selected: list[tuple[str, str, str]] = []
    selected_set: set[tuple[str, str, str]] = set()
    role_counts: dict[str, int] = defaultdict(int)
    scope_counts: dict[tuple[str, str], int] = defaultdict(int)

    def _take(kind: str, quota: int) -> None:
        for carrier_signature, transfer_kind, scope_key in buckets[kind]:
            if len(selected) >= limit or quota <= 0:
                break
            role_signature = str(role_rows.get(carrier_signature, {}).get("role_signature") or carrier_signature)
            scope_tuple = (transfer_kind, scope_key)
            spec = (carrier_signature, transfer_kind, scope_key)
            if spec in selected_set:
                continue
            if role_counts[role_signature] >= int(max_attempts_per_role):
                continue
            if scope_counts[scope_tuple] >= int(max_attempts_per_target_scope):
                continue
            selected.append(spec)
            selected_set.add(spec)
            role_counts[role_signature] += 1
            scope_counts[scope_tuple] += 1
            quota -= 1

    _take("cross_game", quotas["cross_game"])
    _take("cross_context", quotas["cross_context"])
    if len(selected) < limit:
        for spec in target_attempt_specs:
            if len(selected) >= limit:
                break
            carrier_signature, transfer_kind, scope_key = spec
            role_signature = str(role_rows.get(carrier_signature, {}).get("role_signature") or carrier_signature)
            scope_tuple = (transfer_kind, scope_key)
            if spec in selected_set:
                continue
            if role_counts[role_signature] >= int(max_attempts_per_role):
                continue
            if scope_counts[scope_tuple] >= int(max_attempts_per_target_scope):
                continue
            selected.append(spec)
            selected_set.add(spec)
            role_counts[role_signature] += 1
            scope_counts[scope_tuple] += 1
    return selected[:limit]


def derive_concept_candidates(state_conn: sqlite3.Connection, progress_factory: Any | None = None) -> dict[str, Any]:
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
            "roles_seen_for_concept_derivation": 0,
            "roles_skipped_missing_carrier_links": 0,
            "roles_skipped_missing_family_links": 0,
            "roles_skipped_missing_transfer_success": 0,
            "roles_used_for_concepts": 0,
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
    roles_seen_for_concept_derivation = 0
    roles_skipped_missing_carrier_links = 0
    roles_skipped_missing_family_links = 0
    roles_skipped_missing_transfer_success = 0
    roles_used_for_concepts = 0
    role_tracker = progress_factory("derive_concept_candidates roles", len(role_rows), "role", False) if progress_factory else None
    for row in role_rows:
        role_signature = str(row["role_signature"])
        roles_seen_for_concept_derivation += 1
        links = role_links.get(role_signature, {})
        if not links.get("carrier"):
            roles_skipped_missing_carrier_links += 1
            continue
        if not links.get("family"):
            roles_skipped_missing_family_links += 1
            continue
        if int(success_by_role.get(role_signature, 0)) <= 0:
            roles_skipped_missing_transfer_success += 1
            continue
        roles_used_for_concepts += 1
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
        if role_tracker is not None:
            role_tracker.update(1)
    _close_progress_tracker(role_tracker)

    promoted_concepts = 0
    overconcentrated_concepts = 0
    promoted_overconcentrated_concepts = 0
    first_concept_candidate_step: int | None = None
    first_promoted_concept_step: int | None = None
    strong_transfer_total = 0
    concept_strong_counts: list[int] = []
    concept_items = sorted(concept_groups)
    concept_tracker = progress_factory("derive_concept_candidates groups", len(concept_items), "group", False) if progress_factory else None
    for concept_signature in concept_items:
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
    for concept_signature in concept_items:
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
            and linked_role_count >= 1
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
        if concept_tracker is not None:
            concept_tracker.update(1)
    _close_progress_tracker(concept_tracker)
    _write_milestone(state_conn, "first_concept_candidate_step", first_concept_candidate_step, None)
    _write_milestone(state_conn, "first_promoted_concept_step", first_promoted_concept_step, None)
    return {
        "concept_candidate_count": len(concept_groups),
        "promoted_concept_count": promoted_concepts,
        "concept_strong_transfer_success_count": strong_transfer_total,
        "overconcentrated_concept_count": overconcentrated_concepts,
        "promoted_overconcentrated_concept_count": promoted_overconcentrated_concepts,
        "roles_seen_for_concept_derivation": roles_seen_for_concept_derivation,
        "roles_skipped_missing_carrier_links": roles_skipped_missing_carrier_links,
        "roles_skipped_missing_family_links": roles_skipped_missing_family_links,
        "roles_skipped_missing_transfer_success": roles_skipped_missing_transfer_success,
        "roles_used_for_concepts": roles_used_for_concepts,
        "concept_transfer_success_concentration": (
            max(concept_strong_counts) / strong_transfer_total
            if concept_strong_counts and strong_transfer_total > 0
            else None
        ),
    }


def derive_world_model_components(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
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
    candidate_tracker = progress_factory("derive_world_model_components candidates", len(candidate_rows), "candidate", False) if progress_factory else None
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
            and linked_role_count >= 1
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
        if candidate_tracker is not None:
            candidate_tracker.update(1)
    _close_progress_tracker(candidate_tracker)

    _write_milestone(state_conn, "first_world_model_component_step", first_component_step, None)
    _write_milestone(state_conn, "first_coherent_world_model_step", first_coherent_step, None)
    return {
        "world_model_component_count": len(candidate_rows),
        "coherent_world_model_component_count": coherent_count,
        "candidate_only_world_model_component_count": candidate_only_count,
    }


def validate_incremental_promotions_only(
    *,
    memory_dir: Path,
    config: IncrementalPromotionValidationConfig,
    validate_roles_and_concepts: bool,
    validate_world_models: bool,
) -> dict[str, Any]:
    """Apply optional held-out validation without deleting failed candidates."""
    if not config.enabled:
        return {"incremental_promotion_validation_enabled": False}
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as state_conn:
        state_conn.row_factory = sqlite3.Row
        summary = _validate_incremental_promotions(
            state_conn,
            config=config,
            validate_roles_and_concepts=validate_roles_and_concepts,
            validate_world_models=validate_world_models,
        )
        state_conn.execute(
            """
            INSERT INTO memory_summary (key, value_json)
            VALUES ('incremental_promotion_validation_summary', ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (json.dumps(summary, sort_keys=True),),
        )
        state_conn.commit()
    return summary


def _validate_incremental_promotions(
    state_conn: sqlite3.Connection,
    *,
    config: IncrementalPromotionValidationConfig,
    validate_roles_and_concepts: bool,
    validate_world_models: bool,
) -> dict[str, Any]:
    """Calculate Phase 3 metrics from existing structures and later evidence.

    A later global step is the held-out boundary.  This prevents a concept from
    validating itself on the same role-transfer observations used to derive it.
    """
    summary: dict[str, Any] = {
        "incremental_promotion_validation_enabled": True,
        "concept_candidates_evaluated": 0,
        "concepts_rejected_no_incremental_gain": 0,
        "concepts_rejected_no_heldout_lift": 0,
        "concepts_promoted_with_behavioral_lift": 0,
        "concepts_demoted": 0,
        "world_model_components_demoted": 0,
        "role_candidates_evaluated": 0,
        "roles_promoted_with_behavioral_lift": 0,
        "roles_demoted": 0,
    }
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    family_rows = state_conn.execute(
        "SELECT canonical_signature, prediction_lift, last_seen_global_step FROM transformation_families"
    ).fetchall()
    family_prediction = {
        str(row["canonical_signature"]): (
            float(row["prediction_lift"] or 0.0),
            None if row["last_seen_global_step"] is None else int(row["last_seen_global_step"]),
        )
        for row in family_rows
    }
    transfer_rows = state_conn.execute(
        """
        SELECT role_signature, reuse_success, last_seen_global_step
        FROM role_transfer_attempts
        ORDER BY role_signature ASC, attempt_id ASC
        """
    ).fetchall()
    transfers_by_role: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in transfer_rows:
        transfers_by_role[str(row["role_signature"])].append(row)
    future_rows = state_conn.execute(
        """
        SELECT source_role_id, owner_type, owner_key, option_delta
        FROM future_option_events
        """
    ).fetchall()
    future_by_role: dict[str, list[float]] = defaultdict(list)
    for row in future_rows:
        role_signature = row["source_role_id"]
        if role_signature is None and str(row["owner_type"] or "") == "role":
            role_signature = row["owner_key"]
        if role_signature:
            future_by_role[str(role_signature)].append(float(row["option_delta"] or 0.0))

    role_metrics: dict[str, dict[str, float]] = {}
    if validate_roles_and_concepts:
        role_rows = state_conn.execute(
            """
            SELECT role_signature, linked_carrier_count, linked_family_count,
                   linked_context_count, cross_game_count, support_count
            FROM role_candidates ORDER BY role_signature ASC
            """
        ).fetchall()
        for row in role_rows:
            role_signature = str(row["role_signature"])
            links = role_links.get(role_signature, {})
            families = links.get("family", set())
            prediction_values = [family_prediction.get(family, (0.0, None))[0] for family in families]
            attempts = transfers_by_role.get(role_signature, [])
            transfer_lift = (
                sum(int(item["reuse_success"] or 0) for item in attempts) / len(attempts)
                if attempts
                else 0.0
            )
            explanatory_coverage = min(
                1.0,
                (len(families) + len(links.get("context", set())) + len(links.get("game", set())))
                / max(1.0, 3.0 * float(row["linked_carrier_count"] or 0)),
            )
            compression_gain = float(row["linked_carrier_count"] or 0) / max(1.0, float(row["linked_family_count"] or 0))
            prediction_lift = sum(prediction_values) / len(prediction_values) if prediction_values else 0.0
            future_values = future_by_role.get(role_signature, [])
            future_lift = sum(future_values) / len(future_values) if future_values else 0.0
            role_has_cross_evidence = max(
                len(links.get("context", set())), len(links.get("game", set()))
            ) >= int(config.min_cross_context_or_game_evidence)
            role_has_behavioral_lift = max(prediction_lift, future_lift, transfer_lift) >= float(
                config.min_behavioral_or_predictive_lift
            )
            role_promoted = role_has_cross_evidence and explanatory_coverage >= float(
                config.min_incremental_coverage
            ) and role_has_behavioral_lift
            role_status, role_failure_count, role_demoted = _update_promotion_validation_state(
                state_conn,
                candidate_type="role",
                candidate_signature=role_signature,
                passed=role_promoted,
                demotion_failure_limit=int(config.demotion_failure_limit),
                validation_scope="role_link_and_transfer_evidence",
                validation_prediction_lift=prediction_lift,
                validation_action_selection_lift=transfer_lift,
                validation_transfer_lift=transfer_lift,
                updated_global_step=None,
            )
            role_metrics[role_signature] = {
                "coverage": explanatory_coverage,
                "compression": compression_gain,
                "prediction": prediction_lift,
                "future": future_lift,
                "transfer": transfer_lift,
            }
            state_conn.execute(
                """
                UPDATE role_candidates
                SET role_explanatory_coverage = ?, role_compression_gain = ?,
                    role_prediction_lift = ?, role_future_option_lift = ?,
                    role_transfer_lift = ?, promotion_status = ?, promotion_failure_count = ?
                WHERE role_signature = ?
                """,
                (
                    explanatory_coverage, compression_gain, prediction_lift, future_lift,
                    transfer_lift, role_status, role_failure_count, role_signature,
                ),
            )
            if role_promoted:
                summary["roles_promoted_with_behavioral_lift"] += 1
            if role_demoted:
                summary["roles_demoted"] += 1
        summary["role_candidates_evaluated"] = len(role_rows)

        concept_rows = state_conn.execute(
            """
            SELECT concept_signature, compression_gain, explanatory_reach, promotion_score,
                   cross_context_count, cross_game_count, first_seen_global_step, is_promoted
            FROM concept_candidates ORDER BY concept_signature ASC
            """
        ).fetchall()
        for row in concept_rows:
            concept_signature = str(row["concept_signature"])
            links = concept_links.get(concept_signature, {})
            roles = sorted(links.get("role", set()))
            source_metrics = [role_metrics[role] for role in roles if role in role_metrics]
            baseline_coverage = _mean_metric(source_metrics, "coverage")
            baseline_compression = _mean_metric(source_metrics, "compression")
            baseline_prediction = _mean_metric(source_metrics, "prediction")
            baseline_future = _mean_metric(source_metrics, "future")
            baseline_transfer = _mean_metric(source_metrics, "transfer")
            candidate_coverage = min(1.0, float(row["explanatory_reach"] or 0.0) / 8.0)
            incremental_coverage = candidate_coverage - baseline_coverage
            incremental_compression = float(row["compression_gain"] or 0.0) - baseline_compression
            first_seen = None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"])
            heldout_attempts = [
                attempt
                for role in roles
                for attempt in transfers_by_role.get(role, [])
                if first_seen is not None
                and attempt["last_seen_global_step"] is not None
                and int(attempt["last_seen_global_step"]) > first_seen
            ]
            derivation_attempts = [
                attempt
                for role in roles
                for attempt in transfers_by_role.get(role, [])
                if first_seen is not None
                and attempt["last_seen_global_step"] is not None
                and int(attempt["last_seen_global_step"]) <= first_seen
            ]
            derivation_transfer_rate = (
                sum(int(attempt["reuse_success"] or 0) for attempt in derivation_attempts) / len(derivation_attempts)
                if derivation_attempts
                else baseline_transfer
            )
            validation_evidence_count = len(heldout_attempts)
            validation_transfer_lift = (
                (sum(int(attempt["reuse_success"] or 0) for attempt in heldout_attempts) / validation_evidence_count)
                - derivation_transfer_rate
                if validation_evidence_count
                else None
            )
            heldout_prediction = [
                lift
                for family in links.get("family", set())
                for lift, family_last_seen in (family_prediction.get(family, (0.0, None)),)
                if first_seen is not None and family_last_seen is not None and family_last_seen > first_seen
            ]
            validation_prediction_lift = (
                (sum(heldout_prediction) / len(heldout_prediction)) - baseline_prediction
                if heldout_prediction
                else None
            )
            validation_action_selection_lift = validation_transfer_lift
            concept_prediction_lift = (
                sum(family_prediction.get(family, (0.0, None))[0] for family in links.get("family", set()))
                / max(1, len(links.get("family", set())))
                - baseline_prediction
            )
            future_values = [value for role in roles for value in future_by_role.get(role, [])]
            future_lift = (sum(future_values) / len(future_values) if future_values else 0.0) - baseline_future
            cross_game_transfer_lift = validation_transfer_lift if validation_transfer_lift is not None else 0.0
            has_cross_evidence = max(int(row["cross_context_count"] or 0), int(row["cross_game_count"] or 0)) >= int(config.min_cross_context_or_game_evidence)
            has_incremental_gain = incremental_coverage >= float(config.min_incremental_coverage)
            heldout_lifts = [value for value in (validation_prediction_lift, validation_action_selection_lift, validation_transfer_lift) if value is not None]
            has_heldout_lift = bool(heldout_lifts) and max(heldout_lifts) >= float(config.min_behavioral_or_predictive_lift)
            legacy_promoted = bool(int(row["is_promoted"] or 0))
            promoted = legacy_promoted and has_cross_evidence and has_incremental_gain and has_heldout_lift
            status, failure_count, demoted_this_validation = _update_promotion_validation_state(
                state_conn,
                candidate_type="concept",
                candidate_signature=concept_signature,
                passed=promoted,
                demotion_failure_limit=int(config.demotion_failure_limit),
                validation_scope="later_global_step" if validation_evidence_count else "unavailable",
                validation_prediction_lift=validation_prediction_lift,
                validation_action_selection_lift=validation_action_selection_lift,
                validation_transfer_lift=validation_transfer_lift,
                updated_global_step=first_seen,
            )
            adjusted_promotion_score = (
                float(row["promotion_score"] or 0.0)
                if promoted
                else max(0.0, float(row["promotion_score"] or 0.0) - 0.10 * failure_count)
            )
            state_conn.execute(
                """
                UPDATE concept_candidates
                SET concept_incremental_coverage = ?, concept_incremental_compression_gain = ?,
                    concept_prediction_lift = ?, concept_future_option_prediction_lift = ?,
                    concept_cross_game_transfer_lift = ?, validation_scope = ?,
                    validation_prediction_lift = ?, validation_action_selection_lift = ?,
                    validation_transfer_lift = ?, validation_evidence_count = ?,
                    promotion_status = ?, promotion_failure_count = ?, promotion_score = ?, is_promoted = ?
                WHERE concept_signature = ?
                """,
                (
                    incremental_coverage, incremental_compression, concept_prediction_lift, future_lift,
                    cross_game_transfer_lift, "later_global_step" if validation_evidence_count else "unavailable",
                    validation_prediction_lift, validation_action_selection_lift, validation_transfer_lift,
                    validation_evidence_count, status, failure_count, adjusted_promotion_score,
                    int(promoted), concept_signature,
                ),
            )
            demoted = status == "demoted"
            rejection_reasons = _concept_promotion_rejection_reasons(
                promoted=promoted,
                legacy_promoted=legacy_promoted,
                has_incremental_gain=has_incremental_gain,
                has_cross_evidence=has_cross_evidence,
                heldout_lifts=heldout_lifts,
                has_heldout_lift=has_heldout_lift,
                demoted=demoted,
            )
            if promoted:
                assert not rejection_reasons
            else:
                assert rejection_reasons
            diagnostic = {
                "concept_id": concept_signature,
                "candidate_signature": concept_signature,
                "source_role_ids": roles,
                "source_carrier_ids": sorted(links.get("carrier", set())),
                "source_family_ids": sorted(links.get("family", set())),
                "derivation_evidence_count": len(derivation_attempts),
                "validation_evidence_count": validation_evidence_count,
                "validation_scope": "later_global_step" if validation_evidence_count else "unavailable",
                "incremental_explanatory_coverage": incremental_coverage,
                "incremental_compression_gain": incremental_compression,
                "prediction_lift": concept_prediction_lift,
                "future_option_prediction_lift": future_lift,
                "transfer_lift": cross_game_transfer_lift,
                "heldout_prediction_lift": validation_prediction_lift,
                "heldout_action_selection_lift": validation_action_selection_lift,
                "heldout_transfer_lift": validation_transfer_lift,
                "cross_context_evidence_count": int(row["cross_context_count"] or 0),
                "cross_game_evidence_count": int(row["cross_game_count"] or 0),
                "promotion_score": adjusted_promotion_score,
                "promotion_threshold": 0.55,
                "validation_pass": promoted,
                "promoted": promoted,
                "rejection_reasons": rejection_reasons,
                "consecutive_validation_failures": failure_count,
                "demoted": demoted,
                "demotion_reason": "demoted_after_repeated_failure" if demoted else None,
            }
            state_conn.execute(
                """
                INSERT INTO concept_promotion_validation_diagnostics (concept_signature, payload_json)
                VALUES (?, ?)
                ON CONFLICT(concept_signature) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (concept_signature, json.dumps(diagnostic, sort_keys=True)),
            )
            summary["concept_candidates_evaluated"] += 1
            if not has_incremental_gain:
                summary["concepts_rejected_no_incremental_gain"] += 1
            if not has_heldout_lift:
                summary["concepts_rejected_no_heldout_lift"] += 1
            if promoted:
                summary["concepts_promoted_with_behavioral_lift"] += 1
            if demoted_this_validation:
                summary["concepts_demoted"] += 1

    if validate_world_models:
        component_rows = state_conn.execute(
            """
            SELECT component_signature, linked_concept_count, first_seen_global_step, coherence_score, is_coherent
            FROM world_model_components ORDER BY component_signature ASC
            """
        ).fetchall()
        concept_validation = {
            str(row["concept_signature"]): dict(row)
            for row in state_conn.execute(
                """
                SELECT concept_signature, is_promoted, validation_prediction_lift,
                       validation_action_selection_lift, validation_transfer_lift
                FROM concept_candidates
                """
            ).fetchall()
        }
        component_links = _links_by_signature(state_conn, "world_model_links", "component_signature")
        for row in component_rows:
            component_signature = str(row["component_signature"])
            concepts = component_links.get(component_signature, {}).get("concept", set())
            validations = [concept_validation[item] for item in concepts if item in concept_validation]
            prediction_lift = _mean_row_metric(validations, "validation_prediction_lift")
            action_lift = _mean_row_metric(validations, "validation_action_selection_lift")
            transfer_lift = _mean_row_metric(validations, "validation_transfer_lift")
            passed = bool(validations) and all(int(item.get("is_promoted", 0) or 0) == 1 for item in validations)
            status, failure_count, demoted = _update_promotion_validation_state(
                state_conn,
                candidate_type="world_model",
                candidate_signature=component_signature,
                passed=passed,
                demotion_failure_limit=int(config.demotion_failure_limit),
                validation_scope="derived_concept_heldout",
                validation_prediction_lift=prediction_lift,
                validation_action_selection_lift=action_lift,
                validation_transfer_lift=transfer_lift,
                updated_global_step=None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"]),
            )
            coherent = bool(int(row["is_coherent"] or 0)) and passed
            adjusted_coherence_score = (
                float(row["coherence_score"] or 0.0)
                if passed
                else max(0.0, float(row["coherence_score"] or 0.0) - 0.10 * failure_count)
            )
            state_conn.execute(
                """
                UPDATE world_model_components
                SET validation_prediction_lift = ?, validation_action_selection_lift = ?,
                    validation_transfer_lift = ?, promotion_status = ?, promotion_failure_count = ?,
                    coherence_score = ?, candidate_only = ?, is_coherent = ?
                WHERE component_signature = ?
                """,
                (
                    prediction_lift, action_lift, transfer_lift, status, failure_count,
                    adjusted_coherence_score, int(not coherent), int(coherent), component_signature,
                ),
            )
            if demoted:
                summary["world_model_components_demoted"] += 1
    return summary


def _mean_metric(items: list[dict[str, float]], key: str) -> float:
    return sum(float(item.get(key, 0.0) or 0.0) for item in items) / max(1, len(items))


def _mean_row_metric(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return (sum(values) / len(values)) if values else None


def _concept_promotion_rejection_reasons(
    *,
    promoted: bool,
    legacy_promoted: bool,
    has_incremental_gain: bool,
    has_cross_evidence: bool,
    heldout_lifts: list[float],
    has_heldout_lift: bool,
    demoted: bool,
) -> list[str]:
    """Expose the exact Phase 3 gate outcomes without changing the gates."""
    if promoted:
        return []
    reasons: list[str] = []
    if not has_incremental_gain:
        reasons.append("no_incremental_coverage")
    if not has_cross_evidence:
        reasons.append("insufficient_cross_context_or_game_evidence")
    if not heldout_lifts:
        reasons.append("no_heldout_samples")
    elif not has_heldout_lift:
        reasons.extend(("no_predictive_or_behavioral_lift", "heldout_validation_failed"))
    if not legacy_promoted:
        reasons.append("below_promotion_threshold")
    if demoted:
        reasons.append("demoted_after_repeated_failure")
    return list(dict.fromkeys(reasons)) or ["below_promotion_threshold"]


def _update_promotion_validation_state(
    state_conn: sqlite3.Connection,
    *,
    candidate_type: str,
    candidate_signature: str,
    passed: bool,
    demotion_failure_limit: int,
    validation_scope: str,
    validation_prediction_lift: float | None,
    validation_action_selection_lift: float | None,
    validation_transfer_lift: float | None,
    updated_global_step: int | None,
) -> tuple[str, int, bool]:
    previous = state_conn.execute(
        """
        SELECT failure_count, promotion_status FROM promotion_validation_state
        WHERE candidate_type = ? AND candidate_signature = ?
        """,
        (candidate_type, candidate_signature),
    ).fetchone()
    previous_failures = int(previous["failure_count"] or 0) if previous is not None else 0
    previous_status = str(previous["promotion_status"] or "candidate") if previous is not None else "candidate"
    if passed:
        status, failure_count, demoted = "promoted", 0, False
    else:
        failure_count = previous_failures + 1
        previously_promoted = previous_status in {"promoted", "validation_failed"}
        demoted = previously_promoted and failure_count >= max(1, demotion_failure_limit)
        status = "demoted" if demoted else ("validation_failed" if previously_promoted else "candidate")
    state_conn.execute(
        """
        INSERT INTO promotion_validation_state (
            candidate_type, candidate_signature, failure_count, promotion_status,
            last_validation_scope, last_validation_prediction_lift,
            last_validation_action_selection_lift, last_validation_transfer_lift, updated_global_step
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_type, candidate_signature) DO UPDATE SET
            failure_count = excluded.failure_count,
            promotion_status = excluded.promotion_status,
            last_validation_scope = excluded.last_validation_scope,
            last_validation_prediction_lift = excluded.last_validation_prediction_lift,
            last_validation_action_selection_lift = excluded.last_validation_action_selection_lift,
            last_validation_transfer_lift = excluded.last_validation_transfer_lift,
            updated_global_step = excluded.updated_global_step
        """,
        (
            candidate_type, candidate_signature, failure_count, status, validation_scope,
            validation_prediction_lift, validation_action_selection_lift, validation_transfer_lift,
            updated_global_step,
        ),
    )
    return status, failure_count, demoted


def _run_higher_order_stage(
    *,
    memory_dir: Path,
    clear_tables: tuple[str, ...],
    runner: Any,
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
    ):
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        for table in clear_tables:
            state_conn.execute(f"DELETE FROM {table}")
        summary = dict(runner(state_conn, graph_conn) or {})
        state_conn.commit()
        graph_conn.commit()
        return summary


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
    best: dict[str, Any] | None = None
    second: dict[str, Any] | None = None
    best_similarity = -1.0
    second_similarity = -1.0
    candidate_role_count = 0
    for profile in profile_cache.get((transfer_kind, target_scope_key), []):
        if (
            int(profile.get("source_carrier_count") or 0) == 1
            and str(profile.get("role_signature") or "") == str(target["role_signature"])
        ):
            continue
        candidate_role_count += 1
        similarity_score = _jaccard(profile["profile_token_set"], target_tokens)
        if _transfer_candidate_is_better(profile, similarity_score, best, best_similarity):
            second = best
            second_similarity = best_similarity
            best = profile
            best_similarity = similarity_score
        elif _transfer_candidate_is_better(profile, similarity_score, second, second_similarity):
            second = profile
            second_similarity = similarity_score
    if best is None:
        return _no_source_profile_attempt(target, transfer_kind, target_scope_key)
    best_margin = None if second is None else float(best_similarity) - float(second_similarity)
    predicted_role_signature = str(best["role_signature"])
    observed_role_signature = str(target["role_signature"])
    source_carrier_count = int(best["source_carrier_count"])
    similarity_score = float(best_similarity)
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


def _attempt_to_insert_tuple(attempt: dict[str, Any]) -> tuple[Any, ...]:
    return (
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
    )


def _derive_role_transfer_attempts_chunk(
    *,
    chunk: list[tuple[str, str, str]],
    role_rows: dict[str, dict[str, Any]],
    profile_cache: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for carrier_signature, transfer_kind, target_scope_key in chunk:
        attempt = _predict_transfer_attempt(
            profile_cache=profile_cache,
            role_rows=role_rows,
            target_carrier_signature=carrier_signature,
            transfer_kind=transfer_kind,
            target_scope_key=target_scope_key,
        )
        rows.append(_attempt_to_insert_tuple(attempt))
    return rows


def _build_transfer_profile_cache(
    *,
    role_rows: dict[str, dict[str, Any]],
    carrier_contexts: dict[str, set[str]],
    carrier_games: dict[str, set[str]],
    target_scopes: list[tuple[str, str]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    limited_target_scopes = list(target_scopes)
    if len(limited_target_scopes) > 25000:
        limited_target_scopes = limited_target_scopes[:25000]
    role_to_carriers: dict[str, list[str]] = defaultdict(list)
    carrier_to_tokens: dict[str, tuple[str, ...]] = {}
    cross_game_excluded: dict[str, set[str]] = defaultdict(set)
    cross_context_excluded: dict[str, set[str]] = defaultdict(set)
    for candidate_signature, candidate in sorted(role_rows.items(), key=lambda item: item[0]):
        role_to_carriers[str(candidate["role_signature"])].append(candidate_signature)
        carrier_to_tokens[candidate_signature] = tuple(candidate["tokens"])
        for scope_key in carrier_games.get(candidate_signature, set()):
            cross_game_excluded[str(scope_key)].add(candidate_signature)
        for scope_key in carrier_contexts.get(candidate_signature, set()):
            cross_context_excluded[str(scope_key)].add(candidate_signature)
    for transfer_kind, target_scope_key in limited_target_scopes:
        excluded_carriers = (
            cross_game_excluded.get(target_scope_key, set())
            if transfer_kind == "cross_game"
            else cross_context_excluded.get(target_scope_key, set())
        )
        profiles: list[dict[str, Any]] = []
        for role_signature in sorted(role_to_carriers):
            included_carriers = [carrier for carrier in role_to_carriers[role_signature] if carrier not in excluded_carriers]
            if not included_carriers:
                continue
            token_set: set[str] = set()
            context_set: set[str] = set()
            game_set: set[str] = set()
            for carrier_signature in included_carriers:
                token_set.update(carrier_to_tokens.get(carrier_signature, ()))
                context_set.update(carrier_contexts.get(carrier_signature, set()))
                game_set.update(carrier_games.get(carrier_signature, set()))
            profiles.append(
                {
                    "role_signature": role_signature,
                    "profile_tokens": sorted(token_set),
                    "profile_token_set": token_set,
                    "source_carrier_count": len(included_carriers),
                    "source_context_count": len(context_set),
                    "source_game_count": len(game_set),
                }
            )
        cache[(transfer_kind, target_scope_key)] = profiles
    return cache


def _transfer_candidate_is_better(
    profile: dict[str, Any],
    similarity_score: float,
    incumbent: dict[str, Any] | None,
    incumbent_similarity: float,
) -> bool:
    if incumbent is None:
        return True
    if float(similarity_score) != float(incumbent_similarity):
        return float(similarity_score) > float(incumbent_similarity)
    profile_count = int(profile.get("source_carrier_count") or 0)
    incumbent_count = int(incumbent.get("source_carrier_count") or 0)
    if profile_count != incumbent_count:
        return profile_count > incumbent_count
    return str(profile.get("role_signature") or "") < str(incumbent.get("role_signature") or "")


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


def _fetch_edges_for_nodes(
    graph_conn: sqlite3.Connection,
    node_ids: set[str],
    batch_size: int = 500,
    progress_factory: Any | None = None,
) -> list[sqlite3.Row]:
    if not node_ids:
        return []
    ordered = sorted(node_ids)
    rows: list[sqlite3.Row] = []
    batch_total = (len(ordered) + batch_size - 1) // batch_size
    tracker = progress_factory("derive_role_candidates fetch_edges", batch_total, "batch", False) if progress_factory else None
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
        if tracker is not None:
            tracker.update(1)
    _close_progress_tracker(tracker)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[sqlite3.Row] = []
    for row in rows:
        key = (str(row["source_node_id"]), str(row["target_node_id"]), str(row["edge_type"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _graph_nodes_for_ids(graph_conn: sqlite3.Connection, node_ids: set[str], progress_factory: Any | None = None) -> dict[str, str]:
    if not node_ids:
        return {}
    ordered = sorted(node_ids)
    results: dict[str, str] = {}
    batch_total = (len(ordered) + 499) // 500
    tracker = progress_factory("derive_role_candidates graph_nodes", batch_total, "batch", False) if progress_factory else None
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
        if tracker is not None:
            tracker.update(1)
    _close_progress_tracker(tracker)
    return results


def _context_games_for_context_nodes(
    graph_conn: sqlite3.Connection,
    context_node_ids: set[str],
    progress_factory: Any | None = None,
) -> dict[str, set[str]]:
    context_games: dict[str, set[str]] = defaultdict(set)
    if not context_node_ids:
        return context_games
    ordered = sorted(context_node_ids)
    batch_total = (len(ordered) + 499) // 500
    tracker = progress_factory("derive_role_candidates context_games", batch_total, "batch", False) if progress_factory else None
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
        if tracker is not None:
            tracker.update(1)
    _close_progress_tracker(tracker)
    return context_games


def _close_progress_tracker(tracker: Any | None, *, extra: dict[str, Any] | None = None) -> None:
    if tracker is None:
        return
    close = getattr(tracker, "close", None)
    if callable(close):
        close(extra=extra)


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

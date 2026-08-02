from __future__ import annotations

import json
import math
import sqlite3
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from v6.memory.compact_memory import (
    INCREMENTAL_PROMOTION_VALIDATION_VERSION,
    ensure_memory_layout,
)


CONCEPT_PROMOTION_SCORE_THRESHOLD = 0.55
MIN_SOURCE_EVIDENCE_SUPPORT = 2
MIN_TRANSFER_SIMILARITY = 0.60
MAX_WORLD_MODEL_FAMILY_LINKS = 50


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean_optional(values: list[Any]) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


ROLE_CLEAR_TABLES = (
    "role_neighborhood_signatures",
    "role_candidates",
    "role_links",
    "role_transfer_attempts",
    "concept_candidates",
    "concept_links",
    "world_model_components",
    "world_model_links",
    "world_model_family_links",
    "future_option_transfer_links",
    "higher_order_milestones",
)

ROLE_ONLY_CLEAR_TABLES = (
    "role_neighborhood_signatures",
    "role_candidates",
    "role_links",
    "role_transfer_attempts",
    "concept_candidates",
    "concept_links",
    "world_model_components",
    "world_model_links",
    "world_model_family_links",
    "future_option_transfer_links",
    "higher_order_milestones",
)

ROLE_TRANSFER_ONLY_CLEAR_TABLES = (
    "role_transfer_attempts",
    "concept_candidates",
    "concept_links",
    "world_model_components",
    "world_model_links",
    "world_model_family_links",
    "future_option_transfer_links",
)

CONCEPT_ONLY_CLEAR_TABLES = (
    "concept_candidates",
    "concept_links",
    "world_model_components",
    "world_model_links",
    "world_model_family_links",
)

WORLD_MODEL_ONLY_CLEAR_TABLES = (
    "world_model_components",
    "world_model_links",
    "world_model_family_links",
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
    min_incremental_explanatory_coverage: float = 0.05
    min_event_prediction_gain: float = 0.01
    min_event_behavioral_gain: float = 0.01
    min_event_compression_gain: float = 0.01
    min_explanation_event_count: int = 1
    min_relevant_heldout_event_count: int = 20
    promotion_population_comparability_threshold: float = 0.80
    min_cross_context_or_game_evidence: int = 2
    min_behavioral_or_predictive_lift: float = 0.01
    demotion_failure_limit: int = 2
    promotion_score_threshold: float = CONCEPT_PROMOTION_SCORE_THRESHOLD


@dataclass(frozen=True)
class _TransferHistorySeries:
    steps: tuple[int, ...]
    success_prefix: tuple[int, ...]

    def rate_before(self, step: int) -> tuple[float, int]:
        count = bisect_left(self.steps, int(step))
        if count <= 0:
            return 0.0, 0
        return float(self.success_prefix[count]) / float(count), count

    def max_step_before(self, step: int) -> int | None:
        count = bisect_left(self.steps, int(step))
        return self.steps[count - 1] if count else None


@dataclass(frozen=True)
class _TransferHistoryIndex:
    by_role: dict[str, _TransferHistorySeries]
    by_source_target_scope: dict[tuple[str, str, str, str, str], _TransferHistorySeries]
    all_rows: _TransferHistorySeries

    def rate_before(
        self,
        *,
        role: str,
        step: int,
        source_game_key: str | None = None,
        source_context_key: str | None = None,
        target_game_key: str | None = None,
        target_context_key: str | None = None,
    ) -> tuple[float, int]:
        scope_key = (role, source_game_key or "", source_context_key or "", target_game_key or "", target_context_key or "")
        series = self.by_source_target_scope.get(scope_key) if any(scope_key[1:]) else self.by_role.get(role)
        return series.rate_before(step) if series is not None else (0.0, 0)

    def max_step_before(self, *, role: str, step: int) -> int | None:
        series = self.by_role.get(role)
        return series.max_step_before(step) if series is not None else None

    def max_any_step_before(self, step: int) -> int | None:
        return self.all_rows.max_step_before(step)


def _build_transfer_history_index(
    transfer_rows: list[sqlite3.Row],
) -> _TransferHistoryIndex:
    """Build immutable pre-event transfer histories once per validation run.

    Phase 3 only consults evidence strictly before an event's global step.  A
    sorted prefix index is therefore equivalent to the old row scans while
    making each role/scope lookup logarithmic rather than linear in all
    transfer attempts.
    """
    grouped: dict[tuple[str, str, str, str, str] | tuple[str], list[tuple[int, int]]] = defaultdict(list)
    all_values: list[tuple[int, int]] = []
    for row in transfer_rows:
        step = row["last_seen_global_step"]
        role = str(row["role_signature"] or "")
        if step is None or not role:
            continue
        value = (int(step), int(row["reuse_success"] or 0))
        grouped[(role,)].append(value)
        grouped[(
            role,
            str(row["source_game_key"] or ""),
            str(row["source_context_key"] or ""),
            str(row["target_game_key"] or ""),
            str(row["target_context_key"] or ""),
        )].append(value)
        all_values.append(value)

    def make_series(values: list[tuple[int, int]]) -> _TransferHistorySeries:
        values.sort()
        successes = [0]
        for _step, success in values:
            successes.append(successes[-1] + success)
        return _TransferHistorySeries(
            steps=tuple(step for step, _success in values),
            success_prefix=tuple(successes),
        )

    by_role: dict[str, _TransferHistorySeries] = {}
    by_source_target_scope: dict[tuple[str, str, str, str, str], _TransferHistorySeries] = {}
    for key, values in grouped.items():
        series = make_series(values)
        if len(key) == 1:
            by_role[key[0]] = series
        else:
            by_source_target_scope[(key[0], key[1], key[2], key[3], key[4])] = series
    return _TransferHistoryIndex(
        by_role=by_role,
        by_source_target_scope=by_source_target_scope,
        all_rows=make_series(all_values),
    )


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
    schema_paths = ensure_memory_layout(memory_dir)
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
            validation_state_reset_applied_this_run=schema_paths.validation_state_reset_applied_this_run,
        )
    world_summary = derive_world_model_components_only(memory_dir=memory_dir, run_dir=run_dir, progress_factory=progress_factory)
    if validation_config.enabled:
        world_promotion_summary = validate_incremental_promotions_only(
            memory_dir=memory_dir,
            config=validation_config,
            validate_roles_and_concepts=False,
            validate_world_models=True,
            validation_state_reset_applied_this_run=False,
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
        # This is a current-state milestone, not a historical first-ever
        # record.  A fresh zero-success derivation must not inherit it.
        state_conn.execute(
            "DELETE FROM higher_order_milestones WHERE milestone_name = 'first_role_transfer_success_step'"
        )
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
    max_world_model_family_links: int = MAX_WORLD_MODEL_FAMILY_LINKS,
) -> dict[str, Any]:
    del run_dir
    if int(max_world_model_family_links) < 1:
        raise ValueError("max_world_model_family_links must be positive")
    return _run_higher_order_stage(
        memory_dir=memory_dir,
        clear_tables=WORLD_MODEL_ONLY_CLEAR_TABLES,
        runner=lambda state_conn, graph_conn: derive_world_model_components(
            state_conn,
            graph_conn,
            progress_factory=progress_factory,
            max_world_model_family_links=int(max_world_model_family_links),
        ),
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

    carrier_contexts, carrier_games, context_games = _carrier_scope_maps(state_conn, graph_conn)
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
    for carrier_signature, profile_row in role_rows.items():
        contexts = tuple(sorted(carrier_contexts.get(carrier_signature, set())))
        profile_row["contexts"] = contexts
        profile_row["games"] = tuple(sorted(carrier_games.get(carrier_signature, set())))
        profile_row["context_games"] = {
            context: tuple(sorted(context_games.get(context, set())))
            for context in contexts
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
        context_games=context_games,
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
    source_evidence_support_count_sum = 0
    candidate_role_count_sum = 0
    cross_game_attempt_count = 0
    cross_game_success_count = 0
    cross_context_attempt_count = 0
    cross_context_success_count = 0
    scope_surrogate_counts: dict[str, int] = defaultdict(int)
    scope_resolution_counts: dict[str, int] = defaultdict(int)

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
    attempt_items: list[dict[str, Any]] = []
    if int(workers or 1) <= 1 or len(chunks) <= 1:
        chunk_tracker = progress_factory("derive_role_transfer chunks", len(chunks), "chunk", False) if progress_factory else None
        for chunk in chunks:
            attempt_items.extend(
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
                attempt_items.extend(list(future.result()))
                if chunk_tracker is not None:
                    chunk_tracker.update(1)
            _close_progress_tracker(chunk_tracker)
    # Expansion converts a selected aggregate profile into concrete
    # source-target evidence.  Keep the configured cap over persisted rows,
    # not merely the pre-expansion target requests.
    attempt_items = [
        _resolve_transfer_attempt_provenance(state_conn, item)
        for item in attempt_items
    ]
    for item in attempt_items:
        _validate_transfer_attempt_provenance(item)
    attempt_ids = [str(item["attempt_id"]) for item in attempt_items]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("role transfer attempt identity collision before persistence")
    attempt_items.sort(key=lambda item: str(item["attempt_id"]))
    if len(attempt_items) > int(max_transfer_attempts):
        attempt_items = attempt_items[: int(max_transfer_attempts)]
    attempt_rows = [_attempt_to_insert_tuple(item) for item in attempt_items]
    state_conn.executemany(
        """
        INSERT INTO role_transfer_attempts (
            attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
            target_scope_type, target_scope_key, source_game_key, target_game_key,
            source_context_key, target_context_key,
            source_interaction_id, target_interaction_id, source_scope_origin, target_scope_origin,
            source_game_is_surrogate, target_game_is_surrogate,
            source_context_is_surrogate, target_context_is_surrogate,
            source_game_resolution_source, target_game_resolution_source,
            source_context_resolution_source, target_context_resolution_source,
            source_carrier_signature, source_role_signature,
            predicted_target_role_signature, observed_target_role_signature,
            source_carrier_signatures_json, source_game_keys_json, source_context_keys_json,
            provenance_mode, provenance_status, target_carrier_signature, predicted_role_signature,
            observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason,
            best_margin, source_carrier_count, source_evidence_support_count,
            support_gate_passed, similarity_gate_passed, role_match_gate_passed,
            candidate_role_count, first_seen_global_step, last_seen_global_step
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        attempt_rows,
    )
    inserted = len(attempt_items)
    for item in attempt_items:
        transfer_kind = str(item["transfer_kind"])
        transfer_score = float(item["transfer_score"] or 0.0)
        reuse_success = int(item["reuse_success"] or 0)
        failure_reason = str(item["failure_reason"] or "")
        best_margin = item["best_margin"]
        source_carrier_count = int(item["source_carrier_count"] or 0)
        source_evidence_support_count = int(item["source_evidence_support_count"] or 0)
        candidate_role_count = int(item["candidate_role_count"] or 0)
        first_seen_global_step = item["first_seen_global_step"]
        role_signature = str(item["source_role_signature"] or "")
        for prefix in ("source_game", "target_game", "source_context", "target_context"):
            if int(item.get(f"{prefix}_is_surrogate") or 0):
                scope_surrogate_counts[prefix] += 1
            scope_resolution_counts[str(item.get(f"{prefix}_resolution_source") or "unknown")] += 1
        transfer_score_sum += transfer_score
        source_carrier_count_sum += source_carrier_count
        source_evidence_support_count_sum += source_evidence_support_count
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
            successful_roles.add(role_signature)
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
    _write_milestone(state_conn, "first_role_transfer_success_step", first_success_step if success_count else None, None)
    if success_count == 0 and first_success_step is not None:
        raise AssertionError("zero successful transfers must not retain a success milestone")
    return {
        "transfer_attempt_count": inserted,
        "candidate_transfer_attempt_count": total_possible_transfer_attempts,
        "sampled_profile_attempt_count": len(target_attempt_specs),
        "expanded_transfer_attempt_count": len(attempt_ids),
        "persisted_transfer_attempt_count": inserted,
        # Deprecated aliases retain the pre-expansion population definition.
        "total_possible_transfer_attempts": total_possible_transfer_attempts,
        "sampled_transfer_attempts": len(target_attempt_specs),
        "skipped_by_cap_count": max(0, total_possible_transfer_attempts - len(target_attempt_specs)),
        "sampled_cross_game_attempt_count": cross_game_attempt_count,
        "sampled_cross_context_attempt_count": cross_context_attempt_count,
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
        "mean_source_evidence_support_count": (source_evidence_support_count_sum / inserted) if inserted else None,
        "candidate_role_count_mean": (candidate_role_count_sum / inserted) if inserted else None,
        "source_game_missing_before_resolution_count": sum(1 for item in attempt_items if item.get("source_game_resolution_source") != "direct_attempt"),
        "target_game_missing_before_resolution_count": sum(1 for item in attempt_items if item.get("target_game_resolution_source") != "direct_attempt"),
        "source_context_missing_before_resolution_count": sum(1 for item in attempt_items if item.get("source_context_resolution_source") != "direct_attempt"),
        "target_context_missing_before_resolution_count": sum(1 for item in attempt_items if item.get("target_context_resolution_source") != "direct_attempt"),
        "source_game_surrogate_count": scope_surrogate_counts["source_game"],
        "target_game_surrogate_count": scope_surrogate_counts["target_game"],
        "source_context_surrogate_count": scope_surrogate_counts["source_context"],
        "target_context_surrogate_count": scope_surrogate_counts["target_context"],
        "scope_resolution_source_counts": dict(sorted(scope_resolution_counts.items())),
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


def _restore_persistent_concept_state(state_conn: sqlite3.Connection) -> None:
    """Reattach durable promotion state to this epoch's candidate snapshot."""
    candidate_rows = state_conn.execute(
        """
        SELECT concept_signature, first_seen_global_step, is_promoted,
               structurally_promoted_this_epoch
        FROM concept_candidates ORDER BY concept_signature ASC
        """
    ).fetchall()
    for candidate in candidate_rows:
        signature = str(candidate["concept_signature"])
        structural_promoted = int(
            candidate["structurally_promoted_this_epoch"]
            if candidate["structurally_promoted_this_epoch"] is not None
            else candidate["is_promoted"] or 0
        )
        state = state_conn.execute(
            "SELECT * FROM concept_promotion_state WHERE concept_signature = ?", (signature,)
        ).fetchone()
        if state is None:
            status = "promoted" if structural_promoted else "candidate"
            state_conn.execute(
                """
                INSERT INTO concept_promotion_state (
                    concept_signature, historically_promoted, currently_promoted,
                    promotion_status, first_promoted_global_step,
                    validation_algorithm_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    signature, structural_promoted, structural_promoted, status,
                    candidate["first_seen_global_step"] if structural_promoted else None,
                    INCREMENTAL_PROMOTION_VALIDATION_VERSION,
                ),
            )
            continue
        historically_promoted = bool(state["historically_promoted"] or 0)
        currently_promoted = bool(state["currently_promoted"] or 0)
        previous_status = str(state["promotion_status"] or "candidate")
        previous_validation = str(state["validation_status"] or "")
        retained = bool(
            historically_promoted
            and currently_promoted
            and previous_status in {"promoted", "retained", "validated"}
            and previous_validation not in {"failed", "demoted", "invalid"}
        )
        if retained:
            state_conn.execute(
                """
                UPDATE concept_candidates
                SET is_promoted = 1, promotion_status = 'retained',
                    validation_status = ?, promotion_failure_count = ?,
                    demotion_reason = ?, promotion_retained_from_history = 1
                WHERE concept_signature = ?
                """,
                (
                    state["validation_status"], state["consecutive_validation_failures"],
                    state["last_demotion_reason"], signature,
                ),
            )
        elif structural_promoted:
            state_conn.execute(
                """
                UPDATE concept_promotion_state
                SET historically_promoted = 1, currently_promoted = 1,
                    promotion_status = 'promoted',
                    first_promoted_global_step = COALESCE(first_promoted_global_step, ?),
                    validation_algorithm_version = ?, updated_at = datetime('now')
                WHERE concept_signature = ?
                """,
                (candidate["first_seen_global_step"], INCREMENTAL_PROMOTION_VALIDATION_VERSION, signature),
            )


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
            "roles_with_transfer_attempts": 0,
            "roles_with_successful_transfers": 0,
            "roles_eligible_for_concept_derivation": 0,
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
        SELECT source_role_signature AS role_signature, reuse_success, similarity_score, best_margin, source_evidence_support_count, candidate_role_count
        FROM role_transfer_attempts
        WHERE provenance_mode IN ('single_source', 'multi_source')
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
                int(row["source_evidence_support_count"] or 0) >= MIN_SOURCE_EVIDENCE_SUPPORT
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
            and promotion_score >= CONCEPT_PROMOTION_SCORE_THRESHOLD
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
                   first_seen_global_step, last_seen_global_step, is_promoted,
                   structurally_promoted_this_epoch, promotion_retained_from_history,
                   promotion_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                is_promoted,
                0,
                "promoted" if is_promoted else "candidate",
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
    _restore_persistent_concept_state(state_conn)
    active_promoted_step = state_conn.execute(
        "SELECT MIN(first_seen_global_step) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
    ).fetchone()[0]
    first_promoted_concept_step = (
        None if active_promoted_step is None else int(active_promoted_step)
    )
    promoted_concepts = int(state_conn.execute(
        "SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
    ).fetchone()[0])
    _write_milestone(state_conn, "first_concept_candidate_step", first_concept_candidate_step, None)
    _write_milestone(state_conn, "first_promoted_concept_step", first_promoted_concept_step, None)
    _write_historical_milestone(state_conn, "first_concept_candidate_step", first_concept_candidate_step, None)
    _write_historical_milestone(state_conn, "first_promoted_concept_step", first_promoted_concept_step, None)
    return {
        "concept_candidate_count": len(concept_groups),
        "promoted_concept_count": promoted_concepts,
        "concept_strong_transfer_success_count": strong_transfer_total,
        "overconcentrated_concept_count": overconcentrated_concepts,
        "promoted_overconcentrated_concept_count": promoted_overconcentrated_concepts,
        "roles_seen_for_concept_derivation": roles_seen_for_concept_derivation,
        "roles_with_transfer_attempts": len({
            str(row["role_signature"]) for row in transfer_rows
        }),
        "roles_with_successful_transfers": len(success_by_role),
        "roles_eligible_for_concept_derivation": roles_used_for_concepts,
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
    max_world_model_family_links: int = MAX_WORLD_MODEL_FAMILY_LINKS,
) -> dict[str, Any]:
    concept_rows = [dict(row) for row in state_conn.execute(
        """
        SELECT candidate.concept_signature, candidate.concept_type, candidate.linked_role_count,
               candidate.linked_carrier_count, candidate.linked_family_count,
               candidate.cross_context_count, candidate.cross_game_count, candidate.promotion_score,
               candidate.first_seen_global_step, candidate.last_seen_global_step,
               COALESCE(persistent.currently_promoted, candidate.is_promoted, 0) AS is_promoted
        FROM concept_candidates AS candidate
        LEFT JOIN concept_promotion_state AS persistent
          ON persistent.concept_signature = candidate.concept_signature
        ORDER BY COALESCE(persistent.currently_promoted, candidate.is_promoted, 0) DESC,
                 COALESCE(candidate.promotion_score, 0.0) DESC, candidate.concept_signature ASC
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
        SELECT source_role_signature AS role_signature, reuse_success, source_evidence_support_count, candidate_role_count, similarity_score, best_margin
        FROM role_transfer_attempts
        WHERE provenance_mode IN ('single_source', 'multi_source')
        ORDER BY role_signature ASC, attempt_id ASC
        """
    ).fetchall():
        if (
            int(row["reuse_success"] or 0) == 1
            and int(row["source_evidence_support_count"] or 0) >= MIN_SOURCE_EVIDENCE_SUPPORT
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
        candidate_families = sorted(links.get("family", set()))
        contexts = sorted(links.get("context", set()))
        games = sorted(links.get("game", set()))
        family_candidates: list[dict[str, Any]] = []
        for family in candidate_families:
            support_row = state_conn.execute(
                "SELECT COALESCE(SUM(support_count), 0) FROM family_members WHERE family_signature = ?",
                (family,),
            ).fetchone()
            role_count = sum(
                1 for role in roles if family in role_links.get(role, {}).get("family", set())
            )
            event_count = int(state_conn.execute(
                "SELECT COUNT(*) FROM future_option_events WHERE source_family_id = ?", (family,)
            ).fetchone()[0])
            prediction_gain_row = state_conn.execute(
                "SELECT AVG(prediction_lift) FROM transformation_families WHERE canonical_signature = ?", (family,)
            ).fetchone()
            prediction_gain = float(prediction_gain_row[0] or 0.0)
            family_candidates.append({
                "family": family,
                "support": int(support_row[0] or 0), "role_count": role_count,
                "event_count": event_count, "prediction_gain": prediction_gain,
                "provenance_status": "verified" if event_count > 0 else "proxy",
            })
        eligible_families = [
            item for item in family_candidates
            if item["event_count"] >= 2 or item["prediction_gain"] > 0.0 or item["role_count"] >= 2
        ]
        eligible_families.sort(key=lambda item: (
            0 if item["provenance_status"] == "verified" else 1,
            -float(item["prediction_gain"]), -int(item["support"]), str(item["family"]),
        ))
        retained_family_records = eligible_families[:int(max_world_model_family_links)]
        families = [str(item["family"]) for item in retained_family_records]
        family_links_dropped_low_support = len(family_candidates) - len(eligible_families)
        family_links_dropped_limit = len(eligible_families) - len(retained_family_records)
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
        structural_prediction_support_count = len(families) + sum(successful_transfer_by_role.get(role, 0) for role in roles)
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
        explanatory_coverage = (linked_family_count + linked_carrier_count + contradiction_coverage_count) / max(1, node_count)
        component_signature = f"wm:{sha1(concept_signature.encode('utf-8')).hexdigest()[:20]}"
        prediction_rows = state_conn.execute(
            """
            SELECT * FROM world_model_prediction_events
            WHERE component_signature = ?
              AND observed_event_id IS NOT NULL AND observed_global_step IS NOT NULL
              AND prediction_global_step < observed_global_step
            ORDER BY prediction_event_id ASC
            """,
            (component_signature,),
        ).fetchall()
        unmatched_prediction_event_count = int(state_conn.execute(
            """SELECT COUNT(*) FROM world_model_prediction_events
               WHERE component_signature = ? AND (observed_event_id IS NULL OR observed_global_step IS NULL
                    OR prediction_global_step >= observed_global_step)""",
            (component_signature,),
        ).fetchone()[0])
        observed_outcome_count = len(prediction_rows)
        prediction_event_count = observed_outcome_count + unmatched_prediction_event_count
        correct_prediction_count = sum(int(item["prediction_correct"] or 0) for item in prediction_rows)
        prediction_error_count = observed_outcome_count - correct_prediction_count
        baseline_prediction_score = _mean_optional([item["baseline_prediction_score"] for item in prediction_rows])
        component_prediction_score = _mean_optional([item["component_prediction_score"] for item in prediction_rows])
        prediction_accuracy = correct_prediction_count / max(1, observed_outcome_count)
        heldout_prediction_gain = (
            float(component_prediction_score) - float(baseline_prediction_score)
            if component_prediction_score is not None and baseline_prediction_score is not None
            else 0.0
        )
        prediction_evidence_status = (
            "verified" if observed_outcome_count
            else "proxy" if prediction_event_count
            else "missing"
        )
        edge_density = graph_edge_count / max(1, node_count * max(1, node_count - 1))
        total_link_count = linked_role_count + linked_family_count + linked_carrier_count + len(contexts) + len(games)
        verified_link_count = sum(1 for item in retained_family_records if item["provenance_status"] == "verified") + linked_role_count + linked_carrier_count
        verified_link_ratio = verified_link_count / max(1, total_link_count)
        scope_generalization = (cross_context_count + cross_game_count) / max(1, len(contexts) + len(games))
        contradiction_resolution_rate = 0.0
        structural_coherence_score = _clamp01(0.5 * edge_density + 0.5 * verified_link_ratio)
        functional_coherence_score = _clamp01(
            0.30 * prediction_accuracy
            + 0.30 * max(0.0, heldout_prediction_gain)
            + 0.25 * scope_generalization
            + 0.15 * contradiction_resolution_rate
        )
        coherence_score = _clamp01(0.4 * structural_coherence_score + 0.6 * functional_coherence_score)
        is_promoted = int(row.get("is_promoted", 0) or 0) == 1
        candidate_only = 0 if is_promoted else 1
        predicted_outcome_count = observed_outcome_count
        predicted_outcome_count_is_proxy = 0
        is_coherent = int(
            is_promoted
            and linked_role_count >= 1
            and linked_family_count >= 2
            and linked_carrier_count >= 2
            and (cross_context_count >= 3 or cross_game_count >= 2)
            and observed_outcome_count > 0
            and functional_coherence_score > 0.0
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
        component_type = "promoted_concept_component" if is_promoted else "candidate_concept_component"
        state_conn.execute(
            """
            INSERT INTO world_model_components (
                component_signature, component_type, node_count, edge_count, linked_concept_count,
                linked_role_count, linked_family_count, linked_carrier_count, cross_context_count,
                cross_game_count, explanatory_coverage, prediction_support_count, contradiction_coverage_count,
                coherence_score, candidate_only, predicted_outcome_count, predicted_outcome_count_is_proxy,
                first_seen_global_step, last_seen_global_step, is_coherent,
                structural_prediction_support_count, observed_outcome_count, correct_prediction_count,
                prediction_error_count, prediction_evidence_status, baseline_prediction_score,
                component_prediction_score, heldout_prediction_gain, matched_prediction_event_count,
                unmatched_prediction_event_count, structural_coherence_score, functional_coherence_score,
                combined_coherence_score, candidate_family_link_count, retained_family_link_count,
                dropped_family_link_count, family_links_dropped_low_support, family_links_dropped_limit
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                # The legacy column is retained for readers of existing
                # compact stores, but it is no longer a structural proxy for
                # prediction evidence.  Concrete matched rows above are the
                # only source for predicted_outcome_count.
                0,
                contradiction_coverage_count,
                coherence_score,
                candidate_only,
                predicted_outcome_count,
                predicted_outcome_count_is_proxy,
                first_seen,
                last_seen,
                is_coherent,
                structural_prediction_support_count,
                observed_outcome_count,
                correct_prediction_count,
                prediction_error_count,
                prediction_evidence_status,
                baseline_prediction_score,
                component_prediction_score,
                heldout_prediction_gain,
                observed_outcome_count,
                unmatched_prediction_event_count,
                structural_coherence_score,
                functional_coherence_score,
                coherence_score,
                len(family_candidates),
                len(retained_family_records),
                family_links_dropped_low_support + family_links_dropped_limit,
                family_links_dropped_low_support,
                family_links_dropped_limit,
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
        for family_record in retained_family_records:
            state_conn.execute(
                """
                INSERT INTO world_model_family_links (
                    component_signature, family_signature, family_link_support_count,
                    family_link_role_count, family_link_event_count, family_link_prediction_gain,
                    family_link_provenance_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(component_signature, family_signature) DO UPDATE SET
                    family_link_support_count = excluded.family_link_support_count,
                    family_link_role_count = excluded.family_link_role_count,
                    family_link_event_count = excluded.family_link_event_count,
                    family_link_prediction_gain = excluded.family_link_prediction_gain,
                    family_link_provenance_status = excluded.family_link_provenance_status
                """,
                (
                    component_signature, family_record["family"], family_record["support"],
                    family_record["role_count"], family_record["event_count"],
                    family_record["prediction_gain"], family_record["provenance_status"],
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
                historically_coherent = MAX(world_model_component_state.historically_coherent, excluded.historically_coherent),
                currently_coherent = excluded.currently_coherent,
                first_coherent_global_step = COALESCE(world_model_component_state.first_coherent_global_step, excluded.first_coherent_global_step),
                last_validated_global_step = excluded.last_validated_global_step,
                consecutive_validation_failures = CASE WHEN excluded.currently_coherent = 1 THEN 0 ELSE world_model_component_state.consecutive_validation_failures END,
                validation_status = excluded.validation_status,
                updated_at = excluded.updated_at
            """,
            (
                component_signature, is_coherent, is_coherent,
                first_seen if is_coherent else None, last_seen, 0,
                "passed" if is_coherent else "missing_prediction_evidence" if not observed_outcome_count else "failed",
            ),
        )
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
    diagnostic_epoch_id: str | int | None = None,
    explanation_events_path: Path | None = None,
    validation_state_reset_applied_this_run: bool | None = None,
) -> dict[str, Any]:
    """Apply optional held-out validation without deleting failed candidates."""
    if not config.enabled:
        return {"incremental_promotion_validation_enabled": False}
    paths = ensure_memory_layout(memory_dir)
    if validation_state_reset_applied_this_run is None:
        validation_state_reset_applied_this_run = paths.validation_state_reset_applied_this_run
    with sqlite3.connect(paths.current_state) as state_conn:
        state_conn.row_factory = sqlite3.Row
        summary = _validate_incremental_promotions(
            state_conn,
            config=config,
            validate_roles_and_concepts=validate_roles_and_concepts,
            validate_world_models=validate_world_models,
            diagnostic_epoch_id=diagnostic_epoch_id,
            explanation_events_path=explanation_events_path,
            validation_state_reset_applied_this_run=bool(validation_state_reset_applied_this_run),
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
    diagnostic_epoch_id: str | int | None,
    explanation_events_path: Path | None,
    validation_state_reset_applied_this_run: bool,
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
    last_reset_version_row = state_conn.execute(
        "SELECT value_json FROM memory_summary WHERE key = 'incremental_promotion_validation_last_reset_version'"
    ).fetchone()
    try:
        validation_state_last_reset_version = (
            int(json.loads(str(last_reset_version_row[0]))) if last_reset_version_row else None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        validation_state_last_reset_version = None
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    role_explained_structures_by_role = {
        role_signature: _linked_explained_structures(links)
        for role_signature, links in role_links.items()
    }
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
        SELECT attempt_id, source_role_signature AS role_signature, reuse_success, last_seen_global_step,
               observed_target_role_signature AS observed_role_signature,
               predicted_target_role_signature AS predicted_role_signature,
               source_carrier_signature, target_carrier_signature,
               source_game_key, target_game_key, source_context_key, target_context_key,
               provenance_mode
        FROM role_transfer_attempts
        WHERE provenance_mode = 'single_source'
        ORDER BY role_signature ASC, attempt_id ASC
        """
    ).fetchall()
    transfer_history = _build_transfer_history_index(transfer_rows)
    transfers_by_role: dict[str, list[sqlite3.Row]] = defaultdict(list)
    successful_transfer_target_roles_by_role: dict[str, set[str]] = defaultdict(set)
    for row in transfer_rows:
        source_role = str(row["role_signature"])
        transfers_by_role[source_role].append(row)
        if int(row["reuse_success"] or 0) != 1:
            continue
        target_role = row["observed_role_signature"] or row["predicted_role_signature"]
        if target_role and str(target_role) != source_role:
            successful_transfer_target_roles_by_role[source_role].add(str(target_role))
    future_rows = state_conn.execute(
        """
        SELECT event_id, source_role_id, owner_type, owner_key, option_delta,
               first_seen_global_step, last_seen_global_step
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
    previous_coverage_states = _load_incremental_coverage_states(state_conn)
    explanation_event_rows: list[dict[str, Any]] = []

    role_metrics: dict[str, dict[str, Any]] = {}
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
                validation_epoch=diagnostic_epoch_id,
            )
            role_metrics[role_signature] = {
                "coverage": explanatory_coverage,
                "compression": compression_gain,
                "prediction": prediction_lift,
                "future": future_lift,
                "transfer": transfer_lift,
                "families": sorted(families),
                "structures": sorted(role_explained_structures_by_role.get(role_signature, set())),
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
            SELECT concept_candidates.concept_signature, compression_gain, explanatory_reach, promotion_score, raw_promotion_score,
                   cross_context_count, cross_game_count, first_seen_global_step, is_promoted,
                   structurally_promoted_this_epoch, promotion_retained_from_history,
                   COALESCE(persistent.historically_promoted, 0) AS persistent_historically_promoted,
                   COALESCE(persistent.currently_promoted, 0) AS persistent_currently_promoted,
                   persistent.promotion_status AS persistent_promotion_status,
                   persistent.validation_status AS persistent_validation_status,
                   persistent.consecutive_validation_failures AS persistent_failure_count,
                   persistent.total_validation_failures AS persistent_total_failure_count
            FROM concept_candidates
            LEFT JOIN concept_promotion_state AS persistent
              ON persistent.concept_signature = concept_candidates.concept_signature
            ORDER BY concept_candidates.concept_signature ASC
            """
        ).fetchall()
        source_role_structures_by_candidate: dict[str, set[str]] = {}
        current_structure_ids_by_candidate: dict[str, set[str]] = {}
        relevance_links_by_candidate: dict[str, dict[str, set[str]]] = {}
        for concept_row in concept_rows:
            concept_signature = str(concept_row["concept_signature"])
            links = concept_links.get(concept_signature, {})
            roles = sorted(links.get("role", set()))
            relevance_links: dict[str, set[str]] = defaultdict(set)
            for link_type, identifiers in links.items():
                relevance_links[link_type].update(identifiers)
            relevance_links["_candidate_family"].update(links.get("family", set()))
            for role in roles:
                for link_type, identifiers in role_links.get(role, {}).items():
                    relevance_links[link_type].update(identifiers)
                    if link_type == "family":
                        relevance_links["_candidate_role_family"].update(identifiers)
            relevance_links_by_candidate[concept_signature] = {
                link_type: set(identifiers)
                for link_type, identifiers in relevance_links.items()
            }
            source_role_structures = set().union(
                *(role_explained_structures_by_role.get(role, set()) for role in roles)
            ) if roles else set()
            target_roles = set().union(
                *(successful_transfer_target_roles_by_role.get(role, set()) for role in roles)
            ) if roles else set()
            target_role_structures = set().union(
                *(role_explained_structures_by_role.get(role, set()) for role in target_roles)
            ) if target_roles else set()
            current_structure_ids_by_candidate[concept_signature] = (
                _linked_explained_structures(links)
                | source_role_structures
                | target_role_structures
            )
            source_role_structures_by_candidate[concept_signature] = source_role_structures
        for row in concept_rows:
            concept_signature = str(row["concept_signature"])
            links = concept_links.get(concept_signature, {})
            roles = sorted(links.get("role", set()))
            concept_explained_structures = current_structure_ids_by_candidate[concept_signature]
            source_role_explained_structures = source_role_structures_by_candidate[concept_signature]
            source_metrics = [role_metrics[role] for role in roles if role in role_metrics]
            baseline_compression = _mean_metric(source_metrics, "compression")
            baseline_prediction = _mean_metric(source_metrics, "prediction")
            baseline_future = _mean_metric(source_metrics, "future")
            baseline_transfer = _mean_metric(source_metrics, "transfer")
            first_seen = None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"])
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
            explanation_events, functional_diagnostics, functional_state = _build_functional_explanation_diagnostics(
                state_conn=state_conn,
                candidate_signature=concept_signature,
                source_roles=roles,
                first_seen_global_step=first_seen,
                transfer_rows=transfer_rows,
                transfer_history=transfer_history,
                future_rows=future_rows,
                previous_state=previous_coverage_states.get(concept_signature),
                diagnostic_epoch_id=diagnostic_epoch_id,
                config=config,
                candidate_links=relevance_links_by_candidate.get(concept_signature, {}),
            )
            explanation_event_rows.extend(explanation_events)
            incremental_coverage = float(functional_diagnostics["relevant_incremental_coverage"])
            incremental_compression = float(functional_diagnostics["incremental_compression_gain"])
            validation_evidence_count = int(functional_diagnostics["relevant_heldout_event_count"])
            validation_prediction_lift = functional_diagnostics["mean_prediction_gain"] if validation_evidence_count else None
            validation_action_selection_lift = functional_diagnostics["mean_behavioral_gain"] if validation_evidence_count else None
            validation_transfer_lift = _mean_event_gain(
                [
                    event for event in explanation_events
                    if not bool(event.get("invalid")) and bool(event.get("is_relevant"))
                ],
                "prediction_gain",
                event_type="transfer",
            )
            concept_prediction_lift = (
                sum(family_prediction.get(family, (0.0, None))[0] for family in links.get("family", set()))
                / max(1, len(links.get("family", set())))
                - baseline_prediction
            )
            future_values = [value for role in roles for value in future_by_role.get(role, [])]
            future_lift = (sum(future_values) / len(future_values) if future_values else 0.0) - baseline_future
            cross_game_transfer_lift = validation_transfer_lift if validation_transfer_lift is not None else 0.0
            has_cross_evidence = max(int(row["cross_context_count"] or 0), int(row["cross_game_count"] or 0)) >= int(config.min_cross_context_or_game_evidence)
            required_functional_coverage = max(
                float(config.min_incremental_coverage),
                float(config.min_incremental_explanatory_coverage),
            )
            population_matched = (
                int(functional_diagnostics["baseline_event_count"])
                == int(functional_diagnostics["candidate_event_count"])
                == int(functional_diagnostics["common_event_count"])
            )
            population_status = str(functional_diagnostics["functional_coverage_longitudinal_change"]["population_comparison"])
            population_change = str(functional_diagnostics["functional_coverage_longitudinal_change"]["population_change"])
            has_relevant_samples = validation_evidence_count >= int(config.min_relevant_heldout_event_count)
            has_incremental_gain = has_relevant_samples and incremental_coverage >= required_functional_coverage
            heldout_lifts = [value for value in (validation_prediction_lift, validation_action_selection_lift) if value is not None]
            has_heldout_lift = any(
                float(value) >= float(config.min_behavioral_or_predictive_lift)
                for value in heldout_lifts
            )
            has_positive_compression = incremental_compression > 0.0
            historically_promoted = bool(int(row["persistent_historically_promoted"] or 0))
            legacy_promoted = bool(
                int(row["persistent_currently_promoted"] or 0)
                or int(row["is_promoted"] or 0)
            )
            promotion_score_before_validation = float(
                row["raw_promotion_score"]
                if row["raw_promotion_score"] is not None
                else row["promotion_score"] or 0.0
            )
            coverage_component = min(1.0, incremental_coverage / max(required_functional_coverage, 1e-12))
            prediction_component = min(
                1.0,
                max(0.0, float(validation_prediction_lift or 0.0))
                / max(float(config.min_behavioral_or_predictive_lift), 1e-12),
            )
            behavior_component = min(
                1.0,
                max(0.0, float(validation_action_selection_lift or 0.0))
                / max(float(config.min_behavioral_or_predictive_lift), 1e-12),
            )
            lift_component = max(prediction_component, behavior_component)
            compression_component = 1.0 if has_positive_compression else 0.0
            cross_scope_component = min(
                1.0,
                max(int(row["cross_context_count"] or 0), int(row["cross_game_count"] or 0))
                / max(1, int(config.min_cross_context_or_game_evidence)),
            )
            validation_score = (
                0.35 * coverage_component
                + 0.30 * lift_component
                + 0.20 * compression_component
                + 0.15 * cross_scope_component
            )
            raw_score_weight = 0.5
            validation_score_weight = 0.5
            raw_score_contribution = raw_score_weight * promotion_score_before_validation
            validation_score_contribution = validation_score_weight * validation_score
            adjusted_promotion_score = raw_score_contribution + validation_score_contribution
            assert all(
                0.0 <= value <= 1.0
                for value in (
                    coverage_component, prediction_component, behavior_component, lift_component,
                    compression_component, cross_scope_component, validation_score,
                )
            )
            assert math.isclose(
                adjusted_promotion_score,
                raw_score_contribution + validation_score_contribution,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            meets_promotion_score_threshold = adjusted_promotion_score >= float(config.promotion_score_threshold)
            contribution_threshold = 0.5 * float(config.promotion_score_threshold)
            promotion_blocked_only_by_raw_score_contribution = bool(
                not meets_promotion_score_threshold
                and validation_score_contribution >= contribution_threshold
                and raw_score_contribution < contribution_threshold
            )
            promotion_blocked_only_by_validation_score_contribution = bool(
                not meets_promotion_score_threshold
                and raw_score_contribution >= contribution_threshold
                and validation_score_contribution < contribution_threshold
            )
            promoted_by_validation = (
                has_cross_evidence
                and has_incremental_gain
                and has_heldout_lift
                and has_positive_compression
                and meets_promotion_score_threshold
            )
            comparable_failure = (
                population_matched
                and has_relevant_samples
                and population_status == "comparable"
                and not promoted_by_validation
            )
            if not population_matched:
                validation_status = "invalid_baseline_population"
            elif not has_relevant_samples:
                validation_status = "insufficient_relevant_samples"
            elif population_status != "comparable":
                validation_status = population_status
            elif promoted_by_validation:
                validation_status = "passed"
            else:
                validation_status = "failed"
            status, failure_count, demoted_this_validation = _update_promotion_validation_state(
                state_conn,
                candidate_type="concept",
                candidate_signature=concept_signature,
                passed=promoted_by_validation,
                demotion_failure_limit=int(config.demotion_failure_limit),
                validation_scope="relevant_later_global_step" if validation_evidence_count else "unavailable",
                validation_prediction_lift=validation_prediction_lift,
                validation_action_selection_lift=validation_action_selection_lift,
                validation_transfer_lift=validation_transfer_lift,
                updated_global_step=first_seen,
                validation_epoch=diagnostic_epoch_id,
                count_failure=comparable_failure,
                retain_previous_promotion=legacy_promoted and not comparable_failure,
                previously_promoted=legacy_promoted,
                validation_result=validation_status,
            )
            retained_from_valid_history = bool(
                historically_promoted
                and legacy_promoted
                and str(row["persistent_validation_status"] or "") not in {"failed", "demoted", "invalid"}
                and failure_count < int(config.demotion_failure_limit)
                and not demoted_this_validation
            )
            promoted = bool(promoted_by_validation or retained_from_valid_history)
            validation_penalty = 0.0
            persisted_status = (
                "promoted" if promoted_by_validation else "retained"
                if retained_from_valid_history else "demoted"
                if demoted_this_validation else "candidate"
            )
            state_conn.execute(
                """
                UPDATE concept_candidates
                SET concept_incremental_coverage = ?, concept_incremental_compression_gain = ?,
                    concept_prediction_lift = ?, concept_future_option_prediction_lift = ?,
                    concept_cross_game_transfer_lift = ?, validation_scope = ?,
                    validation_prediction_lift = ?, validation_action_selection_lift = ?,
                    validation_transfer_lift = ?, validation_evidence_count = ?,
                    promotion_status = ?, promotion_failure_count = ?, promotion_score = ?, raw_promotion_score = ?, is_promoted = ?,
                    validation_status = ?, validation_penalty = ?, demotion_reason = ?
                WHERE concept_signature = ?
                """,
                (
                    incremental_coverage, incremental_compression, concept_prediction_lift, future_lift,
                    cross_game_transfer_lift, "relevant_later_global_step" if validation_evidence_count else "unavailable",
                    validation_prediction_lift, validation_action_selection_lift, validation_transfer_lift,
                    validation_evidence_count,
                    persisted_status,
                    failure_count, adjusted_promotion_score, promotion_score_before_validation,
                    int(promoted), validation_status, validation_penalty,
                    "demoted_after_repeated_failure" if demoted_this_validation else None,
                    concept_signature,
                ),
            )
            if promoted and persisted_status not in {"promoted", "retained", "validated"}:
                raise AssertionError("promoted concept must have an active promotion status")
            if persisted_status == "retained" and not historically_promoted:
                raise AssertionError("retained concept must be historically promoted")
            first_promoted_step = first_seen if promoted_by_validation else None
            state_conn.execute(
                """
                INSERT INTO concept_promotion_state (
                    concept_signature, historically_promoted, currently_promoted,
                    promotion_status, validation_status, first_promoted_global_step,
                    last_validated_global_step, consecutive_validation_failures,
                    total_validation_failures, last_demotion_reason,
                    validation_algorithm_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(concept_signature) DO UPDATE SET
                    historically_promoted = MAX(concept_promotion_state.historically_promoted, excluded.historically_promoted),
                    currently_promoted = excluded.currently_promoted,
                    promotion_status = excluded.promotion_status,
                    validation_status = excluded.validation_status,
                    first_promoted_global_step = COALESCE(concept_promotion_state.first_promoted_global_step, excluded.first_promoted_global_step),
                    last_validated_global_step = excluded.last_validated_global_step,
                    consecutive_validation_failures = excluded.consecutive_validation_failures,
                    total_validation_failures = MAX(concept_promotion_state.total_validation_failures, excluded.total_validation_failures),
                    last_demotion_reason = excluded.last_demotion_reason,
                    validation_algorithm_version = excluded.validation_algorithm_version,
                    updated_at = excluded.updated_at
                """,
                (
                    concept_signature, int(historically_promoted or promoted), int(promoted),
                    persisted_status, validation_status, first_promoted_step, first_seen,
                    failure_count,
                    int(row["persistent_total_failure_count"] or 0) + int(comparable_failure and not promoted_by_validation),
                    "demoted_after_repeated_failure" if demoted_this_validation else None,
                    INCREMENTAL_PROMOTION_VALIDATION_VERSION,
                ),
            )
            demoted = status == "demoted"
            rejection_reasons = _concept_promotion_rejection_reasons(
                promoted=promoted,
                meets_promotion_score_threshold=meets_promotion_score_threshold,
                has_incremental_gain=has_incremental_gain,
                has_relevant_samples=has_relevant_samples,
                has_cross_evidence=has_cross_evidence,
                heldout_lifts=heldout_lifts,
                has_heldout_lift=has_heldout_lift,
                has_positive_compression=has_positive_compression,
                population_matched=population_matched,
                population_status=population_status,
                demoted=demoted,
            )
            if promoted:
                assert not rejection_reasons
            else:
                assert rejection_reasons
            structural_diagnostics = _build_structural_provenance_diagnostics(
                candidate_signature=concept_signature,
                explanatory_reach=float(row["explanatory_reach"] or 0.0),
                role_metrics=role_metrics,
                source_roles=roles,
                links=links,
                concept_explained_structures=concept_explained_structures,
                source_role_explained_structures=source_role_explained_structures,
            )
            diagnostic = {
                "concept_id": concept_signature,
                "candidate_signature": concept_signature,
                **_compact_source_id_summary("role", roles),
                **_compact_source_id_summary("carrier", sorted(links.get("carrier", set()))),
                **_compact_source_id_summary("family", sorted(links.get("family", set()))),
                "derivation_evidence_count": len(derivation_attempts),
                "validation_evidence_count": validation_evidence_count,
                "validation_scope": "relevant_later_global_step" if validation_evidence_count else "unavailable",
                "validation_status": validation_status,
                "promotion_status": persisted_status,
                "historical_promotion_status": "promoted" if historically_promoted else "candidate",
                "explanatory_reach": float(row["explanatory_reach"] or 0.0),
                "global_explanatory_reach": int(functional_diagnostics["global_explanatory_reach"]),
                "relevant_incremental_coverage": incremental_coverage,
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
                "raw_promotion_score": promotion_score_before_validation,
                "raw_score_weight": raw_score_weight,
                "validation_score_weight": validation_score_weight,
                "raw_score_contribution": raw_score_contribution,
                "validation_score_contribution": validation_score_contribution,
                "adjusted_promotion_score": adjusted_promotion_score,
                "promotion_score_before_validation": promotion_score_before_validation,
                "coverage_component": coverage_component,
                "prediction_component": prediction_component,
                "behavior_component": behavior_component,
                "lift_component": lift_component,
                "compression_component": compression_component,
                "cross_scope_component": cross_scope_component,
                "validation_score": validation_score,
                "validation_penalty": validation_penalty,
                "validation_penalty_reasons": [],
                "promotion_threshold": float(config.promotion_score_threshold),
                "promotion_gate_score_used": "adjusted_promotion_score",
                "promotion_score_gate_passed": meets_promotion_score_threshold,
                "promotion_blocked_only_by_raw_score_contribution": promotion_blocked_only_by_raw_score_contribution,
                "promotion_blocked_only_by_validation_score_contribution": promotion_blocked_only_by_validation_score_contribution,
                "population_comparable": population_status == "comparable",
                "population_change": population_change,
                "retained_fraction_previous": functional_diagnostics["retained_fraction_previous"],
                "retained_fraction_current": functional_diagnostics["retained_fraction_current"],
                "relevant_prediction_event_count": functional_diagnostics["relevant_prediction_event_count"],
                "relevant_contradiction_event_count": functional_diagnostics["relevant_behavior_event_count"],
                "validation_algorithm_version": INCREMENTAL_PROMOTION_VALIDATION_VERSION,
                "validation_state_reset_applied_this_run": validation_state_reset_applied_this_run,
                "validation_state_last_reset_version": validation_state_last_reset_version,
                "legacy_promoted": legacy_promoted,
                "incremental_validation_promoted": promoted_by_validation,
                "promotion_retained_from_history": retained_from_valid_history,
                "current_validation_passed": promoted_by_validation,
                "currently_promoted": promoted,
                "historically_promoted": historically_promoted or promoted,
                "demoted_this_epoch": demoted_this_validation,
                "validation_pass": promoted_by_validation,
                "promoted": promoted,
                "rejection_reasons": rejection_reasons,
                "consecutive_validation_failures": failure_count,
                "demoted": demoted,
                "demotion_reason": "demoted_after_repeated_failure" if demoted else None,
                **functional_diagnostics,
                **structural_diagnostics,
            }
            state_conn.execute(
                """
                INSERT INTO concept_promotion_validation_diagnostics (
                    diagnostic_epoch_id, concept_signature, validation_algorithm_version, payload_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(diagnostic_epoch_id, concept_signature, validation_algorithm_version)
                DO UPDATE SET payload_json = excluded.payload_json
                """,
                (
                    "" if diagnostic_epoch_id is None else str(diagnostic_epoch_id),
                    concept_signature, INCREMENTAL_PROMOTION_VALIDATION_VERSION,
                    json.dumps(diagnostic, sort_keys=True),
                ),
            )
            _store_incremental_coverage_state(
                state_conn,
                candidate_signature=concept_signature,
                epoch_id=diagnostic_epoch_id,
                state=functional_state,
                updated_global_step=first_seen,
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
        if explanation_events_path is not None:
            _write_explanation_event_artifact(Path(explanation_events_path), explanation_event_rows)
        active_signatures = [str(row["concept_signature"]) for row in concept_rows]
        diagnostic_epoch_key = "" if diagnostic_epoch_id is None else str(diagnostic_epoch_id)
        if active_signatures:
            placeholders = ", ".join("?" for _ in active_signatures)
            state_conn.execute(
                f"""
                DELETE FROM concept_promotion_validation_diagnostics
                WHERE diagnostic_epoch_id = ? AND validation_algorithm_version = ?
                  AND concept_signature NOT IN ({placeholders})
                """,
                [diagnostic_epoch_key, INCREMENTAL_PROMOTION_VALIDATION_VERSION, *active_signatures],
            )
        else:
            state_conn.execute(
                "DELETE FROM concept_promotion_validation_diagnostics "
                "WHERE diagnostic_epoch_id = ? AND validation_algorithm_version = ?",
                (diagnostic_epoch_key, INCREMENTAL_PROMOTION_VALIDATION_VERSION),
            )
        active_promoted_step = state_conn.execute(
            "SELECT MIN(first_seen_global_step) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
        ).fetchone()[0]
        active_promoted_step = None if active_promoted_step is None else int(active_promoted_step)
        _write_milestone(state_conn, "first_promoted_concept_step", active_promoted_step, None)
        _write_historical_milestone(state_conn, "first_promoted_concept_step", active_promoted_step, None)
        phase3_promoted_count = int(summary["concepts_promoted_with_behavioral_lift"])
        active_promoted_count = int(state_conn.execute(
            "SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1"
        ).fetchone()[0])
        summary["phase3_promoted_count"] = phase3_promoted_count
        summary["active_promoted_concept_count"] = active_promoted_count
        # Existing promoted concepts remain active during non-comparable or
        # insufficient-sample validation; only comparable repeated failures
        # can demote them.  Therefore the active count is not necessarily the
        # number newly passed in this validation call.
        summary["phase3_active_promoted_count"] = active_promoted_count

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
                validation_epoch=diagnostic_epoch_id,
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


def _mean_metric(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item.get(key, 0.0) or 0.0) for item in items) / max(1, len(items))


def _mean_row_metric(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return (sum(values) / len(values)) if values else None


_DIAGNOSTIC_ID_SAMPLE_LIMIT = 20


def _compact_source_id_summary(kind: str, identifiers: list[str]) -> dict[str, Any]:
    values = sorted(dict.fromkeys(str(identifier) for identifier in identifiers))
    return {
        f"source_{kind}_count": len(values),
        f"source_{kind}_ids_sample": values[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
    }


def _linked_explained_structures(links: dict[str, set[str]]) -> set[str]:
    """Return the stable linked structures used by set-based concept coverage."""
    values: set[str] = set()
    for kind in ("family", "context", "game", "carrier"):
        values.update(f"{kind}:{identifier}" for identifier in links.get(kind, set()))
    return values


def _structure_fingerprint(structure_ids: set[str]) -> str:
    return sha1("\n".join(sorted(structure_ids)).encode("utf-8")).hexdigest()


def _structure_type_counts(structure_ids: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for structure_id in structure_ids:
        kind, _separator, _identifier = structure_id.partition(":")
        if kind:
            counts[kind] += 1
    return {kind: counts[kind] for kind in sorted(counts)}


def _load_incremental_coverage_states(state_conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    table_exists = state_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'concept_incremental_coverage_state'"
    ).fetchone()
    if table_exists is None:
        return {}
    states: dict[str, dict[str, Any]] = {}
    for row in state_conn.execute(
        """
        SELECT candidate_signature, epoch_id, structure_fingerprint, payload_json
        FROM concept_incremental_coverage_state
        ORDER BY candidate_signature ASC
        """
    ).fetchall():
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload["candidate_signature"] = str(row["candidate_signature"])
        payload["epoch_id"] = row["epoch_id"]
        payload["structure_fingerprint"] = str(row["structure_fingerprint"])
        states[str(row["candidate_signature"])] = payload
    return states


def _store_incremental_coverage_state(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    epoch_id: str | int | None,
    state: dict[str, Any],
    updated_global_step: int | None,
) -> None:
    state_conn.execute(
        """
        INSERT INTO concept_incremental_coverage_state (
            candidate_signature, epoch_id, structure_fingerprint, payload_json, updated_global_step
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(candidate_signature) DO UPDATE SET
            epoch_id = excluded.epoch_id,
            structure_fingerprint = excluded.structure_fingerprint,
            payload_json = excluded.payload_json,
            updated_global_step = excluded.updated_global_step
        """,
        (
            candidate_signature,
            None if epoch_id is None else str(epoch_id),
            str(state["structure_fingerprint"]),
            json.dumps(state, sort_keys=True),
            updated_global_step,
        ),
    )


def _build_structural_provenance_diagnostics(
    *,
    candidate_signature: str,
    explanatory_reach: float,
    role_metrics: dict[str, dict[str, Any]],
    source_roles: list[str],
    links: dict[str, set[str]],
    concept_explained_structures: set[str],
    source_role_explained_structures: set[str],
) -> dict[str, Any]:
    """Keep membership sets as provenance-only diagnostics, never as the gate."""
    overlap = concept_explained_structures & source_role_explained_structures
    provenance_coverage = (
        float(len(concept_explained_structures))
        / float(len(concept_explained_structures | source_role_explained_structures))
        if concept_explained_structures or source_role_explained_structures
        else 0.0
    )
    return {
        "structural_provenance_count": len(concept_explained_structures),
        "structural_provenance_coverage": provenance_coverage,
        "structural_overlap_count": len(overlap),
        "structural_overlap_ratio": (
            float(len(overlap)) / float(len(concept_explained_structures))
            if concept_explained_structures else 0.0
        ),
        "structural_provenance_type_counts": _structure_type_counts(concept_explained_structures),
        "structural_provenance_ids_sample": sorted(concept_explained_structures)[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
        "structural_overlap_ids_sample": sorted(overlap)[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
        "structural_provenance_diagnostics": {
            "candidate_signature": candidate_signature,
            "explanatory_reach_descriptive": float(explanatory_reach),
            "source_role_structure_count": len(source_role_explained_structures),
            "source_role_count": len(source_roles),
            "source_carrier_count": len(links.get("carrier", set())),
            "source_family_count": len(links.get("family", set())),
            "role_overlap": [
                {
                    "role_id": role,
                    "overlap_count": len(concept_explained_structures & set(role_metrics.get(role, {}).get("structures", []))),
                }
                for role in sorted(source_roles)
                if role in role_metrics
            ][: _DIAGNOSTIC_ID_SAMPLE_LIMIT],
        },
    }


def _prior_role_success_rate(
    transfer_rows: list[sqlite3.Row],
    *,
    role: str,
    before_step: int,
    source_game_key: str | None = None,
    source_context_key: str | None = None,
    target_game_key: str | None = None,
    target_context_key: str | None = None,
    transfer_history: _TransferHistoryIndex | None = None,
) -> tuple[float, int]:
    if transfer_history is not None:
        return transfer_history.rate_before(
            role=role,
            step=before_step,
            source_game_key=source_game_key,
            source_context_key=source_context_key,
            target_game_key=target_game_key,
            target_context_key=target_context_key,
        )
    matching = [
        row for row in transfer_rows
        if str(row["role_signature"] or "") == role
        and row["last_seen_global_step"] is not None
        and int(row["last_seen_global_step"]) < before_step
        and (source_game_key is None or str(row["source_game_key"] or "") == source_game_key)
        and (source_context_key is None or str(row["source_context_key"] or "") == source_context_key)
        and (target_game_key is None or str(row["target_game_key"] or "") == target_game_key)
        and (target_context_key is None or str(row["target_context_key"] or "") == target_context_key)
    ]
    if not matching:
        return 0.0, 0
    return sum(int(row["reuse_success"] or 0) for row in matching) / len(matching), len(matching)


def _combined_role_score(rates: list[float]) -> float:
    """Probability that a concept's role combination supports the event."""
    if not rates:
        return 0.0
    remaining = 1.0
    for rate in rates:
        remaining *= max(0.0, min(1.0, 1.0 - float(rate)))
    return 1.0 - remaining


def _mean_event_gain(events: list[dict[str, Any]], key: str, *, event_type: str | None = None) -> float | None:
    values = [
        float(event.get(key, 0.0) or 0.0)
        for event in events
        if event_type is None or str(event.get("event_type")) == event_type
    ]
    return sum(values) / len(values) if values else None


def _transfer_explanation_events(
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    transfer_history: _TransferHistoryIndex | None = None,
) -> list[dict[str, Any]]:
    if first_seen_global_step is None or len(source_roles) < 1:
        return []
    source_role_set = set(source_roles)
    events: list[dict[str, Any]] = []
    for row in transfer_rows:
        source_role = str(row["role_signature"] or "")
        step = row["last_seen_global_step"]
        if step is None or int(step) <= first_seen_global_step:
            continue
        target_role = str(row["observed_role_signature"] or row["predicted_role_signature"] or "unknown")
        source_game_key = str(row["source_game_key"] or "")
        source_context_key = str(row["source_context_key"] or "")
        target_game_key = str(row["target_game_key"] or "")
        target_context_key = str(row["target_context_key"] or "")
        generic_rates = [
            _prior_role_success_rate(
                transfer_rows,
                role=role,
                before_step=int(step),
                transfer_history=transfer_history,
            )[0]
            for role in source_roles
        ]
        scoped_rates = [
            rate for role in source_roles
            for rate, count in [
                _prior_role_success_rate(
                    transfer_rows,
                    role=role,
                    before_step=int(step),
                    source_game_key=source_game_key,
                    source_context_key=source_context_key,
                    target_game_key=target_game_key,
                    target_context_key=target_context_key,
                    transfer_history=transfer_history,
                )
            ]
            if count > 0
        ]
        best_single = max(generic_rates, default=0.0)
        baseline = best_single
        # Older transfer rows can lack scope columns.  Their historical role
        # evidence remains legitimate lower-level evidence; use it only when
        # scoped evidence cannot establish a multi-role profile.
        combination_rates = scoped_rates if len(scoped_rates) >= 2 else generic_rates
        concept_score = _combined_role_score(combination_rates) if len(combination_rates) >= 2 else baseline
        event_id = (
            f"transfer:{source_role}:{target_role}:{source_game_key}:{source_context_key}:"
            f"{target_game_key}:{target_context_key}:{row['attempt_id']}"
        )
        outcome = float(int(row["reuse_success"] or 0))
        feature_step = (
            max(
                (
                    prior_step
                    for role in source_roles
                    for prior_step in [transfer_history.max_step_before(role=role, step=int(step))]
                    if prior_step is not None
                ),
                default=None,
            )
            if transfer_history is not None
            else max(
                (
                    int(item["last_seen_global_step"])
                    for item in transfer_rows
                    if str(item["role_signature"] or "") in source_role_set
                    and item["last_seen_global_step"] is not None
                    and int(item["last_seen_global_step"]) < int(step)
                ),
                default=None,
            )
        )
        events.append({
            "concept_id": candidate_signature,
            "event_id": event_id,
            "event_type": "transfer",
            "evaluation_scope": "later_global_step",
            "best_single_role_score": best_single,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
            "_outcome": outcome,
            "_evaluation_global_step": int(step),
            "_feature_global_step_max": feature_step,
            "_label_used_as_feature": False,
            "_source_role_signature": source_role,
            "_predicted_target_role_signature": str(row["predicted_role_signature"] or ""),
            "_observed_target_role_signature": str(row["observed_role_signature"] or ""),
            "_carrier_ids": [
                str(value) for value in (
                    row["source_carrier_signature"] if "source_carrier_signature" in row.keys() else None,
                    row["target_carrier_signature"] if "target_carrier_signature" in row.keys() else None,
                )
                if value not in (None, "")
            ],
            "_context_keys": [value for value in (source_context_key, target_context_key) if value],
            "_game_keys": [value for value in (source_game_key, target_game_key) if value],
        })
    return events


def _future_option_motif_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    future_rows: list[sqlite3.Row],
) -> list[dict[str, Any]]:
    if first_seen_global_step is None or not source_roles:
        return []
    motif_table = state_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'future_option_motifs'"
    ).fetchone()
    if motif_table is None:
        return []
    role_set = set(source_roles)
    role_rates: dict[str, float] = {}
    for role in source_roles:
        values = [
            1.0 if float(row["option_delta"] or 0.0) > 0.0 else 0.0
            for row in future_rows
            if (str(row["source_role_id"] or row["owner_key"] or "") == role)
            and row["last_seen_global_step"] is not None
            and int(row["last_seen_global_step"]) < first_seen_global_step
        ]
        role_rates[role] = sum(values) / len(values) if values else 0.0
    events: list[dict[str, Any]] = []
    for row in state_conn.execute(
        """
        SELECT motif_signature, source_role_ids_json, motif_stability_score, is_emergent, last_seen_global_step
        FROM future_option_motifs ORDER BY motif_signature ASC
        """
    ).fetchall():
        if row["last_seen_global_step"] is None or int(row["last_seen_global_step"]) <= first_seen_global_step:
            continue
        try:
            motif_roles = {str(value) for value in json.loads(str(row["source_role_ids_json"] or "[]"))}
        except (TypeError, ValueError, json.JSONDecodeError):
            motif_roles = set()
        matching_roles = sorted(role_set & motif_roles)
        rates = [role_rates[role] for role in matching_roles]
        baseline = max(rates, default=0.0)
        concept_score = _combined_role_score(rates) if len(rates) >= 2 else baseline
        outcome = float(int(row["is_emergent"] or 0))
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"future_option_motif:{row['motif_signature']}",
            "event_type": "future_option_motif",
            "evaluation_scope": "later_global_step",
            "best_single_role_score": baseline,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
            "_outcome": outcome,
            "_evaluation_global_step": int(row["last_seen_global_step"]),
            "_feature_global_step_max": first_seen_global_step - 1,
            "_label_used_as_feature": False,
            "_source_role_ids": sorted(motif_roles),
        })
    return events


def _prediction_result_columns(state_conn: sqlite3.Connection) -> set[str]:
    """Return the local prediction schema without assuming raw tables exist."""
    table = state_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'prediction_results'"
    ).fetchone()
    if table is None:
        return set()
    return {
        str(row[1])
        for row in state_conn.execute("PRAGMA table_info(prediction_results)").fetchall()
    }


def _prediction_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    role_links: dict[str, dict[str, set[str]]],
    transfer_history: _TransferHistoryIndex | None = None,
) -> list[dict[str, Any]]:
    """Evaluate later lower-level predictions without candidate-concept features.

    The raw prediction result supplies only a held-out label.  Scores are
    derived from pre-event role-transfer evidence, so the observed prediction
    result is never an input feature.
    """
    columns = _prediction_result_columns(state_conn)
    required = {"id", "global_step", "context_signature", "predicted_family", "actual_family"}
    if first_seen_global_step is None or not source_roles or not required <= columns:
        return []
    selected = "id, global_step, context_signature, predicted_family, actual_family"
    source_role_family_ids = sorted({
        str(family)
        for role in source_roles
        for family in role_links.get(role, {}).get("family", set())
        if family not in (None, "")
    })
    events: list[dict[str, Any]] = []
    for row in state_conn.execute(
        f"SELECT {selected} FROM prediction_results WHERE global_step > ? ORDER BY global_step ASC, id ASC",
        (first_seen_global_step,),
    ).fetchall():
        context = str(row["context_signature"] or "")
        predicted_family = str(row["predicted_family"] or "")
        actual_family = str(row["actual_family"] or "")
        step = int(row["global_step"])
        rates = [
            _prior_role_success_rate(
                transfer_rows,
                role=role,
                before_step=step,
                transfer_history=transfer_history,
            )[0]
            for role in source_roles
        ]
        baseline = max(rates, default=0.0)
        concept_score = _combined_role_score(rates) if len(rates) >= 2 else baseline
        outcome = float(predicted_family == actual_family)
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"prediction:prediction_result:{row['id']}:later_global_step",
            "event_type": "prediction",
            "evaluation_scope": "later_global_step",
            "predicted_family": predicted_family,
            "actual_family": actual_family,
            "candidate_role_family_ids": source_role_family_ids,
            "best_single_role_score": baseline,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
            "_outcome": outcome,
            "_evaluation_global_step": step,
            "_feature_global_step_max": (
                transfer_history.max_any_step_before(step)
                if transfer_history is not None
                else max(
                    (int(item["last_seen_global_step"]) for item in transfer_rows
                     if item["last_seen_global_step"] is not None and int(item["last_seen_global_step"]) < step),
                    default=None,
                )
            ),
            "_label_used_as_feature": False,
            "_context_keys": [context] if context else [],
            "_family_ids": [family for family in (predicted_family, actual_family) if family],
        })
    return events


def _contradiction_resolution_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    role_links: dict[str, dict[str, set[str]]],
    transfer_history: _TransferHistoryIndex | None = None,
) -> list[dict[str, Any]]:
    """Evaluate held-out contradiction detection/resolution opportunities."""
    columns = _prediction_result_columns(state_conn)
    required = {"id", "global_step", "context_signature", "context_contradiction"}
    if first_seen_global_step is None or not source_roles or not required <= columns:
        return []
    events: list[dict[str, Any]] = []
    source_role_family_ids = sorted({
        str(family)
        for role in source_roles
        for family in role_links.get(role, {}).get("family", set())
        if family not in (None, "")
    })
    available_family_columns = {"predicted_family", "actual_family"} <= columns
    selected = "id, global_step, context_signature, context_contradiction"
    if available_family_columns:
        selected += ", predicted_family, actual_family"
    for row in state_conn.execute(
        f"""
        SELECT {selected}
        FROM prediction_results
        WHERE global_step > ? AND COALESCE(context_contradiction, 0) = 1
        ORDER BY global_step ASC, id ASC
        """,
        (first_seen_global_step,),
    ).fetchall():
        step = int(row["global_step"])
        predicted_family = str(row["predicted_family"] or "") if available_family_columns else ""
        actual_family = str(row["actual_family"] or "") if available_family_columns else ""
        failure_rates = [
            1.0 - _prior_role_success_rate(
                transfer_rows,
                role=role,
                before_step=step,
                transfer_history=transfer_history,
            )[0]
            for role in source_roles
        ]
        baseline = max(failure_rates, default=0.0)
        concept_score = _combined_role_score(failure_rates) if len(failure_rates) >= 2 else baseline
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"contradiction_resolution:{row['id']}:later_global_step",
            "event_type": "contradiction_resolution",
            "evaluation_scope": "later_global_step",
            "predicted_family": predicted_family,
            "actual_family": actual_family,
            "candidate_role_family_ids": source_role_family_ids,
            "best_single_role_score": baseline,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": concept_score - baseline,
            "_outcome": 1.0,
            "_evaluation_global_step": step,
            "_feature_global_step_max": (
                transfer_history.max_any_step_before(step)
                if transfer_history is not None
                else max(
                    (int(item["last_seen_global_step"]) for item in transfer_rows
                     if item["last_seen_global_step"] is not None and int(item["last_seen_global_step"]) < step),
                    default=None,
                )
            ),
            "_label_used_as_feature": False,
            "_context_keys": [str(row["context_signature"] or "")] if row["context_signature"] else [],
            "_family_ids": [family for family in (predicted_family, actual_family) if family],
        })
    return events


def _classify_functional_explanation_event(
    event: dict[str, Any],
    *,
    config: IncrementalPromotionValidationConfig,
    definition_cost_share: float,
) -> dict[str, Any]:
    """Validate and classify held-out evidence before it enters coverage."""
    outcome = float(event.pop("_outcome"))
    baseline_score = float(event["lower_level_baseline_score"])
    concept_score = float(event["concept_enabled_score"])
    baseline_cost = float(event.get("baseline_description_cost", abs(baseline_score - outcome)))
    concept_cost = float(event.get("concept_description_cost", abs(concept_score - outcome)))
    event["baseline_description_cost"] = baseline_cost
    event["concept_description_cost"] = concept_cost
    event["invalid"] = False
    event["explanation_channels"] = []
    evaluation_step = event.get("_evaluation_global_step")
    feature_step = event.get("_feature_global_step_max")
    leakage = (
        bool(event.get("_label_used_as_feature", False))
        or (evaluation_step is not None and feature_step is not None and int(feature_step) >= int(evaluation_step))
    )
    if leakage:
        event.update({
            "invalid": True, "explained": False,
            "rejection_reason": "label_leakage_detected",
            "compression_gain": 0.0, "prediction_gain": 0.0,
            "behavioral_gain": 0.0, "concept_incremental_gain": 0.0,
        })
        return event
    if not (math.isfinite(baseline_cost) and math.isfinite(concept_cost)) or baseline_cost < 0.0 or concept_cost < 0.0:
        event.update({
            "invalid": True, "explained": False,
            "rejection_reason": "invalid_description_cost",
            "compression_gain": 0.0, "prediction_gain": 0.0,
            "behavioral_gain": 0.0, "concept_incremental_gain": 0.0,
        })
        return event
    prediction_gain = float(event.get("_prediction_gain", baseline_cost - concept_cost))
    behavioral_gain = float(event.get("_behavioral_gain", prediction_gain))
    compression_gain = float(event.get("_compression_gain", prediction_gain - float(definition_cost_share)))
    event.update({
        "prediction_gain": prediction_gain,
        "behavioral_gain": behavioral_gain,
        "compression_gain": compression_gain,
        "concept_incremental_gain": concept_score - baseline_score,
    })
    if not all(math.isfinite(value) for value in (prediction_gain, behavioral_gain, compression_gain)):
        event.update({"invalid": True, "explained": False, "rejection_reason": "non_finite_explanatory_gain"})
        return event
    prediction_explained = prediction_gain >= float(config.min_event_prediction_gain)
    behavioral_explained = behavioral_gain >= float(config.min_event_behavioral_gain)
    compression_explained = compression_gain > 0.0 and compression_gain >= float(config.min_event_compression_gain)
    event["explanation_channels"] = [
        name for name, passed in (
            ("prediction", prediction_explained),
            ("behavioral", behavioral_explained),
            ("compression", compression_explained),
        ) if passed
    ]
    event["explained"] = bool(event["explanation_channels"])
    event["rejection_reason"] = None if event["explained"] else "no_incremental_explanatory_gain"
    return event


def _event_relevance_reasons(
    event: dict[str, Any],
    *,
    source_roles: list[str],
    candidate_links: dict[str, set[str]],
) -> list[str]:
    """Return concrete candidate/event overlaps used by the Phase 3 denominator."""
    reasons: list[str] = []
    roles = set(source_roles)
    source_role = str(event.get("_source_role_signature") or "")
    if not source_role and str(event.get("event_type")) == "transfer":
        source_role = str(event.get("event_id", "")).split(":", 2)[1] if ":" in str(event.get("event_id", "")) else ""
    if source_role in roles or roles & {
        str(value) for value in event.get("_source_role_ids", ()) if value not in (None, "")
    }:
        reasons.append("source_role_overlap")
    if {
        str(event.get("_predicted_target_role_signature") or ""),
        str(event.get("_observed_target_role_signature") or ""),
    } & roles:
        reasons.append("target_role_overlap")
    family_ids = set(str(value) for value in event.get("_family_ids", ()) if value)
    candidate_family_overlap = bool(family_ids & set(candidate_links.get("_candidate_family", set())))
    candidate_role_family_overlap = bool(family_ids & set(candidate_links.get("_candidate_role_family", set())))
    event["candidate_family_overlap"] = candidate_family_overlap
    event["candidate_role_family_overlap"] = candidate_role_family_overlap
    if family_ids & set(candidate_links.get("family", set())):
        reasons.append("family_overlap")
    if set(str(value) for value in event.get("_carrier_ids", ()) if value) & set(candidate_links.get("carrier", set())):
        reasons.append("carrier_overlap")
    if set(str(value) for value in event.get("_context_keys", ()) if value) & set(candidate_links.get("context", set())):
        reasons.append("context_overlap")
    if set(str(value) for value in event.get("_game_keys", ()) if value) & set(candidate_links.get("game", set())):
        reasons.append("game_overlap")
    return reasons


def _event_is_relevant(reasons: list[str]) -> bool:
    """Require a concrete structural link; broad scope alone is diagnostic only."""
    reason_set = set(reasons)
    structural = {
        "source_role_overlap",
        "target_role_overlap",
        "family_overlap",
        "carrier_overlap",
    }
    return bool(reason_set & structural)


def _relevant_event_type_diagnostics(events: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    labels = ("transfer", "future_option_motif", "prediction", "contradiction_resolution")
    output: dict[str, dict[str, float | int]] = {}
    for label in labels:
        items = [item for item in events if str(item.get("event_type")) == label and not bool(item.get("invalid"))]
        explained = [item for item in items if bool(item.get("explained"))]
        count = len(items)
        output[label] = {
            "eligible": count,
            "explained": len(explained),
            "coverage": float(len(explained)) / count if count else 0.0,
            "prediction_lift": _mean_event_gain(items, "prediction_gain") or 0.0,
            "behavioral_lift": _mean_event_gain(items, "behavioral_gain") or 0.0,
            "compression_gain": _mean_event_gain(items, "compression_gain") or 0.0,
        }
    return output


def _build_functional_explanation_diagnostics(
    *,
    state_conn: sqlite3.Connection,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    transfer_history: _TransferHistoryIndex,
    future_rows: list[sqlite3.Row],
    previous_state: dict[str, Any] | None,
    diagnostic_epoch_id: str | int | None,
    config: IncrementalPromotionValidationConfig,
    candidate_links: dict[str, set[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Evaluate held-out events with and without this concept's role combination."""
    events = _transfer_explanation_events(
        candidate_signature=candidate_signature,
        source_roles=source_roles,
        first_seen_global_step=first_seen_global_step,
        transfer_rows=transfer_rows,
        transfer_history=transfer_history,
    )
    events.extend(_future_option_motif_explanation_events(
        state_conn,
        candidate_signature=candidate_signature,
        source_roles=source_roles,
        first_seen_global_step=first_seen_global_step,
        future_rows=future_rows,
    ))
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    events.extend(_prediction_explanation_events(
        state_conn,
        candidate_signature=candidate_signature,
        source_roles=source_roles,
        first_seen_global_step=first_seen_global_step,
        transfer_rows=transfer_rows,
        role_links=role_links,
        transfer_history=transfer_history,
    ))
    events.extend(_contradiction_resolution_explanation_events(
        state_conn,
        candidate_signature=candidate_signature,
        source_roles=source_roles,
        first_seen_global_step=first_seen_global_step,
        transfer_rows=transfer_rows,
        role_links=role_links,
        transfer_history=transfer_history,
    ))
    events.sort(key=lambda item: (str(item["event_type"]), str(item["event_id"])))
    candidate_links = candidate_links or {}
    definition_cost = 0.05 * max(1, len(source_roles))
    event_count = len(events)
    seen_event_ids: set[str] = set()
    for event in events:
        event_id = str(event["event_id"])
        if event_id in seen_event_ids:
            event.pop("_outcome", None)
            event.update({
                "invalid": True, "explained": False, "explanation_channels": [],
                "rejection_reason": "label_leakage_detected",
                "baseline_description_cost": 0.0, "concept_description_cost": 0.0,
                "compression_gain": 0.0, "prediction_gain": 0.0,
                "behavioral_gain": 0.0, "concept_incremental_gain": 0.0,
            })
            reasons = _event_relevance_reasons(
                event,
                source_roles=source_roles,
                candidate_links=candidate_links or {},
            )
            event["candidate_id"] = candidate_signature
            event["is_relevant"] = _event_is_relevant(reasons)
            event["relevance_reasons"] = reasons
            continue
        seen_event_ids.add(event_id)
        _classify_functional_explanation_event(
            event,
            config=config,
            definition_cost_share=definition_cost / max(1, event_count),
        )
        if not bool(event.get("invalid")):
            feature_step = event.get("_feature_global_step_max")
            evaluation_step = event.get("_evaluation_global_step")
            if feature_step is not None and evaluation_step is not None:
                assert int(feature_step) < int(evaluation_step)
        reasons = _event_relevance_reasons(
            event,
            source_roles=source_roles,
            candidate_links=candidate_links,
        )
        event["candidate_id"] = candidate_signature
        event["is_relevant"] = _event_is_relevant(reasons)
        event["relevance_reasons"] = reasons

    # This is explicitly an ablation of a combined linked-role signal, not a
    # generic baseline that might accidentally include the concept.
    for event in events:
        event["baseline_type"] = "best_single_linked_role"
        event["candidate_type"] = "combined_linked_roles"
        event["validation_method"] = "role_combination_ablation"
        event["single_role_baseline_score"] = event.get("lower_level_baseline_score")
        event["role_combination_concept_score"] = event.get("concept_enabled_score")
        event["role_combination_lift"] = event.get("concept_incremental_gain")

    def _score_available(event: dict[str, Any], flag: str, score_key: str) -> bool:
        if not bool(event.get(flag, True)):
            return False
        try:
            return math.isfinite(float(event.get(score_key)))
        except (TypeError, ValueError):
            return False

    pre_population_events = [event for event in events if not bool(event.get("invalid"))]
    baseline_event_ids = {
        str(event["event_id"])
        for event in pre_population_events
        if _score_available(event, "_baseline_available", "lower_level_baseline_score")
    }
    candidate_event_ids = {
        str(event["event_id"])
        for event in pre_population_events
        if _score_available(event, "_candidate_available", "concept_enabled_score")
    }
    common_event_ids = baseline_event_ids & candidate_event_ids
    # A lift is defined only on a matched event population.  Retain the
    # separately observed populations for diagnostics, but exclude their
    # one-sided rows from coverage and promotion decisions.
    for event in pre_population_events:
        if str(event["event_id"]) not in common_event_ids:
            event.update({
                "invalid": True,
                "explained": False,
                "explanation_channels": [],
                "rejection_reason": "baseline_population_mismatch",
            })

    eligible_events = [event for event in events if not bool(event.get("invalid"))]
    invalid_events = [event for event in events if bool(event.get("invalid"))]
    relevant_events = [event for event in eligible_events if bool(event.get("is_relevant"))]
    unrelated_events = [event for event in eligible_events if not bool(event.get("is_relevant"))]
    explained = [event for event in relevant_events if bool(event["explained"])]
    rejected = [event for event in relevant_events if not bool(event["explained"])]
    global_explained = [event for event in eligible_events if bool(event["explained"])]
    eligible_count = len(relevant_events)
    baseline_cost = sum(float(event["baseline_description_cost"]) for event in relevant_events)
    concept_cost = definition_cost + sum(float(event["concept_description_cost"]) for event in relevant_events)
    compression_gain = baseline_cost - concept_cost
    explained_ids = [str(event["event_id"]) for event in explained]
    event_ids = [str(event["event_id"]) for event in relevant_events]
    previous_eligible = None if previous_state is None else previous_state.get("relevant_event_count")
    if previous_eligible is None and previous_state is not None:
        previous_eligible = previous_state.get("eligible_event_count")
    previous_explained = None if previous_state is None else previous_state.get("explained_event_count")
    previous_coverage = None if previous_state is None else previous_state.get("incremental_coverage")
    coverage = float(len(explained)) / float(eligible_count) if eligible_count else 0.0
    previous_ids = set(previous_state.get("relevant_event_ids", previous_state.get("explanation_event_ids", ())) or ()) if previous_state else set()
    previous_explained_ids = set(previous_state.get("explained_relevant_event_ids", ()) or ()) if previous_state else set()
    current_ids = set(event_ids)
    retained_ids = previous_ids & current_ids
    new_ids = current_ids - previous_ids
    expired_ids = previous_ids - current_ids
    retained_fraction_previous = len(retained_ids) / max(1, len(previous_ids))
    retained_fraction_current = len(retained_ids) / max(1, len(current_ids))
    if previous_state is None:
        population_change = "population_definition_changed"
    elif current_ids == previous_ids:
        population_change = "unchanged"
    elif retained_ids == previous_ids:
        population_change = "population_expansion"
    elif retained_ids == current_ids:
        population_change = "population_contraction"
    else:
        population_change = "population_definition_changed"
    comparable = bool(
        previous_state is not None
        and retained_fraction_previous >= float(config.promotion_population_comparability_threshold)
    )
    population_status = "comparable" if comparable else population_change
    retained_events = [event for event in relevant_events if str(event["event_id"]) in retained_ids]
    retained_explained = [event for event in retained_events if bool(event["explained"])]
    retained_coverage = float(len(retained_explained)) / len(retained_events) if retained_events else 0.0
    previous_retained_coverage = (
        float(len(retained_ids & previous_explained_ids)) / len(retained_ids)
        if retained_ids else None
    )
    new_events = [event for event in relevant_events if str(event["event_id"]) in new_ids]
    new_explained = [event for event in new_events if bool(event["explained"])]
    new_coverage = float(len(new_explained)) / len(new_events) if new_events else 0.0
    retained_lift = _mean_event_gain(retained_events, "concept_incremental_gain") or 0.0
    coverage_delta = (
        retained_coverage - previous_retained_coverage
        if comparable and previous_retained_coverage is not None else None
    )
    evaluation_population_id = sha1("\n".join(sorted(event_ids)).encode("utf-8")).hexdigest()
    for event in events:
        event["evaluation_population_id"] = evaluation_population_id
        event["baseline_event_count"] = len(baseline_event_ids)
        event["candidate_event_count"] = len(candidate_event_ids)
        event["common_event_count"] = len(common_event_ids)
        event["baseline_only_event_count"] = len(baseline_event_ids - candidate_event_ids)
        event["candidate_only_event_count"] = len(candidate_event_ids - baseline_event_ids)
    state = {
        "eligible_event_count": eligible_count,
        "relevant_event_count": eligible_count,
        "explained_event_count": len(explained),
        "incremental_coverage": coverage,
        "relevant_event_ids": sorted(current_ids),
        "explained_relevant_event_ids": sorted(explained_ids),
        "explanation_event_ids": sorted(current_ids),
        "relevant_population_signature": _structure_fingerprint(current_ids),
        "structure_fingerprint": _structure_fingerprint(set(event_ids)),
    }
    longitudinal = {
        "previous_epoch": None if previous_state is None else previous_state.get("epoch_id"),
        "previous_explained_event_count": previous_explained,
        "current_explained_event_count": len(explained),
        "explained_event_count_delta": None if previous_explained is None else len(explained) - int(previous_explained or 0),
        "previous_eligible_event_count": previous_eligible,
        "current_eligible_event_count": eligible_count,
        "eligible_event_count_delta": None if previous_eligible is None else eligible_count - int(previous_eligible or 0),
        "previous_incremental_explanatory_coverage": previous_coverage,
        "current_incremental_explanatory_coverage": coverage,
        "coverage_delta": coverage_delta,
        "relevant_population_signature": state["relevant_population_signature"],
        "relevant_event_count": eligible_count,
        "retained_event_count": len(retained_ids),
        "new_relevant_event_count": len(new_ids),
        "expired_relevant_event_count": len(expired_ids),
        "population_comparison": population_status,
        "population_change": population_change,
        "retained_fraction_previous": retained_fraction_previous,
        "retained_fraction_current": retained_fraction_current,
        "current_population_coverage": coverage,
        "retained_population_coverage": retained_coverage,
        "retained_population_lift": retained_lift,
        "new_population_coverage": new_coverage,
        # Compatibility aliases for existing aggregate/report consumers.
        "previous_incremental_coverage": previous_coverage,
        "current_incremental_coverage": coverage,
        "incremental_coverage_delta": coverage_delta,
        "classification": population_status,
    }
    relevance_counts = {
        reason: sum(reason in event.get("relevance_reasons", ()) for event in eligible_events)
        for reason in (
            "source_role_overlap", "target_role_overlap", "family_overlap", "carrier_overlap",
            "context_overlap", "game_overlap",
        )
    }
    game_only_overlap_count = sum(
        event.get("relevance_reasons") == ["game_overlap"] for event in eligible_events
    )
    context_only_overlap_count = sum(
        event.get("relevance_reasons") == ["context_overlap"] for event in eligible_events
    )
    diagnostics_errors: list[str] = []
    if len(relevant_events) + len(unrelated_events) + len(invalid_events) != event_count:
        diagnostics_errors.append("event_population_accounting_mismatch")
    if len(explained) > len(relevant_events):
        diagnostics_errors.append("explained_event_count_exceeds_relevant_count")
    if len(common_event_ids) > len(baseline_event_ids) or len(common_event_ids) > len(candidate_event_ids):
        diagnostics_errors.append("matched_population_count_mismatch")
    diagnostics = {
        "total_event_count": event_count,
        "total_later_event_count": event_count,
        "eligible_explanation_event_count": eligible_count,
        "relevant_heldout_event_count": eligible_count,
        "explained_relevant_event_count": len(explained),
        "unrelated_event_count": len(unrelated_events),
        "invalid_explanation_event_count": len(invalid_events),
        "explained_event_count": len(explained),
        "rejected_event_count": len(rejected),
        "incremental_explanatory_coverage": coverage,
        "relevant_incremental_coverage": coverage,
        "global_explanatory_reach": len(global_explained),
        "global_coverage_descriptive": float(len(global_explained)) / len(eligible_events) if eligible_events else 0.0,
        "explained_event_type_counts": _event_type_counts(explained),
        "rejected_event_type_counts": _event_type_counts(rejected),
        "invalid_event_type_counts": _event_type_counts(invalid_events),
        "prediction_explained_event_count": sum("prediction" in event["explanation_channels"] for event in explained),
        "behavioral_explained_event_count": sum("behavioral" in event["explanation_channels"] for event in explained),
        "compression_explained_event_count": sum("compression" in event["explanation_channels"] for event in explained),
        "multi_channel_explained_event_count": sum(len(event["explanation_channels"]) > 1 for event in explained),
        "mean_prediction_gain": _mean_event_gain(relevant_events, "prediction_gain") or 0.0,
        "mean_behavioral_gain": _mean_event_gain(relevant_events, "behavioral_gain") or 0.0,
        "mean_compression_gain": _mean_event_gain(relevant_events, "compression_gain") or 0.0,
        "baseline_description_cost": baseline_cost,
        "concept_description_cost": concept_cost,
        "incremental_compression_gain": compression_gain,
        "explained_event_ids_sample": sorted(explained_ids)[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
        "rejected_event_ids_sample": sorted(
            str(event["event_id"]) for event in rejected
        )[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
        "functional_coverage_longitudinal_change": longitudinal,
        "coverage_longitudinal_change": longitudinal,
        "coverage_change_classification": population_status,
        "relevant_event_type_diagnostics": _relevant_event_type_diagnostics(relevant_events),
        "relevant_transfer_event_count": sum(1 for event in relevant_events if event["event_type"] == "transfer"),
        "relevant_future_option_event_count": sum(1 for event in relevant_events if event["event_type"] == "future_option_motif"),
        "relevant_prediction_event_count": sum(1 for event in relevant_events if event["event_type"] == "prediction"),
        "relevant_behavior_event_count": sum(1 for event in relevant_events if event["event_type"] == "contradiction_resolution"),
        "evaluation_population_id": evaluation_population_id,
        "baseline_type": "best_single_linked_role",
        "candidate_type": "combined_linked_roles",
        "validation_method": "role_combination_ablation",
        "role_combination_lift": _mean_event_gain(relevant_events, "concept_incremental_gain") or 0.0,
        "baseline_event_count": len(baseline_event_ids),
        "candidate_event_count": len(candidate_event_ids),
        "common_event_count": len(common_event_ids),
        "baseline_only_event_count": len(baseline_event_ids - candidate_event_ids),
        "candidate_only_event_count": len(candidate_event_ids - baseline_event_ids),
        "baseline_population_matched": baseline_event_ids == candidate_event_ids,
        "retained_event_count": len(retained_ids),
        "retained_fraction_previous": retained_fraction_previous,
        "retained_fraction_current": retained_fraction_current,
        "current_population_coverage": coverage,
        "retained_population_coverage": retained_coverage,
        "retained_population_lift": retained_lift,
        "new_population_coverage": new_coverage,
        "relevant_by_source_role_count": relevance_counts["source_role_overlap"],
        "relevant_by_target_role_count": relevance_counts["target_role_overlap"],
        "relevant_by_family_count": relevance_counts["family_overlap"],
        "relevant_by_carrier_count": relevance_counts["carrier_overlap"],
        "relevant_by_context_count": relevance_counts["context_overlap"],
        "relevant_by_game_count": relevance_counts["game_overlap"],
        "game_only_overlap_count": game_only_overlap_count,
        "context_only_overlap_count": context_only_overlap_count,
        "diagnostics_errors": diagnostics_errors,
    }
    return events, diagnostics, state


def _event_type_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        counts[str(event["event_type"])] += 1
    return {key: counts[key] for key in sorted(counts)}


def _write_explanation_event_artifact(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for event in sorted(events, key=lambda item: (str(item["concept_id"]), str(item["event_type"]), str(item["event_id"]))):
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    temporary.replace(path)


def _build_incremental_coverage_diagnostics(
    *,
    candidate_signature: str,
    diagnostic_epoch_id: str | int | None,
    explanatory_reach: float,
    role_metrics: dict[str, dict[str, Any]],
    source_roles: list[str],
    links: dict[str, set[str]],
    concept_explained_structures: set[str],
    source_role_explained_structures: set[str],
    newly_explained_structures: set[str],
    previous_state: dict[str, Any] | None,
    previous_states_with_same_structure: dict[str, list[dict[str, Any]]],
    all_current_structure_ids: dict[str, set[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expose the same set arithmetic used by the Phase 3 coverage gate."""
    coverage_denominator = concept_explained_structures | source_role_explained_structures
    overlapping_structures = concept_explained_structures & source_role_explained_structures
    candidate_count = len(concept_explained_structures)
    source_role_count = len(source_role_explained_structures)
    overlap_count = len(overlapping_structures)
    newly_count = len(newly_explained_structures)
    denominator_count = len(coverage_denominator)
    expected_incremental = float(newly_count) / float(denominator_count) if denominator_count else 0.0
    diagnostics_errors: list[str] = []
    if newly_count + overlap_count != candidate_count:
        diagnostics_errors.append("new_structure_count_plus_overlap_structure_count_mismatch")
    if not 0.0 <= expected_incremental <= 1.0:
        diagnostics_errors.append("incremental_explanatory_coverage_out_of_range")

    role_overlap = [
        {
            "role_id": role,
            "overlap_count": len(concept_explained_structures & set(role_metrics.get(role, {}).get("structures", []))),
            "overlap_weight": float(len(concept_explained_structures & set(role_metrics.get(role, {}).get("structures", [])))),
        }
        for role in sorted(source_roles)
        if role in role_metrics
    ]
    family_overlap_counts: dict[str, int] = defaultdict(int)
    for structure_id in overlapping_structures:
        if structure_id.startswith("family:"):
            family_overlap_counts[structure_id.removeprefix("family:")] += 1
    family_overlap = [
        {
            "family_id": family,
            "overlap_count": count,
            "overlap_weight": float(count),
        }
        for family, count in sorted(family_overlap_counts.items())
    ]
    overlap_ratio = float(overlap_count) / float(candidate_count) if candidate_count else 0.0
    breakdown = _structure_type_counts(coverage_denominator)
    structure_fingerprint = _structure_fingerprint(concept_explained_structures)
    state = {
        "candidate_explained_count": candidate_count,
        "candidate_explained_weight": float(candidate_count),
        "candidate_raw_explained_weight": float(explanatory_reach),
        "already_explained_count": source_role_count,
        "already_explained_weight": float(source_role_count),
        "newly_explained_count": newly_count,
        "coverage_denominator_count": denominator_count,
        "coverage_denominator_weight": float(denominator_count),
        "incremental_coverage": expected_incremental,
        "overlap_count": overlap_count,
        "overlap_weight": float(overlap_count),
        "explanation_structure_ids": sorted(concept_explained_structures),
        "structure_fingerprint": structure_fingerprint,
    }
    longitudinal = _coverage_longitudinal_change(
        current_state=state,
        previous_state=previous_state,
        diagnostic_epoch_id=diagnostic_epoch_id,
    )
    explanation_change = _explanation_set_change(
        current_ids=concept_explained_structures,
        previous_state=previous_state,
    )
    causes = _coverage_change_causes(
        longitudinal=longitudinal,
        explanation_change=explanation_change,
        previous_state=previous_state,
        structure_fingerprint=structure_fingerprint,
        candidate_signature=candidate_signature,
        previous_states_with_same_structure=previous_states_with_same_structure,
        all_current_structure_ids=all_current_structure_ids,
    )
    decline = _coverage_decline_classification(longitudinal)
    return (
        {
            "incremental_coverage_diagnostics": {
                **{key: state[key] for key in (
                    "candidate_explained_count",
                    "candidate_explained_weight",
                    "already_explained_count",
                    "already_explained_weight",
                    "coverage_denominator_count",
                    "coverage_denominator_weight",
                    "incremental_coverage",
                )},
                "newly_explained_count": newly_count,
                "newly_explained_weight": float(newly_count),
                "candidate_raw_explained_weight": float(explanatory_reach),
                "concept_explained_structure_count": candidate_count,
                "source_role_explained_structure_count": source_role_count,
                "overlapping_structure_count": overlap_count,
                "newly_explained_structure_count": newly_count,
                "concept_explained_structure_type_counts": _structure_type_counts(concept_explained_structures),
                "newly_explained_structure_type_counts": _structure_type_counts(newly_explained_structures),
                "newly_explained_structure_ids_sample": sorted(newly_explained_structures)[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
                "overlapping_structure_ids_sample": sorted(overlapping_structures)[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
            },
            "coverage_denominator_breakdown": breakdown,
            "coverage_overlap": {
                "overlap_count": overlap_count,
                "overlap_ratio": overlap_ratio,
                "overlap_by_existing_concept": [],
                "overlap_by_role": role_overlap[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
                "overlap_by_role_count": len(role_overlap),
                "overlap_by_family": family_overlap[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
                "overlap_by_family_count": len(family_overlap),
            },
            "coverage_longitudinal_change": longitudinal,
            "coverage_change_causes": causes,
            "explanation_set_change": explanation_change,
            "coverage_decline_classification": decline,
            "diagnostics_errors": diagnostics_errors,
        },
        state,
    )


def _coverage_longitudinal_change(
    *,
    current_state: dict[str, Any],
    previous_state: dict[str, Any] | None,
    diagnostic_epoch_id: str | int | None,
) -> dict[str, Any]:
    if previous_state is None:
        return {
            "previous_epoch": None,
            "current_epoch": None if diagnostic_epoch_id is None else str(diagnostic_epoch_id),
            "previous_candidate_explained_count": None,
            "current_candidate_explained_count": current_state["candidate_explained_count"],
            "candidate_explained_count_delta": None,
            "previous_denominator_count": None,
            "current_denominator_count": current_state["coverage_denominator_count"],
            "denominator_count_delta": None,
            "previous_incremental_coverage": None,
            "current_incremental_coverage": current_state["incremental_coverage"],
            "incremental_coverage_delta": None,
            "previous_overlap_weight": None,
            "current_overlap_weight": current_state["overlap_weight"],
            "previous_newly_explained_count": None,
            "current_newly_explained_count": current_state["newly_explained_count"],
        }
    previous_candidate_count = float(previous_state.get("candidate_explained_count", 0.0) or 0.0)
    previous_denominator_count = float(previous_state.get("coverage_denominator_count", 0.0) or 0.0)
    previous_incremental = float(previous_state.get("incremental_coverage", 0.0) or 0.0)
    previous_overlap = float(previous_state.get("overlap_weight", 0.0) or 0.0)
    previous_newly_explained_count = float(previous_state.get("newly_explained_count", 0.0) or 0.0)
    return {
        "previous_epoch": previous_state.get("epoch_id"),
        "current_epoch": None if diagnostic_epoch_id is None else str(diagnostic_epoch_id),
        "previous_candidate_explained_count": previous_state.get("candidate_explained_count"),
        "current_candidate_explained_count": current_state["candidate_explained_count"],
        "candidate_explained_count_delta": current_state["candidate_explained_count"] - previous_candidate_count,
        "previous_denominator_count": previous_state.get("coverage_denominator_count"),
        "current_denominator_count": current_state["coverage_denominator_count"],
        "denominator_count_delta": current_state["coverage_denominator_count"] - previous_denominator_count,
        "previous_incremental_coverage": previous_incremental,
        "current_incremental_coverage": current_state["incremental_coverage"],
        "incremental_coverage_delta": current_state["incremental_coverage"] - previous_incremental,
        "previous_overlap_weight": previous_overlap,
        "current_overlap_weight": current_state["overlap_weight"],
        "previous_newly_explained_count": previous_newly_explained_count,
        "current_newly_explained_count": current_state["newly_explained_count"],
    }


def _explanation_set_change(
    *,
    current_ids: set[str],
    previous_state: dict[str, Any] | None,
) -> dict[str, Any]:
    previous_ids = set(previous_state.get("explanation_structure_ids", [])) if previous_state else set()
    added = sorted(current_ids - previous_ids)
    removed = sorted(previous_ids - current_ids)
    return {
        "added_structure_count": len(added),
        "removed_structure_count": len(removed),
        "retained_structure_count": len(current_ids & previous_ids),
        "added_structure_ids_sample": added[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
        "removed_structure_ids_sample": removed[:_DIAGNOSTIC_ID_SAMPLE_LIMIT],
    }


def _coverage_change_causes(
    *,
    longitudinal: dict[str, Any],
    explanation_change: dict[str, Any],
    previous_state: dict[str, Any] | None,
    structure_fingerprint: str,
    candidate_signature: str,
    previous_states_with_same_structure: dict[str, list[dict[str, Any]]],
    all_current_structure_ids: dict[str, set[str]],
) -> list[str]:
    if previous_state is None:
        changed_signature = any(
            str(item.get("candidate_signature")) != candidate_signature
            for item in previous_states_with_same_structure.get(structure_fingerprint, [])
        )
        causes = ["candidate_signature_changed"] if changed_signature else []
        causes.append("missing_previous_epoch_baseline")
        return causes
    candidate_delta = float(longitudinal["current_newly_explained_count"] or 0.0) - float(
        longitudinal["previous_newly_explained_count"] or 0.0
    )
    denominator_delta = float(longitudinal["denominator_count_delta"] or 0.0)
    overlap_delta = float(longitudinal["current_overlap_weight"] or 0.0) - float(
        longitudinal["previous_overlap_weight"] or 0.0
    )
    causes: list[str] = []
    if denominator_delta > 0 and candidate_delta < denominator_delta:
        causes.append("denominator_growth_exceeded_candidate_growth")
    if overlap_delta > 0:
        causes.append("increased_overlap_with_existing_structures")
    if int(explanation_change["removed_structure_count"] or 0) > 0:
        causes.append("candidate_explanation_set_shrank")
        removed = set(explanation_change["removed_structure_ids_sample"])
        if removed and any(
            candidate != candidate_signature and bool(removed & structures)
            for candidate, structures in all_current_structure_ids.items()
        ):
            causes.append("evidence_reassigned_to_other_structure")
    return causes


def _coverage_decline_classification(longitudinal: dict[str, Any]) -> dict[str, Any]:
    previous_incremental = longitudinal.get("previous_incremental_coverage")
    current_incremental = longitudinal.get("current_incremental_coverage")
    if previous_incremental is None or current_incremental is None or current_incremental >= previous_incremental:
        return {
            "classification": "not_declining",
            "numerator_growth_rate": 0.0,
            "denominator_growth_rate": 0.0,
            "overlap_growth_rate": 0.0,
        }
    previous_candidate = float(longitudinal.get("previous_newly_explained_count") or 0.0)
    current_candidate = float(longitudinal.get("current_newly_explained_count") or 0.0)
    previous_denominator = float(longitudinal.get("previous_denominator_count") or 0.0)
    current_denominator = float(longitudinal.get("current_denominator_count") or 0.0)
    previous_overlap = float(longitudinal.get("previous_overlap_weight") or 0.0)
    current_overlap = float(longitudinal.get("current_overlap_weight") or 0.0)
    numerator_rate = (current_candidate - previous_candidate) / max(abs(previous_candidate), 1.0)
    denominator_rate = (current_denominator - previous_denominator) / max(abs(previous_denominator), 1.0)
    overlap_rate = (current_overlap - previous_overlap) / max(abs(previous_overlap), 1.0)
    drivers = []
    if numerator_rate < 0:
        drivers.append("numerator_decline")
    if denominator_rate > 0 and numerator_rate < denominator_rate:
        drivers.append("denominator_growth")
    if overlap_rate > 0:
        drivers.append("overlap_growth")
    return {
        "classification": drivers[0] if len(drivers) == 1 else ("mixed" if drivers else "not_declining"),
        "numerator_growth_rate": numerator_rate,
        "denominator_growth_rate": denominator_rate,
        "overlap_growth_rate": overlap_rate,
    }


def _concept_promotion_rejection_reasons(
    *,
    promoted: bool,
    meets_promotion_score_threshold: bool,
    has_incremental_gain: bool,
    has_relevant_samples: bool,
    has_cross_evidence: bool,
    heldout_lifts: list[float],
    has_heldout_lift: bool,
    has_positive_compression: bool,
    population_matched: bool,
    population_status: str,
    demoted: bool,
) -> list[str]:
    """Expose the exact Phase 3 gate outcomes without changing the gates."""
    if promoted:
        return []
    reasons: list[str] = []
    if not population_matched:
        reasons.append("baseline_population_mismatch")
    if not has_relevant_samples:
        reasons.append("insufficient_relevant_samples")
    elif not has_incremental_gain:
        reasons.append("relevant_coverage_below_threshold")
    if not has_cross_evidence:
        reasons.append("insufficient_cross_context_or_game_evidence")
    if not has_positive_compression:
        reasons.append("compression_gain_non_positive")
    if population_status != "comparable":
        reasons.append("population_not_comparable")
    if not heldout_lifts or not has_heldout_lift:
        reasons.extend(("prediction_lift_below_threshold", "behavioral_lift_below_threshold"))
    if not meets_promotion_score_threshold:
        reasons.append("below_promotion_score_threshold")
    if demoted:
        reasons.append("demoted_after_repeated_failure")
    return list(dict.fromkeys(reasons)) or ["below_promotion_score_threshold"]


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
    validation_epoch: str | int | None = None,
    count_failure: bool = True,
    retain_previous_promotion: bool = False,
    previously_promoted: bool = False,
    validation_result: str | None = None,
) -> tuple[str, int, bool]:
    previous = state_conn.execute(
        """
        SELECT failure_count, promotion_status, last_validation_epoch FROM promotion_validation_state
        WHERE candidate_type = ? AND candidate_signature = ?
        """,
        (candidate_type, candidate_signature),
    ).fetchone()
    previous_failures = int(previous["failure_count"] or 0) if previous is not None else 0
    previous_status = str(previous["promotion_status"] or "candidate") if previous is not None else "candidate"
    epoch_value = None if validation_epoch is None else str(validation_epoch)
    same_epoch = bool(previous is not None and epoch_value is not None and previous["last_validation_epoch"] == epoch_value)
    if passed:
        status, failure_count, demoted = "promoted", 0, False
    elif not count_failure:
        status = "retained" if retain_previous_promotion else previous_status
        failure_count, demoted = previous_failures, False
    elif same_epoch:
        status, failure_count, demoted = previous_status, previous_failures, previous_status == "demoted"
    else:
        failure_count = previous_failures + 1
        was_promoted = bool(previously_promoted or previous_status in {"promoted", "validation_failed"})
        demoted = was_promoted and failure_count >= max(1, demotion_failure_limit)
        status = "demoted" if demoted else ("validation_failed" if was_promoted else "candidate")
    state_conn.execute(
        """
        INSERT INTO promotion_validation_state (
            candidate_type, candidate_signature, failure_count, promotion_status,
            last_validation_scope, last_validation_prediction_lift,
            last_validation_action_selection_lift, last_validation_transfer_lift,
            last_validation_epoch, last_validation_global_step, last_validation_result, updated_global_step
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_type, candidate_signature) DO UPDATE SET
            failure_count = excluded.failure_count,
            promotion_status = excluded.promotion_status,
            last_validation_scope = excluded.last_validation_scope,
            last_validation_prediction_lift = excluded.last_validation_prediction_lift,
            last_validation_action_selection_lift = excluded.last_validation_action_selection_lift,
            last_validation_transfer_lift = excluded.last_validation_transfer_lift,
            last_validation_epoch = excluded.last_validation_epoch,
            last_validation_global_step = excluded.last_validation_global_step,
            last_validation_result = excluded.last_validation_result,
            updated_global_step = excluded.updated_global_step
        """,
        (
            candidate_type, candidate_signature, failure_count, status, validation_scope,
            validation_prediction_lift, validation_action_selection_lift, validation_transfer_lift,
            epoch_value, updated_global_step,
            validation_result or ("passed" if passed else "failed"), updated_global_step,
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
        profile_carriers = tuple(str(value) for value in profile.get("source_carrier_signatures", ()))
        if (
            len(profile_carriers) == 1
            and profile_carriers[0] == str(target_carrier_signature)
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
    # Older synthetic callers and pre-migration profile caches only carry the
    # aggregate carrier count.  It is the best available support signal for
    # those in-memory inputs; persisted attempts always carry the explicit
    # field below.
    source_evidence_support_count = int(
        best.get("source_evidence_support_count", best.get("source_carrier_count", 0)) or 0
    )
    source_carrier_count = int(best["source_carrier_count"])
    similarity_score = float(best_similarity)
    support_gate_passed = int(source_evidence_support_count >= MIN_SOURCE_EVIDENCE_SUPPORT)
    similarity_gate_passed = int(similarity_score >= MIN_TRANSFER_SIMILARITY)
    role_match_gate_passed = int(predicted_role_signature == observed_role_signature)
    if not support_gate_passed:
        failure_reason = "insufficient_source_support"
    elif not similarity_gate_passed:
        failure_reason = "low_similarity"
    elif not role_match_gate_passed:
        failure_reason = "role_mismatch"
    else:
        failure_reason = "success"
    reuse_success = int(failure_reason == "success")
    target_scope_type = "game" if transfer_kind == "cross_game" else "context"
    transfer_score = similarity_score * min(
        1.0,
        source_evidence_support_count / MIN_SOURCE_EVIDENCE_SUPPORT,
    )
    source_carriers = tuple(str(value) for value in best.get("source_carrier_signatures", ()))
    source_games = tuple(str(value) for value in best.get("source_game_keys", ()))
    source_contexts = tuple(str(value) for value in best.get("source_context_keys", ()))
    source_context_game_keys = {
        str(context): tuple(str(game) for game in games)
        for context, games in dict(best.get("source_context_game_keys", {})).items()
    }
    # A profile can still match structurally while lacking the concrete scope
    # required to make the requested transfer claim.  Treat it as an
    # unusable source profile instead of manufacturing a source identity or
    # letting the worker abort on the persistence validation below.
    if transfer_kind == "cross_game" and not source_games:
        return _no_source_profile_attempt(target, transfer_kind, target_scope_key)
    if transfer_kind == "cross_context":
        source_contexts = tuple(value for value in source_contexts if value != str(target_scope_key))
        if not source_contexts:
            return _no_source_profile_attempt(target, transfer_kind, target_scope_key)
    source_carrier_signature = source_carriers[0] if len(source_carriers) == 1 else None
    source_game_key = source_games[0] if len(source_games) == 1 else None
    source_context_key = source_contexts[0] if len(source_contexts) == 1 else None
    target_games = (
        (str(target_scope_key),)
        if transfer_kind == "cross_game"
        else tuple(str(value) for value in target.get("context_games", {}).get(target_scope_key, ()))
    )
    target_game_key = target_games[0] if len(target_games) == 1 else None
    target_context_key = target_scope_key if transfer_kind == "cross_context" else None
    provenance_mode = "single_source" if len(source_carriers) == 1 else "multi_source"
    # Include the complete concrete provenance identity.  A multi-carrier
    # profile is explicit in the JSON fields rather than collapsed into an
    # arbitrary carrier or a negative pseudo-scope.
    attempt_seed = "|".join((
        transfer_kind,
        predicted_role_signature,
        source_game_key or ",".join(source_games),
        target_game_key or "",
        source_context_key or ",".join(source_contexts),
        target_context_key or "",
        source_carrier_signature or ",".join(source_carriers),
        target_carrier_signature,
        predicted_role_signature,
    ))
    return {
        "attempt_id": sha1(attempt_seed.encode("utf-8")).hexdigest(),
        "role_signature": observed_role_signature,
        "transfer_kind": transfer_kind,
        "source_scope_type": "game" if transfer_kind == "cross_game" else "context",
        "source_scope_key": source_game_key if transfer_kind == "cross_game" else source_context_key,
        "target_scope_type": target_scope_type,
        "target_scope_key": target_scope_key,
        "source_game_key": source_game_key,
        "target_game_key": target_game_key,
        "source_context_key": source_context_key,
        "target_context_key": target_context_key,
        "source_carrier_signature": source_carrier_signature,
        "source_role_signature": predicted_role_signature,
        "predicted_target_role_signature": predicted_role_signature,
        "observed_target_role_signature": observed_role_signature,
        "source_carrier_signatures_json": json.dumps(source_carriers),
        "source_game_keys_json": json.dumps(source_games),
        "source_context_keys_json": json.dumps(source_contexts),
        "source_context_game_keys_json": json.dumps(source_context_game_keys, sort_keys=True),
        "provenance_mode": provenance_mode,
        "provenance_status": "verified" if provenance_mode == "single_source" else "aggregate_source",
        "target_carrier_signature": target_carrier_signature,
        "predicted_role_signature": predicted_role_signature,
        "observed_role_signature": observed_role_signature,
        "similarity_score": similarity_score,
        "transfer_score": transfer_score,
        "reuse_success": reuse_success,
        "failure_reason": failure_reason,
        "best_margin": best_margin,
        "source_carrier_count": source_carrier_count,
        "source_evidence_support_count": source_evidence_support_count,
        "support_gate_passed": support_gate_passed,
        "similarity_gate_passed": similarity_gate_passed,
        "role_match_gate_passed": role_match_gate_passed,
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
        attempt["source_game_key"],
        attempt["target_game_key"],
        attempt["source_context_key"],
        attempt["target_context_key"],
        attempt["source_interaction_id"],
        attempt["target_interaction_id"],
        attempt["source_scope_origin"],
        attempt["target_scope_origin"],
        attempt["source_game_is_surrogate"],
        attempt["target_game_is_surrogate"],
        attempt["source_context_is_surrogate"],
        attempt["target_context_is_surrogate"],
        attempt["source_game_resolution_source"],
        attempt["target_game_resolution_source"],
        attempt["source_context_resolution_source"],
        attempt["target_context_resolution_source"],
        attempt["source_carrier_signature"],
        attempt["source_role_signature"],
        attempt["predicted_target_role_signature"],
        attempt["observed_target_role_signature"],
        attempt["source_carrier_signatures_json"],
        attempt["source_game_keys_json"],
        attempt["source_context_keys_json"],
        attempt["provenance_mode"],
        attempt["provenance_status"],
        attempt["target_carrier_signature"],
        attempt["predicted_role_signature"],
        attempt["observed_role_signature"],
        attempt["similarity_score"],
        attempt["transfer_score"],
        attempt["reuse_success"],
        attempt["failure_reason"],
        attempt["best_margin"],
        attempt["source_carrier_count"],
        attempt["source_evidence_support_count"],
        attempt["support_gate_passed"],
        attempt["similarity_gate_passed"],
        attempt["role_match_gate_passed"],
        attempt["candidate_role_count"],
        attempt["first_seen_global_step"],
        attempt["last_seen_global_step"],
    )


def _scope_surrogate(prefix: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:" + sha1(encoded.encode("utf-8")).hexdigest()[:20]


def _transfer_scope_candidates(
    connection: sqlite3.Connection,
    *,
    interaction_id: str | None,
    carrier_signature: str | None,
    role_signature: str | None,
) -> list[dict[str, object]]:
    """Return deterministic concrete evidence candidates for one transfer side."""
    candidates: list[dict[str, object]] = []
    try:
        rows = connection.execute(
            """
            SELECT source_interaction_id, COALESCE(source_game_id, game) AS game_key,
                   COALESCE(source_context_signature, context_key) AS context_key,
                   source_carrier_id, source_role_id, source_family_id,
                   first_seen_global_step
            FROM future_option_events
            WHERE source_interaction_id IS NOT NULL
              AND (
                    source_interaction_id = ?
                 OR source_carrier_id = ?
                 OR source_role_id = ?
              )
            ORDER BY CASE WHEN source_interaction_id = ? THEN 0
                          WHEN source_carrier_id = ? THEN 1
                          WHEN source_role_id = ? THEN 2 ELSE 3 END,
                     COALESCE(first_seen_global_step, 9223372036854775807) ASC,
                     source_interaction_id ASC
            """,
            (interaction_id, carrier_signature, role_signature, interaction_id, carrier_signature, role_signature),
        ).fetchall()
        for row in rows:
            candidates.append(
                {
                    "interaction_id": str(row["source_interaction_id"]),
                    "game_key": row["game_key"],
                    "context_key": row["context_key"],
                    "origin": (
                        "interaction" if interaction_id not in (None, "") and str(row["source_interaction_id"]) == str(interaction_id)
                        else "carrier_interaction" if row["source_carrier_id"] == carrier_signature
                        else "role_interaction"
                    ),
                    "step": row["first_seen_global_step"],
                }
            )
    except sqlite3.Error:
        pass
    if candidates or not carrier_signature:
        return candidates
    try:
        families = [
            str(row[0])
            for row in connection.execute(
                "SELECT linked_key FROM carrier_links WHERE carrier_signature = ? AND linked_type = 'family' ORDER BY linked_key ASC",
                (carrier_signature,),
            ).fetchall()
        ]
        if families:
            placeholders = ",".join("?" for _ in families)
            rows = connection.execute(
                f"""
                SELECT source_interaction_id, COALESCE(source_game_id, game) AS game_key,
                       COALESCE(source_context_signature, context_key) AS context_key,
                       first_seen_global_step
                FROM future_option_events
                WHERE source_family_id IN ({placeholders}) AND source_interaction_id IS NOT NULL
                ORDER BY source_family_id ASC, COALESCE(first_seen_global_step, 9223372036854775807) ASC,
                         source_interaction_id ASC
                """,
                tuple(families),
            ).fetchall()
            candidates.extend(
                {
                    "interaction_id": str(row["source_interaction_id"]),
                    "game_key": row["game_key"],
                    "context_key": row["context_key"],
                    "origin": "family_interaction",
                    "step": row["first_seen_global_step"],
                }
                for row in rows
            )
    except sqlite3.Error:
        pass
    return candidates


def resolve_transfer_scope(
    connection: sqlite3.Connection,
    *,
    interaction_id: str | None,
    carrier_signature: str | None,
    role_signature: str | None,
    existing_game_key: str | None,
    existing_context_key: str | None,
    side: str = "source",
) -> dict[str, str | int | None]:
    """Resolve a transfer side without losing concrete evidence provenance.

    The public fields follow the stable transfer-validation normalization.  The
    extra interaction/origin fields let the caller persist the selected proof.
    """
    candidates = _transfer_scope_candidates(
        connection,
        interaction_id=interaction_id,
        carrier_signature=carrier_signature,
        role_signature=role_signature,
    )
    candidate = candidates[0] if candidates else {}
    chosen_interaction_id = interaction_id or candidate.get("interaction_id")
    game_key = existing_game_key or candidate.get("game_key")
    context_key = existing_context_key or candidate.get("context_key")
    game_source = "direct_attempt" if existing_game_key else (str(candidate.get("origin")) if candidate.get("game_key") else None)
    context_source = "direct_attempt" if existing_context_key else (str(candidate.get("origin")) if candidate.get("context_key") else None)
    if chosen_interaction_id in (None, ""):
        chosen_interaction_id = _scope_surrogate(
            "surrogate_interaction",
            {
                "side": side,
                "role_signature": role_signature,
                "carrier_signature": carrier_signature,
                "game_key": game_key,
                "context_key": context_key,
            },
        )
        interaction_origin = "surrogate"
    else:
        interaction_origin = str(candidate.get("origin") or "direct_attempt")
    game_is_surrogate = 0
    if game_key in (None, ""):
        game_key = _scope_surrogate(
            "surrogate_game",
            {
                "side": side,
                "role_signature": role_signature,
                "carrier_signature": carrier_signature,
                "context_key": context_key,
                "interaction_id": chosen_interaction_id,
            },
        )
        game_source = "surrogate"
        game_is_surrogate = 1
    context_is_surrogate = 0
    if context_key in (None, ""):
        context_key = _scope_surrogate(
            "surrogate_context",
            {
                "side": side,
                "game_key": game_key,
                "role_signature": role_signature,
                "carrier_signature": carrier_signature,
                "interaction_id": chosen_interaction_id,
            },
        )
        context_source = "surrogate"
        context_is_surrogate = 1
    return {
        "game_key": str(game_key),
        "context_key": str(context_key),
        "game_resolution_source": str(game_source or "surrogate"),
        "context_resolution_source": str(context_source or "surrogate"),
        "interaction_id": str(chosen_interaction_id),
        "scope_origin": interaction_origin,
        "game_is_surrogate": game_is_surrogate,
        "context_is_surrogate": context_is_surrogate,
    }


def _resolve_transfer_attempt_provenance(connection: sqlite3.Connection, attempt: dict[str, Any]) -> dict[str, Any]:
    source = resolve_transfer_scope(
        connection,
        interaction_id=attempt.get("source_interaction_id"),
        carrier_signature=attempt.get("source_carrier_signature"),
        role_signature=attempt.get("source_role_signature"),
        existing_game_key=attempt.get("source_game_key"),
        existing_context_key=attempt.get("source_context_key"),
        side="source",
    )
    target = resolve_transfer_scope(
        connection,
        interaction_id=attempt.get("target_interaction_id"),
        carrier_signature=attempt.get("target_carrier_signature"),
        role_signature=attempt.get("observed_target_role_signature") or attempt.get("role_signature"),
        existing_game_key=attempt.get("target_game_key"),
        existing_context_key=attempt.get("target_context_key"),
        side="target",
    )
    attempt.update(
        {
            "source_interaction_id": source["interaction_id"], "target_interaction_id": target["interaction_id"],
            "source_scope_origin": source["scope_origin"], "target_scope_origin": target["scope_origin"],
            "source_game_key": source["game_key"], "target_game_key": target["game_key"],
            "source_context_key": source["context_key"], "target_context_key": target["context_key"],
            "source_game_is_surrogate": source["game_is_surrogate"], "target_game_is_surrogate": target["game_is_surrogate"],
            "source_context_is_surrogate": source["context_is_surrogate"], "target_context_is_surrogate": target["context_is_surrogate"],
            "source_game_resolution_source": source["game_resolution_source"], "target_game_resolution_source": target["game_resolution_source"],
            "source_context_resolution_source": source["context_resolution_source"], "target_context_resolution_source": target["context_resolution_source"],
        }
    )
    real_cross_game = not source["game_is_surrogate"] and not target["game_is_surrogate"] and source["game_key"] != target["game_key"]
    real_cross_context = not source["context_is_surrogate"] and not target["context_is_surrogate"] and source["context_key"] != target["context_key"]
    if str(attempt.get("provenance_mode")) == "single_source" and (real_cross_game or real_cross_context):
        attempt["provenance_status"] = "verified"
    elif all((source["game_key"], target["game_key"], source["context_key"], target["context_key"])) and any((source["game_is_surrogate"], target["game_is_surrogate"], source["context_is_surrogate"], target["context_is_surrogate"])):
        attempt["provenance_status"] = "resolved_with_surrogate"
    else:
        attempt["provenance_status"] = "proxy"
    attempt["attempt_id"] = sha1(json.dumps({
        "transfer_kind": attempt.get("transfer_kind"), "source_role_signature": attempt.get("source_role_signature"),
        "source_game_key": source["game_key"], "target_game_key": target["game_key"],
        "source_context_key": source["context_key"], "target_context_key": target["context_key"],
        "source_carrier_signature": attempt.get("source_carrier_signature"),
        "target_carrier_signature": attempt.get("target_carrier_signature"),
        "predicted_target_role_signature": attempt.get("predicted_target_role_signature"),
        "source_interaction_id": source["interaction_id"], "target_interaction_id": target["interaction_id"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return attempt


def _expand_transfer_attempt_provenance(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a multi-scope source profile into concrete source-target attempts.

    The transfer score remains the score of the selected profile; expansion
    records every concrete source scope that contributed to that profile so no
    target scope is ever stored as synthetic source provenance.
    """
    kind = str(attempt["transfer_kind"])
    source_keys = (
        tuple(json.loads(str(attempt["source_game_keys_json"] or "[]")))
        if kind == "cross_game"
        else tuple(json.loads(str(attempt["source_context_keys_json"] or "[]")))
    )
    if not source_keys:
        return [attempt]
    source_carriers = tuple(json.loads(str(attempt["source_carrier_signatures_json"] or "[]")))
    source_games = tuple(json.loads(str(attempt["source_game_keys_json"] or "[]")))
    source_context_game_keys = json.loads(str(attempt.get("source_context_game_keys_json") or "{}"))
    target_games = (str(attempt["target_game_key"]),) if attempt.get("target_game_key") else ()
    result: list[dict[str, Any]] = []
    for source_key in sorted(str(value) for value in source_keys):
        # One concrete carrier/scope pair is a verified provenance row.  It
        # retains the aggregate profile's independent evidence support, so a
        # single carrier identity does not erase the evidence that selected
        # the source role.
        for source_carrier in (source_carriers or (None,)):
            concrete = dict(attempt)
            if kind == "cross_game":
                concrete["source_scope_type"] = "game"
                concrete["source_scope_key"] = source_key
                concrete["source_game_key"] = source_key
            else:
                concrete["source_scope_type"] = "context"
                concrete["source_scope_key"] = source_key
                concrete["source_context_key"] = source_key
                candidate_games = tuple(sorted(str(value) for value in source_context_game_keys.get(source_key, ())))
                target_game = str(concrete.get("target_game_key") or "")
                concrete["source_game_key"] = (
                    target_game if target_game and target_game in candidate_games
                    else candidate_games[0] if candidate_games
                    else source_games[0] if len(source_games) == 1 else None
                )
            concrete["source_carrier_signature"] = source_carrier
            concrete["source_carrier_count"] = 1 if source_carrier else 0
            concrete["source_evidence_support_count"] = int(attempt.get("source_evidence_support_count") or 0)
            concrete["provenance_mode"] = "single_source" if source_carrier else "multi_source"
            concrete["provenance_status"] = "verified" if source_carrier else "aggregate_source"
            seed = "|".join((
                kind,
                str(concrete.get("source_role_signature") or ""),
                str(concrete.get("source_game_key") or ""),
                str(concrete.get("target_game_key") or ""),
                str(concrete.get("source_context_key") or ""),
                str(concrete.get("target_context_key") or ""),
                str(source_carrier or ""),
                str(concrete["target_carrier_signature"]),
                str(concrete.get("predicted_role_signature") or ""),
            ))
            concrete["attempt_id"] = sha1(seed.encode("utf-8")).hexdigest()
            result.append(concrete)
    return result


def _validate_transfer_attempt_provenance(attempt: dict[str, Any]) -> None:
    """Reject malformed concrete provenance before it reaches SQLite."""
    mode = str(attempt.get("provenance_mode") or "")
    if mode not in {"single_source", "multi_source", "missing_source"}:
        raise ValueError(f"unsupported transfer provenance mode: {mode}")
    kind = str(attempt.get("transfer_kind") or "")
    if kind not in {"cross_game", "cross_context"}:
        raise ValueError(f"unsupported transfer kind: {kind}")
    for key in (
        "source_interaction_id", "target_interaction_id", "source_game_key", "target_game_key",
        "source_context_key", "target_context_key",
    ):
        if attempt.get(key) in (None, ""):
            raise ValueError(f"transfer attempt is missing resolved {key}")
    for prefix in ("source_game", "target_game", "source_context", "target_context"):
        is_surrogate = bool(attempt.get(f"{prefix}_is_surrogate"))
        source = str(attempt.get(f"{prefix}_resolution_source") or "")
        if is_surrogate != (source == "surrogate"):
            raise ValueError(f"transfer attempt has inconsistent {prefix} surrogate resolution")
    real_cross_game = (
        not bool(attempt.get("source_game_is_surrogate"))
        and not bool(attempt.get("target_game_is_surrogate"))
        and str(attempt["source_game_key"]) != str(attempt["target_game_key"])
    )
    real_cross_context = (
        not bool(attempt.get("source_context_is_surrogate"))
        and not bool(attempt.get("target_context_is_surrogate"))
        and str(attempt["source_context_key"]) != str(attempt["target_context_key"])
    )
    if str(attempt.get("provenance_status")) == "verified" and not (
        mode == "single_source" and (real_cross_game or real_cross_context)
    ):
        raise ValueError("verified transfer attempt lacks a real cross-scope dimension")
    if int(attempt.get("reuse_success") or 0) == 1 and not attempt.get("source_role_signature"):
        raise ValueError("successful transfer attempt is missing a source role")
    if mode == "single_source" and not attempt.get("source_carrier_signature"):
        raise ValueError("single-source transfer attempt is missing a source carrier")


def _derive_role_transfer_attempts_chunk(
    *,
    chunk: list[tuple[str, str, str]],
    role_rows: dict[str, dict[str, Any]],
    profile_cache: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for carrier_signature, transfer_kind, target_scope_key in chunk:
        attempt = _predict_transfer_attempt(
            profile_cache=profile_cache,
            role_rows=role_rows,
            target_carrier_signature=carrier_signature,
            transfer_kind=transfer_kind,
            target_scope_key=target_scope_key,
        )
        for item in _expand_transfer_attempt_provenance(attempt):
            rows.append(item)
    return rows


def _build_transfer_profile_cache(
    *,
    role_rows: dict[str, dict[str, Any]],
    carrier_contexts: dict[str, set[str]],
    carrier_games: dict[str, set[str]],
    context_games: dict[str, set[str]],
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
            if transfer_kind == "cross_context":
                target_games = set(context_games.get(target_scope_key, set()))
                included_carriers = [
                    carrier for carrier in included_carriers
                    if not target_games or bool(carrier_games.get(carrier, set()) & target_games)
                ]
                if not included_carriers:
                    continue
            token_set: set[str] = set()
            context_set: set[str] = set()
            game_set: set[str] = set()
            source_carriers: list[str] = []
            for carrier_signature in included_carriers:
                source_carriers.append(carrier_signature)
                token_set.update(carrier_to_tokens.get(carrier_signature, ()))
                context_set.update(carrier_contexts.get(carrier_signature, set()))
                game_set.update(carrier_games.get(carrier_signature, set()))
            if transfer_kind == "cross_game":
                game_set.discard(target_scope_key)
            elif target_games:
                context_set = {
                    context for context in context_set
                    if context_games.get(context, set()) & target_games
                }
            profiles.append(
                {
                    "role_signature": role_signature,
                    "profile_tokens": sorted(token_set),
                    "profile_token_set": token_set,
                    "source_carrier_signatures": tuple(sorted(source_carriers)),
                    "source_game_keys": tuple(sorted(game_set)),
                    "source_context_keys": tuple(sorted(context_set)),
                    "source_context_game_keys": {
                        context: tuple(sorted(context_games.get(context, set())))
                        for context in sorted(context_set)
                    },
                    "source_carrier_count": len(source_carriers),
                    "source_evidence_support_count": len(source_carriers),
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
    target_scope_type = "game" if transfer_kind == "cross_game" else "context"
    attempt_seed = "|".join((transfer_kind, target_scope_key, str(target["carrier_signature"]), "no_source_profile"))
    return {
        "attempt_id": sha1(attempt_seed.encode("utf-8")).hexdigest(),
        "role_signature": str(target["role_signature"]),
        "transfer_kind": transfer_kind,
        "source_scope_type": None,
        "source_scope_key": None,
        "target_scope_type": target_scope_type,
        "target_scope_key": target_scope_key,
        "source_game_key": None,
        "target_game_key": target_scope_key if transfer_kind == "cross_game" else None,
        "source_context_key": None,
        "target_context_key": target_scope_key if transfer_kind == "cross_context" else None,
        "source_carrier_signature": None,
        "source_role_signature": None,
        "predicted_target_role_signature": None,
        "observed_target_role_signature": str(target["role_signature"]),
        "source_carrier_signatures_json": "[]",
        "source_game_keys_json": "[]",
        "source_context_keys_json": "[]",
        "provenance_mode": "missing_source",
        "provenance_status": "missing_source_profile",
        "target_carrier_signature": str(target["carrier_signature"]),
        "predicted_role_signature": None,
        "observed_role_signature": str(target["role_signature"]),
        "similarity_score": 0.0,
        "transfer_score": 0.0,
        "reuse_success": 0,
        "failure_reason": "no_source_profile",
        "best_margin": None,
        "source_carrier_count": 0,
        "source_evidence_support_count": 0,
        "support_gate_passed": 0,
        "similarity_gate_passed": 0,
        "role_match_gate_passed": 0,
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
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    links_by_carrier = _carrier_links_by_carrier(state_conn)
    carrier_contexts = {carrier: set(link_types.get("context", set())) for carrier, link_types in links_by_carrier.items()}
    carrier_games: dict[str, set[str]] = defaultdict(set)
    context_node_ids = {f"context:{context}" for contexts in carrier_contexts.values() for context in contexts}
    context_node_games = _context_games_for_context_nodes(graph_conn, context_node_ids)
    for carrier_signature, contexts in carrier_contexts.items():
        for context in contexts:
            carrier_games[carrier_signature].update(context_node_games.get(f"context:{context}", set()))
    context_games = {
        str(node_id).removeprefix("context:"): set(games)
        for node_id, games in context_node_games.items()
    }
    return carrier_contexts, carrier_games, context_games


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


def _write_historical_milestone(
    connection: sqlite3.Connection,
    name: str,
    first_global_step: int | None,
    evidence_key: str | None,
) -> None:
    if first_global_step is None:
        return
    connection.execute(
        """
        INSERT INTO higher_order_milestone_history (milestone_name, first_global_step, evidence_key)
        VALUES (?, ?, ?)
        ON CONFLICT(milestone_name) DO UPDATE SET
            first_global_step = MIN(higher_order_milestone_history.first_global_step, excluded.first_global_step),
            evidence_key = CASE
                WHEN excluded.first_global_step < higher_order_milestone_history.first_global_step
                THEN excluded.evidence_key ELSE higher_order_milestone_history.evidence_key END
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
        "candidate_transfer_attempt_count": 0,
        "sampled_profile_attempt_count": 0,
        "expanded_transfer_attempt_count": 0,
        "persisted_transfer_attempt_count": 0,
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
        "mean_source_evidence_support_count": None,
        "candidate_role_count_mean": None,
    }

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import MIN_SOURCE_EVIDENCE_SUPPORT, _links_by_signature
from v6.memory.compact_memory import ensure_memory_layout


FUTURE_OPTION_CLEAR_TABLES = (
    "future_option_events",
    "future_option_motifs",
    "future_option_links",
    "future_option_attention_links",
    "future_option_transfer_links",
    "future_option_motif_observations",
)


class FutureOptionDevelopmentStage(str, Enum):
    AUTO = "auto"
    SURVIVAL = "survival"
    MOVEMENT_FREEDOM = "movement_freedom"
    ENVIRONMENTAL_INFLUENCE = "environmental_influence"
    GRAPH_EXPANSION = "graph_expansion"
    ROLE_DISCOVERY = "role_discovery"
    CONCEPT_TRANSFER = "concept_transfer"


@dataclass(frozen=True)
class FutureOptionDevelopmentThresholds:
    stable_contingencies: int = 1
    transformation_families: int = 1
    carriers: int = 1
    roles: int = 1
    promoted_concepts: int = 1
    successful_transfers: int = 1


@dataclass(frozen=True)
class FutureOptionSet:
    option_set_id: str
    state_signature: str
    available_actions: tuple[int, ...]
    reachable_signatures: tuple[str, ...]
    estimated_branching_factor: int
    depth: int


def is_complete_context_key(context_key: object) -> bool:
    """Reject serialized partial contexts; they cannot support scope claims."""
    if context_key in (None, ""):
        return False
    value = str(context_key).strip().lower()
    if not value or "null" in value or "none" in value:
        return False
    return value not in {"[]", "{}"}


@dataclass(frozen=True)
class FutureOptionDelta:
    interaction_id: int
    before_option_set_id: str
    after_option_set_id: str
    added_options: tuple[str, ...]
    removed_options: tuple[str, ...]
    preserved_options: tuple[str, ...]
    delta_score: float


@dataclass(frozen=True)
class TrajectoryOutcomeEquivalence:
    outcome_signature: str
    trajectory_ids: tuple[str, ...]
    best_cost: float
    equivalent_count: int


class FutureOptionEstimator:
    def estimate_option_set(self, env_or_state: Any, *, depth: int = 1, available_actions: list[int] | tuple[int, ...] | None = None) -> FutureOptionSet:
        if hasattr(env_or_state, "tolist"):
            state_signature = json.dumps(env_or_state.tolist(), separators=(",", ":"))
        else:
            state_signature = json.dumps(env_or_state, sort_keys=True, default=str, separators=(",", ":"))
        actions = tuple(sorted(int(item) for item in (available_actions or ())))
        reachable_signatures = tuple(
            f"{state_signature}|a{action}|d{int(depth)}"
            for action in actions
        )
        option_set_id = "fos:" + sha1(
            json.dumps(
                {"state": state_signature, "actions": actions, "depth": int(depth)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        return FutureOptionSet(
            option_set_id=option_set_id,
            state_signature=state_signature,
            available_actions=actions,
            reachable_signatures=reachable_signatures,
            estimated_branching_factor=len(actions),
            depth=int(depth),
        )

    def compare(self, before: FutureOptionSet, after: FutureOptionSet, interaction_id: int) -> FutureOptionDelta:
        before_set = set(before.reachable_signatures)
        after_set = set(after.reachable_signatures)
        added = tuple(sorted(after_set - before_set))
        removed = tuple(sorted(before_set - after_set))
        preserved = tuple(sorted(before_set & after_set))
        delta_score = float(len(added) - len(removed))
        return FutureOptionDelta(
            interaction_id=int(interaction_id),
            before_option_set_id=before.option_set_id,
            after_option_set_id=after.option_set_id,
            added_options=added,
            removed_options=removed,
            preserved_options=preserved,
            delta_score=delta_score,
        )


def derive_future_option_memory(
    *,
    memory_dir: Path,
    run_dir: Path | None = None,
    max_events: int = 500_000,
    max_motifs: int = 100_000,
    development_stage: str | FutureOptionDevelopmentStage = FutureOptionDevelopmentStage.AUTO,
    development_thresholds: FutureOptionDevelopmentThresholds | None = None,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    del run_dir
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as state_conn, sqlite3.connect(paths.graph) as graph_conn:
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        for table in FUTURE_OPTION_CLEAR_TABLES:
            state_conn.execute(f"DELETE FROM {table}")
        resolved_stage = resolve_future_option_development_stage(
            state_conn,
            requested_stage=development_stage,
            thresholds=development_thresholds,
        )
        t0 = time.time()
        events = derive_future_option_events(
            state_conn,
            graph_conn,
            max_events=max_events,
            development_stage=resolved_stage,
            progress_factory=progress_factory,
        )
        events["derive_future_option_events_seconds"] = float(time.time() - t0)
        t0 = time.time()
        motifs = derive_future_option_motifs(
            state_conn,
            graph_conn,
            max_motifs=max_motifs,
            development_stage=resolved_stage,
            progress_factory=progress_factory,
        )
        motifs["derive_future_option_motifs_seconds"] = float(time.time() - t0)
        attention = derive_future_option_attention_links(state_conn)
        transfer = derive_future_option_transfer_links(state_conn)
        summary = {
            "future_option_stage": resolved_stage.value,
            **events,
            **motifs,
            **attention,
            **transfer,
        }
        state_conn.execute(
            """
            INSERT INTO memory_summary (key, value_json)
            VALUES ('future_option_derivation_summary', ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (json.dumps(summary, sort_keys=True),),
        )
        state_conn.commit()
        graph_conn.commit()
        return summary


def derive_future_option_events(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_events: int,
    development_stage: FutureOptionDevelopmentStage = FutureOptionDevelopmentStage.SURVIVAL,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    inserted = 0
    first_event_step: int | None = None
    stable_contingency_rows_seen = 0
    stable_contingency_events_inserted = 0
    transformation_family_rows_seen = 0
    transformation_family_events_inserted = 0
    carrier_rows_seen = 0
    carrier_events_inserted = 0
    role_rows_seen = 0
    role_events_inserted = 0

    def add_event(payload: dict[str, Any]) -> None:
        nonlocal inserted, first_event_step
        if inserted >= int(max_events):
            return
        state_conn.execute(
            """
            INSERT INTO future_option_events (
                event_id, owner_type, owner_key, game, sampler, context_key, action_key, source_kind,
                motif_type, option_delta, option_delta_bucket, option_count_before, option_count_after,
                novelty_score, reversibility_score, branching_score, termination_score, contradiction_score,
                replay_priority_score, memory_priority_score, first_seen_global_step, last_seen_global_step,
                source_interaction_id, source_family_id, source_carrier_id, source_role_id, source_concept_id,
                source_context_signature, source_action, source_game_id, source_sampler,
                future_option_development_stage, survival_delta, movement_freedom_delta,
                environmental_influence_delta, graph_expansion_delta, role_discovery_delta,
                concept_transfer_delta, developmental_option_value, motif_classification_reason,
                classification_source, classification_rule, classification_evidence_id, evidence_json
                , classification_type, classification_provenance_status,
                source_game_key, target_game_key, source_context_key, target_context_key,
                target_interaction_id, source_game_is_surrogate, target_game_is_surrogate,
                source_context_is_surrogate, target_context_is_surrogate,
                context_resolution_source, context_is_surrogate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["event_id"],
                payload["owner_type"],
                payload["owner_key"],
                payload["game"],
                payload["sampler"],
                payload["context_key"],
                payload["action_key"],
                payload["source_kind"],
                payload["motif_type"],
                payload["option_delta"],
                payload["option_delta_bucket"],
                payload["option_count_before"],
                payload["option_count_after"],
                payload["novelty_score"],
                payload["reversibility_score"],
                payload["branching_score"],
                payload["termination_score"],
                payload["contradiction_score"],
                payload["replay_priority_score"],
                payload["memory_priority_score"],
                payload["first_seen_global_step"],
                payload["last_seen_global_step"],
                payload["source_interaction_id"],
                payload["source_family_id"],
                payload["source_carrier_id"],
                payload["source_role_id"],
                payload["source_concept_id"],
                payload["source_context_signature"],
                payload["source_action"],
                payload["source_game_id"],
                payload["source_sampler"],
                payload["future_option_development_stage"],
                payload["survival_delta"],
                payload["movement_freedom_delta"],
                payload["environmental_influence_delta"],
                payload["graph_expansion_delta"],
                payload["role_discovery_delta"],
                payload["concept_transfer_delta"],
                payload["developmental_option_value"],
                payload["motif_classification_reason"],
                payload["classification_source"],
                payload["classification_rule"],
                payload["classification_evidence_id"],
                json.dumps(payload["evidence_json"], sort_keys=True),
                payload["classification_type"],
                payload["classification_provenance_status"],
                payload["source_game_key"],
                payload["target_game_key"],
                payload["source_context_key"],
                payload["target_context_key"],
                payload["target_interaction_id"],
                payload["source_game_is_surrogate"],
                payload["target_game_is_surrogate"],
                payload["source_context_is_surrogate"],
                payload["target_context_is_surrogate"],
                payload["context_resolution_source"],
                payload["context_is_surrogate"],
            ),
        )
        inserted += 1
        if payload["first_seen_global_step"] is not None:
            step = int(payload["first_seen_global_step"])
            first_event_step = step if first_event_step is None else min(first_event_step, step)

    carrier_links = _links_by_signature(state_conn, "carrier_links", "carrier_signature")
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    families_by_contingency: dict[str, set[str]] = defaultdict(set)
    for row in state_conn.execute(
        "SELECT family_signature, contingency_key FROM family_members ORDER BY family_signature ASC, contingency_key ASC"
    ).fetchall():
        families_by_contingency[str(row["contingency_key"])].add(str(row["family_signature"]))
    carriers_by_family: dict[str, set[str]] = defaultdict(set)
    roles_by_carrier: dict[str, set[str]] = defaultdict(set)
    concepts_by_role: dict[str, set[str]] = defaultdict(set)
    for carrier_signature, links in carrier_links.items():
        for family_signature in links.get("family", set()):
            carriers_by_family[family_signature].add(carrier_signature)
    for role_signature, links in role_links.items():
        for carrier_signature in links.get("carrier", set()):
            roles_by_carrier[carrier_signature].add(role_signature)
    for concept_signature, links in concept_links.items():
        for role_signature in links.get("role", set()):
            concepts_by_role[role_signature].add(concept_signature)
    family_meta = {
        str(row["canonical_signature"]): {
            "effect_type": None if row["effect_type"] is None else str(row["effect_type"]),
            "action_group": None if row["action_group"] is None else str(row["action_group"]),
            "polarity": None if row["polarity"] is None else str(row["polarity"]),
        }
        for row in state_conn.execute(
            """
            SELECT canonical_signature, effect_type, action_group, polarity
            FROM transformation_families
            ORDER BY canonical_signature ASC
            """
        ).fetchall()
    }
    live_option_by_interaction = {
        str(row["node_id"]): float(row["future_option_delta"])
        for row in state_conn.execute(
            """
            SELECT node_id, future_option_delta
            FROM memory_scores
            WHERE node_id LIKE 'M0:interaction:%'
              AND future_option_delta IS NOT NULL
            ORDER BY node_id ASC
            """
        ).fetchall()
    }
    live_delta_threshold = _future_option_live_delta_threshold(list(live_option_by_interaction.values()))
    future_edge_by_interaction: dict[str, str] = {}
    future_edge_rows = state_conn.execute(
        """
        SELECT source_node_id, target_node_id, edge_type
        FROM memory_edges
        WHERE edge_type IN (
            'expands_future_options',
            'restricts_future_options',
            'preserves_future_options'
        )
        ORDER BY source_node_id ASC, edge_type ASC
        """
    ).fetchall()
    edge_scan_tracker = progress_factory("derive_future_option_events memory_edges", len(future_edge_rows), "edge", False) if progress_factory else None
    future_edge_rows = [
        row for row in future_edge_rows
        if str(row["source_node_id"]).startswith("M0:interaction:")
    ]
    future_edge_rows = [
        row for row in future_edge_rows
        if str(row["source_node_id"]).startswith("M0:interaction:")
    ]
    future_edge_rows = [
        row for row in future_edge_rows
        if str(row["source_node_id"]).startswith("M0:interaction:")
    ]
    future_edge_rows = [
        row for row in future_edge_rows
        if str(row["source_node_id"]).startswith("M0:interaction:")
    ]
    future_edge_rows = [
        row for row in future_edge_rows
        if str(row["source_node_id"]).startswith("M0:interaction:")
    ]
    for row in future_edge_rows:
        future_edge_by_interaction.setdefault(str(row["source_node_id"]), str(row["edge_type"]))
        if edge_scan_tracker is not None:
            edge_scan_tracker.update(1)
    _close_progress_tracker(edge_scan_tracker)

    # Preserve concrete future-option graph-edge provenance as first-class
    # future-option events. These rows must not depend on a corresponding
    # memory_scores row or higher-order substrate candidate.
    for row in future_edge_rows:
        source_node_id = str(row["source_node_id"])
        target_node_id = str(row["target_node_id"])
        source_interaction_id = (
            source_node_id.rsplit(":", 1)[-1]
            if source_node_id.startswith("M0:interaction:")
            else source_node_id
        )
        target_interaction_id = (
            target_node_id.rsplit(":", 1)[-1]
            if target_node_id.startswith("M0:interaction:")
            else target_node_id
        )
        edge_type = str(row["edge_type"])
        payload = _build_future_option_event(
            owner_type="interaction",
            owner_key=source_interaction_id,
            source_kind="future_option_edge",
            game=None,
            sampler=None,
            context_key=None,
            action_key=None,
            text_fragments=[edge_type],
            support_count=1,
            polarity=None,
            first_seen=None,
            last_seen=None,
            mean_prediction_error=0.0,
            mean_replay_priority=0.0,
            stability_score=0.0,
            event_id_seed=(
                f"future_option_edge|{source_node_id}|"
                f"{target_node_id}|{edge_type}"
            ),
            evidence_json={
                "source_table": "memory_edges",
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "edge_type": edge_type,
            },
            future_option_edge_type=edge_type,
            source_interaction_ids={source_interaction_id},
            development_stage=development_stage,
        )
        payload["target_interaction_id"] = target_interaction_id
        payload["classification_source"] = "future_option_edge"
        payload["classification_rule"] = "future_option_edge"
        payload["classification_provenance_status"] = "verified"
        payload["evidence_json"]["classification_source"] = "future_option_edge"
        payload["evidence_json"]["classification_rule"] = "future_option_edge"
        payload["evidence_json"]["classification_provenance_status"] = "verified"
        add_event(payload)

    interaction_ids_by_family: dict[str, set[str]] = defaultdict(set)
    carrier_interaction_ids: dict[str, set[str]] = defaultdict(set)
    carrier_family_ids: dict[str, set[str]] = defaultdict(set)
    role_carrier_ids: dict[str, set[str]] = defaultdict(set)
    edge_rows = state_conn.execute(
        """
        SELECT source_node_id, target_node_id, edge_type
        FROM memory_edges
        ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC
        """
    ).fetchall()
    for row in edge_rows:
        source = str(row["source_node_id"])
        target = str(row["target_node_id"])
        edge_type = str(row["edge_type"])
        if source.startswith("M0:interaction:") and edge_type == "supports" and target.startswith("M2:family:"):
            interaction_ids_by_family[target.split("M2:family:", 1)[1]].add(source)
        if source.startswith("M3:carrier:") and edge_type == "carried_by" and target.startswith("M0:interaction:"):
            carrier_interaction_ids[source].add(target)
        if source.startswith("M3:carrier:") and edge_type == "associated_with_family" and target.startswith("M2:family:"):
            carrier_family_ids[source].add(target.split("M2:family:", 1)[1])
        if source.startswith("M3:carrier:") and edge_type == "plays_role" and target.startswith("M3:role:"):
            role_carrier_ids[target].add(source)

    contingency_rows = state_conn.execute(
        """
        SELECT canonical_key, game, sampler, context_level, action, effect_signature, support_count,
               first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error, mean_replay_priority
        FROM stable_contingencies
        ORDER BY canonical_key ASC
        """
    ).fetchall()
    contingency_tracker = progress_factory("derive_future_option_events stable_contingencies", len(contingency_rows), "contingency", False) if progress_factory else None
    for row in contingency_rows:
        stable_contingency_rows_seen += 1
        before_inserted = inserted
        add_event(
            _build_future_option_event(
                owner_type="contingency",
                owner_key=str(row["canonical_key"]),
                source_kind="stable_contingency",
                game=row["game"],
                sampler=row["sampler"],
                context_key=_context_key_from_canonical(str(row["canonical_key"])),
                action_key=None if row["action"] is None else str(row["action"]),
                text_fragments=[row["effect_signature"], row["canonical_key"]],
                support_count=int(row["support_count"] or 0),
                polarity=None,
                first_seen=row["first_seen_global_step"],
                last_seen=row["last_seen_global_step"],
                mean_prediction_error=float(row["mean_prediction_error"] or 0.0),
                mean_replay_priority=float(row["mean_replay_priority"] or 0.0),
                stability_score=float(row["stability_score"] or 0.0),
                event_id_seed=f"contingency|{row['canonical_key']}",
                evidence_json={
                    "source_table": "stable_contingencies",
                    "raw_key": str(row["canonical_key"]),
                    "source_family_ids": sorted(families_by_contingency.get(str(row["canonical_key"]), set())),
                    "text_tokens_used": _tokenize_text_fragments([row["effect_signature"], row["canonical_key"]]),
                    "heuristic_note": "future-option counts are heuristic unless true reachable-state columns are available",
                },
                action_group=None if row["action"] is None else str(row["action"]),
                source_family_ids=families_by_contingency.get(str(row["canonical_key"]), set()),
                development_stage=development_stage,
            )
        )
        if inserted > before_inserted:
            stable_contingency_events_inserted += 1
        if contingency_tracker is not None:
            contingency_tracker.update(1)
    _close_progress_tracker(contingency_tracker)
    family_rows = state_conn.execute(
        """
        SELECT canonical_signature, effect_type, action_group, polarity, support_count, member_count,
               first_seen_global_step, last_seen_global_step, stability_score
        FROM transformation_families
        ORDER BY canonical_signature ASC
        """
    ).fetchall()
    family_tracker = progress_factory("derive_future_option_events transformation_families", len(family_rows), "family", False) if progress_factory else None
    for row in family_rows:
        transformation_family_rows_seen += 1
        family_signature = str(row["canonical_signature"])
        before_inserted = inserted
        add_event(
            _build_future_option_event(
                owner_type="family",
                owner_key=family_signature,
                source_kind="transformation_family",
                game=None,
                sampler=None,
                context_key=None,
                action_key=None if row["action_group"] is None else str(row["action_group"]),
                text_fragments=[row["canonical_signature"], row["effect_type"], row["action_group"], row["polarity"]],
                support_count=int(row["support_count"] or 0),
                polarity=row["polarity"],
                first_seen=row["first_seen_global_step"],
                last_seen=row["last_seen_global_step"],
                mean_prediction_error=0.0,
                mean_replay_priority=float(row["stability_score"] or 0.0),
                stability_score=float(row["stability_score"] or 0.0),
                event_id_seed=f"family|{row['canonical_signature']}",
                evidence_json={
                    "source_table": "transformation_families",
                    "raw_key": str(row["canonical_signature"]),
                    "source_interaction_ids": _interaction_ids_from_nodes(interaction_ids_by_family.get(family_signature, set())),
                    "source_carrier_ids": sorted(carriers_by_family.get(family_signature, set())),
                    "text_tokens_used": _tokenize_text_fragments(
                        [row["canonical_signature"], row["effect_type"], row["action_group"], row["polarity"]]
                    ),
                    "heuristic_note": "future-option counts are heuristic unless true reachable-state columns are available",
                },
                effect_type=None if row["effect_type"] is None else str(row["effect_type"]),
                action_group=None if row["action_group"] is None else str(row["action_group"]),
                live_option_delta=_mean_live_delta_for_interactions(
                    interaction_ids_by_family.get(family_signature, set()),
                    live_option_by_interaction,
                ),
                future_option_edge_type=_majority_edge_type_for_interactions(
                    interaction_ids_by_family.get(family_signature, set()),
                    future_edge_by_interaction,
                ),
                live_delta_threshold=live_delta_threshold,
                source_interaction_ids=_interaction_ids_from_nodes(interaction_ids_by_family.get(family_signature, set())),
                source_family_ids={family_signature},
                source_carrier_ids=carriers_by_family.get(family_signature, set()),
                development_stage=development_stage,
            )
        )
        if inserted > before_inserted:
            transformation_family_events_inserted += 1
        if family_tracker is not None:
            family_tracker.update(1)
    _close_progress_tracker(family_tracker)
    carrier_rows = state_conn.execute(
        """
        SELECT carrier_signature, carrier_source, support_count, linked_family_count,
               first_seen_global_step, last_seen_global_step, stability_score, is_emergent
        FROM carrier_candidates
        ORDER BY carrier_signature ASC
        """
    ).fetchall()
    carrier_tracker = progress_factory("derive_future_option_events carrier_candidates", len(carrier_rows), "carrier", False) if progress_factory else None
    for row in carrier_rows:
        carrier_rows_seen += 1
        links = carrier_links.get(str(row["carrier_signature"]), {})
        family_text = sorted(links.get("family", set()))
        contexts = sorted(links.get("context", set()))
        carrier_node = f"M3:carrier:{sha1(str(row['carrier_signature']).encode('utf-8')).hexdigest()[:20]}"
        carrier_family_meta = _majority_family_meta(
            linked_families=family_text or sorted(carrier_family_ids.get(carrier_node, set())),
            family_meta=family_meta,
        )
        carrier_interactions = carrier_interaction_ids.get(carrier_node, set())
        carrier_roles = roles_by_carrier.get(str(row["carrier_signature"]), set())
        carrier_concepts = set().union(*(concepts_by_role.get(role, set()) for role in carrier_roles)) if carrier_roles else set()
        before_inserted = inserted
        add_event(
            _build_future_option_event(
                owner_type="carrier",
                owner_key=str(row["carrier_signature"]),
                source_kind="carrier_candidate",
                game=None,
                sampler=None,
                context_key=contexts[0] if contexts else None,
                action_key=None,
                text_fragments=[row["carrier_source"], *family_text],
                support_count=int(row["support_count"] or 0),
                polarity=carrier_family_meta.get("polarity"),
                first_seen=row["first_seen_global_step"],
                last_seen=row["last_seen_global_step"],
                mean_prediction_error=0.0,
                mean_replay_priority=float(row["stability_score"] or 0.0),
                stability_score=float(row["stability_score"] or 0.0),
                event_id_seed=f"carrier|{row['carrier_signature']}",
                evidence_json={
                    "source_table": "carrier_candidates",
                    "raw_key": str(row["carrier_signature"]),
                    "linked_families": family_text,
                    "linked_contexts": contexts,
                    "source_interaction_ids": _interaction_ids_from_nodes(carrier_interactions),
                    "source_role_ids": sorted(carrier_roles),
                    "source_concept_ids": sorted(carrier_concepts),
                    "text_tokens_used": _tokenize_text_fragments([row["carrier_source"], *family_text]),
                    "heuristic_note": "future-option counts are heuristic unless true reachable-state columns are available",
                },
                effect_type=carrier_family_meta.get("effect_type"),
                action_group=carrier_family_meta.get("action_group"),
                live_option_delta=_mean_live_delta_for_interactions(carrier_interactions, live_option_by_interaction),
                future_option_edge_type=_majority_edge_type_for_interactions(carrier_interactions, future_edge_by_interaction),
                live_delta_threshold=live_delta_threshold,
                source_interaction_ids=_interaction_ids_from_nodes(carrier_interactions),
                source_family_ids=family_text or carrier_family_ids.get(carrier_node, set()),
                source_carrier_ids={str(row["carrier_signature"])},
                source_role_ids=carrier_roles,
                source_concept_ids=carrier_concepts,
                development_stage=development_stage,
            )
        )
        if inserted > before_inserted:
            carrier_events_inserted += 1
        if carrier_tracker is not None:
            carrier_tracker.update(1)
    _close_progress_tracker(carrier_tracker)
    role_rows = state_conn.execute(
        """
        SELECT role_signature, role_type, support_count, first_seen_global_step, last_seen_global_step, role_stability_score
        FROM role_candidates
        ORDER BY role_signature ASC
        """
    ).fetchall()
    role_tracker = progress_factory("derive_future_option_events role_candidates", len(role_rows), "role", False) if progress_factory else None
    for row in role_rows:
        role_rows_seen += 1
        links = role_links.get(str(row["role_signature"]), {})
        families = sorted(links.get("family", set()))
        contexts = sorted(links.get("context", set()))
        games = sorted(links.get("game", set()))
        role_node = f"M3:role:{sha1(str(row['role_signature']).encode('utf-8')).hexdigest()[:20]}"
        role_family_meta = _majority_family_meta(linked_families=families, family_meta=family_meta)
        role_interactions: set[str] = set()
        for carrier_node in role_carrier_ids.get(role_node, set()):
            role_interactions.update(carrier_interaction_ids.get(carrier_node, set()))
        role_carriers = set(links.get("carrier", set()))
        role_concepts = concepts_by_role.get(str(row["role_signature"]), set())
        before_inserted = inserted
        add_event(
            _build_future_option_event(
                owner_type="role",
                owner_key=str(row["role_signature"]),
                source_kind="role_candidate",
                game=games[0] if games else None,
                sampler=None,
                context_key=contexts[0] if contexts else None,
                action_key=None,
                text_fragments=[row["role_type"], *families],
                support_count=int(row["support_count"] or 0),
                polarity=role_family_meta.get("polarity"),
                first_seen=row["first_seen_global_step"],
                last_seen=row["last_seen_global_step"],
                mean_prediction_error=0.0,
                mean_replay_priority=float(row["role_stability_score"] or 0.0),
                stability_score=float(row["role_stability_score"] or 0.0),
                event_id_seed=f"role|{row['role_signature']}",
                evidence_json={
                    "source_table": "role_candidates",
                    "raw_key": str(row["role_signature"]),
                    "linked_families": families,
                    "linked_contexts": contexts,
                    "linked_games": games,
                    "source_interaction_ids": _interaction_ids_from_nodes(role_interactions),
                    "source_carrier_ids": sorted(role_carriers),
                    "source_concept_ids": sorted(role_concepts),
                    "text_tokens_used": _tokenize_text_fragments([row["role_type"], *families]),
                    "heuristic_note": "future-option counts are heuristic unless true reachable-state columns are available",
                },
                effect_type=role_family_meta.get("effect_type"),
                action_group=role_family_meta.get("action_group"),
                live_option_delta=_mean_live_delta_for_interactions(role_interactions, live_option_by_interaction),
                future_option_edge_type=_majority_edge_type_for_interactions(role_interactions, future_edge_by_interaction),
                live_delta_threshold=live_delta_threshold,
                source_interaction_ids=_interaction_ids_from_nodes(role_interactions),
                source_family_ids=families,
                source_carrier_ids=role_carriers,
                source_role_ids={str(row["role_signature"])},
                source_concept_ids=role_concepts,
                development_stage=development_stage,
            )
        )
        if inserted > before_inserted:
            role_events_inserted += 1
        if role_tracker is not None:
            role_tracker.update(1)
    _close_progress_tracker(role_tracker)
    _write_future_milestone(state_conn, "first_future_option_event_step", first_event_step, None)
    return {
        "future_option_event_count": inserted,
        "future_option_events_inserted_total": inserted,
        "stable_contingency_rows_seen": stable_contingency_rows_seen,
        "stable_contingency_events_inserted": stable_contingency_events_inserted,
        "transformation_family_rows_seen": transformation_family_rows_seen,
        "transformation_family_events_inserted": transformation_family_events_inserted,
        "carrier_rows_seen": carrier_rows_seen,
        "carrier_events_inserted": carrier_events_inserted,
        "role_rows_seen": role_rows_seen,
        "role_events_inserted": role_events_inserted,
        "first_future_option_event_step": first_event_step,
    }


def derive_future_option_motifs(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_motifs: int,
    development_stage: FutureOptionDevelopmentStage = FutureOptionDevelopmentStage.SURVIVAL,
    progress_factory: Any | None = None,
) -> dict[str, Any]:
    carrier_links = _links_by_signature(state_conn, "carrier_links", "carrier_signature")
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    carriers_by_family: dict[str, set[str]] = defaultdict(set)
    roles_by_carrier: dict[str, set[str]] = defaultdict(set)
    concepts_by_role: dict[str, set[str]] = defaultdict(set)
    concepts_by_link: dict[tuple[str, str], set[str]] = defaultdict(set)
    for carrier_signature, links in carrier_links.items():
        for family_signature in links.get("family", set()):
            carriers_by_family[family_signature].add(carrier_signature)
    for role_signature, links in role_links.items():
        for carrier_signature in links.get("carrier", set()):
            roles_by_carrier[carrier_signature].add(role_signature)
    for concept_signature, link_groups in concept_links.items():
        for linked_type in ("role", "family", "carrier"):
            for linked_key in link_groups.get(linked_type, set()):
                concepts_by_link[(linked_type, linked_key)].add(concept_signature)
        for role_signature in link_groups.get("role", set()):
            concepts_by_role[role_signature].add(concept_signature)
    _augment_provenance_from_graph(
        graph_conn,
        carriers_by_family=carriers_by_family,
        roles_by_carrier=roles_by_carrier,
    )
    rows = [dict(row) for row in state_conn.execute("SELECT * FROM future_option_events ORDER BY event_id ASC").fetchall()]
    events_with_owner_type_role = 0
    role_linked_event_count = 0
    motifs_with_role_links = 0
    emergent_motifs_with_role_links = 0
    failure_occurrence_count = 0
    unresolved_family_occurrence_count = 0
    unresolved_carrier_occurrence_count = 0
    unresolved_role_occurrence_count = 0
    unique_unresolved_families: set[str] = set()
    unique_unresolved_carriers: set[str] = set()
    unique_unresolved_roles: set[str] = set()
    structural_classification_counts: Counter[str] = Counter()
    unknown_reason_counts: Counter[str] = Counter()
    groups: dict[str, dict[str, Any]] = {}
    row_tracker = progress_factory("derive_future_option_motifs events", len(rows), "event", False) if progress_factory else None
    for row in rows:
        if str(row.get("owner_type") or "") == "role":
            events_with_owner_type_role += 1
        evidence = _load_jsonish(row.get("evidence_json"))
        contexts = set(_coerce_list(evidence.get("linked_contexts")))
        if row.get("context_key") not in (None, ""):
            contexts.add(str(row["context_key"]))
        games = set(_coerce_list(evidence.get("linked_games")))
        if row.get("game") not in (None, ""):
            games.add(str(row["game"]))
        families = set(_coerce_list(evidence.get("linked_families")))
        carriers: set[str] = set()
        roles: set[str] = set()
        source_interactions = _event_provenance_values(row, evidence, "interaction")
        source_families = _event_provenance_values(row, evidence, "family")
        source_carriers = _event_provenance_values(row, evidence, "carrier")
        source_roles = _event_provenance_values(row, evidence, "role")
        source_concepts = _event_provenance_values(row, evidence, "concept")
        if str(row["owner_type"]) == "carrier":
            carriers.add(str(row["owner_key"]))
            source_carriers.add(str(row["owner_key"]))
            families.update(carrier_links.get(str(row["owner_key"]), {}).get("family", set()))
            contexts.update(carrier_links.get(str(row["owner_key"]), {}).get("context", set()))
        if str(row["owner_type"]) == "role":
            roles.add(str(row["owner_key"]))
            source_roles.add(str(row["owner_key"]))
            role_link_map = role_links.get(str(row["owner_key"]), {})
            families.update(role_link_map.get("family", set()))
            contexts.update(role_link_map.get("context", set()))
            games.update(role_link_map.get("game", set()))
        if str(row["owner_type"]) == "family":
            families.add(str(row["owner_key"]))
            source_families.add(str(row["owner_key"]))
        families.update(source_families)
        carriers.update(source_carriers)
        roles.update(source_roles)
        concepts: set[str] = set()
        for role_signature in roles:
            concepts.update(concepts_by_link.get(("role", role_signature), set()))
        for family_signature in families:
            concepts.update(concepts_by_link.get(("family", family_signature), set()))
        for carrier_signature in carriers:
            concepts.update(concepts_by_link.get(("carrier", carrier_signature), set()))
        concepts.update(source_concepts)
        if roles:
            role_linked_event_count += 1
        signature_tokens = sorted(
            {
                f"motif_type:{row['motif_type']}",
                f"option_delta_bucket:{row['option_delta_bucket']}",
                f"source_kind:{row['source_kind']}",
                f"owner_type:{row['owner_type']}",
                f"context_bucket:{_bucket(len(contexts))}",
                f"game_bucket:{_bucket(len(games))}",
                f"action_bucket:{_bucket(0 if row.get('action_key') in (None, '') else 1)}",
            }
        )
        motif_signature = "fom:" + sha1(json.dumps(signature_tokens, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()[:20]
        group = groups.setdefault(
            motif_signature,
            {
                "motif_signature": motif_signature,
                "motif_type": str(row["motif_type"]),
                "events": [],
                "families": set(),
                "carriers": set(),
                "roles": set(),
                "concepts": set(),
                "source_interactions": set(),
                "source_families": set(),
                "source_carriers": set(),
                "source_roles": set(),
                "source_concepts": set(),
                "development_components": [],
                "classification_reasons": Counter(),
                "classification_sources": Counter(),
                "contexts": set(),
                "games": set(),
                "support_count": 0,
                "first_seen": None,
                "last_seen": None,
            },
        )
        group["events"].append(row)
        group["families"].update(families)
        group["carriers"].update(carriers)
        group["roles"].update(roles)
        group["concepts"].update(concepts)
        group["source_interactions"].update(source_interactions)
        group["source_families"].update(source_families)
        group["source_carriers"].update(source_carriers)
        group["source_roles"].update(source_roles)
        group["source_concepts"].update(source_concepts)
        group["development_components"].append(_development_components_from_row(row))
        event_classification_reason = str(row.get("motif_classification_reason") or "unknown")
        group["classification_reasons"][event_classification_reason] += 1
        event_classification_source = str(row.get("classification_source") or evidence.get("classification_source") or "unknown")
        group["classification_sources"][event_classification_source] += 1
        if event_classification_reason.startswith("structural"):
            structural_classification_counts[event_classification_reason] += 1
        group["contexts"].update(contexts)
        group["games"].update(games)
        support_hint = int(evidence.get("support_count", 1) or 1)
        group["support_count"] += support_hint
        if row["first_seen_global_step"] is not None:
            step = int(row["first_seen_global_step"])
            group["first_seen"] = step if group["first_seen"] is None else min(group["first_seen"], step)
        if row["last_seen_global_step"] is not None:
            step = int(row["last_seen_global_step"])
            group["last_seen"] = step if group["last_seen"] is None else max(group["last_seen"], step)
        if row_tracker is not None:
            row_tracker.update(1)
    _close_progress_tracker(row_tracker)
    selected = sorted(groups)[: int(max_motifs)]
    emergent_count = 0
    first_emergent_step: int | None = None
    motif_type_counts: Counter[str] = Counter()
    motif_type_source_counts: Counter[str] = Counter()
    unknown_motif_event_count = 0
    total_future_option_event_count = len(rows)
    motif_tracker = progress_factory("derive_future_option_motifs motifs", len(selected), "motif", False) if progress_factory else None
    for motif_signature in selected:
        group = groups[motif_signature]
        resolution = _resolve_motif_provenance(
            group,
            carriers_by_family=carriers_by_family,
            roles_by_carrier=roles_by_carrier,
            concepts_by_role=concepts_by_role,
            concepts_by_link=concepts_by_link,
        )
        unresolved_families = set(resolution["unresolved_families"])
        unresolved_carriers = set(resolution["unresolved_carriers"])
        unresolved_roles = set(resolution["unresolved_roles"])
        unresolved_family_occurrence_count += len(unresolved_families)
        unresolved_carrier_occurrence_count += len(unresolved_carriers)
        unresolved_role_occurrence_count += len(unresolved_roles)
        failure_occurrence_count += len(unresolved_families) + len(unresolved_carriers) + len(unresolved_roles)
        unique_unresolved_families.update(unresolved_families)
        unique_unresolved_carriers.update(unresolved_carriers)
        unique_unresolved_roles.update(unresolved_roles)
        events = group["events"]
        linked_event_count = len(events)
        linked_family_count = len(group["families"])
        linked_carrier_count = len(group["carriers"])
        linked_role_count = len(group["roles"])
        linked_concept_count = len(group["concepts"])
        if linked_role_count > 0:
            motifs_with_role_links += 1
        cross_context_count = len(group["contexts"])
        cross_game_count = len(group["games"])
        mean_option_delta = _mean([row.get("option_delta") for row in events])
        mean_abs_option_delta = _mean([abs(float(row.get("option_delta") or 0.0)) for row in events])
        mean_novelty_score = _mean([row.get("novelty_score") for row in events])
        mean_reversibility_score = _mean([row.get("reversibility_score") for row in events])
        mean_branching_score = _mean([row.get("branching_score") for row in events])
        mean_termination_score = _mean([row.get("termination_score") for row in events])
        mean_replay_priority_score = _mean([row.get("replay_priority_score") for row in events])
        component_means = _mean_development_components(group["development_components"])
        classification_reason = _majority_counter_value(group["classification_reasons"])
        classification_source = _majority_counter_value(group["classification_sources"])
        if str(group["motif_type"]) != "unknown" and classification_source == "unknown":
            # Legacy/unclassified rows remain observable but are not scientific
            # motif classifications.
            group["motif_type"] = "unknown"
        verified_event_observations = [
            row for row in events
            if str(row.get("classification_provenance_status") or "missing") == "verified"
            and str(row.get("classification_source") or "unknown") != "unknown"
        ]
        motif_provenance_status = (
            "verified" if verified_event_observations
            else "proxy" if events and str(group["motif_type"]) != "unknown"
            else "missing"
        )
        if str(group["motif_type"]) == "unknown":
            unknown_reason_counts[classification_reason] += linked_event_count
        for row in events:
            evidence = _load_jsonish(row.get("evidence_json"))
            motif_type_source_counts[str(evidence.get("motif_type_source") or "unknown")] += 1
            if str(row.get("motif_type") or "unknown") == "unknown":
                unknown_motif_event_count += 1
        motif_stability_score = _clamp01(
            0.25 * min(1.0, linked_event_count / 5.0)
            + 0.20 * min(1.0, cross_context_count / 3.0)
            + 0.20 * min(1.0, cross_game_count / 2.0)
            + 0.20 * min(1.0, (mean_abs_option_delta or 0.0) / 1.5)
            + 0.15 * min(1.0, int(group["support_count"]) / 20.0)
        )
        is_emergent = int(
            linked_event_count >= 3
            and str(group["motif_type"]) != "unknown"
            and motif_stability_score >= 0.50
            and (cross_context_count >= 2 or cross_game_count >= 2)
        )
        motif_type_counts[str(group["motif_type"])] += 1
        state_conn.execute(
            """
            INSERT OR REPLACE INTO future_option_motifs (
                motif_signature, motif_type, support_count, linked_event_count, linked_family_count, linked_carrier_count,
                linked_role_count, linked_concept_count, cross_context_count, cross_game_count, mean_option_delta,
                mean_abs_option_delta, mean_novelty_score, mean_reversibility_score, mean_branching_score,
                mean_termination_score, mean_replay_priority_score, first_seen_global_step, last_seen_global_step,
                motif_stability_score, is_emergent, source_interaction_ids_json, source_family_ids_json,
                source_carrier_ids_json, source_role_ids_json, source_concept_ids_json,
                future_option_development_stage, development_component_means_json, motif_classification_reason,
                classification_source, classification_rule, classification_evidence_id,
                source_game_keys_json, target_game_keys_json, source_context_keys_json, target_context_keys_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                motif_signature,
                group["motif_type"],
                int(group["support_count"]),
                linked_event_count,
                linked_family_count,
                linked_carrier_count,
                linked_role_count,
                linked_concept_count,
                cross_context_count,
                cross_game_count,
                mean_option_delta,
                mean_abs_option_delta,
                mean_novelty_score,
                mean_reversibility_score,
                mean_branching_score,
                mean_termination_score,
                mean_replay_priority_score,
                group["first_seen"],
                group["last_seen"],
                motif_stability_score,
                is_emergent,
                json.dumps(sorted(group["source_interactions"])),
                json.dumps(sorted(group["source_families"])),
                json.dumps(sorted(group["source_carriers"])),
                json.dumps(sorted(group["source_roles"])),
                json.dumps(sorted(group["source_concepts"])),
                development_stage.value,
                json.dumps(component_means, sort_keys=True),
                classification_reason,
                classification_source,
                classification_reason,
                str(events[0]["event_id"]) if events else None,
                json.dumps(sorted(group["games"])),
                json.dumps([]),
                json.dumps(sorted(group["contexts"])),
                json.dumps([]),
            ),
        )
        state_conn.execute(
            "UPDATE future_option_motifs SET provenance_status = ? WHERE motif_signature = ?",
            (motif_provenance_status, motif_signature),
        )
        if is_emergent:
            emergent_count += 1
            if linked_role_count > 0:
                emergent_motifs_with_role_links += 1
            if group["first_seen"] is not None:
                first_emergent_step = group["first_seen"] if first_emergent_step is None else min(first_emergent_step, int(group["first_seen"]))
        for row in events:
            _insert_future_link(state_conn, motif_signature, "event", str(row["event_id"]), 1, group["first_seen"], group["last_seen"])
            state_conn.execute(
                """
                INSERT OR REPLACE INTO future_option_motif_observations (
                    motif_signature, event_id, source_game_key, target_game_key,
                    source_context_key, target_context_key, source_interaction_id,
                    target_interaction_id, source_game_is_surrogate, target_game_is_surrogate,
                    source_context_is_surrogate, target_context_is_surrogate,
                    provenance_status, classification_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    motif_signature, str(row["event_id"]), row.get("source_game_key") or row.get("game"),
                    row.get("target_game_key"), row.get("source_context_key") or row.get("context_key"),
                    row.get("target_context_key"), row.get("source_interaction_id"),
                    row.get("target_interaction_id"), int(row.get("source_game_is_surrogate") or 0),
                    int(row.get("target_game_is_surrogate") or 0), int(row.get("source_context_is_surrogate") or 0),
                    int(row.get("target_context_is_surrogate") or 0),
                    str(row.get("classification_provenance_status") or "missing"),
                    str(row.get("classification_source") or "unknown"),
                ),
            )
        for family in sorted(group["families"]):
            _insert_future_link(state_conn, motif_signature, "family", family, 1, group["first_seen"], group["last_seen"])
            _upsert_future_provenance_link(state_conn, motif_signature, "motif_derived_from_family", family, group["first_seen"], group["last_seen"])
        for carrier in sorted(group["carriers"]):
            _insert_future_link(state_conn, motif_signature, "carrier", carrier, 1, group["first_seen"], group["last_seen"])
            _upsert_future_provenance_link(state_conn, motif_signature, "motif_expressed_by_carrier", carrier, group["first_seen"], group["last_seen"])
        for role in sorted(group["roles"]):
            _insert_future_link(state_conn, motif_signature, "role", role, 1, group["first_seen"], group["last_seen"])
            _upsert_future_provenance_link(state_conn, motif_signature, "motif_associated_with_role", role, group["first_seen"], group["last_seen"])
        for concept in sorted(group["concepts"]):
            _insert_future_link(state_conn, motif_signature, "concept", concept, 1, group["first_seen"], group["last_seen"])
            _upsert_future_provenance_link(state_conn, motif_signature, "motif_supports_concept", concept, group["first_seen"], group["last_seen"])
        for context in sorted(group["contexts"]):
            _insert_future_link(state_conn, motif_signature, "context", context, 1, group["first_seen"], group["last_seen"])
        for game in sorted(group["games"]):
            _insert_future_link(state_conn, motif_signature, "game", game, 1, group["first_seen"], group["last_seen"])
        if motif_tracker is not None:
            motif_tracker.update(1)
    _close_progress_tracker(motif_tracker)
    _write_future_milestone(state_conn, "first_emergent_future_option_motif_step", first_emergent_step, None)
    unknown_motif_count = int(motif_type_counts.get("unknown", 0))
    return {
        "future_option_motif_count": len(selected),
        "future_option_motifs_inserted_total": len(selected),
        "emergent_future_option_motif_count": emergent_count,
        "motif_type_counts": dict(sorted(motif_type_counts.items())),
        "motif_type_source_counts": dict(sorted(motif_type_source_counts.items())),
        "unknown_motif_count": unknown_motif_count,
        "unknown_motif_ratio": (unknown_motif_count / len(selected)) if selected else None,
        "unknown_motif_event_count": unknown_motif_event_count,
        "unknown_motif_event_ratio": (
            unknown_motif_event_count / total_future_option_event_count
            if total_future_option_event_count > 0
            else None
        ),
        "first_emergent_future_option_motif_step": first_emergent_step,
        "events_with_owner_type_role": events_with_owner_type_role,
        "role_linked_event_count": role_linked_event_count,
        "motifs_with_role_links": motifs_with_role_links,
        "emergent_motifs_with_role_links": emergent_motifs_with_role_links,
        "motifs_with_family_provenance": sum(1 for group in groups.values() if group["source_families"]),
        "motifs_with_carrier_provenance": sum(1 for group in groups.values() if group["source_carriers"]),
        "motifs_with_role_provenance": sum(1 for group in groups.values() if group["source_roles"]),
        "motifs_with_concept_provenance": sum(1 for group in groups.values() if group["source_concepts"]),
        "provenance_resolution_failures": failure_occurrence_count,
        "failure_occurrence_count": failure_occurrence_count,
        "unique_unresolved_id_count": len(unique_unresolved_families | unique_unresolved_carriers | unique_unresolved_roles),
        "unresolved_family_to_carrier_count": unresolved_family_occurrence_count,
        "unresolved_carrier_to_role_count": unresolved_carrier_occurrence_count,
        "unresolved_role_to_concept_count": unresolved_role_occurrence_count,
        "unique_unresolved_family_to_carrier_count": len(unique_unresolved_families),
        "unique_unresolved_carrier_to_role_count": len(unique_unresolved_carriers),
        "unique_unresolved_role_to_concept_count": len(unique_unresolved_roles),
        "classified_by_structural_effect_count": int(structural_classification_counts.get("structural_effect", 0)),
        "classified_by_option_delta_count": int(structural_classification_counts.get("structural_option_delta", 0)),
        "classified_by_graph_effect_count": int(structural_classification_counts.get("structural_graph_effect", 0)),
        "classified_by_role_effect_count": int(structural_classification_counts.get("structural_role_effect", 0)),
        "classified_by_concept_effect_count": int(structural_classification_counts.get("structural_concept_effect", 0)),
        "unknown_reason_counts": dict(sorted(unknown_reason_counts.items())),
    }


def _close_progress_tracker(tracker: Any | None, *, extra: dict[str, Any] | None = None) -> None:
    if tracker is None:
        return
    close = getattr(tracker, "close", None)
    if callable(close):
        close(extra=extra)


def derive_future_option_attention_links(state_conn: sqlite3.Connection) -> dict[str, Any]:
    original_row_factory = state_conn.row_factory
    state_conn.row_factory = sqlite3.Row
    event_to_motif = {
        str(row["linked_key"]): str(row["motif_signature"])
        for row in state_conn.execute(
            """
            SELECT motif_signature, linked_key
            FROM future_option_links
            WHERE linked_type = 'event'
            ORDER BY motif_signature ASC, linked_key ASC
            """
        ).fetchall()
    }
    interaction_nodes = [
        dict(row)
        for row in state_conn.execute(
            """
            SELECT node_id, attrs_json
            FROM memory_nodes
            WHERE node_type = 'InteractionMemory'
            ORDER BY node_id ASC
            """
        ).fetchall()
    ]
    score_rows = {
        str(row["node_id"]): dict(row)
        for row in state_conn.execute(
            """
            SELECT node_id, future_option_delta, replay_priority
            FROM memory_scores
            WHERE node_id LIKE 'M0:interaction:%'
            ORDER BY node_id ASC
            """
        ).fetchall()
    }
    edge_rows = [
        dict(row)
        for row in state_conn.execute(
            """
            SELECT source_node_id, target_node_id, edge_type
            FROM memory_edges
            WHERE source_node_id LIKE 'M0:interaction:%'
            ORDER BY source_node_id ASC, edge_type ASC, target_node_id ASC
            """
        ).fetchall()
    ]
    edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_rows:
        edges_by_source[str(row["source_node_id"])].append(row)
    rows: list[dict[str, Any]] = []
    live_future_option_delta_count = 0
    null_future_option_delta_count = 0
    live_high_option_change_count = 0
    live_nonzero_option_delta_count = 0
    for node in interaction_nodes:
        node_id = str(node["node_id"])
        attrs = _load_jsonish(node.get("attrs_json"))
        score_row = score_rows.get(node_id, {})
        future_option_delta = score_row.get("future_option_delta")
        outgoing = edges_by_source.get(node_id, [])
        edge_types = {str(edge["edge_type"]) for edge in outgoing}
        has_live_signal = future_option_delta is not None or bool(
            {"changes_future_options", "expands_future_options", "restricts_future_options", "preserves_future_options"} & edge_types
        )
        if not has_live_signal:
            continue
        if future_option_delta is None:
            null_future_option_delta_count += 1
        else:
            live_future_option_delta_count += 1
        contradiction_score = 1.0 if "violates_prediction" in edge_types else 0.0
        replay_priority = float(score_row.get("replay_priority") or 0.0)
        memory_priority = _clamp01(abs(float(future_option_delta or 0.0)) / 2.0)
        high_option_change = int(
            (future_option_delta is not None and abs(float(future_option_delta or 0.0)) >= 1.0)
            or "expands_future_options" in edge_types
            or "restricts_future_options" in edge_types
        )
        high_attention = int(
            replay_priority >= 0.50
            or "selected_for_replay" in edge_types
            or "violates_prediction" in edge_types
        )
        if replay_priority >= 0.50 and "violates_prediction" in edge_types:
            attention_signal_source = "replay_priority+contradiction"
        elif replay_priority >= 0.50 or "selected_for_replay" in edge_types:
            attention_signal_source = "replay_priority"
        elif "violates_prediction" in edge_types:
            attention_signal_source = "contradiction"
        else:
            attention_signal_source = "none"
        option_delta_abs = abs(float(future_option_delta or 0.0))
        if option_delta_abs > 0.0:
            live_nonzero_option_delta_count += 1
        raw_high_attention = int(
            replay_priority >= 0.50
            or "selected_for_replay" in edge_types
            or "violates_prediction" in edge_types
        )
        live_high_option_change_count += high_option_change
        rows.append(
            {
                "event_id": node_id,
                "motif_signature": None,
                "owner_type": "interaction",
                "owner_key": node_id,
                "option_delta_abs": option_delta_abs,
                "replay_priority_score": replay_priority,
                "memory_priority_score": memory_priority,
                "contradiction_score": contradiction_score,
                "high_option_change": high_option_change,
                "high_attention": raw_high_attention,
                "raw_high_attention": raw_high_attention,
                "attention_signal_source": attention_signal_source,
                "first_seen_global_step": attrs.get("global_step"),
                "last_seen_global_step": attrs.get("global_step"),
                "source_label": "live",
            }
        )
    heuristic_future_option_delta_count = 0
    h10_fallback_reason = None
    if live_future_option_delta_count <= 0:
        h10_fallback_reason = "no_live_future_option_deltas"
    elif live_nonzero_option_delta_count <= 0:
        h10_fallback_reason = "all_live_option_deltas_zero"
    elif live_high_option_change_count <= 0:
        h10_fallback_reason = "no_live_high_option_change"
    if h10_fallback_reason is not None:
        heuristic_rows = state_conn.execute("SELECT * FROM future_option_events ORDER BY event_id ASC").fetchall()
        for row in heuristic_rows:
            payload = dict(row)
            heuristic_future_option_delta_count += 1
            option_delta_abs = abs(float(payload.get("option_delta") or 0.0))
            replay_priority = float(payload.get("replay_priority_score") or 0.0)
            memory_priority = float(payload.get("memory_priority_score") or 0.0)
            contradiction_score = float(payload.get("contradiction_score") or 0.0)
            high_option_change = int(
                option_delta_abs >= 1.0 or str(payload.get("option_delta_bucket") or "") in {"large_negative", "large_positive"}
            )
            high_attention = int(replay_priority >= 0.50 or contradiction_score >= 0.50)
            if replay_priority >= 0.50 and contradiction_score >= 0.50:
                attention_signal_source = "replay_priority+contradiction"
            elif replay_priority >= 0.50:
                attention_signal_source = "replay_priority"
            elif contradiction_score >= 0.50:
                attention_signal_source = "contradiction"
            else:
                attention_signal_source = "none"
            rows.append(
                {
                    "event_id": payload["event_id"],
                    "motif_signature": event_to_motif.get(str(payload["event_id"])),
                    "owner_type": payload["owner_type"],
                    "owner_key": payload["owner_key"],
                    "option_delta_abs": option_delta_abs,
                    "replay_priority_score": replay_priority,
                    "memory_priority_score": memory_priority,
                    "contradiction_score": contradiction_score,
                    "high_option_change": high_option_change,
                    "high_attention": high_attention,
                    "raw_high_attention": high_attention,
                    "attention_signal_source": attention_signal_source,
                    "first_seen_global_step": payload["first_seen_global_step"],
                    "last_seen_global_step": payload["last_seen_global_step"],
                    "source_label": "heuristic",
                }
            )
    attention_scores = [
        max(
            float(row.get("replay_priority_score") or 0.0),
            float(row.get("contradiction_score") or 0.0),
            float(row.get("memory_priority_score") or 0.0),
        )
        for row in rows
    ]
    attention_all_equal = bool(attention_scores) and max(attention_scores) == min(attention_scores)
    percentile_80 = _percentile(attention_scores, 0.80) if attention_scores else None
    percentile_50 = _percentile(attention_scores, 0.50) if attention_scores else None
    attention_threshold_method = "p80_epoch" if attention_scores else "unavailable"
    attention_calibration_degenerate = bool(attention_scores) and (
        attention_all_equal or percentile_80 is None or float(percentile_80) <= 0.0
    )
    high_option_change_count = 0
    high_attention_count = 0
    raw_high_attention_count = 0
    calibrated_high_attention_count = 0
    high_both_count = 0
    low_attention_count = 0
    sources_seen: set[str] = set()
    for row in rows:
        option_delta_abs = abs(float(row.get("option_delta_abs") or 0.0))
        replay_priority = float(row.get("replay_priority_score") or 0.0)
        memory_priority = float(row.get("memory_priority_score") or 0.0)
        contradiction_score = float(row.get("contradiction_score") or 0.0)
        high_option_change = int(row.get("high_option_change") or 0)
        attention_score = max(replay_priority, contradiction_score, memory_priority)
        attention_score_percentile = _percentile_rank(attention_scores, attention_score) if attention_scores else None
        calibrated_high_attention = 0
        if not attention_calibration_degenerate and percentile_80 is not None and float(percentile_80) > 0.0:
            calibrated_high_attention = int(attention_score >= float(percentile_80))
        primary_high_attention = (
            calibrated_high_attention
            if not attention_calibration_degenerate
            else int(row.get("raw_high_attention") or row.get("high_attention") or 0)
        )
        attention_signal_source = str(row.get("attention_signal_source") or "none")
        sources_seen.add(str(row.get("source_label") or "none"))
        state_conn.execute(
            """
            INSERT INTO future_option_attention_links (
                event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score,
                memory_priority_score, contradiction_score, high_option_change, high_attention,
                raw_high_attention, calibrated_high_attention, source_label,
                attention_signal_source, attention_score, attention_score_percentile, attention_threshold_method,
                attention_calibration_degenerate, first_seen_global_step, last_seen_global_step
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["event_id"],
                row.get("motif_signature"),
                row["owner_type"],
                row["owner_key"],
                option_delta_abs,
                replay_priority,
                memory_priority,
                contradiction_score,
                high_option_change,
                primary_high_attention,
                int(row.get("raw_high_attention") or row.get("high_attention") or 0),
                calibrated_high_attention,
                row.get("source_label"),
                attention_signal_source,
                attention_score,
                attention_score_percentile,
                attention_threshold_method,
                int(bool(attention_calibration_degenerate)),
                row["first_seen_global_step"],
                row["last_seen_global_step"],
            ),
        )
        high_option_change_count += high_option_change
        raw_high_attention = int(row.get("raw_high_attention") or row.get("high_attention") or 0)
        raw_high_attention_count += raw_high_attention
        calibrated_high_attention_count += int(calibrated_high_attention or 0)
        high_attention_count += int(primary_high_attention or 0)
        if high_option_change and int(primary_high_attention or 0):
            high_both_count += 1
        if not high_option_change and int(primary_high_attention or 0):
            low_attention_count += 1
    low_option_change_count = max(0, len(rows) - high_option_change_count)
    high_rate = (high_both_count / high_option_change_count) if high_option_change_count else None
    low_rate = (low_attention_count / low_option_change_count) if low_option_change_count else None
    lift_unbounded = False
    lift = None
    if high_rate is not None:
        if low_option_change_count <= 0:
            lift = None
        elif (low_rate or 0.0) <= 0.0:
            if (high_rate or 0.0) > 0.0:
                lift_unbounded = True
        else:
            lift = high_rate / low_rate
    if not rows:
        high_option_change_source = "none"
    elif sources_seen == {"live"}:
        high_option_change_source = "live"
    elif sources_seen == {"heuristic"}:
        high_option_change_source = "heuristic"
    else:
        high_option_change_source = "mixed"
    try:
        return {
            "future_option_attention_link_count": len(rows),
            "live_future_option_delta_count": live_future_option_delta_count,
            "heuristic_future_option_delta_count": heuristic_future_option_delta_count,
            "null_future_option_delta_count": null_future_option_delta_count,
            "high_option_change_count": high_option_change_count,
            "high_option_change_source": high_option_change_source,
            "high_attention_count": high_attention_count,
            "high_option_change_attention_count": high_both_count,
            "low_option_change_attention_count": low_attention_count,
            "option_attention_lift": lift,
            "option_attention_lift_unbounded": lift_unbounded,
            "attention_threshold_method": attention_threshold_method,
            "attention_calibration_degenerate": attention_calibration_degenerate,
            "attention_threshold_p80": percentile_80,
            "attention_threshold_p50": percentile_50,
            "raw_high_attention_count": raw_high_attention_count,
            "calibrated_high_attention_count": calibrated_high_attention_count,
            "attention_primary_signal": "raw_fallback" if attention_calibration_degenerate else "calibrated",
            "h10_live_rows_used": sum(1 for row in rows if str(row.get("source_label")) == "live"),
            "h10_heuristic_rows_used": sum(1 for row in rows if str(row.get("source_label")) == "heuristic"),
            "h10_fallback_reason": h10_fallback_reason,
        }
    finally:
        state_conn.row_factory = original_row_factory


def _resolve_motif_transfer_provenance(
    state_conn: sqlite3.Connection,
    *,
    motif_signature: str,
    motif_links: dict[str, set[str]],
    direct_interaction_ids: set[str],
) -> dict[str, Any]:
    """Resolve explicit motif substrate links back to concrete interactions."""
    families = set(motif_links.get("family", set())) | set(motif_links.get("motif_derived_from_family", set()))
    carriers = set(motif_links.get("carrier", set())) | set(motif_links.get("motif_expressed_by_carrier", set()))
    roles = set(motif_links.get("role", set())) | set(motif_links.get("motif_associated_with_role", set()))
    concepts = set(motif_links.get("concept", set())) | set(motif_links.get("motif_supports_concept", set()))
    initial_families = set(families)
    initial_carriers = set(carriers)
    initial_roles = set(roles)
    initial_concepts = set(concepts)
    interaction_ids = set(direct_interaction_ids) | set(motif_links.get("interaction", set()))
    interaction_ids.update(motif_links.get("source_interaction", set()))
    if interaction_ids:
        return {
            "status": "verified", "resolution_path": "motif_to_interaction",
            "interaction_ids": interaction_ids, "family_ids": families,
            "carrier_ids": carriers, "role_ids": roles, "concept_ids": concepts,
        }

    event_ids = set(motif_links.get("event", set())) | set(motif_links.get("future_option_event", set()))
    if event_ids:
        placeholders = ", ".join("?" for _ in event_ids)
        try:
            event_rows = state_conn.execute(
                "SELECT source_interaction_id FROM future_option_events "
                f"WHERE event_id IN ({placeholders}) AND source_interaction_id IS NOT NULL",
                tuple(sorted(event_ids)),
            ).fetchall()
        except sqlite3.Error:
            event_rows = []
        interaction_ids.update(
            str(row[0]) for row in event_rows if row[0] not in (None, "")
        )
    if interaction_ids:
        return {
            "status": "verified", "resolution_path": "motif_to_interaction",
            "interaction_ids": interaction_ids, "family_ids": families,
            "carrier_ids": carriers, "role_ids": roles, "concept_ids": concepts,
        }

    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    carrier_links = _links_by_signature(state_conn, "carrier_links", "carrier_signature")
    for concept in sorted(concepts):
        roles.update(concept_links.get(concept, {}).get("role", set()))
    for role in sorted(roles):
        role_link_values = role_links.get(role, {})
        carriers.update(role_link_values.get("carrier", set()))
        families.update(role_link_values.get("family", set()))
    for carrier in sorted(carriers):
        families.update(carrier_links.get(carrier, {}).get("family", set()))

    def _event_interactions(column: str, values: set[str]) -> set[str]:
        if not values:
            return set()
        placeholders = ", ".join("?" for _ in values)
        try:
            rows = state_conn.execute(
                f"SELECT source_interaction_id FROM future_option_events "
                f"WHERE {column} IN ({placeholders}) AND source_interaction_id IS NOT NULL",
                tuple(sorted(values)),
            ).fetchall()
        except sqlite3.Error:
            return set()
        return {str(row[0]) for row in rows if row[0] not in (None, "")}

    family_interactions = _event_interactions("source_family_id", families)
    carrier_interactions = _event_interactions("source_carrier_id", carriers)
    role_interactions = _event_interactions("source_role_id", roles)
    concept_interactions = _event_interactions("source_concept_id", concepts)
    if family_interactions:
        interaction_ids.update(family_interactions)
        if initial_concepts:
            path = "motif_to_concept_to_role_to_carrier_to_family_to_interaction"
        elif initial_roles:
            path = "motif_to_role_to_carrier_to_family_to_interaction"
        elif initial_carriers:
            path = "motif_to_carrier_to_family_to_interaction"
        else:
            path = "motif_to_family_to_interaction"
    elif carrier_interactions:
        interaction_ids.update(carrier_interactions)
        path = "motif_to_carrier_to_family_to_interaction"
    elif role_interactions:
        interaction_ids.update(role_interactions)
        path = "motif_to_role_to_carrier_to_family_to_interaction"
    elif concept_interactions:
        interaction_ids.update(concept_interactions)
        path = "motif_to_concept_to_role_to_carrier_to_family_to_interaction"
    else:
        path = "unresolved"
    return {
        "status": "verified" if interaction_ids else "proxy" if (families or carriers or roles or concepts) else "missing",
        "resolution_path": path,
        "interaction_ids": interaction_ids,
        "family_ids": families,
        "carrier_ids": carriers,
        "role_ids": roles,
        "concept_ids": concepts,
    }


def _concept_validation_records(state_conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Use durable promotion state when available, with a legacy fallback."""
    rows = state_conn.execute(
        """
        SELECT candidate.concept_signature, candidate.is_promoted,
               candidate.promotion_status, candidate.validation_status,
               candidate.promotion_score,
               persistent.currently_promoted, persistent.promotion_status AS persistent_promotion_status,
               persistent.validation_status AS persistent_validation_status
        FROM concept_candidates AS candidate
        LEFT JOIN concept_promotion_state AS persistent
          ON persistent.concept_signature = candidate.concept_signature
        ORDER BY candidate.concept_signature ASC
        """
    ).fetchall()
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = str(row["concept_signature"])
        promoted = int(
            row["is_promoted"]
            if row["is_promoted"] is not None else row["is_promoted"] or 0
        )
        promotion_status = str(
            row["persistent_promotion_status"]
            if row["persistent_promotion_status"] not in (None, "") else row["promotion_status"] or "candidate"
        )
        validation_status = str(
            row["persistent_validation_status"]
            if row["persistent_validation_status"] not in (None, "") else row["validation_status"] or ""
        )
        status = (
            "verified"
            if promoted == 1
            and promotion_status in {"promoted", "retained", "validated"}
            and validation_status not in {"failed", "demoted", "invalid"}
            else "demoted"
            if promotion_status == "demoted" or validation_status == "demoted"
            else "proxy"
        )
        records[signature] = {
            "status": status,
            "adjusted_promotion_score": float(row["promotion_score"] or 0.0),
        }
    return records


def _resolve_concepts_for_roles(
    state_conn: sqlite3.Connection,
    *,
    concept_links: dict[str, dict[str, set[str]]],
    concept_records: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return direct concepts, or the strongest concrete indirect fallback."""
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    direct: dict[str, set[str]] = defaultdict(set)
    for concept_signature, links in concept_links.items():
        for role_signature in links.get("role", set()):
            direct[role_signature].add(concept_signature)

    resolutions: dict[str, list[dict[str, Any]]] = {}
    all_roles = set(role_links) | set(direct)
    for role_signature in sorted(all_roles):
        direct_concepts = sorted(direct.get(role_signature, set()))
        if direct_concepts:
            resolutions[role_signature] = [
                {
                    "concept_signature": concept_signature,
                    "mode": "direct_role",
                    "path": "role_to_concept",
                    "shared_carrier_count": 0,
                    "shared_family_count": 0,
                    "status": concept_records.get(concept_signature, {}).get("status", "missing"),
                }
                for concept_signature in direct_concepts
            ]
            continue
        role_carriers = set(role_links.get(role_signature, {}).get("carrier", set()))
        role_families = set(role_links.get(role_signature, {}).get("family", set()))
        candidates: list[dict[str, Any]] = []
        for concept_signature, links in concept_links.items():
            shared_carriers = role_carriers & set(links.get("carrier", set()))
            shared_families = role_families & set(links.get("family", set()))
            if not shared_carriers and not shared_families:
                continue
            if shared_carriers and shared_families:
                mode, path, strength = "shared_carrier_and_family", "role_to_carrier_and_family_to_concept", 0
            elif shared_carriers:
                mode, path, strength = "shared_carrier", "role_to_carrier_to_concept", 1
            else:
                mode, path, strength = "shared_family", "role_to_family_to_concept", 2
            record = concept_records.get(concept_signature, {})
            candidates.append({
                "concept_signature": concept_signature,
                "mode": mode,
                "path": path,
                "shared_carrier_count": len(shared_carriers),
                "shared_family_count": len(shared_families),
                "status": record.get("status", "missing"),
                "_strength": strength,
                "_score": float(record.get("adjusted_promotion_score", 0.0)),
            })
        if candidates:
            candidates.sort(key=lambda item: (
                item["_strength"],
                0 if item["status"] == "verified" else 1,
                -(item["shared_carrier_count"] + item["shared_family_count"]),
                -item["_score"], item["concept_signature"],
            ))
            best = dict(candidates[0])
            best.pop("_strength", None)
            best.pop("_score", None)
            resolutions[role_signature] = [best]
        else:
            resolutions[role_signature] = [{
                "concept_signature": "__none__", "mode": "missing", "path": "unresolved",
                "shared_carrier_count": 0, "shared_family_count": 0, "status": "missing",
            }]
    return resolutions


def derive_future_option_transfer_links(state_conn: sqlite3.Connection) -> dict[str, Any]:
    # Transfer links are fully derived from the current motifs and concrete
    # transfer attempts.  Rebuild atomically so nullable scope keys cannot
    # bypass SQLite's composite-key conflict detection on a rerun.
    state_conn.execute("DELETE FROM future_option_transfer_links")
    motif_links = _links_by_signature(state_conn, "future_option_links", "motif_signature")
    motif_quality = {
        str(row["motif_signature"]): {
            "is_emergent": int(row["is_emergent"] or 0),
            "support_count": int(row["support_count"] or 0),
            "motif_stability_score": float(row["motif_stability_score"] or 0.0),
            "source_interaction_ids_json": row["source_interaction_ids_json"],
        }
        for row in state_conn.execute(
            "SELECT motif_signature, is_emergent, support_count, motif_stability_score, source_interaction_ids_json "
            "FROM future_option_motifs ORDER BY motif_signature ASC"
        ).fetchall()
    }
    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    concept_records = _concept_validation_records(state_conn)
    concept_resolutions_by_role = _resolve_concepts_for_roles(
        state_conn, concept_links=concept_links, concept_records=concept_records,
    )
    transfer_rows = [dict(row) for row in state_conn.execute(
        """
        SELECT source_role_signature AS role_signature, transfer_score, best_margin, reuse_success, similarity_score,
               source_evidence_support_count, candidate_role_count, source_game_key, target_game_key,
               source_context_key, target_context_key, source_interaction_id, target_interaction_id,
               source_game_is_surrogate, target_game_is_surrogate,
               source_context_is_surrogate, target_context_is_surrogate,
               source_game_resolution_source, target_game_resolution_source,
               source_context_resolution_source, target_context_resolution_source,
               provenance_mode, provenance_status
        FROM role_transfer_attempts
        WHERE provenance_mode = 'single_source'
        ORDER BY source_role_signature ASC, attempt_id ASC
        """
    ).fetchall()]
    transfers_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transfer_rows:
        transfers_by_role[str(row["role_signature"])].append(row)
    concept_validation_status = {
        signature: str(record["status"])
        for signature, record in concept_records.items()
    }
    promoted_concepts = {signature for signature, status in concept_validation_status.items() if status == "verified"}
    inserted = 0
    motifs_with_transfer = 0
    motifs_with_strong_transfer = 0
    motifs_with_promoted_concept = 0
    emergent_link_count = 0
    emergent_motifs_with_transfer = 0
    emergent_motifs_with_strong_transfer = 0
    emergent_motifs_with_promoted_concept = 0
    motif_success_numer = 0
    motif_success_denom = 0
    motif_strong_numer = 0
    emergent_success_numer = 0
    emergent_success_denom = 0
    emergent_strong_numer = 0
    motifs_seen_for_transfer = 0
    motifs_skipped_no_role_links = 0
    motifs_skipped_insufficient_support_or_stability = 0
    motifs_with_role_links_for_transfer = 0
    unique_roles_seen_from_motif_links: set[str] = set()
    unique_roles_with_transfer_attempts: set[str] = set()
    unique_roles_with_concepts: set[str] = set()
    motif_role_link_count = 0
    motif_role_transfer_attempt_link_count = 0
    motif_role_concept_link_count = 0
    roles_with_direct_concept_links: set[str] = set()
    roles_resolved_via_shared_carrier: set[str] = set()
    roles_resolved_via_shared_family: set[str] = set()
    roles_resolved_via_carrier_and_family: set[str] = set()
    roles_still_without_concept: set[str] = set()
    h11_links_using_direct_concept_resolution = 0
    h11_links_using_indirect_concept_resolution = 0
    indirect_verified_chain_count = 0
    indirect_proxy_chain_count = 0
    verified_concrete_transfer_link_count = 0
    verified_transfer_pairs: set[tuple[str, str, str, str]] = set()
    all_motifs_with_transfer: set[str] = set()
    verified_motifs_with_transfer: set[str] = set()
    all_motifs_with_strong: set[str] = set()
    verified_motifs_with_strong: set[str] = set()
    all_motifs_with_promoted: set[str] = set()
    verified_motifs_with_promoted: set[str] = set()
    fully_verified_emergent_chain_count = 0
    partially_verified_emergent_chain_count = 0
    unverified_emergent_chain_count = 0
    for motif_signature in sorted(motif_links):
        motifs_seen_for_transfer += 1
        quality = motif_quality.get(motif_signature, {})
        provenance_resolution = _resolve_motif_transfer_provenance(
            state_conn,
            motif_signature=motif_signature,
            motif_links=motif_links[motif_signature],
            direct_interaction_ids=set(_coerce_list_json(quality.get("source_interaction_ids_json"))),
        )
        motif_provenance_status = str(provenance_resolution["status"])
        is_emergent_motif = int(quality.get("is_emergent", 0)) == 1
        roles = sorted(
            set(motif_links[motif_signature].get("role", set()))
            | set(motif_links[motif_signature].get("motif_associated_with_role", set()))
        )
        if not roles:
            motifs_skipped_no_role_links += 1
            continue
        if (
            not is_emergent_motif
            and (
                int(quality.get("support_count", 0)) < 3
                or float(quality.get("motif_stability_score", 0.0)) < 0.50
            )
        ):
            motifs_skipped_insufficient_support_or_stability += 1
            continue
        motifs_with_role_links_for_transfer += 1
        motif_role_link_count += len(roles)
        unique_roles_seen_from_motif_links.update(roles)
        motif_had_transfer = False
        motif_had_strong = False
        motif_had_promoted = False
        for role_signature in roles:
            concept_resolutions = concept_resolutions_by_role.get(role_signature, [{
                "concept_signature": "__none__", "mode": "missing", "path": "unresolved",
                "shared_carrier_count": 0, "shared_family_count": 0, "status": "missing",
            }])
            concepts = [str(item["concept_signature"]) for item in concept_resolutions]
            role_transfer_rows = transfers_by_role.get(role_signature, [])
            if role_transfer_rows:
                unique_roles_with_transfer_attempts.add(role_signature)
                motif_role_transfer_attempt_link_count += 1
            if any(item["mode"] == "direct_role" for item in concept_resolutions):
                roles_with_direct_concept_links.add(role_signature)
                unique_roles_with_concepts.add(role_signature)
                motif_role_concept_link_count += len(concept_resolutions)
            elif concept_resolutions[0]["mode"] == "shared_carrier_and_family":
                roles_resolved_via_carrier_and_family.add(role_signature)
            elif concept_resolutions[0]["mode"] == "shared_carrier":
                roles_resolved_via_shared_carrier.add(role_signature)
            elif concept_resolutions[0]["mode"] == "shared_family":
                roles_resolved_via_shared_family.add(role_signature)
            else:
                roles_still_without_concept.add(role_signature)
            rows_by_pair: dict[tuple[str | None, str | None, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
            for transfer_row in role_transfer_rows:
                pair = (
                    transfer_row.get("source_game_key"), transfer_row.get("target_game_key"),
                    transfer_row.get("source_context_key"), transfer_row.get("target_context_key"),
                )
                rows_by_pair[pair].append(transfer_row)
            if not rows_by_pair:
                rows_by_pair[(None, None, None, None)] = []
            for resolution in concept_resolutions:
                concept_signature = str(resolution["concept_signature"])
                for provenance, pair_rows in sorted(rows_by_pair.items(), key=lambda item: tuple(str(value or "") for value in item[0])):
                    transfer_attempt_count = len(pair_rows)
                    successful_transfer_count = sum(1 for row in pair_rows if int(row["reuse_success"] or 0) == 1)
                    strong_transfer_success_count = sum(
                        1
                        for row in pair_rows
                        if int(row["reuse_success"] or 0) == 1
                        and int(row["source_evidence_support_count"] or 0) >= MIN_SOURCE_EVIDENCE_SUPPORT
                        and int(row["candidate_role_count"] or 0) >= 2
                        and float(row["similarity_score"] or 0.0) >= 0.60
                        and float(row["best_margin"] or 0.0) >= 0.10
                    )
                    promoted_concept_count = int(concept_signature != "__none__" and concept_signature in promoted_concepts)
                    if transfer_attempt_count <= 0 and promoted_concept_count <= 0:
                        continue
                    mean_transfer_score = _mean([row.get("transfer_score") for row in pair_rows])
                    mean_best_margin = _mean([row.get("best_margin") for row in pair_rows if row.get("best_margin") is not None])
                    source_game_key, target_game_key, source_context_key, target_context_key = provenance
                    representative = pair_rows[0] if pair_rows else {}
                    source_game_is_surrogate = int(representative.get("source_game_is_surrogate") or 0)
                    target_game_is_surrogate = int(representative.get("target_game_is_surrogate") or 0)
                    source_context_is_surrogate = int(representative.get("source_context_is_surrogate") or 0)
                    target_context_is_surrogate = int(representative.get("target_context_is_surrogate") or 0)
                    real_cross_game = (
                        not source_game_is_surrogate and not target_game_is_surrogate
                        and source_game_key != target_game_key
                    )
                    real_cross_context = (
                        not source_context_is_surrogate and not target_context_is_surrogate
                        and source_context_key != target_context_key
                    )
                    transfer_provenance_status = (
                        "verified" if pair_rows and all(str(row.get("provenance_status") or "") == "verified" for row in pair_rows)
                        else "resolved_with_surrogate" if any((source_game_is_surrogate, target_game_is_surrogate, source_context_is_surrogate, target_context_is_surrogate))
                        else "proxy"
                    )
                    transfer_scope = (
                        "cross_game_and_context" if real_cross_game and real_cross_context
                        else "cross_game" if real_cross_game
                        else "cross_context" if real_cross_context
                        else "surrogate_resolved" if transfer_provenance_status == "resolved_with_surrogate"
                        else "same_scope"
                    )
                    concept_status = str(resolution["status"]) if concept_signature != "__none__" else "missing"
                    first_seen = _safe_min(state_conn, motif_signature, role_signature, concept_signature, "first")
                    last_seen = _safe_min(state_conn, motif_signature, role_signature, concept_signature, "last")
                    state_conn.execute(
                        """
                        INSERT INTO future_option_transfer_links (
                            motif_signature, role_signature, concept_signature, transfer_attempt_count,
                            successful_transfer_count, strong_transfer_success_count, promoted_concept_count,
                            mean_transfer_score, mean_best_margin, source_role_signature, source_game_key,
                            target_game_key, source_context_key, target_context_key,
                            source_interaction_id, target_interaction_id,
                            source_game_is_surrogate, target_game_is_surrogate,
                            source_context_is_surrogate, target_context_is_surrogate,
                            source_game_resolution_source, target_game_resolution_source,
                            source_context_resolution_source, target_context_resolution_source,
                            transfer_scope, provenance_mode,
                            motif_provenance_status, transfer_provenance_status, concept_validation_status,
                            motif_provenance_resolution_path, motif_resolved_interaction_count,
                            motif_resolved_family_count, motif_resolved_carrier_count,
                            motif_resolved_role_count, motif_resolved_concept_count,
                            concept_resolution_mode, concept_resolution_path,
                            shared_carrier_count, shared_family_count,
                            first_seen_global_step, last_seen_global_step
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            motif_signature, role_signature, concept_signature,
                            transfer_attempt_count, successful_transfer_count, strong_transfer_success_count,
                            promoted_concept_count, mean_transfer_score, mean_best_margin, role_signature,
                            source_game_key, target_game_key, source_context_key, target_context_key,
                            representative.get("source_interaction_id"), representative.get("target_interaction_id"),
                            source_game_is_surrogate, target_game_is_surrogate,
                            source_context_is_surrogate, target_context_is_surrogate,
                            representative.get("source_game_resolution_source"), representative.get("target_game_resolution_source"),
                            representative.get("source_context_resolution_source"), representative.get("target_context_resolution_source"),
                            transfer_scope, "single_source", motif_provenance_status, transfer_provenance_status,
                            concept_status, provenance_resolution["resolution_path"],
                            len(provenance_resolution["interaction_ids"]), len(provenance_resolution["family_ids"]),
                            len(provenance_resolution["carrier_ids"]), len(provenance_resolution["role_ids"]),
                            len(provenance_resolution["concept_ids"]),
                            resolution["mode"], resolution["path"],
                            int(resolution["shared_carrier_count"]), int(resolution["shared_family_count"]),
                            first_seen, last_seen,
                        ),
                    )
                    inserted += 1
                    if resolution["mode"] == "direct_role":
                        h11_links_using_direct_concept_resolution += 1
                    elif resolution["mode"] != "missing":
                        h11_links_using_indirect_concept_resolution += 1
                    motif_had_transfer = motif_had_transfer or transfer_attempt_count > 0
                    motif_had_strong = motif_had_strong or strong_transfer_success_count > 0
                    motif_had_promoted = motif_had_promoted or promoted_concept_count > 0
                    if transfer_attempt_count > 0:
                        all_motifs_with_transfer.add(motif_signature)
                    if strong_transfer_success_count > 0:
                        all_motifs_with_strong.add(motif_signature)
                    if promoted_concept_count > 0:
                        all_motifs_with_promoted.add(motif_signature)
                    fully_verified = (
                        motif_provenance_status == "verified"
                        and transfer_provenance_status == "verified"
                        and concept_status == "verified"
                    )
                    if resolution["mode"] != "direct_role" and resolution["mode"] != "missing":
                        if fully_verified:
                            indirect_verified_chain_count += 1
                        else:
                            indirect_proxy_chain_count += 1
                    if fully_verified and transfer_attempt_count > 0:
                        verified_motifs_with_transfer.add(motif_signature)
                    if fully_verified and strong_transfer_success_count > 0:
                        verified_motifs_with_strong.add(motif_signature)
                    if fully_verified and promoted_concept_count > 0:
                        verified_motifs_with_promoted.add(motif_signature)
                    if transfer_provenance_status == "verified":
                        assert role_signature and (real_cross_game or real_cross_context)
                        verified_concrete_transfer_link_count += 1
                        verified_transfer_pairs.add(tuple(str(value) for value in provenance))
                    if is_emergent_motif:
                        emergent_link_count += 1
                        if fully_verified:
                            fully_verified_emergent_chain_count += 1
                        elif motif_provenance_status == "missing" or transfer_provenance_status == "missing" or concept_status == "missing":
                            unverified_emergent_chain_count += 1
                        else:
                            partially_verified_emergent_chain_count += 1
                        emergent_success_numer += successful_transfer_count
                        emergent_success_denom += transfer_attempt_count
                        emergent_strong_numer += strong_transfer_success_count
                    motif_success_numer += successful_transfer_count
                    motif_success_denom += transfer_attempt_count
                    motif_strong_numer += strong_transfer_success_count
        motifs_with_transfer += int(motif_had_transfer)
        motifs_with_strong_transfer += int(motif_had_strong)
        motifs_with_promoted_concept += int(motif_had_promoted)
        if is_emergent_motif:
            emergent_motifs_with_transfer += int(motif_had_transfer)
            emergent_motifs_with_strong_transfer += int(motif_had_strong)
            emergent_motifs_with_promoted_concept += int(motif_had_promoted)
    return {
        "future_option_transfer_link_count": inserted,
        "all_motifs_with_transfer_count": len(all_motifs_with_transfer),
        "verified_motifs_with_transfer_count": len(verified_motifs_with_transfer),
        "all_motifs_with_strong_transfer_count": len(all_motifs_with_strong),
        "verified_motifs_with_strong_transfer_count": len(verified_motifs_with_strong),
        "all_motifs_with_promoted_concept_count": len(all_motifs_with_promoted),
        "verified_motifs_with_promoted_concept_count": len(verified_motifs_with_promoted),
        # Deprecated aliases retain the inclusive (all-link) meaning.
        "motifs_with_transfer_count": len(all_motifs_with_transfer),
        "motifs_with_strong_transfer_count": len(all_motifs_with_strong),
        "motifs_with_promoted_concept_count": len(all_motifs_with_promoted),
        "motif_transfer_success_rate": (motif_success_numer / motif_success_denom) if motif_success_denom else None,
        "motif_strong_transfer_success_rate": (motif_strong_numer / motif_success_denom) if motif_success_denom else None,
        "promoted_concept_motif_count": motifs_with_promoted_concept,
        "emergent_future_option_transfer_link_count": emergent_link_count,
        "emergent_motif_transfer_link_count": emergent_link_count,
        "all_emergent_motif_transfer_link_count": emergent_link_count,
        "fully_verified_emergent_chain_count": fully_verified_emergent_chain_count,
        "partially_verified_emergent_chain_count": partially_verified_emergent_chain_count,
        "unverified_emergent_chain_count": unverified_emergent_chain_count,
        "verified_concrete_transfer_link_count": verified_concrete_transfer_link_count,
        "verified_transfer_pair_count": len(verified_transfer_pairs),
        "distinct_source_target_pair_count": len(verified_transfer_pairs),
        "emergent_motifs_with_strong_transfer_count": emergent_motifs_with_strong_transfer,
        "emergent_motifs_with_promoted_concept_count": emergent_motifs_with_promoted_concept,
        "emergent_motif_transfer_success_rate": (emergent_success_numer / emergent_success_denom) if emergent_success_denom else None,
        "emergent_motif_strong_transfer_success_rate": (emergent_strong_numer / emergent_success_denom) if emergent_success_denom else None,
        "motifs_seen_for_transfer": motifs_seen_for_transfer,
        "motifs_skipped_no_role_links": motifs_skipped_no_role_links,
        "motifs_skipped_insufficient_support_or_stability": motifs_skipped_insufficient_support_or_stability,
        "motifs_with_role_links_for_transfer": motifs_with_role_links_for_transfer,
        "roles_seen_from_motif_links": len(unique_roles_seen_from_motif_links),
        "roles_with_transfer_attempts": len(unique_roles_with_transfer_attempts),
        "roles_with_concepts": len(unique_roles_with_concepts),
        "unique_roles_seen_from_motif_links": len(unique_roles_seen_from_motif_links),
        "unique_roles_with_transfer_attempts": len(unique_roles_with_transfer_attempts),
        "unique_roles_with_concepts": len(unique_roles_with_concepts),
        "motif_role_link_count": motif_role_link_count,
        "motif_role_transfer_attempt_link_count": motif_role_transfer_attempt_link_count,
        "motif_role_concept_link_count": motif_role_concept_link_count,
        "roles_with_direct_concept_links": len(roles_with_direct_concept_links),
        "roles_resolved_via_shared_carrier": len(roles_resolved_via_shared_carrier),
        "roles_resolved_via_shared_family": len(roles_resolved_via_shared_family),
        "roles_resolved_via_carrier_and_family": len(roles_resolved_via_carrier_and_family),
        "roles_still_without_concept": len(roles_still_without_concept),
        "h11_links_using_direct_concept_resolution": h11_links_using_direct_concept_resolution,
        "h11_links_using_indirect_concept_resolution": h11_links_using_indirect_concept_resolution,
        "indirect_verified_chain_count": indirect_verified_chain_count,
        "indirect_proxy_chain_count": indirect_proxy_chain_count,
    }


def _build_future_option_event(
    *,
    owner_type: str,
    owner_key: str,
    source_kind: str,
    game: Any,
    sampler: Any,
    context_key: str | None,
    action_key: str | None,
    text_fragments: list[Any],
    support_count: int,
    polarity: Any,
    first_seen: Any,
    last_seen: Any,
    mean_prediction_error: float,
    mean_replay_priority: float,
    stability_score: float,
    event_id_seed: str,
    evidence_json: dict[str, Any],
    effect_type: str | None = None,
    action_group: str | None = None,
    live_option_delta: float | None = None,
    future_option_edge_type: str | None = None,
    live_delta_threshold: float | None = None,
    source_interaction_ids: set[str] | list[str] | tuple[str, ...] = (),
    source_family_ids: set[str] | list[str] | tuple[str, ...] = (),
    source_carrier_ids: set[str] | list[str] | tuple[str, ...] = (),
    source_role_ids: set[str] | list[str] | tuple[str, ...] = (),
    source_concept_ids: set[str] | list[str] | tuple[str, ...] = (),
    development_stage: FutureOptionDevelopmentStage = FutureOptionDevelopmentStage.SURVIVAL,
) -> dict[str, Any]:
    fallback_motif_type, text_tokens, fallback_motif_type_source = _detect_motif_type(
        text_fragments,
        owner_type=owner_type,
        source_kind=source_kind,
        effect_type=effect_type,
        action_group=action_group,
        polarity=polarity,
        live_option_delta=live_option_delta,
        future_option_edge_type=future_option_edge_type,
        live_delta_threshold=live_delta_threshold,
    )
    motif_type, motif_classification_reason = _classify_structural_motif(
        fallback_motif_type=fallback_motif_type,
        live_option_delta=live_option_delta,
        future_option_edge_type=future_option_edge_type,
        effect_type=effect_type,
        source_interaction_ids=source_interaction_ids,
        source_family_ids=source_family_ids,
        source_carrier_ids=source_carrier_ids,
        source_role_ids=source_role_ids,
        source_concept_ids=source_concept_ids,
    )
    classification_source = _classification_source_from_rule(motif_classification_reason)
    # Text/legacy fallbacks are useful diagnostics but are not verified motif
    # classifications.
    if classification_source == "unknown" and motif_type != "unknown":
        motif_type = "unknown"
        motif_classification_reason = "unverified_fallback"
    motif_type_source = fallback_motif_type_source
    polarity_text = str(polarity or "").lower()
    combined_text = " ".join(str(item or "") for item in text_fragments).lower()
    if live_option_delta is not None:
        option_delta = float(live_option_delta)
        live_option_delta_used = True
    else:
        option_delta = _base_option_delta(motif_type)
        live_option_delta_used = False
        if "positive" in polarity_text:
            option_delta += 0.25
        if "negative" in polarity_text:
            option_delta -= 0.25
        if "no_change" in combined_text:
            option_delta -= 0.25
        if support_count >= 20:
            option_delta *= 1.10
    option_count_before = 10.0
    option_count_after = max(0.0, option_count_before + option_delta)
    contradiction_score = _clamp01(mean_prediction_error if mean_prediction_error > 0.0 else 0.0)
    replay_priority_score = _clamp01(mean_replay_priority if mean_replay_priority > 0.0 else stability_score)
    novelty_score = _clamp01(abs(option_delta) / 2.0)
    reversibility_score = 1.0 if motif_type == "reversible" else 0.0
    branching_score = 1.0 if motif_type in {"enable", "branch"} else 0.0
    termination_score = 1.0 if motif_type == "terminate" else 0.0
    memory_priority_score = _clamp01(0.5 * novelty_score + 0.3 * contradiction_score + 0.2 * replay_priority_score)
    evidence = dict(evidence_json)
    evidence["support_count"] = int(support_count)
    evidence["text_tokens_used"] = text_tokens
    evidence["motif_type_source"] = motif_type_source
    evidence["live_option_delta_used"] = bool(live_option_delta_used)
    evidence["raw_live_option_delta"] = live_option_delta
    evidence["future_option_edge_type"] = future_option_edge_type
    provenance = {
        "source_interaction_ids": _sorted_ids(source_interaction_ids),
        "source_family_ids": _sorted_ids(source_family_ids),
        "source_carrier_ids": _sorted_ids(source_carrier_ids),
        "source_role_ids": _sorted_ids(source_role_ids),
        "source_concept_ids": _sorted_ids(source_concept_ids),
    }
    for key, values in provenance.items():
        evidence[key] = values
    source_game_key = None if game in (None, "") else str(game)
    if is_complete_context_key(context_key):
        source_context_key = str(context_key)
        context_is_surrogate = 0
        context_resolution_source = "direct_event"
    elif context_key not in (None, ""):
        source_context_key = (
            "surrogate_context:"
            + sha1(str(context_key).encode("utf-8")).hexdigest()[:20]
        )
        context_is_surrogate = 1
        context_resolution_source = "surrogate"
    else:
        source_context_key = None
        context_is_surrogate = 0
        context_resolution_source = "missing"
    concrete_provenance = bool(
        provenance["source_interaction_ids"]
        or provenance["source_family_ids"]
        or provenance["source_carrier_ids"]
        or provenance["source_role_ids"]
        or provenance["source_concept_ids"]
    )
    classification_provenance_status = (
        "verified" if classification_source != "unknown" and concrete_provenance
        else "proxy" if classification_source != "unknown"
        else "missing"
    )
    evidence["classification_type"] = motif_type
    evidence["classification_provenance_status"] = classification_provenance_status
    evidence["source_game_key"] = source_game_key
    evidence["source_context_key"] = source_context_key
    evidence["context_resolution_source"] = context_resolution_source
    evidence["context_is_surrogate"] = bool(context_is_surrogate)
    components = _compute_development_components(
        option_delta=option_delta,
        motif_type=motif_type,
        effect_type=effect_type,
        future_option_edge_type=future_option_edge_type,
        source_interaction_ids=provenance["source_interaction_ids"],
        source_family_ids=provenance["source_family_ids"],
        source_carrier_ids=provenance["source_carrier_ids"],
        source_role_ids=provenance["source_role_ids"],
        source_concept_ids=provenance["source_concept_ids"],
    )
    developmental_option_value = float(components[_development_component_name(development_stage)])
    evidence["future_option_development_stage"] = development_stage.value
    evidence["development_components"] = components
    evidence["developmental_option_value"] = developmental_option_value
    evidence["motif_classification_reason"] = motif_classification_reason
    evidence["classification_source"] = classification_source
    evidence["classification_rule"] = motif_classification_reason
    event_id = "foe:" + sha1(event_id_seed.encode("utf-8")).hexdigest()
    evidence["classification_evidence_id"] = event_id
    return {
        "event_id": event_id,
        "owner_type": owner_type,
        "owner_key": owner_key,
        "game": None if game in (None, "") else str(game),
        "sampler": None if sampler in (None, "") else str(sampler),
        "context_key": context_key,
        "action_key": action_key,
        "source_kind": source_kind,
        "motif_type": motif_type,
        "option_delta": option_delta,
        "option_delta_bucket": _option_delta_bucket(option_delta),
        "option_count_before": option_count_before,
        "option_count_after": option_count_after,
        "novelty_score": novelty_score,
        "reversibility_score": reversibility_score,
        "branching_score": branching_score,
        "termination_score": termination_score,
        "contradiction_score": contradiction_score,
        "replay_priority_score": replay_priority_score,
        "memory_priority_score": memory_priority_score,
        "first_seen_global_step": None if first_seen is None else int(first_seen),
        "last_seen_global_step": None if last_seen is None else int(last_seen),
        "source_interaction_id": _first_id(provenance["source_interaction_ids"]),
        "source_family_id": _first_id(provenance["source_family_ids"]),
        "source_carrier_id": _first_id(provenance["source_carrier_ids"]),
        "source_role_id": _first_id(provenance["source_role_ids"]),
        "source_concept_id": _first_id(provenance["source_concept_ids"]),
        "source_context_signature": context_key,
        "source_action": action_key,
        "source_game_id": None if game in (None, "") else str(game),
        "source_sampler": None if sampler in (None, "") else str(sampler),
        "future_option_development_stage": development_stage.value,
        "survival_delta": components["survival_delta"],
        "movement_freedom_delta": components["movement_freedom_delta"],
        "environmental_influence_delta": components["environmental_influence_delta"],
        "graph_expansion_delta": components["graph_expansion_delta"],
        "role_discovery_delta": components["role_discovery_delta"],
        "concept_transfer_delta": components["concept_transfer_delta"],
        "developmental_option_value": developmental_option_value,
        "motif_classification_reason": motif_classification_reason,
        "classification_source": classification_source,
        "classification_rule": motif_classification_reason,
        "classification_evidence_id": event_id,
        "classification_type": motif_type,
        "classification_provenance_status": classification_provenance_status,
        "source_game_key": source_game_key,
        "target_game_key": None,
        "source_context_key": source_context_key,
        "target_context_key": None,
        "target_interaction_id": None,
        "source_game_is_surrogate": 0,
        "target_game_is_surrogate": 0,
        "source_context_is_surrogate": context_is_surrogate,
        "target_context_is_surrogate": 0,
        "context_resolution_source": context_resolution_source,
        "context_is_surrogate": context_is_surrogate,
        "evidence_json": evidence,
    }


def _context_key_from_canonical(canonical_key: str) -> str | None:
    if "|a" not in canonical_key:
        return None
    return canonical_key.split("|a", 1)[0] or None


def _interaction_ids_from_nodes(node_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for node_id in node_ids:
        value = str(node_id)
        values.append(value.rsplit(":", 1)[-1] if value.startswith("M0:interaction:") else value)
    return _sorted_ids(values)


def _sorted_ids(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _first_id(values: list[str]) -> str | None:
    return values[0] if values else None


def _event_provenance_values(row: dict[str, Any], evidence: dict[str, Any], level: str) -> set[str]:
    column = f"source_{level}_id"
    values = set(_coerce_list(evidence.get(f"source_{level}_ids")))
    if row.get(column) not in (None, ""):
        values.add(str(row[column]))
    return {str(value) for value in values if value not in (None, "")}


def _augment_provenance_from_graph(
    graph_conn: sqlite3.Connection,
    *,
    carriers_by_family: dict[str, set[str]],
    roles_by_carrier: dict[str, set[str]],
) -> None:
    """Use canonical graph edges as an additional structural provenance source."""
    try:
        graph_nodes = {
            str(row["node_id"]): str(row["canonical_key"])
            for row in graph_conn.execute(
                "SELECT node_id, canonical_key FROM graph_nodes WHERE canonical_key IS NOT NULL"
            ).fetchall()
        }
        edge_rows = graph_conn.execute(
            "SELECT source_node_id, target_node_id, edge_type FROM graph_edges "
            "WHERE edge_type IN ('associated_with_family', 'plays_role')"
        ).fetchall()
    except sqlite3.Error:
        return
    for row in edge_rows:
        source = graph_nodes.get(str(row["source_node_id"]))
        target = graph_nodes.get(str(row["target_node_id"]))
        if source in (None, "") or target in (None, ""):
            continue
        if str(row["edge_type"]) == "associated_with_family":
            carriers_by_family[target].add(source)
        elif str(row["edge_type"]) == "plays_role":
            roles_by_carrier[source].add(target)


def _resolve_motif_provenance(
    group: dict[str, Any],
    *,
    carriers_by_family: dict[str, set[str]],
    roles_by_carrier: dict[str, set[str]],
    concepts_by_role: dict[str, set[str]],
    concepts_by_link: dict[tuple[str, str], set[str]],
) -> dict[str, set[str]]:
    """Resolve only explicit substrate links; no name-based association is used."""
    unresolved_families: set[str] = set()
    unresolved_carriers: set[str] = set()
    unresolved_roles: set[str] = set()
    families = set(group["families"]) | set(group["source_families"])
    carriers = set(group["carriers"]) | set(group["source_carriers"])
    roles = set(group["roles"]) | set(group["source_roles"])
    concepts = set(group["concepts"]) | set(group["source_concepts"])
    for family in sorted(families):
        resolved = carriers_by_family.get(family, set())
        if not resolved:
            unresolved_families.add(family)
        carriers.update(resolved)
    for carrier in sorted(carriers):
        resolved = roles_by_carrier.get(carrier, set())
        if not resolved:
            unresolved_carriers.add(carrier)
        roles.update(resolved)
        concepts.update(concepts_by_link.get(("carrier", carrier), set()))
    for role in sorted(roles):
        resolved = concepts_by_role.get(role, set())
        if not resolved:
            unresolved_roles.add(role)
        concepts.update(resolved)
    for family in families:
        concepts.update(concepts_by_link.get(("family", family), set()))
    group["families"].update(families)
    group["carriers"].update(carriers)
    group["roles"].update(roles)
    group["concepts"].update(concepts)
    group["source_families"].update(families)
    group["source_carriers"].update(carriers)
    group["source_roles"].update(roles)
    group["source_concepts"].update(concepts)
    return {
        "unresolved_families": unresolved_families,
        "unresolved_carriers": unresolved_carriers,
        "unresolved_roles": unresolved_roles,
    }


def _upsert_future_provenance_link(
    conn: sqlite3.Connection,
    motif_signature: str,
    linked_type: str,
    linked_key: str | None,
    first_seen: int | None,
    last_seen: int | None,
) -> None:
    if linked_key in (None, ""):
        return
    conn.execute(
        """
        INSERT INTO future_option_links (
            motif_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
        ) VALUES (?, ?, ?, 1, ?, ?)
        ON CONFLICT(motif_signature, linked_type, linked_key) DO UPDATE SET
            first_seen_global_step = CASE
                WHEN future_option_links.first_seen_global_step IS NULL THEN excluded.first_seen_global_step
                WHEN excluded.first_seen_global_step IS NULL THEN future_option_links.first_seen_global_step
                ELSE MIN(future_option_links.first_seen_global_step, excluded.first_seen_global_step)
            END,
            last_seen_global_step = CASE
                WHEN future_option_links.last_seen_global_step IS NULL THEN excluded.last_seen_global_step
                WHEN excluded.last_seen_global_step IS NULL THEN future_option_links.last_seen_global_step
                ELSE MAX(future_option_links.last_seen_global_step, excluded.last_seen_global_step)
            END
        """,
        (motif_signature, linked_type, linked_key, first_seen, last_seen),
    )


def resolve_future_option_development_stage(
    state_conn: sqlite3.Connection,
    *,
    requested_stage: str | FutureOptionDevelopmentStage,
    thresholds: FutureOptionDevelopmentThresholds | None = None,
) -> FutureOptionDevelopmentStage:
    requested = (
        requested_stage
        if isinstance(requested_stage, FutureOptionDevelopmentStage)
        else FutureOptionDevelopmentStage(str(requested_stage).strip().lower())
    )
    if requested is not FutureOptionDevelopmentStage.AUTO:
        return requested
    limits = thresholds or FutureOptionDevelopmentThresholds()
    counts = {
        "stable_contingencies": int(state_conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]),
        "transformation_families": int(state_conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]),
        "carriers": int(state_conn.execute("SELECT COUNT(*) FROM carrier_candidates WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0]),
        "roles": int(state_conn.execute("SELECT COUNT(*) FROM role_candidates WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0]),
        "promoted_concepts": int(state_conn.execute("SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1").fetchone()[0]),
        "successful_transfers": int(state_conn.execute("SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1").fetchone()[0]),
    }
    if counts["promoted_concepts"] >= limits.promoted_concepts and counts["successful_transfers"] >= limits.successful_transfers:
        return FutureOptionDevelopmentStage.CONCEPT_TRANSFER
    if counts["roles"] >= limits.roles:
        return FutureOptionDevelopmentStage.ROLE_DISCOVERY
    if counts["carriers"] >= limits.carriers:
        return FutureOptionDevelopmentStage.GRAPH_EXPANSION
    if counts["transformation_families"] >= limits.transformation_families:
        return FutureOptionDevelopmentStage.ENVIRONMENTAL_INFLUENCE
    if counts["stable_contingencies"] >= limits.stable_contingencies:
        return FutureOptionDevelopmentStage.MOVEMENT_FREEDOM
    return FutureOptionDevelopmentStage.SURVIVAL


def _classify_structural_motif(
    *,
    fallback_motif_type: str,
    live_option_delta: float | None,
    future_option_edge_type: str | None,
    effect_type: str | None,
    source_interaction_ids: set[str] | list[str] | tuple[str, ...],
    source_family_ids: set[str] | list[str] | tuple[str, ...],
    source_carrier_ids: set[str] | list[str] | tuple[str, ...],
    source_role_ids: set[str] | list[str] | tuple[str, ...],
    source_concept_ids: set[str] | list[str] | tuple[str, ...],
) -> tuple[str, str]:
    effect = str(effect_type or "").lower()
    edge = str(future_option_edge_type or "")
    if "terminal_transition" in effect:
        return "terminate", "structural_effect"
    if edge == "expands_future_options":
        return "enable", "structural_option_delta"
    if edge == "restricts_future_options":
        return "block", "structural_option_delta"
    if live_option_delta is not None:
        if float(live_option_delta) > 0.0:
            return "enable", "structural_option_delta"
        if float(live_option_delta) < 0.0:
            return "block", "structural_option_delta"
        return "neutral", "structural_option_delta"
    if effect in {"positive_change", "mixed_change"}:
        return "transform", "structural_effect"
    if effect in {"preserve", "no_change"}:
        return "neutral", "structural_effect"
    if source_concept_ids:
        return "transform", "structural_concept_effect"
    if source_role_ids:
        return "transform", "structural_role_effect"
    if source_interaction_ids or source_family_ids or source_carrier_ids:
        return "transform", "structural_graph_effect"
    return fallback_motif_type, "fallback_" + ("unknown" if fallback_motif_type == "unknown" else "text_or_legacy")


def _classification_source_from_rule(rule: str) -> str:
    mapping = {
        "structural_effect": "structural_effect",
        "structural_option_delta": "option_delta",
        "structural_graph_effect": "graph_effect",
        "structural_role_effect": "role_effect",
        "structural_concept_effect": "concept_effect",
    }
    return mapping.get(str(rule), "unknown")


def _compute_development_components(
    *,
    option_delta: float,
    motif_type: str,
    effect_type: str | None,
    future_option_edge_type: str | None,
    source_interaction_ids: list[str],
    source_family_ids: list[str],
    source_carrier_ids: list[str],
    source_role_ids: list[str],
    source_concept_ids: list[str],
) -> dict[str, float]:
    terminal = motif_type == "terminate" or "terminal_transition" in str(effect_type or "").lower()
    movement = float(option_delta)
    graph_evidence = len(source_interaction_ids) + len(source_family_ids) + len(source_carrier_ids)
    environmental = 1.0 if str(effect_type or "").lower() in {"positive_change", "mixed_change"} else 0.0
    if terminal:
        environmental = -1.0
    elif str(future_option_edge_type or "") == "restricts_future_options":
        environmental = min(environmental, -1.0)
    return {
        "survival_delta": -1.0 if terminal else 0.0,
        "movement_freedom_delta": movement,
        "environmental_influence_delta": environmental,
        "graph_expansion_delta": min(1.0, float(graph_evidence) / 3.0),
        "role_discovery_delta": min(1.0, float(len(source_role_ids))),
        "concept_transfer_delta": min(1.0, float(len(source_concept_ids))),
    }


def _development_component_name(stage: FutureOptionDevelopmentStage) -> str:
    return {
        FutureOptionDevelopmentStage.SURVIVAL: "survival_delta",
        FutureOptionDevelopmentStage.MOVEMENT_FREEDOM: "movement_freedom_delta",
        FutureOptionDevelopmentStage.ENVIRONMENTAL_INFLUENCE: "environmental_influence_delta",
        FutureOptionDevelopmentStage.GRAPH_EXPANSION: "graph_expansion_delta",
        FutureOptionDevelopmentStage.ROLE_DISCOVERY: "role_discovery_delta",
        FutureOptionDevelopmentStage.CONCEPT_TRANSFER: "concept_transfer_delta",
    }.get(stage, "survival_delta")


def _development_components_from_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(row.get(name) or 0.0)
        for name in (
            "survival_delta",
            "movement_freedom_delta",
            "environmental_influence_delta",
            "graph_expansion_delta",
            "role_discovery_delta",
            "concept_transfer_delta",
        )
    }


def _mean_development_components(components: list[dict[str, float]]) -> dict[str, float | None]:
    names = (
        "survival_delta",
        "movement_freedom_delta",
        "environmental_influence_delta",
        "graph_expansion_delta",
        "role_discovery_delta",
        "concept_transfer_delta",
    )
    return {name: _mean([item.get(name) for item in components]) for name in names}


def _majority_counter_value(counter: Counter[str]) -> str:
    if not counter:
        return "unknown"
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _mean_live_delta_for_interactions(
    interaction_ids: set[str],
    live_option_by_interaction: dict[str, float],
) -> float | None:
    values = [float(live_option_by_interaction[interaction_id]) for interaction_id in sorted(interaction_ids) if interaction_id in live_option_by_interaction]
    return _mean(values)


def _majority_edge_type_for_interactions(
    interaction_ids: set[str],
    future_edge_by_interaction: dict[str, str],
) -> str | None:
    counts: Counter[str] = Counter(
        str(future_edge_by_interaction[interaction_id])
        for interaction_id in sorted(interaction_ids)
        if interaction_id in future_edge_by_interaction
    )
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _majority_family_meta(
    *,
    linked_families: list[str],
    family_meta: dict[str, dict[str, str | None]],
) -> dict[str, str | None]:
    result: dict[str, str | None] = {"effect_type": None, "action_group": None, "polarity": None}
    for field in ("effect_type", "action_group", "polarity"):
        counts: Counter[str] = Counter()
        for family_signature in linked_families:
            value = family_meta.get(str(family_signature), {}).get(field)
            if value not in (None, ""):
                counts[str(value)] += 1
        if counts:
            result[field] = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]
    return result


def _detect_motif_type(
    text_fragments: list[Any],
    *,
    owner_type: str,
    source_kind: str,
    effect_type: str | None = None,
    action_group: str | None = None,
    polarity: Any = None,
    live_option_delta: float | None = None,
    future_option_edge_type: str | None = None,
    live_delta_threshold: float | None = None,
) -> tuple[str, list[str], str]:
    text = " ".join(str(item or "") for item in text_fragments).lower()
    effect_text = str(effect_type or "").lower()
    action_text = str(action_group or "").lower()
    polarity_text = str(polarity or "").lower()
    if live_option_delta is not None:
        delta = float(live_option_delta)
        positive_threshold = max(0.01, float(live_delta_threshold or 0.01))
        negative_threshold = -positive_threshold
        if delta > positive_threshold:
            if future_option_edge_type == "expands_future_options" or any(token in text for token in ("enable", "open", "unlock", "allow", "access", "branch", "split")):
                return "enable", _tokenize_text_fragments(text_fragments), "live_delta_rule"
            if any(token in effect_text for token in ("preserve", "no_change")) or "stabil" in text:
                return "stabilize", _tokenize_text_fragments(text_fragments), "live_delta_rule"
            return "transform", _tokenize_text_fragments(text_fragments), "live_delta_rule"
        if delta < negative_threshold:
            return "constrain", _tokenize_text_fragments(text_fragments), "live_delta_rule"
        return "neutral", _tokenize_text_fragments(text_fragments), "live_delta_rule"
    if future_option_edge_type:
        edge_type = str(future_option_edge_type)
        if edge_type == "expands_future_options":
            return ("branch" if "branch" in action_text or source_kind == "role_candidate" else "enable"), _tokenize_text_fragments(text_fragments), "future_option_edge"
        if edge_type == "restricts_future_options":
            return ("terminate" if "terminal" in effect_text else "block"), _tokenize_text_fragments(text_fragments), "future_option_edge"
        if edge_type == "preserves_future_options":
            return "stabilize", _tokenize_text_fragments(text_fragments), "future_option_edge"
    if effect_text:
        if "positive_change" in effect_text or "mixed_change" in effect_text:
            return "transform", _tokenize_text_fragments(text_fragments), "structured_effect"
        if "preserve" in effect_text or "no_change" in effect_text:
            return "stabilize", _tokenize_text_fragments(text_fragments), "structured_effect"
        if "terminal_transition" in effect_text:
            return "terminate", _tokenize_text_fragments(text_fragments), "structured_effect"
    motif_map = {
        "enable": ("enable", "open", "unlock", "add", "create", "reveal", "allow", "access", "expose", "make_available"),
        "block": ("block", "close", "obstacle", "wall", "prevent", "forbid", "stop", "restrict", "hide"),
        "terminate": ("remove", "delete", "clear", "destroy", "consume", "vanish", "end", "eliminate"),
        "reversible": ("reverse", "restore", "toggle", "swap", "move_back", "undo", "inverse", "return"),
        "transform": ("transform", "recolor", "change", "convert", "shift", "move", "rotate", "reflect", "resize"),
        "branch": ("split", "duplicate", "copy", "fork", "branch", "multiply", "expand"),
        "merge": ("merge", "combine", "join", "collapse", "compress", "unify"),
        "stabilize": ("stable", "repeat", "preserve", "maintain", "keep", "no_change"),
    }
    for motif_type, keywords in motif_map.items():
        if any(keyword in text for keyword in keywords):
            return motif_type, _tokenize_text_fragments(text_fragments), "text_keyword"
    if "positive" in polarity_text and owner_type in {"carrier", "role"}:
        return "enable", _tokenize_text_fragments(text_fragments), "structured_effect"
    if "negative" in polarity_text and owner_type in {"carrier", "role"}:
        return "block", _tokenize_text_fragments(text_fragments), "structured_effect"
    return "unknown", _tokenize_text_fragments(text_fragments), "unknown"


def _base_option_delta(motif_type: str) -> float:
    return {
        "enable": 2.0,
        "branch": 2.0,
        "transform": 1.0,
        "reversible": 0.5,
        "stabilize": 0.0,
        "neutral": 0.0,
        "merge": -0.5,
        "constrain": -1.0,
        "block": -1.0,
        "terminate": -2.0,
        "unknown": 0.0,
    }.get(motif_type, 0.0)


def _option_delta_bucket(value: float) -> str:
    if value <= -1.5:
        return "large_negative"
    if value < -0.25:
        return "negative"
    if value <= 0.25:
        return "neutral"
    if value < 1.5:
        return "positive"
    return "large_positive"


def _tokenize_text_fragments(fragments: list[Any]) -> list[str]:
    tokens: list[str] = []
    for item in fragments:
        for token in str(item or "").lower().replace("|", " ").replace(":", " ").split():
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def _insert_future_link(
    conn: sqlite3.Connection,
    motif_signature: str,
    linked_type: str,
    linked_key: str | None,
    support_count: int,
    first_seen: int | None,
    last_seen: int | None,
) -> None:
    if linked_key in (None, ""):
        return
    conn.execute(
        """
        INSERT INTO future_option_links (
            motif_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(motif_signature, linked_type, linked_key)
        DO UPDATE SET
            support_count = COALESCE(future_option_links.support_count, 0) + excluded.support_count,
            first_seen_global_step = CASE
                WHEN future_option_links.first_seen_global_step IS NULL THEN excluded.first_seen_global_step
                WHEN excluded.first_seen_global_step IS NULL THEN future_option_links.first_seen_global_step
                ELSE MIN(future_option_links.first_seen_global_step, excluded.first_seen_global_step)
            END,
            last_seen_global_step = CASE
                WHEN future_option_links.last_seen_global_step IS NULL THEN excluded.last_seen_global_step
                WHEN excluded.last_seen_global_step IS NULL THEN future_option_links.last_seen_global_step
                ELSE MAX(future_option_links.last_seen_global_step, excluded.last_seen_global_step)
            END
        """,
        (motif_signature, linked_type, linked_key, support_count, first_seen, last_seen),
    )


def _write_future_milestone(conn: sqlite3.Connection, name: str, first_global_step: int | None, evidence_key: str | None) -> None:
    conn.execute(
        """
        INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key)
        VALUES (?, ?, ?)
        ON CONFLICT(milestone_name)
        DO UPDATE SET first_global_step = excluded.first_global_step, evidence_key = excluded.evidence_key
        """,
        (name, first_global_step, evidence_key),
    )


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _future_option_live_delta_threshold(values: list[float]) -> float:
    cooked = [abs(float(value)) for value in values if value is not None]
    if not cooked:
        return 0.01
    return max(0.01, float(_percentile(cooked, 0.60) or 0.01))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    cooked = sorted(float(value) for value in values)
    if len(cooked) == 1:
        return cooked[0]
    position = max(0.0, min(1.0, float(fraction))) * float(len(cooked) - 1)
    low = int(position)
    high = min(len(cooked) - 1, low + 1)
    if low == high:
        return cooked[low]
    weight = position - float(low)
    return cooked[low] * (1.0 - weight) + cooked[high] * weight


def _percentile_rank(values: list[float], target: float) -> float | None:
    if not values:
        return None
    cooked = sorted(float(value) for value in values)
    less_equal = sum(1 for value in cooked if value <= float(target))
    return float(less_equal) / float(len(cooked))


def _mean(values: list[Any]) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    if not cooked:
        return None
    return sum(cooked) / len(cooked)


def _coerce_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _coerce_list_json(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    return _coerce_list(parsed)


def _load_jsonish(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _safe_min(
    state_conn: sqlite3.Connection,
    motif_signature: str,
    role_signature: str,
    concept_signature: str,
    which: str,
) -> int | None:
    column = "MIN(first_seen_global_step)" if which == "first" else "MAX(last_seen_global_step)"
    concept_arg = "" if concept_signature == "__none__" else concept_signature
    rows = state_conn.execute(
        f"""
        SELECT {column}
        FROM (
            SELECT first_seen_global_step, last_seen_global_step FROM future_option_motifs WHERE motif_signature = ?
            UNION ALL
            SELECT first_seen_global_step, last_seen_global_step FROM role_candidates WHERE role_signature = ?
            UNION ALL
            SELECT first_seen_global_step, last_seen_global_step FROM concept_candidates WHERE concept_signature = ?
        )
        """,
        (motif_signature, role_signature, concept_arg),
    ).fetchone()
    if not rows or rows[0] is None:
        return None
    return int(rows[0])

# v6.3 canonical current-validation semantics
_concept_validation_records_base = _concept_validation_records

def _concept_validation_records(state_conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    from v6.v63_semantics import _current_validation_records
    previous = state_conn.row_factory
    state_conn.row_factory = sqlite3.Row
    try:
        base = _concept_validation_records_base(state_conn)
    finally:
        state_conn.row_factory = previous
    return _current_validation_records(state_conn, base)


def resolve_future_option_development_stage(
    state_conn: sqlite3.Connection,
    *,
    requested_stage: FutureOptionDevelopmentStage | str,
    thresholds: FutureOptionDevelopmentThresholds | None = None,
) -> FutureOptionDevelopmentStage:
    requested = (
        requested_stage
        if isinstance(requested_stage, FutureOptionDevelopmentStage)
        else FutureOptionDevelopmentStage(str(requested_stage).strip().lower())
    )
    if requested is not FutureOptionDevelopmentStage.AUTO:
        return requested
    limits = thresholds or FutureOptionDevelopmentThresholds()
    records = _concept_validation_records(state_conn)
    verified_concepts = sum(1 for record in records.values() if record.get("status") == "verified")
    counts = {
        "stable_contingencies": int(state_conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]),
        "transformation_families": int(state_conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]),
        "carriers": int(state_conn.execute("SELECT COUNT(*) FROM carrier_candidates WHERE COALESCE(is_emergent,0)=1").fetchone()[0]),
        "roles": int(state_conn.execute("SELECT COUNT(*) FROM role_candidates WHERE COALESCE(is_emergent,0)=1").fetchone()[0]),
        "promoted_concepts": verified_concepts,
        "successful_transfers": int(state_conn.execute("SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success,0)=1").fetchone()[0]),
    }
    if counts["promoted_concepts"] >= limits.promoted_concepts and counts["successful_transfers"] >= limits.successful_transfers:
        return FutureOptionDevelopmentStage.CONCEPT_TRANSFER
    if counts["roles"] >= limits.roles:
        return FutureOptionDevelopmentStage.ROLE_DISCOVERY
    if counts["carriers"] >= limits.carriers:
        return FutureOptionDevelopmentStage.GRAPH_EXPANSION
    if counts["transformation_families"] >= limits.transformation_families:
        return FutureOptionDevelopmentStage.ENVIRONMENTAL_INFLUENCE
    if counts["stable_contingencies"] >= limits.stable_contingencies:
        return FutureOptionDevelopmentStage.MOVEMENT_FREEDOM
    return FutureOptionDevelopmentStage.SURVIVAL

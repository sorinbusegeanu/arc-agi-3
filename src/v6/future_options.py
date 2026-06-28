from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import _links_by_signature
from v6.memory.compact_memory import ensure_memory_layout


FUTURE_OPTION_CLEAR_TABLES = (
    "future_option_events",
    "future_option_motifs",
    "future_option_links",
    "future_option_attention_links",
    "future_option_transfer_links",
)


@dataclass(frozen=True)
class FutureOptionSet:
    option_set_id: str
    state_signature: str
    available_actions: tuple[int, ...]
    reachable_signatures: tuple[str, ...]
    estimated_branching_factor: int
    depth: int


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
) -> dict[str, Any]:
    del run_dir
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as state_conn, sqlite3.connect(paths.graph) as graph_conn:
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        for table in FUTURE_OPTION_CLEAR_TABLES:
            state_conn.execute(f"DELETE FROM {table}")
        events = derive_future_option_events(state_conn, graph_conn, max_events=max_events)
        motifs = derive_future_option_motifs(state_conn, graph_conn, max_motifs=max_motifs)
        attention = derive_future_option_attention_links(state_conn)
        transfer = derive_future_option_transfer_links(state_conn)
        summary = {**events, **motifs, **attention, **transfer}
        state_conn.commit()
        graph_conn.commit()
        return summary


def derive_future_option_events(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_events: int,
) -> dict[str, Any]:
    inserted = 0
    first_event_step: int | None = None

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
                replay_priority_score, memory_priority_score, first_seen_global_step, last_seen_global_step, evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(payload["evidence_json"], sort_keys=True),
            ),
        )
        inserted += 1
        if payload["first_seen_global_step"] is not None:
            step = int(payload["first_seen_global_step"])
            first_event_step = step if first_event_step is None else min(first_event_step, step)

    carrier_links = _links_by_signature(state_conn, "carrier_links", "carrier_signature")
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
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
    for row in state_conn.execute(
        """
        SELECT source_node_id, edge_type
        FROM memory_edges
        WHERE source_node_id LIKE 'M0:interaction:%'
          AND edge_type IN (
            'expands_future_options',
            'restricts_future_options',
            'preserves_future_options'
          )
        ORDER BY source_node_id ASC, edge_type ASC
        """
    ).fetchall():
        future_edge_by_interaction.setdefault(str(row["source_node_id"]), str(row["edge_type"]))
    interaction_ids_by_family: dict[str, set[str]] = defaultdict(set)
    carrier_interaction_ids: dict[str, set[str]] = defaultdict(set)
    carrier_family_ids: dict[str, set[str]] = defaultdict(set)
    role_carrier_ids: dict[str, set[str]] = defaultdict(set)
    for row in state_conn.execute(
        """
        SELECT source_node_id, target_node_id, edge_type
        FROM memory_edges
        ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC
        """
    ).fetchall():
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

    for row in state_conn.execute(
        """
        SELECT canonical_key, game, sampler, context_level, action, effect_signature, support_count,
               first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error, mean_replay_priority
        FROM stable_contingencies
        ORDER BY canonical_key ASC
        """
    ).fetchall():
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
                    "text_tokens_used": _tokenize_text_fragments([row["effect_signature"], row["canonical_key"]]),
                    "heuristic_note": "future-option counts are heuristic unless true reachable-state columns are available",
                },
                action_group=None if row["action"] is None else str(row["action"]),
            )
        )
    for row in state_conn.execute(
        """
        SELECT canonical_signature, effect_type, action_group, polarity, support_count, member_count,
               first_seen_global_step, last_seen_global_step, stability_score
        FROM transformation_families
        ORDER BY canonical_signature ASC
        """
    ).fetchall():
        family_signature = str(row["canonical_signature"])
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
            )
        )
    for row in state_conn.execute(
        """
        SELECT carrier_signature, carrier_source, support_count, linked_family_count,
               first_seen_global_step, last_seen_global_step, stability_score, is_emergent
        FROM carrier_candidates
        ORDER BY carrier_signature ASC
        """
    ).fetchall():
        links = carrier_links.get(str(row["carrier_signature"]), {})
        family_text = sorted(links.get("family", set()))
        contexts = sorted(links.get("context", set()))
        carrier_node = f"M3:carrier:{sha1(str(row['carrier_signature']).encode('utf-8')).hexdigest()[:20]}"
        carrier_family_meta = _majority_family_meta(
            linked_families=family_text or sorted(carrier_family_ids.get(carrier_node, set())),
            family_meta=family_meta,
        )
        carrier_interactions = carrier_interaction_ids.get(carrier_node, set())
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
                    "text_tokens_used": _tokenize_text_fragments([row["carrier_source"], *family_text]),
                    "heuristic_note": "future-option counts are heuristic unless true reachable-state columns are available",
                },
                effect_type=carrier_family_meta.get("effect_type"),
                action_group=carrier_family_meta.get("action_group"),
                live_option_delta=_mean_live_delta_for_interactions(carrier_interactions, live_option_by_interaction),
                future_option_edge_type=_majority_edge_type_for_interactions(carrier_interactions, future_edge_by_interaction),
                live_delta_threshold=live_delta_threshold,
            )
        )
    for row in state_conn.execute(
        """
        SELECT role_signature, role_type, support_count, first_seen_global_step, last_seen_global_step, role_stability_score
        FROM role_candidates
        ORDER BY role_signature ASC
        """
    ).fetchall():
        links = role_links.get(str(row["role_signature"]), {})
        families = sorted(links.get("family", set()))
        contexts = sorted(links.get("context", set()))
        games = sorted(links.get("game", set()))
        role_node = f"M3:role:{sha1(str(row['role_signature']).encode('utf-8')).hexdigest()[:20]}"
        role_family_meta = _majority_family_meta(linked_families=families, family_meta=family_meta)
        role_interactions: set[str] = set()
        for carrier_node in role_carrier_ids.get(role_node, set()):
            role_interactions.update(carrier_interaction_ids.get(carrier_node, set()))
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
                    "text_tokens_used": _tokenize_text_fragments([row["role_type"], *families]),
                    "heuristic_note": "future-option counts are heuristic unless true reachable-state columns are available",
                },
                effect_type=role_family_meta.get("effect_type"),
                action_group=role_family_meta.get("action_group"),
                live_option_delta=_mean_live_delta_for_interactions(role_interactions, live_option_by_interaction),
                future_option_edge_type=_majority_edge_type_for_interactions(role_interactions, future_edge_by_interaction),
                live_delta_threshold=live_delta_threshold,
            )
        )
    _write_future_milestone(state_conn, "first_future_option_event_step", first_event_step, None)
    return {"future_option_event_count": inserted, "first_future_option_event_step": first_event_step}


def derive_future_option_motifs(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    max_motifs: int,
) -> dict[str, Any]:
    del graph_conn
    carrier_links = _links_by_signature(state_conn, "carrier_links", "carrier_signature")
    role_links = _links_by_signature(state_conn, "role_links", "role_signature")
    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    concepts_by_link: dict[tuple[str, str], set[str]] = defaultdict(set)
    for concept_signature, link_groups in concept_links.items():
        for linked_type in ("role", "family", "carrier"):
            for linked_key in link_groups.get(linked_type, set()):
                concepts_by_link[(linked_type, linked_key)].add(concept_signature)
    rows = [dict(row) for row in state_conn.execute("SELECT * FROM future_option_events ORDER BY event_id ASC").fetchall()]
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
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
        if str(row["owner_type"]) == "carrier":
            carriers.add(str(row["owner_key"]))
            families.update(carrier_links.get(str(row["owner_key"]), {}).get("family", set()))
            contexts.update(carrier_links.get(str(row["owner_key"]), {}).get("context", set()))
        if str(row["owner_type"]) == "role":
            roles.add(str(row["owner_key"]))
            role_link_map = role_links.get(str(row["owner_key"]), {})
            families.update(role_link_map.get("family", set()))
            contexts.update(role_link_map.get("context", set()))
            games.update(role_link_map.get("game", set()))
        if str(row["owner_type"]) == "family":
            families.add(str(row["owner_key"]))
        concepts: set[str] = set()
        for role_signature in roles:
            concepts.update(concepts_by_link.get(("role", role_signature), set()))
        for family_signature in families:
            concepts.update(concepts_by_link.get(("family", family_signature), set()))
        for carrier_signature in carriers:
            concepts.update(concepts_by_link.get(("carrier", carrier_signature), set()))
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
    selected = sorted(groups)[: int(max_motifs)]
    emergent_count = 0
    first_emergent_step: int | None = None
    motif_type_counts: Counter[str] = Counter()
    motif_type_source_counts: Counter[str] = Counter()
    unknown_motif_event_count = 0
    total_future_option_event_count = len(rows)
    for motif_signature in selected:
        group = groups[motif_signature]
        events = group["events"]
        linked_event_count = len(events)
        linked_family_count = len(group["families"])
        linked_carrier_count = len(group["carriers"])
        linked_role_count = len(group["roles"])
        linked_concept_count = len(group["concepts"])
        cross_context_count = len(group["contexts"])
        cross_game_count = len(group["games"])
        mean_option_delta = _mean([row.get("option_delta") for row in events])
        mean_abs_option_delta = _mean([abs(float(row.get("option_delta") or 0.0)) for row in events])
        mean_novelty_score = _mean([row.get("novelty_score") for row in events])
        mean_reversibility_score = _mean([row.get("reversibility_score") for row in events])
        mean_branching_score = _mean([row.get("branching_score") for row in events])
        mean_termination_score = _mean([row.get("termination_score") for row in events])
        mean_replay_priority_score = _mean([row.get("replay_priority_score") for row in events])
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
            INSERT INTO future_option_motifs (
                motif_signature, motif_type, support_count, linked_event_count, linked_family_count, linked_carrier_count,
                linked_role_count, linked_concept_count, cross_context_count, cross_game_count, mean_option_delta,
                mean_abs_option_delta, mean_novelty_score, mean_reversibility_score, mean_branching_score,
                mean_termination_score, mean_replay_priority_score, first_seen_global_step, last_seen_global_step,
                motif_stability_score, is_emergent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        if is_emergent:
            emergent_count += 1
            if group["first_seen"] is not None:
                first_emergent_step = group["first_seen"] if first_emergent_step is None else min(first_emergent_step, int(group["first_seen"]))
        for row in events:
            _insert_future_link(state_conn, motif_signature, "event", str(row["event_id"]), 1, group["first_seen"], group["last_seen"])
        for family in sorted(group["families"]):
            _insert_future_link(state_conn, motif_signature, "family", family, 1, group["first_seen"], group["last_seen"])
        for carrier in sorted(group["carriers"]):
            _insert_future_link(state_conn, motif_signature, "carrier", carrier, 1, group["first_seen"], group["last_seen"])
        for role in sorted(group["roles"]):
            _insert_future_link(state_conn, motif_signature, "role", role, 1, group["first_seen"], group["last_seen"])
        for concept in sorted(group["concepts"]):
            _insert_future_link(state_conn, motif_signature, "concept", concept, 1, group["first_seen"], group["last_seen"])
        for context in sorted(group["contexts"]):
            _insert_future_link(state_conn, motif_signature, "context", context, 1, group["first_seen"], group["last_seen"])
        for game in sorted(group["games"]):
            _insert_future_link(state_conn, motif_signature, "game", game, 1, group["first_seen"], group["last_seen"])
    _write_future_milestone(state_conn, "first_emergent_future_option_motif_step", first_emergent_step, None)
    unknown_motif_count = int(motif_type_counts.get("unknown", 0))
    return {
        "future_option_motif_count": len(selected),
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
    }


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
        for row in state_conn.execute("SELECT * FROM future_option_events ORDER BY event_id ASC").fetchall():
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
                int(row.get("raw_high_attention") or row.get("high_attention") or 0),
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
        high_attention_count += raw_high_attention
        if high_option_change and raw_high_attention:
            high_both_count += 1
        if not high_option_change and raw_high_attention:
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
            "h10_live_rows_used": sum(1 for row in rows if str(row.get("source_label")) == "live"),
            "h10_heuristic_rows_used": sum(1 for row in rows if str(row.get("source_label")) == "heuristic"),
            "h10_fallback_reason": h10_fallback_reason,
        }
    finally:
        state_conn.row_factory = original_row_factory


def derive_future_option_transfer_links(state_conn: sqlite3.Connection) -> dict[str, Any]:
    motif_links = _links_by_signature(state_conn, "future_option_links", "motif_signature")
    motif_emergence = {
        str(row["motif_signature"]): int(row["is_emergent"] or 0)
        for row in state_conn.execute(
            "SELECT motif_signature, is_emergent FROM future_option_motifs ORDER BY motif_signature ASC"
        ).fetchall()
    }
    concept_links = _links_by_signature(state_conn, "concept_links", "concept_signature")
    concepts_by_role: dict[str, set[str]] = defaultdict(set)
    for concept_signature, links in concept_links.items():
        for role_signature in links.get("role", set()):
            concepts_by_role[role_signature].add(concept_signature)
    transfer_rows = [dict(row) for row in state_conn.execute(
        """
        SELECT role_signature, transfer_score, best_margin, reuse_success, similarity_score,
               source_carrier_count, candidate_role_count
        FROM role_transfer_attempts
        ORDER BY role_signature ASC, attempt_id ASC
        """
    ).fetchall()]
    transfers_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transfer_rows:
        transfers_by_role[str(row["role_signature"])].append(row)
    promoted_concepts = {
        str(row["concept_signature"])
        for row in state_conn.execute(
            "SELECT concept_signature FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1 ORDER BY concept_signature ASC"
        ).fetchall()
    }
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
    for motif_signature in sorted(motif_links):
        is_emergent_motif = int(motif_emergence.get(motif_signature, 0)) == 1
        roles = sorted(motif_links[motif_signature].get("role", set()))
        if not roles:
            continue
        motif_had_transfer = False
        motif_had_strong = False
        motif_had_promoted = False
        for role_signature in roles:
            concepts = sorted(concepts_by_role.get(role_signature, set()) or {"__none__"})
            role_transfer_rows = transfers_by_role.get(role_signature, [])
            for concept_signature in concepts:
                transfer_attempt_count = len(role_transfer_rows)
                successful_transfer_count = sum(1 for row in role_transfer_rows if int(row["reuse_success"] or 0) == 1)
                strong_transfer_success_count = sum(
                    1
                    for row in role_transfer_rows
                    if int(row["reuse_success"] or 0) == 1
                    and int(row["source_carrier_count"] or 0) >= 2
                    and int(row["candidate_role_count"] or 0) >= 2
                    and float(row["similarity_score"] or 0.0) >= 0.60
                    and float(row["best_margin"] or 0.0) >= 0.10
                )
                promoted_concept_count = int(concept_signature != "__none__" and concept_signature in promoted_concepts)
                if transfer_attempt_count <= 0 and promoted_concept_count <= 0:
                    continue
                mean_transfer_score = _mean([row.get("transfer_score") for row in role_transfer_rows])
                mean_best_margin = _mean([row.get("best_margin") for row in role_transfer_rows if row.get("best_margin") is not None])
                first_seen = _safe_min(state_conn, motif_signature, role_signature, concept_signature, "first")
                last_seen = _safe_min(state_conn, motif_signature, role_signature, concept_signature, "last")
                state_conn.execute(
                    """
                    INSERT INTO future_option_transfer_links (
                        motif_signature, role_signature, concept_signature, transfer_attempt_count,
                        successful_transfer_count, strong_transfer_success_count, promoted_concept_count,
                        mean_transfer_score, mean_best_margin, first_seen_global_step, last_seen_global_step
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        motif_signature,
                        role_signature,
                        concept_signature,
                        transfer_attempt_count,
                        successful_transfer_count,
                        strong_transfer_success_count,
                        promoted_concept_count,
                        mean_transfer_score,
                        mean_best_margin,
                        first_seen,
                        last_seen,
                    ),
                )
                inserted += 1
                motif_had_transfer = motif_had_transfer or transfer_attempt_count > 0
                motif_had_strong = motif_had_strong or strong_transfer_success_count > 0
                motif_had_promoted = motif_had_promoted or promoted_concept_count > 0
                motif_success_numer += successful_transfer_count
                motif_success_denom += transfer_attempt_count
                motif_strong_numer += strong_transfer_success_count
                if is_emergent_motif:
                    emergent_link_count += 1
                    emergent_success_numer += successful_transfer_count
                    emergent_success_denom += transfer_attempt_count
                    emergent_strong_numer += strong_transfer_success_count
        motifs_with_transfer += int(motif_had_transfer)
        motifs_with_strong_transfer += int(motif_had_strong)
        motifs_with_promoted_concept += int(motif_had_promoted)
        if is_emergent_motif:
            emergent_motifs_with_transfer += int(motif_had_transfer)
            emergent_motifs_with_strong_transfer += int(motif_had_strong)
            emergent_motifs_with_promoted_concept += int(motif_had_promoted)
    return {
        "future_option_transfer_link_count": inserted,
        "motifs_with_transfer_count": motifs_with_transfer,
        "motifs_with_strong_transfer_count": motifs_with_strong_transfer,
        "motifs_with_promoted_concept_count": motifs_with_promoted_concept,
        "motif_transfer_success_rate": (motif_success_numer / motif_success_denom) if motif_success_denom else None,
        "motif_strong_transfer_success_rate": (motif_strong_numer / motif_success_denom) if motif_success_denom else None,
        "promoted_concept_motif_count": motifs_with_promoted_concept,
        "emergent_future_option_transfer_link_count": emergent_link_count,
        "emergent_motifs_with_strong_transfer_count": emergent_motifs_with_strong_transfer,
        "emergent_motifs_with_promoted_concept_count": emergent_motifs_with_promoted_concept,
        "emergent_motif_transfer_success_rate": (emergent_success_numer / emergent_success_denom) if emergent_success_denom else None,
        "emergent_motif_strong_transfer_success_rate": (emergent_strong_numer / emergent_success_denom) if emergent_success_denom else None,
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
) -> dict[str, Any]:
    motif_type, text_tokens, motif_type_source = _detect_motif_type(
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
    return {
        "event_id": "foe:" + sha1(event_id_seed.encode("utf-8")).hexdigest(),
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
        "evidence_json": evidence,
    }


def _context_key_from_canonical(canonical_key: str) -> str | None:
    if "|a" not in canonical_key:
        return None
    return canonical_key.split("|a", 1)[0] or None


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

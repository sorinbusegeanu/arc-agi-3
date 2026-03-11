from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import NavigationEdgeV2, NavigationStateCellV2, SCHEMA_VERSION, TrajectoryEpisodeV2


def _cell_from_step(step) -> Optional[Tuple[int, int]]:
    pos = step.actual_avatar_centroid or step.predicted_avatar_centroid
    if pos is None:
        return None
    return (int(round(pos[0])), int(round(pos[1])))


def _edge_transition_type(src: Tuple[int, int], dst: Tuple[int, int], step) -> str:
    if step.done:
        return "transition"
    if src == dst:
        return "blocked"
    if step.area_id and step.info.get("pre_area_id") and step.info.get("pre_area_id") != step.area_id:
        return "transition"
    if step.event_ids:
        return "interaction"
    return "move"


def build_navigation_graph_from_episodes(
    episodes: List[TrajectoryEpisodeV2],
) -> tuple[list[NavigationStateCellV2], list[NavigationEdgeV2]]:
    cell_stats: DefaultDict[Tuple[int, int], Dict[str, int]] = defaultdict(lambda: {"visit": 0, "blocked": 0})
    edge_stats: DefaultDict[Tuple[Tuple[int, int], Tuple[int, int], Optional[int], str], Dict[str, object]] = defaultdict(
        lambda: {"success": 0, "blocked": 0, "uncertain": 0, "evidence": []}
    )
    game_id = episodes[0].game_id if episodes else "unknown_game"

    for episode in episodes:
        prev_cell: Optional[Tuple[int, int]] = None
        for step in episode.steps:
            cell = _cell_from_step(step)
            if cell is None:
                prev_cell = None
                continue
            cell_stats[cell]["visit"] += 1
            if prev_cell is not None:
                transition_type = _edge_transition_type(prev_cell, cell, step)
                edge_key = (prev_cell, cell, step.action.action_id, transition_type)
                if prev_cell == cell:
                    edge_stats[edge_key]["blocked"] += 1
                    cell_stats[cell]["blocked"] += 1
                elif step.actual_avatar_centroid is None and step.predicted_avatar_centroid is not None:
                    edge_stats[edge_key]["uncertain"] += 1
                else:
                    edge_stats[edge_key]["success"] += 1
                edge_stats[edge_key]["evidence"].append(f"{episode.episode_id}:{step.step_idx}")
            prev_cell = cell

    cells: List[NavigationStateCellV2] = []
    for cell, stats in sorted(cell_stats.items()):
        visits = stats["visit"]
        blocked = stats["blocked"]
        status = "blocked" if blocked > visits // 2 and visits > 0 else "confirmed"
        cells.append(
            NavigationStateCellV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                cell=cell,
                status=status,
                visit_count=visits,
                blocked_count=blocked,
                last_seen_round=0,
                confidence=min(1.0, visits / 4.0),
            )
        )

    edge_lookup: Dict[Tuple[Tuple[int, int], Tuple[int, int], Optional[int]], Dict[str, int]] = defaultdict(lambda: {"forward": 0, "backward": 0})
    for (src, dst, action_id, _), stats in edge_stats.items():
        edge_lookup[(src, dst, action_id)]["forward"] += int(stats["success"])
        edge_lookup[(dst, src, action_id)]["backward"] += int(stats["success"])

    edges: List[NavigationEdgeV2] = []
    for (src, dst, action_id, transition_type), stats in sorted(edge_stats.items()):
        success_count = int(stats["success"])
        blocked_count = int(stats["blocked"])
        uncertain_count = int(stats["uncertain"])
        total = success_count + blocked_count + uncertain_count
        opposite = edge_lookup[(src, dst, action_id)]["backward"]
        bidirectional_confidence = min(1.0, min(success_count, opposite) / float(max(1, max(success_count, opposite))))
        edges.append(
            NavigationEdgeV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                edge_id="edge:%s:%s:%s:%s:%s" % (src[0], src[1], dst[0], dst[1], action_id),
                src_cell=src,
                dst_cell=dst,
                action_id=action_id,
                transition_type=transition_type,
                success_count=success_count,
                blocked_count=blocked_count,
                uncertain_count=uncertain_count,
                bidirectional_confidence=bidirectional_confidence,
                confidence=min(1.0, total / 3.0),
                evidence_refs=list(stats["evidence"])[-12:],
            )
        )
    return cells, edges


def merge_navigation_graph(
    existing_cells: List[NavigationStateCellV2],
    existing_edges: List[NavigationEdgeV2],
    new_cells: List[NavigationStateCellV2],
    new_edges: List[NavigationEdgeV2],
) -> tuple[list[NavigationStateCellV2], list[NavigationEdgeV2]]:
    merged_cells: Dict[Tuple[int, int], NavigationStateCellV2] = {row.cell: row for row in existing_cells}
    for cell in new_cells:
        prior = merged_cells.get(cell.cell)
        if prior is None:
            merged_cells[cell.cell] = cell
            continue
        merged_cells[cell.cell] = NavigationStateCellV2(
            schema_version=SCHEMA_VERSION,
            game_id=prior.game_id,
            cell=prior.cell,
            status="blocked" if (prior.blocked_count + cell.blocked_count) > (prior.visit_count + cell.visit_count) // 2 else "confirmed",
            visit_count=prior.visit_count + cell.visit_count,
            blocked_count=prior.blocked_count + cell.blocked_count,
            last_seen_round=max(prior.last_seen_round, cell.last_seen_round),
            confidence=max(prior.confidence, cell.confidence),
        )

    merged_edges: Dict[Tuple[Tuple[int, int], Tuple[int, int], Optional[int], str], NavigationEdgeV2] = {
        (row.src_cell, row.dst_cell, row.action_id, row.transition_type): row for row in existing_edges
    }
    for edge in new_edges:
        key = (edge.src_cell, edge.dst_cell, edge.action_id, edge.transition_type)
        prior = merged_edges.get(key)
        if prior is None:
            merged_edges[key] = edge
            continue
        total = prior.success_count + prior.blocked_count + prior.uncertain_count + edge.success_count + edge.blocked_count + edge.uncertain_count
        merged_edges[key] = NavigationEdgeV2(
            schema_version=SCHEMA_VERSION,
            game_id=prior.game_id,
            edge_id=prior.edge_id,
            src_cell=prior.src_cell,
            dst_cell=prior.dst_cell,
            action_id=prior.action_id,
            transition_type=prior.transition_type,
            success_count=prior.success_count + edge.success_count,
            blocked_count=prior.blocked_count + edge.blocked_count,
            uncertain_count=prior.uncertain_count + edge.uncertain_count,
            bidirectional_confidence=max(prior.bidirectional_confidence, edge.bidirectional_confidence),
            confidence=min(1.0, total / 4.0),
            evidence_refs=sorted(set(prior.evidence_refs + edge.evidence_refs))[-24:],
        )
    return sorted(merged_cells.values(), key=lambda row: row.cell), sorted(
        merged_edges.values(),
        key=lambda row: (row.src_cell, row.dst_cell, -1 if row.action_id is None else row.action_id, row.transition_type),
    )

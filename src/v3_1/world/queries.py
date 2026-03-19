from __future__ import annotations

PLANNER_VISIBLE_POI_BUCKETS = {"structural", "interactable_object"}


def _planner_visible_poi(entity: dict) -> bool:
    return bool(
        entity.get("kind") == "poi"
        and bool(entity.get("planner_visible", True))
        and bool(entity.get("planner_targetable", True))
        and str(entity.get("poi_bucket") or "") in PLANNER_VISIBLE_POI_BUCKETS
    )


def _root_poi_id(row: dict) -> str:
    return str(row.get("parent_poi_id") or row.get("entity_id") or "")


def _detector_backed(row: dict) -> bool:
    provenance = set(str(value) for value in list(row.get("poi_source_provenance", []) or []))
    return "detector" in provenance


def _poi_matches_local_region(row: dict, area_id: str) -> bool:
    if str(row.get("area_id") or "") == str(area_id):
        return True
    if bool(row.get("reachable_now")) or bool(row.get("reachable_later")):
        return True
    if str(row.get("approachable_status") or "") in {"approachable_now", "approachable_later"}:
        return True
    access_profile = dict(row.get("access_profile", {}) or {})
    if bool(access_profile.get("frontier_adjacent")):
        return True
    nearest_bbox_distance = access_profile.get("nearest_bbox_distance")
    return nearest_bbox_distance is not None and float(nearest_bbox_distance) <= 4.0


def _prioritize_poi_targets(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in list(rows or []):
        groups.setdefault(_root_poi_id(row), []).append(row)
    prioritized: list[dict] = []
    for root_id, group in groups.items():
        children = [row for row in group if int(row.get("poi_hierarchy_level", 0) or 0) > 0]
        distinct_children = sorted(
            children,
            key=lambda row: (
                not _detector_backed(row),
                not bool(row.get("reachable_now")),
                not bool(row.get("reachable_later")),
                not str(row.get("approachable_status") or "") in {"approachable_now", "approachable_later"},
                -float(row.get("interact_effect_score", 0.0) or row.get("candidate_effect_score", 0.0) or 0.0),
                -float(row.get("utility", 0.0) or 0.0),
                -float(row.get("confidence", 0.0) or 0.0),
                str(row.get("entity_id", "")),
            ),
        )
        if distinct_children:
            prioritized.extend(distinct_children)
            continue
        parents = sorted(
            group,
            key=lambda row: (
                not _detector_backed(row),
                not bool(row.get("reachable_now")),
                not bool(row.get("reachable_later")),
                not str(row.get("approachable_status") or "") in {"approachable_now", "approachable_later"},
                -float(row.get("utility", 0.0) or 0.0),
                -float(row.get("confidence", 0.0) or 0.0),
                str(row.get("entity_id", "")),
            ),
        )
        if parents:
            prioritized.append(parents[0])
    return prioritized


def target_candidate_queries(state: dict) -> list[dict]:
    retries = state.get("memory_hints", {}).get("retries", {})
    rows = []
    for entity_id, entity in state.get("entities", {}).items():
        if not _planner_visible_poi(entity):
            continue
        rows.append(
            dict(
                entity,
                retry_count=int(retries.get(entity_id, 0)),
                novelty=float(entity.get("novelty", 0.0)),
            )
        )
    return sorted(
        _prioritize_poi_targets(rows),
        key=lambda row: (
            not bool(row.get("reachable_now")),
            not bool(row.get("reachable_later")),
            int(row.get("retry_count", 0)),
            -float(row.get("confidence", 0.0)),
            -float(row.get("utility", 0.0)),
            row.get("entity_id", ""),
        ),
    )


def area_local_pois(state: dict, area_id: str) -> list[dict]:
    entities = dict(state.get("entities", {}))
    poi_ids = state.get("indexes", {}).get("pois_by_area", {}).get(area_id, [])
    rows = [
        dict(entities[poi_id], returned_by_area_local_pois=True)
        for poi_id in poi_ids
        if poi_id in entities and _planner_visible_poi(entities[poi_id])
    ]
    for entity_id, entity in entities.items():
        if entity_id in poi_ids or not _planner_visible_poi(entity):
            continue
        if _poi_matches_local_region(entity, area_id):
            rows.append(dict(entity, returned_by_area_local_pois=True))
    return _prioritize_poi_targets(rows)


def reachable_targets(state: dict) -> list[dict]:
    target_ids = state.get("indexes", {}).get("reachable_targets", [])
    return [state["entities"][target_id] for target_id in target_ids if target_id in state.get("entities", {})]


def unreachable_targets(state: dict) -> list[dict]:
    target_ids = state.get("indexes", {}).get("blocked_targets", [])
    return [state["entities"][target_id] for target_id in target_ids if target_id in state.get("entities", {})]


def novelty_retry_aware_targets(state: dict, retries: dict[str, int] | None = None) -> list[dict]:
    retries = retries or {}
    rows = []
    for row in target_candidate_queries(state):
        rows.append(dict(row, retry_count=int(retries.get(row["entity_id"], 0))))
    return sorted(rows, key=lambda row: (row["retry_count"], -float(row.get("novelty", 0.0)), -float(row.get("confidence", 0.0))))


def frontier_candidates(state: dict) -> list[dict]:
    target_ids = state.get("indexes", {}).get("frontier_candidates", [])
    return [state["entities"][target_id] for target_id in target_ids if target_id in state.get("entities", {})]


def candidate_targets(state: dict) -> list[dict]:
    return target_candidate_queries(state)

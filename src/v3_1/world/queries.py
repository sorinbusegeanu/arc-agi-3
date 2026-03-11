from __future__ import annotations


def target_candidate_queries(state: dict) -> list[dict]:
    retries = state.get("memory_hints", {}).get("retries", {})
    rows = []
    for entity_id, entity in state.get("entities", {}).items():
        if entity.get("kind") != "poi":
            continue
        rows.append(
            dict(
                entity,
                retry_count=int(retries.get(entity_id, 0)),
                novelty=float(entity.get("novelty", 0.0)),
            )
        )
    return sorted(
        rows,
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
    poi_ids = state.get("indexes", {}).get("pois_by_area", {}).get(area_id, [])
    return [state["entities"][poi_id] for poi_id in poi_ids if poi_id in state.get("entities", {})]


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

from __future__ import annotations

try:
    from v5_0.route.map_builder import build_traversability_map
except Exception:  # pragma: no cover - optional module in v5_0
    build_traversability_map = None
try:
    from v5_0.route.pathfinder import plan_route_to_poi, replan_route_after_step
except Exception:  # pragma: no cover - optional module in v5_0
    plan_route_to_poi = None
    replan_route_after_step = None
try:
    from v5_0.route.service import build_routes_multi_reset
except Exception:  # pragma: no cover - optional module in v5_0
    build_routes_multi_reset = None
from v5_0.route.trajectory_enumerator import enumerate_routes_between_points

__all__ = [
    "build_traversability_map",
    "plan_route_to_poi",
    "replan_route_after_step",
    "build_routes_multi_reset",
    "enumerate_routes_between_points",
]

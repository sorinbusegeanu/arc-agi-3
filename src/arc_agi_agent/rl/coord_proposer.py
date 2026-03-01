from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple


def _default_cfg() -> Dict[str, Any]:
    return {
        "coord_topK": 16,
        "max_objects_used": 32,
    }


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    return {}


def _grid_dims(fp_report: Dict[str, Any]) -> Tuple[int, int, str]:
    feats = fp_report.get("features_v1", {})
    if isinstance(feats, dict):
        grid_index = feats.get("grid_index", {})
        grids = grid_index.get("grids") if isinstance(grid_index, dict) else None
        if isinstance(grids, list) and grids:
            g0 = grids[0]
            return int(g0.get("height", 64)), int(g0.get("width", 64)), str(g0.get("name", "frame_0"))

    state = fp_report.get("state_summary", {})
    if isinstance(state, dict):
        gs = state.get("grid_summaries")
        if isinstance(gs, list) and gs:
            g0 = gs[0]
            return int(g0.get("height", 64)), int(g0.get("width", 64)), str(g0.get("name", "frame_0"))

    return 64, 64, "frame_0"


def _add(points: List[Dict[str, Any]], x: int, y: int, tag: str, source: str, priority: int) -> None:
    points.append({
        "x": int(x),
        "y": int(y),
        "tag": str(tag),
        "source_grid_name": str(source),
        "priority": int(priority),
    })


class CoordProposer:
    def propose(
        self,
        fp_current: Any,
        fp_prev: Optional[Any] = None,
        last_event: Optional[Dict[str, Any]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_eff = {**_default_cfg(), **(cfg or {})}
        top_k = int(cfg_eff["coord_topK"])

        curr = _as_dict(fp_current)
        h, w, grid_name = _grid_dims(curr)

        points: List[Dict[str, Any]] = []

        feats = curr.get("features_v1", {})
        if isinstance(feats, dict):
            obj_idx = feats.get("object_index")
            if isinstance(obj_idx, list):
                objs = sorted(obj_idx, key=lambda o: (-int(o.get("area", 0)), str(o.get("object_id", ""))))
                for obj in objs[: int(cfg_eff["max_objects_used"])]:
                    cy, cx = obj.get("centroid", (0.0, 0.0))
                    src = str(obj.get("grid_name") or grid_name)
                    _add(points, int(round(cx)), int(round(cy)), "centroid", src, 0)
                    bbox = obj.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        y0, x0, y1, x1 = [int(v) for v in bbox]
                        _add(points, x0, y0, "bbox_corner", src, 1)
                        _add(points, x1, y0, "bbox_corner", src, 1)
                        _add(points, x0, y1, "bbox_corner", src, 1)
                        _add(points, x1, y1, "bbox_corner", src, 1)

            inter = feats.get("interaction_points")
            if isinstance(inter, list):
                for p in inter:
                    point = p.get("point")
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        x, y = int(point[0]), int(point[1])
                        _add(points, x, y, str(p.get("tag", "interaction")), str(p.get("source_grid_name") or grid_name), 2)

        diff_summary = curr.get("diff_summary", {})
        if isinstance(diff_summary, dict):
            bbox = diff_summary.get("changed_bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                y0, x0, y1, x1 = [int(v) for v in bbox]
                _add(points, int((x0 + x1) / 2), int((y0 + y1) / 2), "changed_bbox_focus", grid_name, 0)

        anchors = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
        for x, y in anchors:
            _add(points, x, y, "grid_anchor", grid_name, 3)

        # Deterministic order: priority -> y -> x -> tag
        points.sort(key=lambda p: (int(p["priority"]), int(p["y"]), int(p["x"]), str(p["tag"])))

        # Dedupe by coordinate.
        out: List[Dict[str, Any]] = []
        seen: set[Tuple[int, int]] = set()
        for p in points:
            x, y = int(p["x"]), int(p["y"])
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "x": x,
                    "y": y,
                    "tag": str(p["tag"]),
                    "source_grid_name": str(p.get("source_grid_name", grid_name)),
                }
            )
            if len(out) >= top_k:
                break

        return {"schema_version": "COORD_CANDS_V1", "coords": out}

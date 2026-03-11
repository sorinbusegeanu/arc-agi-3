from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def dataclass_to_dict(obj: Any) -> Any:
    if dataclass_isinstance(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return [dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def dataclass_isinstance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        return max(0, self.x2 - self.x1 + 1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1 + 1)

    def area(self) -> int:
        return self.width() * self.height()

    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> Dict[str, int]:
        return {"x1": int(self.x1), "y1": int(self.y1), "x2": int(self.x2), "y2": int(self.y2)}

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "BBox":
        return BBox(int(payload["x1"]), int(payload["y1"]), int(payload["x2"]), int(payload["y2"]))


def bbox_from_points(points: Sequence[Tuple[int, int]]) -> Optional[BBox]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    union = a.area() + b.area() - inter
    return float(inter) / float(max(1, union))


def bbox_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.centroid()
    bx, by = b.centroid()
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def point_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def point_neighbors4(pt: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = int(pt[0]), int(pt[1])
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def normalize_palette(values: Iterable[int]) -> List[int]:
    out = []
    seen = set()
    for value in values:
        iv = int(value)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    out.sort()
    return out


def stable_tuple_key(value: Any) -> Tuple:
    if isinstance(value, dict):
        return tuple((k, stable_tuple_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(stable_tuple_key(v) for v in value)
    if isinstance(value, tuple):
        return tuple(stable_tuple_key(v) for v in value)
    return (value,)


def compact_context_key(area_id: Optional[str], local_occupancy_code: str, target_bucket: str, action_id: Optional[int] = None) -> str:
    parts = [str(area_id or "none"), str(local_occupancy_code), str(target_bucket)]
    if action_id is not None:
        parts.append(str(int(action_id)))
    return "|".join(parts)


def merge_ranges(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (min(int(a[0]), int(b[0])), max(int(a[1]), int(b[1])))


def point_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def point_neighbors4(pt: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = int(pt[0]), int(pt[1])
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def normalize_palette(values: Iterable[int]) -> List[int]:
    return sorted({int(v) for v in values})


def stable_tuple_key(value: Any) -> Tuple:
    if isinstance(value, dict):
        return tuple((k, stable_tuple_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(stable_tuple_key(v) for v in value)
    return (value,)


def compact_context_key(
    area_id: Optional[str],
    local_occupancy_code: Any,
    target_bucket: Any,
    action_id: Optional[int] = None,
) -> str:
    area = area_id or "none"
    action_part = f":a{int(action_id)}" if action_id is not None else ""
    return f"{area}:{local_occupancy_code}:{target_bucket}{action_part}"


def merge_ranges(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (min(int(a[0]), int(b[0])), max(int(a[1]), int(b[1])))


def merge_bboxes(bboxes: Iterable[BBox]) -> Optional[BBox]:
    xs1 = []
    ys1 = []
    xs2 = []
    ys2 = []
    for b in bboxes:
        xs1.append(b.x1)
        ys1.append(b.y1)
        xs2.append(b.x2)
        ys2.append(b.y2)
    if not xs1:
        return None
    return BBox(min(xs1), min(ys1), max(xs2), max(ys2))


def grid_palette(grid: Sequence[Sequence[int]]) -> List[int]:
    palette = set()
    for row in grid:
        for v in row:
            palette.add(int(v))
    return sorted(palette)


def grid_diff(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> Tuple[int, List[Tuple[int, int]]]:
    changed = []
    for y, (row_a, row_b) in enumerate(zip(a, b)):
        for x, (va, vb) in enumerate(zip(row_a, row_b)):
            if va != vb:
                changed.append((x, y))
    return len(changed), changed


def connected_components_per_color(grid: Sequence[Sequence[int]]) -> Dict[int, List[List[Tuple[int, int]]]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    comps: Dict[int, List[List[Tuple[int, int]]]] = {}

    def neighbors(x: int, y: int) -> Iterable[Tuple[int, int]]:
        if x > 0:
            yield (x - 1, y)
        if x + 1 < width:
            yield (x + 1, y)
        if y > 0:
            yield (x, y - 1)
        if y + 1 < height:
            yield (x, y + 1)

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue
            visited[y][x] = True
            color = int(grid[y][x])
            stack = [(x, y)]
            comp = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                for nx, ny in neighbors(cx, cy):
                    if visited[ny][nx]:
                        continue
                    if int(grid[ny][nx]) != color:
                        continue
                    visited[ny][nx] = True
                    stack.append((nx, ny))
                    comp.append((nx, ny))
            comps.setdefault(color, []).append(comp)
    return comps

from __future__ import annotations

from collections import Counter
from typing import Any

from v4.agentContract.types import V4Observation


class StateAdapter:
    reused_modules = ("src/v4/state/*", "src/v4/exploration/*", "src/v4/affordance/*")

    def summarize_observation(self, observation) -> dict:
        if isinstance(observation, V4Observation):
            payload = dict(observation.raw_payload)
            candidate_clickable_regions = ()  # Disabled for now: clickable-region detection is not implemented yet.
            hud_regions = self._collect_regions(payload, ("hud_regions",))
            life_regions = self._collect_regions(payload, ("life_regions",))
            progress_regions = self._collect_regions(payload, ("progress_regions",))
            candidate_hazards = self._collect_regions(payload, ("hazard_cells", "hazards", "enemy_cells", "danger_cells"))
            frame_avatar_cells, frame_poi_cells, frame_click_cells, frame_hazard_cells = self._derive_cells_from_frame(observation.frame)
            if not candidate_hazards:
                candidate_hazards = tuple(f"cell:{x},{y}" for x, y in frame_hazard_cells)
            salient = self._collect_regions(payload, ("changed_regions", "salient_changed_regions"))
            observed_affordances = self._derive_affordances(candidate_clickable_regions)
            avatar_cells = self._collect_cells(payload, ("avatar_position", "avatar_positions", "avatar", "player_position", "player")) or frame_avatar_cells
            clickable_cells = ()
            poi_cells = self._collect_cells(payload, ("poi_cells", "goal_cells", "targets", "target_cells")) or frame_poi_cells
            hazard_cells = self._collect_cells(payload, ("hazard_cells", "enemy_cells", "danger_cells")) or frame_hazard_cells
            return {
                "avatar_bbox": self._bbox_from_cells(avatar_cells),
                "avatar_position": self._center_from_bbox(self._bbox_from_cells(avatar_cells)),
                "candidate_clickable_regions": candidate_clickable_regions,
                "hud_regions": hud_regions,
                "life_regions": life_regions,
                "progress_regions": progress_regions,
                "salient_changed_regions": salient,
                "frame_poi_cells": frame_poi_cells,
                "candidate_hazards": candidate_hazards,
                "candidate_mode_hints": self._derive_mode_hints(candidate_clickable_regions, hud_regions),
                "observed_affordances": observed_affordances,
                "probe_effects": {},
                "levels_completed": int(observation.levels_completed),
                "win_levels": int(observation.win_levels),
                "terminal_flag": self._terminal_flag(observation.state),
                "raw_observation_payload": {
                    "avatar_cells": avatar_cells,
                    "clickable_cells": clickable_cells,
                    "poi_cells": poi_cells,
                    "hazard_cells": hazard_cells,
                    "frame_size": self._frame_size(observation.frame),
                    "pre_frame": self.extract_frame_plane(observation.frame),
                    "state": observation.state,
                    **self._build_background_payload(
                        frame=self.extract_frame_plane(observation.frame),
                        avatar_cells=avatar_cells,
                        poi_cells=poi_cells,
                        hazard_cells=hazard_cells,
                    ),
                },
            }
        payload = observation if isinstance(observation, dict) else {}
        return {
            "avatar_bbox": payload.get("avatar_bbox"),
            "avatar_position": payload.get("avatar_position"),
            "candidate_clickable_regions": (),
            "hud_regions": tuple(payload.get("hud_regions", ())),
            "life_regions": tuple(payload.get("life_regions", ())),
            "progress_regions": tuple(payload.get("progress_regions", ())),
            "salient_changed_regions": tuple(payload.get("salient_changed_regions", ())),
            "frame_poi_cells": tuple(payload.get("frame_poi_cells", ())),
            "candidate_hazards": tuple(payload.get("candidate_hazards", ())),
            "candidate_mode_hints": tuple(payload.get("candidate_mode_hints", ())),
            "observed_affordances": tuple(payload.get("observed_affordances", ())),
            "probe_effects": dict(payload.get("probe_effects", {})),
            "levels_completed": payload.get("levels_completed"),
            "win_levels": payload.get("win_levels"),
            "terminal_flag": bool(payload.get("terminal_flag", False)),
            "raw_observation_payload": dict(payload.get("raw_observation_payload", {})),
        }

    def _build_background_payload(
        self,
        *,
        frame: tuple[tuple[int, ...], ...],
        avatar_cells: tuple[tuple[int, int], ...],
        poi_cells: tuple[tuple[int, int], ...],
        hazard_cells: tuple[tuple[int, int], ...],
    ) -> dict[str, Any]:
        if not frame:
            return {}
        dominant_colors = self._estimate_dominant_background_colors(frame)
        excluded = set(avatar_cells) | set(poi_cells) | set(hazard_cells)
        traversable = self._derive_traversable_regions(frame, avatar_cells=avatar_cells, excluded=excluded)
        blocking = self._derive_blocking_regions(frame, avatar_cells=avatar_cells, excluded=excluded, dominant_colors=dominant_colors)
        unknown = self._derive_unknown_regions(frame, excluded=excluded, traversable=traversable, blocking=blocking)
        return {
            "traversable_regions": tuple(self._serialize_region(component) for component in traversable),
            "blocking_regions": tuple(self._serialize_region(component) for component in blocking),
            "unknown_regions": tuple(self._serialize_region(component) for component in unknown),
            "playfield_bbox": self._playfield_bbox(frame),
        }

    def _collect_regions(self, payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
        regions: list[str] = []
        for key in keys:
            if key not in payload:
                continue
            for item in self._flatten(payload[key]):
                region = self._to_region(item)
                if region is not None and region not in regions:
                    regions.append(region)
        return tuple(regions)

    def _collect_cells(self, payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
        cells: list[tuple[int, int]] = []
        for key in keys:
            if key not in payload:
                continue
            for item in self._flatten(payload[key]):
                cell = self._to_cell(item)
                if cell is not None and cell not in cells:
                    cells.append(cell)
        return tuple(cells)

    def _flatten(self, value: Any):
        if isinstance(value, dict):
            yield value
            return
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(part, int) for part in value):
            yield value
            return
        if isinstance(value, list) and len(value) == 2 and all(isinstance(part, int) for part in value):
            yield tuple(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (list, tuple, dict)):
                    yield from self._flatten(item)
                else:
                    yield item
            return
        yield value

    def _to_region(self, value: Any) -> str | None:
        cell = self._to_cell(value)
        if cell is not None:
            return f"cell:{cell[0]},{cell[1]}"
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            cell = self._to_cell(value)
            if cell is not None:
                return f"cell:{cell[0]},{cell[1]}"
        return None

    def _to_cell(self, value: Any) -> tuple[int, int] | None:
        if isinstance(value, dict):
            if {"x", "y"} <= set(value):
                return (int(value["x"]), int(value["y"]))
            if {"col", "row"} <= set(value):
                return (int(value["col"]), int(value["row"]))
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(part, int) for part in value):
            return (int(value[0]), int(value[1]))
        if isinstance(value, list) and len(value) == 2 and all(isinstance(part, int) for part in value):
            return (int(value[0]), int(value[1]))
        return None

    def _derive_affordances(self, clickable_regions: tuple[str, ...]) -> tuple[str, ...]:
        affordances: list[str] = []
        if clickable_regions:
            affordances.append("click")
        return tuple(affordances)

    def _derive_mode_hints(self, clickable_regions: tuple[str, ...], hud_regions: tuple[str, ...]) -> tuple[str, ...]:
        hints = []
        if clickable_regions:
            hints.append("click")
        if hud_regions:
            hints.append("progress")
        return tuple(hints)

    def _terminal_flag(self, state: str) -> bool:
        lowered = str(state).lower()
        return any(token in lowered for token in ("win", "success", "lose", "fail", "game_over", "complete"))

    def _frame_size(self, frame: Any) -> tuple[int, int] | None:
        if not isinstance(frame, tuple) or not frame:
            return None
        plane = frame[0]
        if not isinstance(plane, tuple) or not plane:
            return None
        return (len(plane[0]), len(plane))

    def extract_frame_plane(self, frame: Any) -> tuple[tuple[int, ...], ...]:
        if not isinstance(frame, tuple) or not frame or not isinstance(frame[0], tuple):
            return ()
        return tuple(tuple(int(cell) for cell in row) for row in frame[0])

    def _bbox_from_cells(self, cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int] | None:
        if not cells:
            return None
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return (min(xs), min(ys), max(xs), max(ys))

    def _center_from_bbox(self, bbox: tuple[int, int, int, int] | None) -> tuple[float, float] | None:
        if bbox is None:
            return None
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _derive_cells_from_frame(self, frame: Any) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        if not isinstance(frame, tuple) or not frame or not isinstance(frame[0], tuple):
            return (), (), (), ()
        avatar_cells = []
        poi_cells = []
        click_cells = []
        hazard_cells = []
        for y, row in enumerate(frame[0]):
            if not isinstance(row, tuple):
                continue
            for x, value in enumerate(row):
                if value == 1:
                    avatar_cells.append((x, y))
                elif value == 2:
                    poi_cells.append((x, y))
                elif value == 3:
                    click_cells.append((x, y))
                elif value == 9:
                    hazard_cells.append((x, y))
        return tuple(avatar_cells), tuple(poi_cells), tuple(click_cells), tuple(hazard_cells)

    def _estimate_dominant_background_colors(self, frame: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        counts = Counter(int(value) for row in frame for value in row)
        return tuple(color for color, _ in counts.most_common(2))

    def _connected_regions_for_colors(self, frame: tuple[tuple[int, ...], ...], *, allowed_colors: set[int], excluded: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
        cells = {(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) in allowed_colors and (x, y) not in excluded}
        components = []
        remaining = set(cells)
        while remaining:
            start = remaining.pop()
            stack = [start]
            component = {start}
            while stack:
                x, y = stack.pop()
                for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def _derive_traversable_regions(
        self,
        frame: tuple[tuple[int, ...], ...],
        *,
        avatar_cells: tuple[tuple[int, int], ...],
        excluded: set[tuple[int, int]],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        if not avatar_cells:
            return ()
        adjacent_colors = set()
        for x, y in avatar_cells:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= ny < len(frame) and 0 <= nx < len(frame[0]) and (nx, ny) not in excluded:
                    adjacent_colors.add(int(frame[ny][nx]))
        if not adjacent_colors:
            adjacent_colors.add(0)
        regions = self._connected_regions_for_colors(frame, allowed_colors=adjacent_colors, excluded=excluded)
        return tuple(region for region in regions if any(self._touches_avatar_neighbor(region, avatar_cells) for _ in (0,)))

    def _touches_avatar_neighbor(self, region: tuple[tuple[int, int], ...], avatar_cells: tuple[tuple[int, int], ...]) -> bool:
        avatar_neighbors = set()
        for x, y in avatar_cells:
            avatar_neighbors.update(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
        return any(cell in avatar_neighbors for cell in region)

    def _derive_blocking_regions(
        self,
        frame: tuple[tuple[int, ...], ...],
        *,
        avatar_cells: tuple[tuple[int, int], ...],
        excluded: set[tuple[int, int]],
        dominant_colors: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        height = len(frame)
        width = len(frame[0]) if height else 0
        blocking = []
        for color in {int(value) for row in frame for value in row if int(value) != 0}:
            regions = self._connected_regions_for_colors(frame, allowed_colors={color}, excluded=excluded)
            for region in regions:
                xs = [cell[0] for cell in region]
                ys = [cell[1] for cell in region]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                touches_perimeter = bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width - 1 or bbox[3] == height - 1
                elongated = (bbox[2] - bbox[0] + 1) >= max(3, width // 4) or (bbox[3] - bbox[1] + 1) >= max(3, height // 4)
                dominant = color in set(dominant_colors)
                if touches_perimeter or elongated or len(region) >= max(4, (width * height) // 12) or dominant:
                    blocking.append(region)
        return tuple(blocking)

    def _derive_unknown_regions(
        self,
        frame: tuple[tuple[int, ...], ...],
        *,
        excluded: set[tuple[int, int]],
        traversable: tuple[tuple[tuple[int, int], ...], ...],
        blocking: tuple[tuple[tuple[int, int], ...], ...],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        known = set(excluded)
        for region in traversable + blocking:
            known.update(region)
        unknown_cells = {
            (x, y)
            for y, row in enumerate(frame)
            for x, value in enumerate(row)
            if (x, y) not in known and int(value) != 0
        }
        return self._connected_regions_for_colors(frame, allowed_colors={int(frame[y][x]) for x, y in unknown_cells}, excluded=known) if unknown_cells else ()

    def _serialize_region(self, region: tuple[tuple[int, int], ...]) -> str:
        return "|".join(f"cell:{x},{y}" for x, y in region)

    def _playfield_bbox(self, frame: tuple[tuple[int, ...], ...]) -> tuple[int, int, int, int] | None:
        nonzero = [(x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) != 0]
        if not nonzero:
            return None
        xs = [x for x, _ in nonzero]
        ys = [y for _, y in nonzero]
        return (min(xs), min(ys), max(xs), max(ys))

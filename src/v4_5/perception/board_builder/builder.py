from __future__ import annotations

from v4_5.contracts.boardState import BoardState
from v4_5.perception.board_builder.avatarExtractor import extract_avatar
from v4_5.perception.board_builder.backgroundExtractor import extract_background_masks
from v4_5.perception.board_builder.clickableExtractor import extract_clickable_items
from v4_5.perception.board_builder.hazardExtractor import extract_hazard_mask
from v4_5.perception.board_builder.hudExtractor import extract_hud_regions
from v4_5.perception.board_builder.poiExtractor import extract_pois
from v4_5.perception.board_geometry import BoardGeometryResult


class DeterministicBoardBuilder:
    module_name = "DeterministicBoardBuilder"

    def build(self, *, geometry_result: BoardGeometryResult, round_id: str, schema_version: str = "v4.5") -> BoardState:
        current_frame = geometry_result.frames[-1]
        avatar = extract_avatar(current_frame)
        pois = extract_pois(current_frame)
        clickables = extract_clickable_items(current_frame)
        hud_region, progress_region, lives_region = extract_hud_regions(current_frame)
        traversable, blocking = extract_background_masks(current_frame)
        hazard_mask = extract_hazard_mask(current_frame)
        objects = tuple(item for item in (avatar,) if item is not None) + pois + clickables
        gaps = []
        if avatar is None:
            gaps.append("avatar_absent")
        if not pois:
            gaps.append("poi_absent")
        return BoardState(
            schema_version=schema_version,
            round_id=round_id,
            board_geometry_summary=geometry_result.summary,
            source_observation_window_size=geometry_result.window_size,
            avatar_object=avatar,
            objects=objects,
            traversable_background=traversable,
            blocking_background=blocking,
            hazard_representation=hazard_mask,
            hud_region=hud_region,
            progress_bar_region=progress_region,
            lives_region=lives_region,
            advisory_only=True,
            gaps=tuple(gaps),
        )


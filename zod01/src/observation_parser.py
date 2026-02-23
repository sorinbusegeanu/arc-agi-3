from __future__ import annotations

from arcengine import FrameData, FrameDataRaw

from .types import ParsedObservation


def _to_layer_tuple(layer: object) -> tuple[tuple[int, ...], ...]:
    if hasattr(layer, "tolist"):
        rows = layer.tolist()
    else:
        rows = layer  # type: ignore[assignment]
    return tuple(tuple(int(v) for v in row) for row in rows)


def _normalize_action_name(raw: object) -> str:
    if hasattr(raw, "name"):
        return str(raw.name)
    text = str(raw)
    if text.isdigit():
        if text == "0":
            return "RESET"
        return f"ACTION{text}"
    if text.startswith("ACTION") or text == "RESET":
        return text
    return text


def parse_observation(raw: FrameDataRaw | FrameData) -> ParsedObservation:
    frame_layers = tuple(_to_layer_tuple(layer) for layer in raw.frame)
    return ParsedObservation(
        game_id=raw.game_id,
        state=raw.state.name,
        levels_completed=int(raw.levels_completed),
        win_levels=int(raw.win_levels),
        guid=str(raw.guid or ""),
        full_reset=bool(getattr(raw, "full_reset", False)),
        available_actions=tuple(_normalize_action_name(a) for a in raw.available_actions),
        frame_layers=frame_layers,
    )

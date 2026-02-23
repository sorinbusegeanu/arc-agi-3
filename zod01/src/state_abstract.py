from __future__ import annotations

import hashlib
import json

from .types import CanonicalState, ParsedObservation


def _grid_shape(frame_layers: tuple[tuple[tuple[int, ...], ...], ...]) -> tuple[int, int]:
    if not frame_layers:
        return (0, 0)
    h = len(frame_layers[0])
    w = len(frame_layers[0][0]) if h else 0
    return (h, w)


def canonicalize_state(obs: ParsedObservation) -> CanonicalState:
    payload_obj = {
        "state": obs.state,
        "levels_completed": obs.levels_completed,
        "win_levels": obs.win_levels,
        "available_actions": obs.available_actions,
        "frame_layers": obs.frame_layers,
    }
    payload = json.dumps(payload_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    state_hash = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return CanonicalState(
        state_hash=state_hash,
        payload=payload,
        grid_shape=_grid_shape(obs.frame_layers),
        state=obs.state,
        levels_completed=obs.levels_completed,
        win_levels=obs.win_levels,
        available_actions=obs.available_actions,
    )

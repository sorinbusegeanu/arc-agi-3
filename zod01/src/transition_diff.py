from __future__ import annotations

from .types import CanonicalState, TransitionDelta


def _unpack_grid(s: CanonicalState) -> tuple[tuple[int, ...], ...]:
    # Use first frame layer as primary world state.
    import json

    parsed = json.loads(s.payload.decode("utf-8"))
    layers = parsed.get("frame_layers") or []
    if not layers:
        return tuple()
    return tuple(tuple(int(v) for v in row) for row in layers[0])


def compute_delta(prev: CanonicalState, curr: CanonicalState) -> TransitionDelta:
    g0 = _unpack_grid(prev)
    g1 = _unpack_grid(curr)
    if not g0 or not g1:
        return TransitionDelta(0, 0, 0.0, True, True, ("empty-grid",))

    h = min(len(g0), len(g1))
    w = min(len(g0[0]), len(g1[0]))
    total = h * w
    changed = 0
    for y in range(h):
        r0 = g0[y]
        r1 = g1[y]
        for x in range(w):
            if r0[x] != r1[x]:
                changed += 1

    no_op = changed == 0 and prev.state == curr.state and prev.levels_completed == curr.levels_completed
    reversible_guess = not no_op and changed <= max(1, total // 32)

    tags: list[str] = []
    if no_op:
        tags.append("no-op")
    if curr.levels_completed > prev.levels_completed:
        tags.append("progress")
    if curr.state != prev.state:
        tags.append(f"state:{prev.state}->{curr.state}")

    return TransitionDelta(
        changed_cells=changed,
        total_cells=total,
        change_ratio=(changed / total) if total else 0.0,
        no_op=no_op,
        reversible_guess=reversible_guess,
        tags=tuple(tags),
    )

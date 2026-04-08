from __future__ import annotations

from v4_5.contracts.boardObject import BoardObject


def extract_clickable_items(frame: tuple[tuple[int, ...], ...]) -> tuple[BoardObject, ...]:
    cells = tuple((x, y) for y, row in enumerate(frame) for x, value in enumerate(row) if int(value) == 7)
    return tuple(
        BoardObject(
            object_id=f"clickable:{index}",
            object_type="clickable_item",
            bbox=(x, y, x, y),
            center=(float(x), float(y)),
            position_x=float(x),
            position_y=float(y),
            color=7,
        )
        for index, (x, y) in enumerate(cells)
    )


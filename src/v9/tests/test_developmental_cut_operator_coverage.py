from __future__ import annotations

from v9.runtime.developmental_cut import REQUIRED_DEVELOPMENTAL_OPERATORS, default_developmental_pipeline


def test_registry_covers_every_actor_visible_operator() -> None:
    assert tuple(row.name for row in default_developmental_pipeline().operators) == REQUIRED_DEVELOPMENTAL_OPERATORS

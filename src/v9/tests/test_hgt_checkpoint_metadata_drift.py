from __future__ import annotations

from v9.hgt import training


def _edges(count: int, source: str, target: str) -> dict[tuple[str, str, str], object]:
    return {
        (source, f"REL_{index:03d}", target): object()
        for index in range(count)
    }


def test_hgt_training_cut_can_expand_relation_metadata_within_one_run() -> None:
    old_x = {"M0_EPISODE": object(), "TEXT": object()}
    old_edges = _edges(110, "M0_EPISODE", "TEXT")
    old_metadata = training._metadata_for_graph(old_x, old_edges)

    new_x = {
        "M0_EPISODE": object(),
        "M1_NORMALIZED_RELATION": object(),
        "ACTION": object(),
        "STATE": object(),
        "EFFECT": object(),
    }
    new_edges = _edges(135, "M1_NORMALIZED_RELATION", "ACTION")
    new_metadata = training._metadata_for_graph(new_x, new_edges)

    assert len(old_metadata[1]) == 110
    assert len(new_metadata[1]) == 135
    assert old_metadata != new_metadata
    assert "TEXT" in old_metadata[0]
    assert "TEXT" not in new_metadata[0]
    assert "M1_NORMALIZED_RELATION" in new_metadata[0]


def test_checkpoint_metadata_must_match_current_hgt_architecture() -> None:
    checkpoint_metadata = (
        ["M0_EPISODE", "TEXT"],
        sorted(_edges(110, "M0_EPISODE", "TEXT")),
    )
    current_metadata = (
        ["ACTION", "EFFECT", "M0_EPISODE", "M1_NORMALIZED_RELATION", "STATE"],
        sorted(_edges(135, "M1_NORMALIZED_RELATION", "ACTION")),
    )

    assert checkpoint_metadata != current_metadata

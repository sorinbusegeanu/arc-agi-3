from __future__ import annotations

import numpy as np
import pytest

from v11.tools.vision import changed_cell_summary, compare_regions


@pytest.mark.parametrize(
    ("transformation", "source", "expected"),
    [
        ("identity", [[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ("mirror_horizontal", [[1, 2], [3, 4]], [[2, 1], [4, 3]]),
        ("mirror_vertical", [[1, 2], [3, 4]], [[3, 4], [1, 2]]),
        ("rotate90", [[1, 2, 3], [4, 5, 6]], [[4, 1], [5, 2], [6, 3]]),
        ("rotate180", [[1, 2], [3, 4]], [[4, 3], [2, 1]]),
        ("rotate270", [[1, 2, 3], [4, 5, 6]], [[3, 6], [2, 5], [1, 4]]),
    ],
)
def test_compare_regions_matches_transformations_including_dimension_swaps(
    transformation: str, source: list[list[int]], expected: list[list[int]],
) -> None:
    source_array, expected_array = np.array(source), np.array(expected)
    frame = np.zeros((8, 12), dtype=np.int64)
    frame[1:1 + source_array.shape[0], 1:1 + source_array.shape[1]] = source_array
    frame[2:2 + expected_array.shape[0], 7:7 + expected_array.shape[1]] = expected_array
    result = compare_regions(
        frame, [1, 1, 1 + source_array.shape[0], 1 + source_array.shape[1]],
        [2, 7, 2 + expected_array.shape[0], 7 + expected_array.shape[1]], transformation,
    )
    assert result["shape_compatible"] is True
    assert result["transformed_match"] is True
    assert result["mismatched_cells"] == 0
    assert result["mismatches"] == []


def test_compare_regions_reports_bounded_native_coordinate_mappings_and_shape_mismatch() -> None:
    frame = np.zeros((10, 10), dtype=np.int64)
    frame[1:3, 1:3] = [[1, 2], [3, 4]]
    frame[5:7, 5:7] = [[1, 9], [3, 4]]
    result = compare_regions(frame, [1, 1, 3, 3], [5, 5, 7, 7], max_mismatches=1)
    assert result["mismatched_cells"] == 1
    assert result["mismatches_truncated"] is False
    assert result["mismatches"][0] == {
        "source_xy": [2, 1], "target_xy": [6, 5],
        "transformed_source_value": 2, "target_value": 9,
    }
    incompatible = compare_regions(frame, [1, 1, 3, 3], [5, 5, 6, 8], "rotate90")
    assert incompatible["shape_compatible"] is False
    assert incompatible["mismatched_cells"] is None
    with pytest.raises(ValueError, match="in-frame"):
        compare_regions(frame, [0, 0, 2, 2], [9, 9, 11, 11])


def test_changed_cell_summary_separates_masked_ui_from_board_and_bounds_details() -> None:
    before = np.zeros((4, 8), dtype=np.int64)
    after = before.copy()
    after[0, 0] = 1
    after[3, 7] = 2
    result = changed_cell_summary(
        before, after,
        ui_mask={"frame_shape": [4, 8], "excluded_regions": [{"bounds": [0, 0, 1, 1]}]},
        max_cells=1,
    )
    assert result["changed_cells"] == 2
    assert result["board_changed_cells"] == 1
    assert result["ui_changed_cells"] == 1
    assert result["cell_details_truncated"] is True
    assert result["cell_details"][0]["region"] == "ui"
    assert changed_cell_summary(before, after)["board_changed_cells"] is None

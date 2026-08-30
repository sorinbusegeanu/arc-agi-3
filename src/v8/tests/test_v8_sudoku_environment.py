from __future__ import annotations

import unittest

from v8.environments.sudoku_env import (
    SudokuAdapter,
    decode_sudoku_action,
    encode_sudoku_action,
)


class SudokuEnvironmentTests(unittest.TestCase):
    def test_action_round_trip(self):
        for row, column, digit in ((0, 0, 1), (4, 5, 7), (8, 8, 9)):
            token = encode_sudoku_action(row, column, digit)
            self.assertEqual(decode_sudoku_action(token), (row, column, digit))

    def test_seeded_puzzle_has_expected_shape_and_clues(self):
        adapter = SudokuAdapter(seed=3)
        try:
            board = adapter.observe()
            self.assertEqual(len(board), 81)
            self.assertEqual(adapter.clue_count, 36)
            self.assertEqual(sum(value != 0 for value in board), 36)
            self.assertTrue(adapter.available_actions())
            self.assertTrue(adapter.cognitive_boundary_event().continuation)
        finally:
            adapter.close()

    def test_legal_placement_changes_exactly_one_cell(self):
        adapter = SudokuAdapter(seed=5)
        try:
            before = adapter.observe()
            action = adapter.available_actions()[0]
            after = adapter.step(action)
            self.assertEqual(adapter.cognitive_changed_extent(before, after), 1)
            row, column, digit = decode_sudoku_action(action)
            self.assertEqual(after[row * 9 + column], digit)
        finally:
            adapter.close()

    def test_generated_solution_is_reachable_and_positive(self):
        adapter = SudokuAdapter(seed=11)
        try:
            safety = 0
            while adapter.cognitive_boundary_event().continuation:
                safety += 1
                self.assertLessEqual(safety, 81)
                board = adapter.observe()
                action = next(
                    token
                    for token in adapter.available_actions()
                    if (
                        lambda decoded: adapter._solution[decoded[0] * 9 + decoded[1]] == decoded[2]
                    )(decode_sudoku_action(token))
                )
                adapter.step(action)
            self.assertGreater(adapter.cognitive_boundary_event().primary_valence, 0)
            self.assertTrue(adapter.cognitive_target_reached())
        finally:
            adapter.close()

    def test_different_actor_seeds_have_distinct_provenance(self):
        left = SudokuAdapter(seed=1)
        right = SudokuAdapter(seed=2)
        try:
            self.assertNotEqual(left.identity.source_hash, right.identity.source_hash)
        finally:
            left.close()
            right.close()

    def test_reset_produces_new_seeded_instance(self):
        adapter = SudokuAdapter(seed=7)
        try:
            first = adapter.observe()
            second = adapter.reset()
            self.assertNotEqual(first, second)
            self.assertEqual(sum(value != 0 for value in second), 36)
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()

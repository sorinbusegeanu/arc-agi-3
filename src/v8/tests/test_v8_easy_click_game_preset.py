from __future__ import annotations

import unittest
from unittest.mock import patch

from v7.game_sets import EASY_CLICK_GAMES, V7_GAME_PRESETS, resolve_game_selector


EXPECTED_EASY_CLICK_GAMES = ("gp01", "gp02")


class EasyClickGamePresetTests(unittest.TestCase):
    def test_easy_click_preset_has_two_simple_click_games(self) -> None:
        self.assertEqual(EASY_CLICK_GAMES, EXPECTED_EASY_CLICK_GAMES)
        self.assertEqual(V7_GAME_PRESETS["easy_click"], EXPECTED_EASY_CLICK_GAMES)

    def test_easy_click_selector_resolves_through_v8_cli_selector_path(self) -> None:
        with patch("v7.game_sets.registered_game_ids", return_value=EXPECTED_EASY_CLICK_GAMES):
            self.assertEqual(resolve_game_selector("easy_click"), EXPECTED_EASY_CLICK_GAMES)


if __name__ == "__main__":
    unittest.main()

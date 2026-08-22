from __future__ import annotations

import unittest
from unittest.mock import patch

from v7.environment.arc_adapter import ArcGridEnvironment
from v7.game_sets import (
    LEARNING_GAMES,
    LEARNING_GP_GAMES,
    LEARNING_PURE_CLICK_GAMES,
    V7_GAME_PRESETS,
    resolve_game_selector,
)
from v8 import click_exploration_v848 as click


EXPECTED_LEARNING_GAMES = (
    "pb02", "pb03", "sk01", "sk02", "sk03", "ci01", "op01",
    "ic02", "ic03", "nw01", "nw02", "nw03", "tc01",
    "tb02", "tb03", "cr01", "wl01", "rn01",
    "ul02", "ul03", "fs02", "fs03", "tp02", "tp03", "ex02", "ex03",
    "fi01", "hz01", "vi01",
    "as01", "tw01", "cq01", "ez01", "ez02", "ez03", "ez04",
    "cv01", "dm01", "mm01", "pt01", "sq01",
    "gp01", "gp02", "gp03", "gp04",
)

EXPECTED_PURE_CLICK_GAMES = ("cv01", "dm01", "mm01", "pt01", "sq01")
EXPECTED_GP_GAMES = ("gp01", "gp02", "gp03", "gp04")


class LearningGamePresetTests(unittest.TestCase):
    def test_learning_preset_has_exact_requested_games(self) -> None:
        self.assertEqual(LEARNING_GAMES, EXPECTED_LEARNING_GAMES)
        self.assertEqual(V7_GAME_PRESETS["learning"], EXPECTED_LEARNING_GAMES)
        self.assertEqual(len(LEARNING_GAMES), 45)
        self.assertEqual(len(set(LEARNING_GAMES)), 45)

    def test_learning_additions_have_requested_click_scopes(self) -> None:
        self.assertEqual(LEARNING_PURE_CLICK_GAMES, EXPECTED_PURE_CLICK_GAMES)
        self.assertEqual(LEARNING_GP_GAMES, EXPECTED_GP_GAMES)
        for game_id in LEARNING_PURE_CLICK_GAMES:
            environment = ArcGridEnvironment(game_id=game_id, seed=0)
            environment.game_wait_seconds = 0.0
            actions = tuple(environment.available_actions())
            self.assertTrue(actions, game_id)
            self.assertTrue(
                all(click._is_click_token(action) for action in actions),
                game_id,
            )

    def test_learning_selector_resolves_through_v8_cli_selector_path(self) -> None:
        with patch("v7.game_sets.registered_game_ids", return_value=EXPECTED_LEARNING_GAMES):
            self.assertEqual(resolve_game_selector("learning"), EXPECTED_LEARNING_GAMES)


if __name__ == "__main__":
    unittest.main()

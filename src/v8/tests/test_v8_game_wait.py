from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v7.environment.arc_adapter import ArcGridEnvironment
from v8.cli import main


class _FakeArcEnv:
    def __init__(self, steps):
        self._steps = iter(steps)

    def reset(self):
        return SimpleNamespace(
            frame=np.zeros((2, 2), dtype=np.int64),
            state="NOT_FINISHED",
            levels_completed=0,
            available_actions=[1],
        )

    def step(self, _action):
        return next(self._steps)


class GameWaitTests(unittest.TestCase):
    def test_wait_occurs_only_after_complete_game_terminal_state(self) -> None:
        level_complete = SimpleNamespace(
            frame=np.ones((2, 2), dtype=np.int64),
            state="NOT_FINISHED",
            levels_completed=1,
            available_actions=[1],
        )
        game_complete = SimpleNamespace(
            frame=np.ones((2, 2), dtype=np.int64),
            state="WIN",
            levels_completed=5,
            available_actions=[1],
        )
        fake = _FakeArcEnv([level_complete, game_complete])
        factory = lambda **_kwargs: fake

        with patch.dict(os.environ, {"ARC_AGI3_GAME_WAIT_SECONDS": "1"}, clear=False):
            env = ArcGridEnvironment(game_id="fake", env_factory=factory)
            with patch("v7.environment.arc_adapter.time.sleep") as sleep:
                env.step(1)
                sleep.assert_not_called()
                env.step(1)
                sleep.assert_called_once_with(1.0)

    def test_cli_wait_defaults_to_one_second(self) -> None:
        with patch("v8.cli.run_continuous", return_value=0) as run:
            self.assertEqual(main(["continuous-run", "--games", "diverse"]), 0)
        self.assertEqual(run.call_args.args[0].wait, 1.0)

    def test_cli_wait_can_be_overridden(self) -> None:
        with patch("v8.cli.run_continuous", return_value=0) as run:
            self.assertEqual(
                main(["continuous-run", "--games", "diverse", "--wait", "0.25"]),
                0,
            )
        self.assertEqual(run.call_args.args[0].wait, 0.25)


if __name__ == "__main__":
    unittest.main()

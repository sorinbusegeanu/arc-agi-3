from __future__ import annotations

import os
import unittest

import v8
from v8 import click_transition_exploration_v860 as v860


class ClickTransitionExplorationV860RegressionTests(unittest.TestCase):
    def test_invalid_repeat_cap_falls_back_to_bounded_default(self) -> None:
        prior = os.environ.get(v860._REPEAT_CAP_ENV)
        try:
            os.environ[v860._REPEAT_CAP_ENV] = "invalid"
            self.assertEqual(v860._repeat_cap(), v860._DEFAULT_REPEAT_CAP)
            os.environ[v860._REPEAT_CAP_ENV] = "999"
            self.assertEqual(v860._repeat_cap(), 16)
            os.environ[v860._REPEAT_CAP_ENV] = "0"
            self.assertEqual(v860._repeat_cap(), 1)
        finally:
            if prior is None:
                os.environ.pop(v860._REPEAT_CAP_ENV, None)
            else:
                os.environ[v860._REPEAT_CAP_ENV] = prior


if __name__ == "__main__":
    unittest.main()

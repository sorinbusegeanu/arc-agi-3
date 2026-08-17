from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import v8
from v8 import cli_v819


class AdaptiveAllocationCliV819Tests(unittest.TestCase):
    def test_allocation_options_are_applied_only_during_delegated_cli_call(self) -> None:
        env_name = "ARC_AGI3_V8_ALLOCATION_LEASE_STEPS"
        prior = os.environ.pop(env_name, None)
        observed = {}
        try:
            def delegated(argv):
                observed["argv"] = list(argv)
                observed["lease"] = os.environ.get(env_name)
                observed["plateau"] = os.environ.get(
                    "ARC_AGI3_V8_PLATEAU_PRIORITY_ENABLED"
                )
                return 0

            with patch("v8.cli.main", delegated):
                result = cli_v819.main(
                    [
                        "continuous-run",
                        "--games",
                        "learning",
                        "--allocation-lease-steps",
                        "512",
                        "--allocation-plateau-priority",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                observed["argv"],
                ["continuous-run", "--games", "learning"],
            )
            self.assertEqual(observed["lease"], "512")
            self.assertEqual(observed["plateau"], "1")
            self.assertNotIn(env_name, os.environ)
            self.assertNotIn("ARC_AGI3_V8_PLATEAU_PRIORITY_ENABLED", os.environ)
        finally:
            if prior is not None:
                os.environ[env_name] = prior

    def test_invalid_allocation_weight_is_rejected_before_base_cli(self) -> None:
        with patch("v8.cli.main") as delegated:
            with self.assertRaises(ValueError):
                cli_v819.main(
                    [
                        "continuous-run",
                        "--games",
                        "learning",
                        "--allocation-stable-weight",
                        "0",
                    ]
                )
            delegated.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8 import primary_valence
from v8 import progress_reporting_v054 as progress


class ProgressReportingV054Tests(unittest.TestCase):
    def test_first_win_step_uses_producer_sequence_and_ignores_retry_duplicates(self) -> None:
        original_base = progress._BASE_EXPERIENCE
        original_active = primary_valence._CAPTURE_ACTIVE
        try:
            progress._BASE_EXPERIENCE = lambda **kwargs: SimpleNamespace(**kwargs)
            primary_valence._CAPTURE_ACTIVE = True
            progress._FIRST_PRODUCER_SEQUENCE = 0
            progress._FIRST_WIN_STEP = 0

            progress._experience_with_progress(producer_sequence=1001, terminal_polarity=0)
            progress._experience_with_progress(producer_sequence=1002, terminal_polarity=0)
            # Same producer sequence models a ring-buffer retry for the same action.
            progress._experience_with_progress(producer_sequence=1002, terminal_polarity=0)
            progress._experience_with_progress(producer_sequence=1003, terminal_polarity=1)
            progress._experience_with_progress(producer_sequence=1004, terminal_polarity=1)

            self.assertEqual(progress._FIRST_WIN_STEP, 3)
        finally:
            progress._BASE_EXPERIENCE = original_base
            primary_valence._CAPTURE_ACTIVE = original_active
            progress._FIRST_PRODUCER_SEQUENCE = 0
            progress._FIRST_WIN_STEP = 0


if __name__ == "__main__":
    unittest.main()

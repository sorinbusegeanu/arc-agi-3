from __future__ import annotations

import unittest
from unittest.mock import patch

import v8  # noqa: F401 - install production runtime stack
from v8 import adaptive_memory_transfer_grounding_v856 as grounding
from v8 import adaptive_memory_transfer_integrity_v856 as v856
from v8 import environment_neutrality_v837 as v837
from v8 import sampling_transfer_v833 as transfer
from v8.model import MemoryUid


class AdaptiveMemoryTransferGroundingV856Tests(unittest.TestCase):
    def setUp(self) -> None:
        v856._TARGET_TRANSFER_FAILURES.clear()

    def tearDown(self) -> None:
        v856._TARGET_TRANSFER_FAILURES.clear()

    def test_final_grounded_path_applies_target_local_penalty_once(self) -> None:
        uid = MemoryUid(7, 31)
        v856._TARGET_TRANSFER_FAILURES[v856._transfer_failure_key("target", uid)] = 2
        m7 = {2: ((3.0, uid, "M7_CORRESPONDENCE"),)}
        m1n = {5: ((1.0, None, "M1N_GROUNDED"),)}
        with patch.object(grounding, "_BASE_GROUNDED_TRANSFER", return_value=(m7, m1n)):
            adjusted_m7, adjusted_m1n = grounding._grounded_transfer_v856(object(), "target")
        self.assertEqual(adjusted_m1n, m1n)
        self.assertAlmostEqual(adjusted_m7[2][0][0], 2.70)

    def test_final_grounded_path_blocks_repeated_failed_foreign_strategy(self) -> None:
        blocked = MemoryUid(7, 32)
        viable = MemoryUid(7, 33)
        v856._TARGET_TRANSFER_FAILURES[
            v856._transfer_failure_key("target", blocked)
        ] = v856._TRANSFER_BACKOFF_LIMIT
        m7 = {
            2: (
                (4.0, blocked, "M7_CORRESPONDENCE"),
                (3.0, viable, "M7_CORRESPONDENCE"),
            )
        }
        with patch.object(grounding, "_BASE_GROUNDED_TRANSFER", return_value=(m7, {})):
            adjusted, _m1n = grounding._grounded_transfer_v856(object(), "target")
        self.assertEqual(len(adjusted[2]), 1)
        self.assertEqual(adjusted[2][0][1], viable)

    def test_v837_and_v833_use_same_final_grounded_authority(self) -> None:
        self.assertIs(v837._grounded_transfer_index, grounding._grounded_transfer_v856)
        self.assertIs(v837._grounded_m7_index_v837, grounding._grounded_m7_v856)
        self.assertIs(transfer._lineage_transfer_index, grounding._grounded_m7_v856)


if __name__ == "__main__":
    unittest.main()

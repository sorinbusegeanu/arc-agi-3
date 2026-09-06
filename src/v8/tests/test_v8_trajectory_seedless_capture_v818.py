from __future__ import annotations

import unittest
from types import SimpleNamespace

import v8
from v8.environments.schemas import EnvironmentIdentity
from v8.model import stable_u64
from v8.persistent_identity import trajectory_identity
from v8 import trajectory_optimizer_v814 as optimizer


class SeedlessTrajectoryCaptureV818Tests(unittest.TestCase):
    def test_capture_does_not_retain_actor_seed(self) -> None:
        prior = (
            optimizer._CAPTURE_ACTIVE,
            optimizer._CAPTURE_SOURCE_ID,
            optimizer._CAPTURE_SEED,
            optimizer._CAPTURE_ENV_ROOT,
            list(optimizer._CAPTURE_PREFIX),
            list(optimizer._CAPTURE_SEGMENT),
            list(optimizer._ACTOR_ACTION_HISTORY),
        )
        try:
            optimizer._reset_capture(
                SimpleNamespace(game_id="world", seed=987654, env_root=None)
            )
            self.assertTrue(optimizer._CAPTURE_ACTIVE)
            self.assertEqual(optimizer._CAPTURE_SOURCE_ID, "world")
            self.assertEqual(optimizer._CAPTURE_SEED, 0)
        finally:
            optimizer._CAPTURE_ACTIVE = prior[0]
            optimizer._CAPTURE_SOURCE_ID = prior[1]
            optimizer._CAPTURE_SEED = prior[2]
            optimizer._CAPTURE_ENV_ROOT = prior[3]
            optimizer._CAPTURE_PREFIX = prior[4]
            optimizer._CAPTURE_SEGMENT = prior[5]
            optimizer._ACTOR_ACTION_HISTORY = prior[6]

    def test_actor_trajectory_signature_ignores_execution_seed(self) -> None:
        first_world = EnvironmentIdentity(
            "gymnasium", "FrozenLake-v1", "default", "seed=101"
        ).source_hash
        second_world = EnvironmentIdentity(
            "gymnasium", "FrozenLake-v1", "default", "seed=909"
        ).source_hash
        first = trajectory_identity(
            first_world,
            producer_id=4,
            episode_ordinal=0,
            sequence_base=33,
            namespace=b"v8-traj-seed",
        )
        second = trajectory_identity(
            second_world,
            producer_id=4,
            episode_ordinal=0,
            sequence_base=33,
            namespace=b"v8-traj-seed",
        )
        self.assertEqual(first, second)

    def test_other_actor_hashes_still_include_all_parts(self) -> None:
        first = stable_u64(7, 101, 33, person=b"v8-other")
        second = stable_u64(7, 909, 33, person=b"v8-other")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()

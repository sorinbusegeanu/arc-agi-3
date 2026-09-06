from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v8.arena import EdgeRecord, NodeRecord
from v8.environments.schemas import EnvironmentIdentity
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    stable_u64,
)
from v8.persistent_identity import (
    PERSISTENT_IDENTITY_MARKER,
    arc_world_id,
    prepare_persistent_identity_root,
    trajectory_identity,
)
from v8.transfer import TransferValidator


def _role(index: int) -> NodeRecord:
    uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (index,))
    return NodeRecord(
        uid=uid,
        fingerprint=index,
        level=int(MemoryLevel.M3),
        memory_type=int(MemoryType.ROLE),
        key_parts=(index,),
        support_count=4,
        significance_sum=1.0,
        prediction_error_sum=0.0,
        learning_value_sum=1.0,
        transfer_prior_sum=1.0,
        explanatory_sum=1.0,
        future_option_sum=0.0,
        score_weight=1.0,
        updated_watermark=index,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.STRUCTURAL),
    )


class PersistentIdentityTests(unittest.TestCase):
    def test_two_seeds_share_one_environment_world_id(self) -> None:
        first = EnvironmentIdentity(
            "gymnasium", "FrozenLake-v1", "is_slippery=False", "seed=12"
        )
        second = EnvironmentIdentity(
            "gymnasium", "FrozenLake-v1", "is_slippery=False", "seed=47"
        )

        self.assertEqual(first.source_hash, second.source_hash)
        self.assertEqual(first.instance_id, second.instance_id)
        self.assertEqual(first.instance, "default")
        self.assertEqual(second.instance, "default")

    def test_different_environments_keep_distinct_world_ids(self) -> None:
        frozen = EnvironmentIdentity("gymnasium", "FrozenLake-v1", "default")
        chess = EnvironmentIdentity("chess", "ArcAgi/Chess-v0", "default")

        self.assertNotEqual(frozen.source_hash, chess.source_hash)

    def test_arc_world_identity_is_bit_for_bit_unchanged(self) -> None:
        for game_id in ("tp01", "gp03", "Sudoku-v0"):
            self.assertEqual(
                arc_world_id(game_id),
                stable_u64(game_id, person=b"v8-game"),
            )

    def test_transfer_provenance_distinct_uses_seed_free_worlds(self) -> None:
        left = _role(1)
        right = _role(2)
        edge = EdgeRecord(
            left.uid,
            int(RelationType.TRANSFER_CORRESPONDENCE),
            right.uid,
            1,
            1,
            1.0,
            1.0,
        )
        frozen_seed_12 = EnvironmentIdentity(
            "gymnasium", "FrozenLake-v1", "default", "seed=12"
        ).source_hash
        frozen_seed_47 = EnvironmentIdentity(
            "gymnasium", "FrozenLake-v1", "default", "seed=47"
        ).source_hash
        chess = EnvironmentIdentity(
            "chess", "ArcAgi/Chess-v0", "default", "seed=12"
        ).source_hash

        same_world = {left.uid: frozen_seed_12, right.uid: frozen_seed_47}
        self.assertEqual(
            TransferValidator().candidates(
                (left, right),
                (edge,),
                provenance=lambda uid: frozenset((same_world[uid],)),
            ),
            (),
        )

        distinct_worlds = {left.uid: frozen_seed_12, right.uid: chess}
        candidates = TransferValidator().candidates(
            (left, right),
            (edge,),
            provenance=lambda uid: frozenset((distinct_worlds[uid],)),
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(
            all(
                set(row.formation_games) != set(row.correspondence_games)
                for row in candidates
            )
        )

    def test_replay_and_deduplication_identity_is_deterministic(self) -> None:
        world = EnvironmentIdentity(
            "gymnasium", "FrozenLake-v1", "default", "seed=12"
        ).source_hash
        first = trajectory_identity(
            world,
            producer_id=7,
            episode_ordinal=3,
            sequence_base=100,
            namespace=b"v8-test-trajectory",
        )
        second = trajectory_identity(
            world,
            producer_id=7,
            episode_ordinal=3,
            sequence_base=100,
            namespace=b"v8-test-trajectory",
        )

        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            trajectory_identity(
                world,
                producer_id=7,
                episode_ordinal=4,
                sequence_base=100,
                namespace=b"v8-test-trajectory",
            ),
        )

    def test_legacy_store_requires_explicit_recoverable_reset(self) -> None:
        with tempfile.TemporaryDirectory() as raw_parent:
            root = Path(raw_parent) / "continuous"
            legacy = root / "snapshots" / "snapshot-legacy"
            legacy.mkdir(parents=True)
            (legacy / "COMPLETE").write_text("legacy\n", encoding="ascii")

            with self.assertRaisesRegex(RuntimeError, "seed-scoped provenance"):
                prepare_persistent_identity_root(root)

            archive = prepare_persistent_identity_root(root, reset_legacy=True)

            self.assertIsNotNone(archive)
            assert archive is not None
            self.assertTrue((archive / "snapshots" / "snapshot-legacy" / "COMPLETE").is_file())
            self.assertTrue((root / PERSISTENT_IDENTITY_MARKER).is_file())
            self.assertFalse((root / "snapshots").exists())


if __name__ == "__main__":
    unittest.main()

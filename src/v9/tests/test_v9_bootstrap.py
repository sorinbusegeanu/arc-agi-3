from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError

from v8.environments.schemas import EnvironmentIdentity
from v9.environment_registry import EnvironmentIdentityRegistry
from v9.modalities.symbols import DeterministicSymbolCodec, ModalityId
from v9.scientific_config import (
    BASELINE_GIT_COMMIT,
    ScientificConfig,
    write_scientific_config_manifest,
)


class V9BootstrapTests(unittest.TestCase):
    def test_scientific_config_is_immutable_and_id_is_deterministic(self) -> None:
        first = ScientificConfig(runtime_stack_layers=("a", "b"))
        second = ScientificConfig(runtime_stack_layers=("a", "b"))
        self.assertEqual(first.config_id, second.config_id)
        self.assertEqual(len(first.config_id), 64)
        with self.assertRaises(FrozenInstanceError):
            first.design_version = "changed"  # type: ignore[misc]

    def test_current_capture_records_live_v8_authorities(self) -> None:
        config = ScientificConfig.capture_current()
        self.assertEqual(config.implementation_baseline_git_commit, BASELINE_GIT_COMMIT)
        self.assertIn("information_flow_integrity_v879", config.runtime_stack_layers)
        self.assertGreaterEqual(len(config.arena_packet_sizes), 4)
        self.assertGreaterEqual(len(config.arena_record_sizes), 3)
        self.assertTrue(all(size > 0 for _name, size in config.arena_packet_sizes))
        self.assertTrue(all(size > 0 for _name, size in config.arena_record_sizes))
        self.assertIn("v8.cli_v819.main", config.default_cli_entrypoint)

    def test_manifest_rejects_different_config_in_same_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = ScientificConfig(runtime_stack_layers=("a",))
            target = write_scientific_config_manifest(directory, first)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["scientific_config_id"], first.config_id)
            write_scientific_config_manifest(directory, first)
            with self.assertRaises(RuntimeError):
                write_scientific_config_manifest(
                    directory,
                    ScientificConfig(runtime_stack_layers=("b",)),
                )

    def test_symbol_ids_are_deterministic_and_vocab_scoped(self) -> None:
        alpha = DeterministicSymbolCodec("alpha")
        alpha_again = DeterministicSymbolCodec("alpha")
        beta = DeterministicSymbolCodec("beta")
        self.assertEqual(alpha.symbol_id("A"), alpha_again.symbol_id("A"))
        self.assertNotEqual(alpha.symbol_id("A"), beta.symbol_id("A"))
        self.assertEqual(ModalityId.WORLD.value, 1)
        self.assertEqual(ModalityId.SYMBOL.value, 2)

    def test_symbol_stream_preserves_order_and_round_trips_codec_identity(self) -> None:
        codec = DeterministicSymbolCodec("synthetic")
        observations = codec.encode_stream(("A7", "B2", "A7"), stream_name="episode-1")
        self.assertEqual([row.position for row in observations], [0, 1, 2])
        self.assertEqual(observations[0].symbol_id, observations[2].symbol_id)
        self.assertNotEqual(observations[0].symbol_id, observations[1].symbol_id)
        restored = DeterministicSymbolCodec.from_state_dict(codec.state_dict())
        self.assertEqual(restored.vocabulary_id, codec.vocabulary_id)
        self.assertEqual(restored.symbol_id("A7"), codec.symbol_id("A7"))

    def test_environment_registry_round_trip_and_source_hash_compatibility(self) -> None:
        identity = EnvironmentIdentity("gym", "FrozenLake-v1", "map=4x4", "seed=7")
        registry = EnvironmentIdentityRegistry()
        instance_id = registry.register(identity)
        self.assertEqual(instance_id, identity.source_hash)
        self.assertEqual(registry.resolve_source_hash(identity.source_hash), identity)
        first_episode = registry.next_episode(instance_id)
        second_episode = registry.next_episode(instance_id)
        self.assertNotEqual(first_episode, second_episode)

        restored = EnvironmentIdentityRegistry.from_state_dict(registry.state_dict())
        self.assertEqual(restored.resolve(instance_id), identity)
        third_episode = restored.next_episode(instance_id)
        self.assertNotIn(third_episode, {first_episode, second_episode})

    def test_environment_registry_detects_tampered_identity(self) -> None:
        identity = EnvironmentIdentity("gym", "FrozenLake-v1", "default", "seed=1")
        registry = EnvironmentIdentityRegistry()
        registry.register(identity)
        state = registry.state_dict()
        rows = state["identities"]
        self.assertIsInstance(rows, list)
        row = rows[0]
        self.assertIsInstance(row, dict)
        row["instance_id"] = int(row["instance_id"]) + 1
        with self.assertRaises(ValueError):
            EnvironmentIdentityRegistry.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()

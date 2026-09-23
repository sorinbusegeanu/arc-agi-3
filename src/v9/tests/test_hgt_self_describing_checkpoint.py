from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v9.hgt.training import (
    MODEL_SCHEMA_VERSION,
    _architecture_sidecar_path,
    _checkpoint_architecture,
    _load_compatible_training_parent,
    _write_architecture_sidecar,
)


class HGTSelfDescribingCheckpointTests(unittest.TestCase):
    def test_checkpoint_architecture_preserves_evolved_relation_shape(self):
        old_edges = [("M0_EPISODE", f"REL_{index}", "M0_EPISODE") for index in range(110)]
        new_edges = old_edges + [
            ("M1_NORMALIZED_RELATION", f"SEMANTIC_{index}", "ACTION")
            for index in range(25)
        ]
        old = _checkpoint_architecture({
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "metadata": (["M0_EPISODE"], old_edges),
            "input_dim": 64,
            "hidden_dim": 64,
            "layers": 3,
            "heads": 4,
        })
        current = (["M0_EPISODE", "M1_NORMALIZED_RELATION", "ACTION"], new_edges)
        self.assertEqual(len(old["metadata"][1]), 110)
        self.assertEqual(len(current[1]), 135)
        self.assertNotEqual(old["metadata"], current)
        self.assertEqual(old["hidden_dim"], 64)
        self.assertEqual(old["layers"], 3)

    def test_architecture_sidecar_round_trips_checkpoint_description(self):
        architecture = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "metadata": (
                ["ACTION", "M1_NORMALIZED_RELATION"],
                [
                    ("ACTION", "SEMANTIC_OF", "M1_NORMALIZED_RELATION"),
                    ("M1_NORMALIZED_RELATION", "SEMANTIC", "ACTION"),
                ],
            ),
            "input_dim": 64,
            "hidden_dim": 64,
            "layers": 3,
            "heads": 4,
        }
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "hgt-000002.pt"
            _write_architecture_sidecar(checkpoint, architecture)
            sidecar = _architecture_sidecar_path(checkpoint)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"][0], architecture["metadata"][0])
            self.assertEqual(
                [tuple(edge) for edge in payload["metadata"][1]],
                architecture["metadata"][1],
            )
            self.assertEqual(payload["hidden_dim"], 64)

    def test_missing_metadata_is_rejected_before_weight_loading(self):
        with self.assertRaisesRegex(RuntimeError, "self-describing metadata"):
            _checkpoint_architecture({
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "input_dim": 64,
                "hidden_dim": 64,
                "layers": 3,
                "heads": 4,
            })

    def test_older_accepted_checkpoint_is_behavior_only_training_parent(self):
        class Runtime:
            def __init__(self):
                self.gauges = {}

            def set_telemetry_gauge(self, name, value):
                self.gauges[name] = value

        runtime = Runtime()
        checkpoint, architecture = _load_compatible_training_parent(
            {"model_schema_version": MODEL_SCHEMA_VERSION - 2, "model_state": object()},
            device=object(),
            model=object(),
            current_architecture={},
            runtime=runtime,
        )
        self.assertIsNone(checkpoint)
        self.assertIsNone(architecture)
        self.assertEqual(runtime.gauges["hgt_training_parent_behavior_only"], 1)
        self.assertEqual(runtime.gauges["hgt_training_parent_schema_version"], MODEL_SCHEMA_VERSION - 2)
        self.assertEqual(runtime.gauges["hgt_training_schema_version"], MODEL_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v9.runtime.runtime import ContinuousMemoryRuntime


class _Telemetry:
    model_version = "untrained"


class _Runtime:
    _restore_hgt_checkpoint = ContinuousMemoryRuntime._restore_hgt_checkpoint

    def __init__(self, root):
        self.root = Path(root)
        self.unified_telemetry = _Telemetry()
        self.scores = None
        self.gauges = {}

    def set_hgt_action_scores(self, scores, **kwargs):
        self.scores = scores

    def set_telemetry_gauge(self, key, value):
        self.gauges[key] = value


class HGTRestoreTests(unittest.TestCase):
    def test_missing_manifest_keeps_runtime_untrained(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = _Runtime(root)
            runtime._restore_hgt_checkpoint()
            self.assertEqual(runtime.unified_telemetry.model_version, "untrained")
            self.assertIsNone(runtime.scores)

    @unittest.skipUnless(__import__('importlib').util.find_spec('torch'), 'torch not installed')
    def test_manifest_restores_accepted_checkpoint_policy(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "models"
            model_dir.mkdir()
            checkpoint = model_dir / "hgt-000002.pt"
            checkpoint.write_bytes(b"checkpoint")
            (model_dir / "hgt_manifest.json").write_text(json.dumps({
                "current_model_version": "hgt-000003",
                "current_checkpoint": "models/hgt-000003.pt",
                "last_accepted_model_version": "hgt-000002",
                "accepted_checkpoint": "models/hgt-000002.pt",
            }))
            runtime = _Runtime(root)
            fake = {"action_scores": {"17": {"2": 0.75}}}
            with patch("torch.load", return_value=fake):
                runtime._restore_hgt_checkpoint()
            self.assertEqual(runtime.unified_telemetry.model_version, "hgt-000002")
            self.assertEqual(runtime.scores, {17: {2: 0.75}})
            self.assertEqual(runtime.gauges["hgt_restored_on_startup"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import dataclass

from v8.learning_fixes_v088_transfer_pass_evidence_fix import (
    _persist_new_passed_trials,
)
from v8.model import MemoryUid


@dataclass
class _Trial:
    target_game_hash: int
    effect: float
    passed: bool
    formation_games: tuple[int, ...]
    intervention: str = "matched_arc_target_memory_vs_memory_free"


class _Transfer:
    effect_threshold = 0.0

    def __init__(self, uid):
        self._trials = {
            uid: [
                _Trial(10, 0.4, True, (1, 2)),
                _Trial(11, 0.0, False, (1, 2)),
            ]
        }


class _ReadView:
    def __init__(self, uid, row):
        self._node_by_uid = {uid: row}

    def _refresh_strategy_cache(self):
        return None


class _Peers:
    def __init__(self, transfer):
        self.transfer = transfer
        self.calls = []

    def _append_evidence(self, kind, row, value, **kwargs):
        self.calls.append((kind, row, value, kwargs))


class _Runtime:
    def __init__(self, uid, row):
        self.peers = _Peers(_Transfer(uid))
        self.read_view = _ReadView(uid, row)


class TransferPassEvidencePersistenceTests(unittest.TestCase):
    def test_only_new_positive_pass_is_persisted(self):
        uid = MemoryUid(1, 2)
        row = object()
        runtime = _Runtime(uid, row)

        written = _persist_new_passed_trials(runtime, {uid: 0})

        self.assertEqual(written, 1)
        self.assertEqual(len(runtime.peers.calls), 1)
        kind, persisted_row, value, fields = runtime.peers.calls[0]
        self.assertEqual(kind, "transfer_trial_pass")
        self.assertIs(persisted_row, row)
        self.assertEqual(value, 0.4)
        self.assertEqual(fields["target_game_hash"], 10)
        self.assertEqual(fields["provenance_games"], (1, 2))
        self.assertEqual(fields["causal_intervention"], "matched_arc_target_memory_vs_memory_free")
        self.assertEqual(fields["effect_direction"], 1)
        self.assertTrue(fields["unique"])

    def test_prior_trials_are_not_rewritten(self):
        uid = MemoryUid(3, 4)
        runtime = _Runtime(uid, object())

        written = _persist_new_passed_trials(runtime, {uid: 2})

        self.assertEqual(written, 0)
        self.assertEqual(runtime.peers.calls, [])


if __name__ == "__main__":
    unittest.main()

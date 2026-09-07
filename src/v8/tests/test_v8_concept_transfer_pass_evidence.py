from types import SimpleNamespace

from v8.learning_fixes_v088_transfer_pass_evidence_fix import _persist_new_passed_trials
from v8.model import MemoryLevel, MemoryUid


def _trial(*, passed=True, effect=0.8, target=123, formation=(1, 2)):
    return SimpleNamespace(
        passed=passed,
        effect=effect,
        target_game_hash=target,
        formation_games=formation,
        intervention="matched_arc_target_memory_vs_memory_free",
    )


def _runtime(level, trials):
    uid = MemoryUid(1, 2)
    row = SimpleNamespace(uid=uid, level=level)
    emitted = []

    class Peers:
        transfer = SimpleNamespace(_trials={uid: list(trials)}, effect_threshold=0.0)

        @staticmethod
        def _append_evidence(kind, row_arg, value, **kwargs):
            emitted.append((kind, row_arg, value, kwargs))

    read_view = SimpleNamespace(_node_by_uid={uid: row})
    return SimpleNamespace(peers=Peers(), read_view=read_view), emitted, uid


def test_m4_pass_persists_concept_transfer_pass():
    runtime, emitted, uid = _runtime(MemoryLevel.M4, [_trial()])

    written = _persist_new_passed_trials(runtime, {})

    assert written == 1
    assert [item[0] for item in emitted] == [
        "transfer_trial_pass",
        "concept_transfer_pass",
    ]
    for _kind, _row, value, fields in emitted:
        assert value == 0.8
        assert fields["target_game_hash"] == 123
        assert fields["provenance_games"] == (1, 2)
        assert fields["causal_intervention"] == "matched_arc_target_memory_vs_memory_free"
        assert fields["effect_direction"] == 1
        assert fields["unique"] is True


def test_non_m4_pass_does_not_persist_concept_transfer_pass():
    runtime, emitted, uid = _runtime(MemoryLevel.M3, [_trial()])

    _persist_new_passed_trials(runtime, {})

    assert [item[0] for item in emitted] == ["transfer_trial_pass"]


def test_failed_or_previous_trials_do_not_create_concept_pass():
    runtime, emitted, uid = _runtime(
        MemoryLevel.M4,
        [_trial(passed=True), _trial(passed=False, effect=0.0, target=456)],
    )

    written = _persist_new_passed_trials(runtime, {uid: 1})

    assert written == 0
    assert emitted == []

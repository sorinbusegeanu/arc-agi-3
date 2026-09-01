from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v8 import information_flow_diagnostics as flow
from v8 import learning_fixes_v088 as learning
from v8.arena import NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    ValidationState,
    stable_u64,
)
from v8.publication import PlannedAction, _StrategyRow
from v8.transfer import TransferCandidate, TransferValidator


def _memory(
    index: int,
    level: MemoryLevel,
    memory_type: MemoryType,
    key_parts: tuple[int, ...] | None = None,
) -> NodeRecord:
    key = key_parts or (index,)
    return NodeRecord(
        uid=MemoryUid.from_key(level, memory_type, key),
        fingerprint=index,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
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


def _candidate(row: NodeRecord) -> TransferCandidate:
    return TransferCandidate(
        row.uid,
        1,
        1.0,
        (stable_u64("source", person=b"v8-game"),),
        MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (999,)),
        (stable_u64("correspondence", person=b"v8-game"),),
    )


def _correspondence_ancestor(index: int = 999) -> NodeRecord:
    return _memory(index, MemoryLevel.M3, MemoryType.ROLE, (999,))


def _empty_inventory() -> dict[str, object]:
    return {
        "identity_index": "GAME_PROVENANCE target UID low word",
        "identities_actually_present_count": 0,
        "memory_counts_by_level": {},
        "m3_exists": False,
        "m4_exists": False,
        "executable_lower_level_memory_exists": False,
    }


def test_target_memory_exists_and_is_executable() -> None:
    ancestor = _memory(1, MemoryLevel.M3, MemoryType.ROLE)
    outcome = _memory(2, MemoryLevel.M6, MemoryType.OUTCOME)
    strategy = _memory(
        3,
        MemoryLevel.M7,
        MemoryType.STRATEGY,
        (1, outcome.uid.hi, outcome.uid.lo, 77),
    )
    cache_row = _StrategyRow(1, outcome.uid, strategy.uid, 4, 0.8, 2.0, 77, False, True)
    view = SimpleNamespace(
        _node_by_uid={row.uid: row for row in (ancestor, outcome, strategy)},
        _parents={strategy.uid: {ancestor.uid}},
        _strategy_by_context={77: (cache_row,)},
        _strategy_fallback=(cache_row,),
    )

    snapshot = learning._candidate_execution_snapshot(
        view, (ancestor, outcome, strategy), ancestor.uid
    )
    event = learning._target_resolution_event(
        read_view=view,
        nodes=(ancestor, outcome, strategy),
        candidate=_candidate(ancestor),
        game_id="target",
        target_hash=stable_u64("target", person=b"v8-game"),
        probe_diagnostic={"available_action_ids": [1], "observed_context_buckets": [77]},
        candidate_snapshot=snapshot,
        target_inventory=_empty_inventory(),
        trajectory={"successful_trajectory_exists": False},
        executable_reference=learning._executable_reference(view, (ancestor, outcome, strategy)),
        used=1,
    )

    assert snapshot["cached_executable_descendant_count"] == 1
    assert event["executable_predicate_result"] is True
    assert event["exact_executable_predicate_failure_reason"] is None


def test_target_memory_exists_but_fails_executable_predicate() -> None:
    ancestor = _memory(10, MemoryLevel.M3, MemoryType.ROLE)
    view = SimpleNamespace(
        _node_by_uid={ancestor.uid: ancestor},
        _parents={},
        _strategy_by_context={},
        _strategy_fallback=(),
    )
    snapshot = learning._candidate_execution_snapshot(view, (ancestor,), ancestor.uid)

    assert snapshot["lookup_result_status"] == "found_structural_ancestor"
    assert snapshot["executable_predicate_failure_reason"] == (
        "no_grounded_action_evidence"
    )
    assert "grounded lineage action evidence or replayable source trajectory" in snapshot[
        "missing_or_invalid_executable_fields"
    ]


def test_target_memory_lookup_returns_no_memory() -> None:
    missing = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (404,))
    view = SimpleNamespace(
        _node_by_uid={}, _parents={}, _strategy_by_context={}, _strategy_fallback=()
    )
    snapshot = learning._candidate_execution_snapshot(view, (), missing)

    assert snapshot["lookup_result_status"] == "not_found"
    assert snapshot["required_ancestor_memory"] is None
    assert snapshot["executable_predicate_failure_reason"] == (
        "required_ancestor_lookup_returned_none"
    )


def test_world_provenance_identity_mismatch_is_visible() -> None:
    source_hash = stable_u64("source", person=b"v8-game")
    target_hash = stable_u64("target", person=b"v8-game")
    ancestor = _memory(20, MemoryLevel.M3, MemoryType.ROLE)
    view = SimpleNamespace(
        _node_by_uid={ancestor.uid: ancestor},
        _parents={},
        _strategy_by_context={},
        _strategy_fallback=(),
        _v839_direct_games={ancestor.uid: {source_hash}},
    )
    inventories = learning._direct_target_memory_inventories(
        view, (ancestor,), (target_hash,)
    )
    snapshot = learning._candidate_execution_snapshot(view, (ancestor,), ancestor.uid)
    event = learning._target_resolution_event(
        read_view=view,
        nodes=(ancestor,),
        candidate=_candidate(ancestor),
        game_id="target",
        target_hash=target_hash,
        probe_diagnostic={},
        candidate_snapshot=snapshot,
        target_inventory=inventories[target_hash],
        trajectory={"successful_trajectory_exists": False},
        executable_reference={},
        used=0,
    )

    assert event["requested_lookup_identity"]["provenance_world_hash"] == target_hash
    assert source_hash in event["candidate_memory_identities_actually_present"][
        "formation_world_hashes"
    ]
    assert event["target_specific_memory_lookup_performed"] is False
    assert event["target_world_memory"]["identities_actually_present_count"] == 0


def test_m3_present_m4_absent_is_diagnosed_explicitly() -> None:
    target_hash = stable_u64("target", person=b"v8-game")
    m3 = _memory(30, MemoryLevel.M3, MemoryType.ROLE)
    view = SimpleNamespace(
        _node_by_uid={m3.uid: m3},
        _v839_direct_games={m3.uid: {target_hash}},
    )
    inventory = learning._direct_target_memory_inventories(
        view, (m3,), (target_hash,)
    )[target_hash]

    assert inventory["m3_exists"] is True
    assert inventory["m4_exists"] is False
    assert inventory["memory_counts_by_level"] == {"M3": 1}


def test_successful_trajectory_exists_but_is_not_linked_to_target_memory() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        optimizer_root = Path(raw_root) / "trajectory_optimizer"
        optimizer_root.mkdir(parents=True)
        (optimizer_root / "best_successful.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "games": {
                        "target": {
                            "trajectory_id": "target-win",
                            "successes": 1,
                            "levels": [{"level": 0, "actions": [1, 2]}],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        row = learning._trajectory_inventory(("target",))["target"]

    assert row["successful_trajectory_exists"] is True
    assert row["trajectory_id"] == "target-win"
    assert row["executable_representation_available"] is True
    assert row["linked_memory_id"] is None


def test_resolution_instrumentation_does_not_change_scheduler_result() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        ancestor = _memory(40, MemoryLevel.M3, MemoryType.ROLE)
        candidate = _candidate(ancestor)
        transfer = TransferValidator()
        transfer.candidates = lambda *_args, **_kwargs: (candidate,)
        peers = SimpleNamespace(
            transfer=transfer,
            record_transfer_trial=transfer.record_trial,
            _append_evidence=lambda *_args, **_kwargs: None,
        )
        view = SimpleNamespace(
            node_records=lambda: (ancestor,),
            source_games=lambda _uid: frozenset(),
            _node_by_uid={ancestor.uid: ancestor},
            _parents={},
            _strategy_by_context={},
            _strategy_fallback=(),
            _v839_direct_games={},
        )
        runtime = SimpleNamespace(peers=peers, read_view=view)

        with patch.object(learning, "_held_out_games", return_value=("target",)), patch.object(
            learning, "_probe_policy_v088", return_value=(0.0, 0)
        ):
            result = learning._run_automatic_transfer_experiments_v088(
                runtime,
                games=("source",),
                env_root=None,
                seed=1,
                steps_per_trial=2,
                max_trials=1,
            )

        records = [
            json.loads(line)
            for line in (Path(raw_root) / flow.LOG_NAME).read_text(encoding="utf-8").splitlines()
        ]
        detail = [row for row in records if row["stage"] == "target_memory_resolution"]

    assert (result.attempted, result.completed, result.passed) == (0, 0, 0)
    assert detail[0]["exact_executable_predicate_failure_reason"] == (
        "no_grounded_action_evidence"
    )
    assert detail[0]["target_memory_level"] == "M3"


def test_target_resolution_detail_collection_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ):
        flow.reset_for_tests()
        for index in range(100):
            flow.emit_bounded(
                "transfer",
                "target_memory_resolution",
                input_count=1,
                output_count=0,
                rejection_counts={"not_executable": 1},
                examples=({"target_world": f"target-{index}"},),
            )
        records = [
            json.loads(line)
            for line in (Path(raw_root) / flow.LOG_NAME).read_text(encoding="utf-8").splitlines()
        ]

    assert len(records) == flow.MAX_EXAMPLES


class _TransferProbeEnvironment:
    def __init__(self, **_kwargs):
        self.last_levels_completed = 0
        self.last_outcome_polarity = "neutral"
        self.actions = []

    def observe(self):
        return ((0,),)

    def available_actions(self):
        return (2,)

    def step(self, action):
        self.actions.append(int(action))
        return ((int(action),),)

    def reset(self):
        return ((0,),)


class _EffectProbeEnvironment:
    def __init__(self, *, game_id, seed, **_kwargs):
        self.game_id = str(game_id)
        self.seed = int(seed)
        self.last_levels_completed = 0
        self.last_outcome_polarity = "neutral"
        self.actions = []

    def observe(self):
        return ((0,),)

    def available_actions(self):
        return (1, 2)

    def step(self, action):
        action = int(action)
        self.actions.append(action)
        self.last_outcome_polarity = "positive" if action == 2 else "neutral"
        return ((action,),)

    def reset(self):
        self.last_outcome_polarity = "neutral"
        return ((0,),)


def _transfer_runtime(candidate_row, lineage_rows, parents):
    candidate = _candidate(candidate_row)
    transfer = TransferValidator()
    transfer.candidates = lambda *_args, **_kwargs: (candidate,)
    recorded = []
    evidence = []

    def record(uid, **kwargs):
        trial = transfer.record_trial(uid, **kwargs)
        recorded.append(trial)
        return trial

    class View:
        _node_by_uid = {row.uid: row for row in lineage_rows}
        _parents = parents
        _strategy_by_context = {}
        _strategy_fallback = ()
        _v839_direct_games = {}

        def node_records(self):
            return tuple(lineage_rows)

        def source_games(self, uid):
            return frozenset(candidate.formation_games) if uid == candidate.uid else frozenset()

        def planned_action(self, *_args, **_kwargs):
            return None

    peers = SimpleNamespace(
        transfer=transfer,
        record_transfer_trial=record,
        _append_evidence=lambda kind, *_args, **_kwargs: evidence.append(kind),
    )
    return SimpleNamespace(peers=peers, read_view=View()), recorded, evidence


def test_grounded_m3_and_m4_lineage_schedule_without_m7() -> None:
    grounded = _memory(
        60,
        MemoryLevel.M1,
        MemoryType.CONTINGENCY,
        (101, 2, 202, 303),
    )
    role = _memory(61, MemoryLevel.M3, MemoryType.ROLE)
    concept = _memory(62, MemoryLevel.M4, MemoryType.CONCEPT)
    mapped = _memory(
        63,
        MemoryLevel.M1,
        MemoryType.CONTINGENCY,
        (401, 2, 402, 403),
    )
    correspondence = _correspondence_ancestor()

    for candidate_row, rows, parents in (
        (
            role,
            (grounded, mapped, role, correspondence),
            {role.uid: {grounded.uid}, correspondence.uid: {mapped.uid}},
        ),
        (
            concept,
            (grounded, mapped, role, concept, correspondence),
            {
                concept.uid: {role.uid},
                role.uid: {grounded.uid},
                correspondence.uid: {mapped.uid},
            },
        ),
    ):
        with tempfile.TemporaryDirectory() as raw_root, patch.dict(
            os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
        ), patch.object(
            learning, "_held_out_games", return_value=("target",)
        ), patch(
            "v7.environment.arc_adapter.ArcGridEnvironment", _TransferProbeEnvironment
        ):
            flow.reset_for_tests()
            runtime, recorded, evidence = _transfer_runtime(candidate_row, rows, parents)
            snapshot = learning._candidate_execution_snapshot(
                runtime.read_view, rows, candidate_row.uid
            )
            result = learning._run_automatic_transfer_experiments_v088(
                runtime,
                games=("source",),
                env_root=None,
                seed=3,
                steps_per_trial=4,
                max_trials=1,
            )

        assert (result.attempted, result.completed, result.passed) == (1, 1, 0)
        assert snapshot["m7_descendant_count"] == 0
        assert snapshot["cached_executable_descendant_count"] == 0
        assert len(recorded) == 1
        assert recorded[0].passed is False
        assert "transfer_trial_fail" in evidence
        assert "transfer_trial_pass" not in evidence


def test_candidate_without_grounded_or_trajectory_evidence_remains_rejected() -> None:
    ancestor = _memory(70, MemoryLevel.M3, MemoryType.ROLE)
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ), patch.object(
        learning, "_held_out_games", return_value=("target",)
    ), patch(
        "v7.environment.arc_adapter.ArcGridEnvironment", _TransferProbeEnvironment
    ):
        flow.reset_for_tests()
        runtime, recorded, evidence = _transfer_runtime(ancestor, (ancestor,), {})
        result = learning._run_automatic_transfer_experiments_v088(
            runtime,
            games=("source",),
            env_root=None,
            seed=4,
            steps_per_trial=4,
            max_trials=1,
        )
        records = [
            json.loads(line)
            for line in (Path(raw_root) / flow.LOG_NAME).read_text(encoding="utf-8").splitlines()
        ]
        detail = [row for row in records if row["stage"] == "target_memory_resolution"]

    assert (result.attempted, result.completed, result.passed) == (0, 0, 0)
    assert not recorded
    assert "transfer_trial_pass" not in evidence
    assert "transfer_trial_fail" not in evidence
    assert detail[0]["exact_executable_predicate_failure_reason"] == (
        "no_grounded_action_evidence"
    )
    assert detail[0]["lower_level_resolution_failures"] == [
        "no_grounded_action_evidence",
        "no_replayable_source_trajectory",
    ]


def test_replayable_source_trajectory_can_schedule_without_m7() -> None:
    ancestor = _memory(75, MemoryLevel.M3, MemoryType.ROLE)
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ), patch.object(
        learning, "_held_out_games", return_value=("target",)
    ), patch(
        "v7.environment.arc_adapter.ArcGridEnvironment", _TransferProbeEnvironment
    ):
        optimizer_root = Path(raw_root) / "trajectory_optimizer"
        optimizer_root.mkdir(parents=True)
        (optimizer_root / "best_successful.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "games": {
                        "source": {
                            "trajectory_id": "source-win",
                            "successes": 1,
                            "levels": [{"level": 0, "actions": [2]}],
                        },
                        "correspondence": {
                            "trajectory_id": "correspondence-win",
                            "successes": 1,
                            "levels": [{"level": 0, "actions": [2]}],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        flow.reset_for_tests()
        runtime, recorded, evidence = _transfer_runtime(ancestor, (ancestor,), {})
        result = learning._run_automatic_transfer_experiments_v088(
            runtime,
            games=("source",),
            env_root=None,
            seed=7,
            steps_per_trial=4,
            max_trials=1,
        )

    assert (result.attempted, result.completed, result.passed) == (1, 1, 0)
    assert len(recorded) == 1
    assert "transfer_trial_pass" not in evidence


def test_grounded_actions_unsupported_by_target_are_rejected_exactly() -> None:
    grounded = _memory(
        76,
        MemoryLevel.M1,
        MemoryType.CONTINGENCY,
        (101, 9, 202, 303),
    )
    ancestor = _memory(77, MemoryLevel.M3, MemoryType.ROLE)
    mapped = _memory(
        78,
        MemoryLevel.M1,
        MemoryType.CONTINGENCY,
        (401, 9, 402, 403),
    )
    correspondence = _correspondence_ancestor()
    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ), patch.object(
        learning, "_held_out_games", return_value=("target",)
    ), patch(
        "v7.environment.arc_adapter.ArcGridEnvironment", _TransferProbeEnvironment
    ):
        flow.reset_for_tests()
        runtime, recorded, evidence = _transfer_runtime(
            ancestor,
            (grounded, mapped, ancestor, correspondence),
            {
                ancestor.uid: {grounded.uid},
                correspondence.uid: {mapped.uid},
            },
        )
        result = learning._run_automatic_transfer_experiments_v088(
            runtime,
            games=("source",),
            env_root=None,
            seed=8,
            steps_per_trial=4,
            max_trials=1,
        )
        records = [
            json.loads(line)
            for line in (Path(raw_root) / flow.LOG_NAME).read_text(encoding="utf-8").splitlines()
        ]
        detail = [row for row in records if row["stage"] == "target_memory_resolution"]

    assert (result.attempted, result.completed, result.passed) == (0, 0, 0)
    assert not recorded
    assert "transfer_trial_pass" not in evidence
    assert "transfer_trial_fail" not in evidence
    assert detail[0]["exact_executable_predicate_failure_reason"] == (
        "target_action_unsupported"
    )
    assert detail[0]["target_action_failure_detail"] == (
        "mapped_action_not_in_target_action_set"
    )


def test_formation_provenance_still_excludes_held_out_target() -> None:
    target_hash = stable_u64("target", person=b"v8-game")
    grounded = _memory(
        80,
        MemoryLevel.M1,
        MemoryType.CONTINGENCY,
        (101, 2, 202, 303),
    )
    ancestor = _memory(81, MemoryLevel.M3, MemoryType.ROLE)
    candidate = TransferCandidate(
        ancestor.uid,
        1,
        1.0,
        (target_hash,),
        MemoryUid.zero(),
        (stable_u64("other", person=b"v8-game"),),
    )
    transfer = TransferValidator()
    transfer.candidates = lambda *_args, **_kwargs: (candidate,)
    view = SimpleNamespace(
        node_records=lambda: (grounded, ancestor),
        source_games=lambda _uid: frozenset((target_hash,)),
        _node_by_uid={grounded.uid: grounded, ancestor.uid: ancestor},
        _parents={ancestor.uid: {grounded.uid}},
        _strategy_by_context={},
        _strategy_fallback=(),
        _v839_direct_games={},
    )
    runtime = SimpleNamespace(
        peers=SimpleNamespace(
            transfer=transfer,
            record_transfer_trial=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("provenance-rejected target scheduled")
            ),
        ),
        read_view=view,
    )

    with patch.object(learning, "_held_out_games", return_value=("target",)), patch.object(
        learning, "_probe_policy_v088", side_effect=AssertionError("probe should not run")
    ):
        result = learning._run_automatic_transfer_experiments_v088(
            runtime,
            games=("source",),
            env_root=None,
            seed=5,
            steps_per_trial=4,
            max_trials=1,
        )

    assert (result.attempted, result.completed, result.passed) == (0, 0, 0)


def test_existing_m7_plan_remains_first_execution_path() -> None:
    strategy_uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (2, 3, 4, 5))
    plan = PlannedAction(2, MemoryUid.zero(), strategy_uid, 1.0)
    view = SimpleNamespace(planned_action=lambda *_args, **_kwargs: plan)
    diagnostic = {}

    with patch("v7.environment.arc_adapter.ArcGridEnvironment", _TransferProbeEnvironment):
        metric, used = learning._probe_policy_v088(
            read_view=view,
            game_id="target",
            env_root=None,
            seed=6,
            steps=4,
            required_ancestor=MemoryUid.zero(),
            execution_evidence={"kind": "grounded_lineage", "action_ids": [2]},
            diagnostic=diagnostic,
        )

    assert metric == 0.0
    assert used == 4
    assert diagnostic["m7_planned_steps"] == 4
    assert diagnostic["lower_level_evidence_steps"] == 0


def test_correspondence_conditions_actions_and_positive_effect_passes() -> None:
    source_grounded = _memory(
        90,
        MemoryLevel.M1,
        MemoryType.CONTINGENCY,
        (101, 9, 202, 303),
    )
    ancestor = _memory(91, MemoryLevel.M3, MemoryType.ROLE)
    mapped_grounded = _memory(
        92,
        MemoryLevel.M1,
        MemoryType.CONTINGENCY,
        (401, 2, 402, 403),
    )
    correspondence = _correspondence_ancestor()
    rows = (source_grounded, mapped_grounded, ancestor, correspondence)
    parents = {
        ancestor.uid: {source_grounded.uid},
        correspondence.uid: {mapped_grounded.uid},
    }

    with tempfile.TemporaryDirectory() as raw_root, patch.dict(
        os.environ, {"ARC_AGI3_V8_ROOT": raw_root}, clear=False
    ), patch.object(
        learning, "_held_out_games", return_value=("target",)
    ), patch.object(
        learning, "_memory_free_action", return_value=1
    ), patch(
        "v7.environment.arc_adapter.ArcGridEnvironment", _EffectProbeEnvironment
    ):
        flow.reset_for_tests()
        runtime, recorded, evidence = _transfer_runtime(ancestor, rows, parents)
        result = learning._run_automatic_transfer_experiments_v088(
            runtime,
            games=("source",),
            env_root=None,
            seed=9,
            steps_per_trial=4,
            max_trials=1,
        )
        records = [
            json.loads(line)
            for line in (Path(raw_root) / flow.LOG_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    evaluation = next(
        row for row in records if row["stage"] == "transfer_trial_evaluation"
    )
    scheduling = next(
        row for row in records if row["stage"] == "transfer_experiment_scheduling"
    )
    assert (result.attempted, result.completed, result.passed) == (1, 1, 1)
    assert len(recorded) == 1
    assert recorded[0].effect == 5.0
    assert recorded[0].passed is True
    assert "transfer_trial_fail" not in evidence
    assert scheduling["examples"][0]["scheduler_decision"] == "transfer_trial_pass"

    assert evaluation["exact_source_action_sequence"] == [9]
    assert evaluation["exact_actions_executed_on_target"] == [2, 2, 2, 2]
    assert evaluation["correspondence_mapping"]["mapped_action_sequence"] == [2]
    assert evaluation["correspondence_conditioned_mapping"] is True
    assert evaluation["generic_source_actions_reused_without_correspondence"] is False
    assert evaluation["grounded_source_memory_ids_actually_used"] == [
        source_grounded.uid.hex()
    ]
    assert evaluation["grounded_correspondence_memory_ids_actually_used"] == [
        mapped_grounded.uid.hex()
    ]
    assert evaluation["intervention"]["score"] == 5.0
    assert evaluation["matched_memory_free_control"]["score"] == 0.0
    assert evaluation["matched_memory_free_control"]["actions"] == [1, 1, 1, 1]
    assert evaluation["intervention"]["initial_state_signature"] == evaluation[
        "matched_memory_free_control"
    ]["initial_state_signature"]
    assert evaluation["intervention"]["initial_available_action_ids"] == [1, 2]
    assert evaluation["matched_memory_free_control"][
        "initial_available_action_ids"
    ] == [1, 2]
    assert evaluation["matched_memory_free_control"]["target_memory_query_count"] == 0
    assert evaluation["matched_memory_free_control"]["target_memory_leakage"] is False
    assert evaluation["computed_transfer_effect"] == 5.0
    assert evaluation["existing_pass_threshold"] == 0.0
    assert evaluation["exact_failure_reason"] is None
    assert evaluation["matched_control_checks"] == {
        "same_target_world": True,
        "same_initial_target_state": True,
        "same_initial_available_actions": True,
        "same_seed": True,
        "same_horizon": True,
        "intervention_memory_policy_enabled": True,
        "control_memory_policy_enabled": False,
        "control_target_memory_query_count": 0,
        "control_target_memory_leakage": False,
        "only_transfer_policy_differs": True,
    }


def test_transfer_effect_calculation_preserves_strict_existing_threshold() -> None:
    uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (1000,))
    validator = TransferValidator(effect_threshold=0.0)

    positive = validator.record_trial(
        uid, target_game_hash=1, metric_on=2.0, metric_off=0.5
    )
    zero = validator.record_trial(
        uid, target_game_hash=2, metric_on=1.0, metric_off=1.0
    )
    negative = validator.record_trial(
        uid, target_game_hash=3, metric_on=0.25, metric_off=1.0
    )

    assert positive.effect == 1.5
    assert positive.passed is True
    assert learning._transfer_trial_failure_reason(positive, 0.0) is None
    assert zero.effect == 0.0
    assert zero.passed is False
    assert learning._transfer_trial_failure_reason(zero, 0.0) == (
        "transfer_effect_equal_to_existing_threshold"
    )
    assert negative.effect == -0.75
    assert negative.passed is False
    assert learning._transfer_trial_failure_reason(negative, 0.0) == (
        "transfer_effect_below_existing_threshold"
    )

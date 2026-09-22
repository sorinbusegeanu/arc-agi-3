from __future__ import annotations

from types import SimpleNamespace

import pytest

from v9.hgt import training
from v9.hgt.epoch_dataset import transition_training_rows_from_records
from v9.memory.model import MemoryLevel, MemoryType


def _torch():
    return pytest.importorskip("torch")


def test_policy_node_features_do_not_contain_outcome_labels() -> None:
    torch = _torch()
    node = SimpleNamespace(
        level=MemoryLevel.M0,
        memory_type=MemoryType.EPISODE,
        created_watermark=17,
        structural_key=(1, 2, 3),
    )
    common = {
        "recurrence": 2,
        "support": 3,
        "semantic_before": ((7, 1, 2, 3, 1.0),),
        "semantic_action": ((8, 4, 5, 6, 1.0),),
        "semantic_options": ((8, 7, 8, 9, 1.0),),
    }
    positive = {
        **common,
        "task_success": True,
        "task_failure": False,
        "task_truncated": False,
        "level_index": 4,
        "levels_completed": 3,
        "primary_valence": 1,
        "semantic_after": ((9, 10, 11, 12, 1.0),),
        "semantic_effects": ((9, 13, 14, 15, 1.0),),
    }
    negative = {
        **common,
        "task_success": False,
        "task_failure": True,
        "task_truncated": True,
        "level_index": 0,
        "levels_completed": 0,
        "primary_valence": -1,
        "semantic_after": ((9, 20, 21, 22, -1.0),),
        "semantic_effects": ((9, 23, 24, 25, -1.0),),
    }
    assert torch.equal(
        training._node_feature(node, positive, 64, torch),
        training._node_feature(node, negative, 64, torch),
    )


def test_higher_memory_auxiliary_objectives_are_not_masked_by_action_mask() -> None:
    torch = _torch()
    node_type = "M4_CONCEPT"
    logits = {node_type: torch.zeros((1, 3), requires_grad=True)}
    values = {node_type: torch.zeros(1, requires_grad=True)}
    auxiliary = {
        objective: {node_type: torch.zeros(1, requires_grad=True)}
        for objective in training.AUX_OBJECTIVES
    }
    y = {node_type: torch.zeros(1, dtype=torch.long)}
    policy_masks = {node_type: torch.zeros(1, dtype=torch.bool)}
    action_masks = {node_type: torch.zeros(1, dtype=torch.bool)}
    action_targets = {node_type: torch.zeros(1)}
    task_targets = {
        objective: {node_type: torch.zeros(1)}
        for objective in training.AUX_OBJECTIVES
    }
    task_masks = {
        objective: {node_type: torch.zeros(1, dtype=torch.bool)}
        for objective in training.AUX_OBJECTIVES
    }
    task_masks["correspondence"][node_type][0] = True
    log_vars = {
        objective: torch.tensor(0.0, requires_grad=True)
        for objective in training.OBJECTIVE_NAMES
    }

    loss, _accuracy, by_head = training._loss(
        logits,
        values,
        auxiliary,
        y,
        policy_masks,
        action_targets,
        action_masks,
        task_targets,
        task_masks,
        torch,
        objective_weights=(1.0,) * 9,
        log_vars=log_vars,
        dynamic_weighting=False,
    )
    assert loss is not None
    assert "correspondence" in by_head


def test_explicit_action_ranking_uses_hgt_value_predictions() -> None:
    torch = _torch()
    node_type = training.NODE_TYPE
    values = {node_type: torch.tensor([1.0, -1.0], requires_grad=True)}
    meta = {
        node_type: [
            (7, 99, 1, 1, 10, "game"),
            (7, 99, 2, 1, 11, "game"),
        ]
    }
    masks = {node_type: torch.tensor([True, True])}
    best = {"game_scenario": "game", "context_signature": 99, "action_id": 1}
    worst = {"game_scenario": "game", "context_signature": 99, "action_id": 2}

    loss, correct, total = training._explicit_action_ranking_loss(
        values, meta, masks, [(best, worst)], torch
    )
    assert loss is not None
    assert total == 1
    assert correct == 1
    loss.backward()
    assert values[node_type].grad is not None


def test_policy_gate_is_closed_at_random_ranking_and_bounded_when_open() -> None:
    config = SimpleNamespace(
        hgt_min_validation_ranking_accuracy=0.5,
        hgt_max_policy_score=0.05,
    )
    assert training._policy_score_scale(
        config, validation_accuracy=0.5, validation_pairs=100
    ) == 0.0
    assert training._policy_score_scale(
        config, validation_accuracy=1.0, validation_pairs=100
    ) == pytest.approx(0.05)
    assert training._policy_score_scale(
        config, validation_accuracy=1.0, validation_pairs=0
    ) == 0.0

    scores = training._bounded_advantage_scores({1: 10.0, 2: -5.0, 3: 2.0}, 0.05)
    assert max(abs(value) for value in scores.values()) <= 0.05 + 1e-12
    assert sum(scores.values()) == pytest.approx(0.0)


def test_transition_returns_reward_level_progress_consistently() -> None:
    rows = transition_training_rows_from_records(
        [
            {
                "game_scenario": "arc",
                "actor_id": 1,
                "episode_id": 1,
                "global_step": 0,
                "before_signature": 10,
                "after_signature": 11,
                "action_id": 1,
                "primary_valence": 0,
                "levels_completed": 0,
            },
            {
                "game_scenario": "arc",
                "actor_id": 1,
                "episode_id": 1,
                "global_step": 1,
                "before_signature": 11,
                "after_signature": 12,
                "action_id": 2,
                "primary_valence": 0,
                "levels_completed": 1,
            },
        ]
    )
    assert rows[1]["target_return"] == pytest.approx(0.5)
    assert rows[0]["target_return"] == pytest.approx(0.97 * 0.5)


def test_policy_validation_split_is_context_level_and_disjoint() -> None:
    torch = _torch()
    validation_context = next(
        value
        for value in range(10000)
        if training._context_is_validation("game", value, 0.2)
    )
    training_context = next(
        value
        for value in range(10000)
        if not training._context_is_validation("game", value, 0.2)
    )
    node_type = training.NODE_TYPE
    meta = {
        node_type: [
            (1, validation_context, 1, 1, 1, "game"),
            (1, validation_context, 2, 1, 2, "game"),
            (1, training_context, 1, 2, 3, "game"),
        ]
    }
    masks = {node_type: torch.tensor([True, True, True])}
    train_masks, validation_masks = training._split_policy_masks(
        meta, masks, torch, validation_fraction=0.2
    )
    assert train_masks[node_type].tolist() == [False, False, True]
    assert validation_masks[node_type].tolist() == [True, True, False]


def test_obsolete_raw_transition_value_head_loss_is_removed() -> None:
    source = __import__("inspect").getsource(training)
    assert "_transition_batch_loss" not in source
    assert "hgt_validation_ranking_accuracy" in source
    assert "policy_score_scale" in source



def test_policy_feature_source_excludes_post_action_outcomes_without_torch() -> None:
    import inspect

    node_source = inspect.getsource(training._node_feature)
    predictive_source = inspect.getsource(training._predictive_semantic_rows)
    for forbidden in (
        "task_success",
        "task_failure",
        "task_truncated",
        "level_index",
        "levels_completed",
        "primary_valence",
    ):
        assert forbidden not in node_source
    assert "semantic_after" not in predictive_source
    assert "semantic_effects" not in predictive_source


def test_auxiliary_mask_contract_keeps_non_action_memories_without_torch() -> None:
    import inspect

    source = inspect.getsource(training._loss)
    assert "(~action_mask) | policy_mask" in source
    assert "task_masks[objective][node_type]" in source


def test_ranking_contract_uses_full_hgt_values_without_raw_encoder_path() -> None:
    import inspect

    source = inspect.getsource(training._explicit_action_ranking_loss)
    assert "value_dict" in source
    assert "model.encoders" not in source
    module_source = inspect.getsource(training)
    assert "_transition_batch_loss" not in module_source


def test_sparse_contexts_generate_game_level_action_ranking_pair() -> None:
    from v9.hgt.epoch_dataset import action_ranking_pairs

    rows = [
        {
            "game_scenario": "game",
            "context_signature": 101,
            "action_id": 1,
            "target_return": 0.75,
        },
        {
            "game_scenario": "game",
            "context_signature": 202,
            "action_id": 2,
            "target_return": -0.25,
        },
    ]
    pairs = action_ranking_pairs(rows)
    assert len(pairs) == 1
    best, worst = pairs[0]
    assert best["ranking_scope"] == "game"
    assert worst["ranking_scope"] == "game"
    assert best["action_id"] == 1
    assert worst["action_id"] == 2

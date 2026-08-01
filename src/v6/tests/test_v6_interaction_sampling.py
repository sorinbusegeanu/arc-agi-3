from __future__ import annotations

import os

import v6.evaluation.interaction_sampling as sampling
from v6.evaluation.interaction_sampling import InteractionSamplingConfig


def _payload(seed: int, labels: tuple[str, ...] = ("PRESERVE", "EXPAND")) -> dict:
    examples = []
    for index, label in enumerate(labels):
        examples.append(
            {
                "contingency_id": index + 1,
                "contingency_key": [["ctx", index], "action"],
                "features": {
                    "context_level": 1,
                    "confidence": 0.5 + index * 0.1,
                    "support_count_log": 1.0 + index,
                    "prediction_error_rate": 0.1,
                    "context_support": 2.0,
                    "action_entropy_at_context": 0.2,
                    "transformation_entropy_at_context": 0.3,
                    "transformation_family_support_log": 1.0,
                    "changed_cells": float(index),
                    "dx": 0.0,
                    "dy": 1.0,
                    "colors_added_count": 1.0,
                    "colors_removed_count": 0.0,
                    "contingency_in_degree": 1.0,
                    "contingency_out_degree": 2.0,
                    "follows_in_degree": 1.0,
                    "follows_out_degree": 2.0,
                    "cooccurrence_degree": 2.0,
                    "clustering_coefficient": 0.0,
                    "degree_centrality": 0.1,
                    "pagerank": 0.1,
                },
                "label": label,
            }
        )
    return {"seed": seed, "examples": examples}


def _task(game: str, sampler_name: str, config: InteractionSamplingConfig, *, valid: bool = True) -> dict:
    payloads = [_payload(0), _payload(1)] if valid else []
    return {
        "game": game,
        "sampler_name": sampler_name,
        "payloads": payloads,
        "config": sampling._validation_task_config(config),
    }


def test_validation_workers_default_to_eight_and_are_bounded() -> None:
    assert InteractionSamplingConfig().validation_workers == 8
    assert sampling._effective_validation_workers(8, 2) == 2
    assert sampling._effective_validation_workers(99, 32) == 16
    assert sampling._effective_validation_workers(0, 5) == 1


def test_validation_worker_initializer_limits_numerical_threads(monkeypatch) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    sampling._validation_worker_initializer()
    assert all(
        os.environ[name] == "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    )


def test_streamed_validation_parallel_and_sequential_results_are_equivalent_and_ordered() -> None:
    base = InteractionSamplingConfig(train_seeds=(0,), test_seed=1, validation_workers=1)
    tasks = [_task("zb01", "sampler_b", base), _task("za01", "sampler_a", base)]
    sequential, sequential_metrics = sampling._run_streamed_validation_tasks(base, tasks)
    parallel_config = InteractionSamplingConfig(train_seeds=(0,), test_seed=1, validation_workers=8)
    parallel, parallel_metrics = sampling._run_streamed_validation_tasks(parallel_config, tasks)

    assert [(row["game"], row["sampler_name"]) for row in parallel] == [("za01", "sampler_a"), ("zb01", "sampler_b")]
    timing_keys = {"validation_payload_decode_seconds", "validation_feature_preparation_seconds", "validation_classifier_seconds"}
    assert [
        {key: value for key, value in row["validation"].items() if key not in timing_keys}
        for row in parallel
    ] == [
        {key: value for key, value in row["validation"].items() if key not in timing_keys}
        for row in sequential
    ]
    assert sequential_metrics["validation_worker_count"] == 1
    assert parallel_metrics["validation_worker_count"] == 2
    assert parallel_metrics["validation_task_count"] == 2
    assert parallel_metrics["validation_parallel_wall_seconds"] >= 0.0
    assert parallel_metrics["validation_payload_decode_seconds_total"] >= 0.0
    assert parallel_metrics["validation_feature_preparation_seconds_total"] >= 0.0
    assert parallel_metrics["validation_classifier_seconds_total"] >= 0.0


def test_streamed_validation_failure_does_not_stop_other_groups() -> None:
    config = InteractionSamplingConfig(train_seeds=(0,), test_seed=1, validation_workers=2)
    results, _metrics = sampling._run_streamed_validation_tasks(
        config,
        [_task("good", "sampler", config), _task("bad", "sampler", config, valid=False)],
    )
    assert {row["game"] for row in results} == {"good", "bad"}
    assert next(row for row in results if row["game"] == "good")["run_status"] == "ok"
    failed = next(row for row in results if row["game"] == "bad")
    assert failed["run_status"] == "failed"
    assert failed["failure_reason"].startswith("RuntimeError:")


def test_feature_and_baseline_preparation_are_reused_per_validation_task(monkeypatch) -> None:
    import v6.evaluation.id_free_prefuture_validation as id_free

    config = InteractionSamplingConfig(train_seeds=(0,), test_seed=1)
    calls = {"group": 0, "features": 0, "contingency_baseline": 0}
    original_group = sampling.prepare_id_free_validation_group
    original_features = sampling.prepare_id_free_feature_set
    original_contingency_baseline = id_free.contingency_baseline_predictions

    def wrapped_group(*args, **kwargs):
        calls["group"] += 1
        return original_group(*args, **kwargs)

    def wrapped_features(*args, **kwargs):
        calls["features"] += 1
        return original_features(*args, **kwargs)

    def wrapped_contingency_baseline(*args, **kwargs):
        calls["contingency_baseline"] += 1
        return original_contingency_baseline(*args, **kwargs)

    monkeypatch.setattr(sampling, "prepare_id_free_validation_group", wrapped_group)
    monkeypatch.setattr(sampling, "prepare_id_free_feature_set", wrapped_features)
    monkeypatch.setattr(id_free, "contingency_baseline_predictions", wrapped_contingency_baseline)
    result = sampling._run_streamed_validation_task(_task("game", "sampler", config))

    assert result["run_status"] == "ok"
    assert calls == {
        "group": 1,
        "features": len(sampling.ID_FREE_FEATURE_SETS),
        "contingency_baseline": 1,
    }

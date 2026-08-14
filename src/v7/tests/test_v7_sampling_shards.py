from __future__ import annotations

from types import SimpleNamespace

from v7.environment.runner import ArcGameRunResult
from v7.experiment import (
    _append_aggregated_game_results,
    _epoch_game_rates,
    _sampling_shards,
)


def test_sampling_shards_fill_worker_budget_without_multiplying_steps() -> None:
    shards = _sampling_shards(
        games=("g0", "g1", "g2"),
        epoch=2,
        steps_per_game=10,
        workers=8,
        seed=100,
        env_root=None,
        epsilon=0.1,
        ablation_mask=0,
        first_job_index=50,
        epoch_count=5,
    )
    assert len(shards) == 8
    assert {job.game_id for job in shards} == {"g0", "g1", "g2"}
    assert sum(job.steps for job in shards if job.game_id == "g0") == 10
    assert sum(job.steps for job in shards if job.game_id == "g1") == 10
    assert sum(job.steps for job in shards if job.game_id == "g2") == 10
    assert len({job.job_index for job in shards}) == len(shards)
    ranges = sorted(
        (job.global_step_offset + 1, job.global_step_offset + job.steps)
        for job in shards
    )
    assert all(left[1] < right[0] for left, right in zip(ranges, ranges[1:]))


def test_sampling_shards_do_not_split_single_step_games() -> None:
    shards = _sampling_shards(
        games=("g0", "g1"),
        epoch=0,
        steps_per_game=1,
        workers=40,
        seed=0,
        env_root=None,
        epsilon=0.1,
        ablation_mask=0,
        first_job_index=0,
        epoch_count=1,
    )
    assert len(shards) == 2
    assert [job.steps for job in shards] == [1, 1]


def test_epoch_game_rates_count_games_not_shards() -> None:
    sampled = (
        SimpleNamespace(game_id="g0", wins=40, levels_completed=50),
        SimpleNamespace(game_id="g0", wins=30, levels_completed=40),
        SimpleNamespace(game_id="g1", wins=0, levels_completed=3),
        SimpleNamespace(game_id="g1", wins=0, levels_completed=0),
        SimpleNamespace(game_id="g2", wins=0, levels_completed=0),
    )
    win_rate, level_rate = _epoch_game_rates(sampled)
    assert win_rate == 100.0 / 3.0
    assert level_rate == 200.0 / 3.0


def test_aggregated_game_results_remain_one_row_per_game_after_sharding() -> None:
    sampled = (
        SimpleNamespace(game_id="g0", steps=4, wins=2, failures=1, levels_completed=3, resets=1),
        SimpleNamespace(game_id="g0", steps=6, wins=5, failures=2, levels_completed=7, resets=2),
        SimpleNamespace(game_id="g1", steps=10, wins=0, failures=4, levels_completed=1, resets=3),
    )
    results: list[ArcGameRunResult] = []
    _append_aggregated_game_results(
        results,
        sampled=sampled,
        games=("g0", "g1"),
        generation=9,
        memories=123,
    )
    assert len(results) == 2
    assert results[0].game_id == "g0"
    assert results[0].steps == 10
    assert results[0].wins == 7
    assert results[0].failures == 3
    assert results[0].levels_completed == 10
    assert results[0].resets == 3
    assert results[1].game_id == "g1"
    assert results[1].steps == 10
    assert results[1].wins == 0
    assert results[1].failures == 4
    assert results[1].levels_completed == 1
    assert results[1].resets == 3

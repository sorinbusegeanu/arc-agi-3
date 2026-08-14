from __future__ import annotations

from v7.experiment import _sampling_shards


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

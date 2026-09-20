from __future__ import annotations

from v9.research.experiment_manifest import (
    ExperimentManifest,
    InteractionOpportunityManifest,
    ScientificEvidenceId,
    TrialSpec,
    derive_environment_instance_id,
    derive_episode_id,
    derive_producer_id,
    derive_sampling_epoch_id,
    derive_stable_environment_job_id,
)
from v9.runtime import ScientificConfig


def _identities(process_run_epoch: int, worker_slot: int) -> tuple[int, ...]:
    manifest = ExperimentManifest(
        ScientificConfig().config_id,
        InteractionOpportunityManifest((TrialSpec("stable-job", 3, 4, 0, 0, 16),)),
        root_random_seed=9,
        replicate_seeds=(17,),
    )
    experiment = manifest.experiment_id
    replicate = manifest.replicate_id(0)
    epoch = derive_sampling_epoch_id(experiment, replicate, 2)
    job = derive_stable_environment_job_id(experiment, replicate, "synthetic", 0)
    producer = derive_producer_id(experiment, replicate, epoch, job, 0)
    environment = derive_environment_instance_id(experiment, replicate, job, 0)
    episode = derive_episode_id(experiment, replicate, epoch, producer, 5)
    evidence = ScientificEvidenceId(experiment, replicate, epoch, producer, 23, 1)
    # Process-only inputs are deliberately not passed to any derivation.
    assert process_run_epoch >= 0 and worker_slot >= 0
    return experiment, replicate, epoch, producer, environment, episode, int(evidence.checksum[:16], 16)


def test_process_restart_and_worker_schedule_do_not_change_scientific_ids() -> None:
    assert _identities(1, 0) == _identities(99, 31)

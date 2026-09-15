from __future__ import annotations

from v9.environments.schemas import EnvironmentIdentity
from v9.memory.identity import EpisodeId, EventUid, MemoryUid, stable_u64
from v9.modalities.contract import PassiveSymbolEvent, SYMBOL_MODALITY, TimelineIdentity
from v9.modalities.symbols import DeterministicSymbolCodec

from .multiprocess import EncodedTransition


def publish_transition_symbols(runtime, transition: EncodedTransition, *, base_watermark: int | None = None) -> None:
    if not transition.symbols:
        return
    identity = EnvironmentIdentity(*transition.environment_identity)
    environment = runtime.environments.register(identity).value
    episode_id = EpisodeId(int(transition.episode_id))
    codec = DeterministicSymbolCodec(f"{identity.family}-raw-symbols")
    runtime.symbol_codecs[codec.vocabulary_id.value] = codec
    observations = codec.encode_stream(
        transition.symbols,
        stream_name=f"{environment}:{episode_id.value}:{transition.producer_sequence}",
    )
    base = runtime.watermark if base_watermark is None else int(base_watermark)
    for index, row in enumerate(observations):
        event = PassiveSymbolEvent(
            TimelineIdentity(
                EventUid.from_producer(
                    transition.actor_id + 1_000_000,
                    transition.producer_sequence * 10_000 + index,
                ),
                base + index + 1,
                transition.actor_id + 1_000_000,
                transition.producer_sequence * 10_000 + index,
                environment,
                episode_id,
                SYMBOL_MODALITY,
            ),
            row.vocabulary_id,
            row.stream_id,
            row.symbol_id,
            row.position.value,
        )
        runtime.submit_event(event)


def publish_encoded_transition(runtime, transition: EncodedTransition) -> None:
    identity = EnvironmentIdentity(*transition.environment_identity)
    environment = runtime.environments.register(identity).value
    episode_id = EpisodeId(int(transition.episode_id))
    if not transition.symbols_only:
        experience = runtime.make_experience(
            producer_id=transition.actor_id,
            producer_sequence=transition.producer_sequence,
            environment_instance_id=environment,
            global_step=transition.global_step,
            context_signature=transition.before_signature,
            action_id=transition.action_id,
            outcome_signature=transition.after_signature,
            family_signature=stable_u64(
                transition.observation_schema_id,
                int(transition.before_signature != transition.after_signature),
                person=b"v9-family",
            ),
            carrier_signature=stable_u64(
                transition.observation_schema_id,
                transition.before_signature,
                person=b"v9-carrier",
            ),
            future_option_delta=float(transition.available_actions_after),
            changed_cells=int(transition.before_signature != transition.after_signature),
            primary_valence=transition.primary_valence,
            trajectory_signature=stable_u64(
                environment,
                transition.episode_id,
                person=b"v9-trajectory",
            ),
            next_context_signature=transition.after_signature,
        )
        runtime.submit(experience, episode_id=episode_id)

    publish_transition_symbols(runtime, transition)

    if transition.task_success and not transition.symbols_only:
        # submit() may use deferred ingestion; force this terminal experience
        # through the canonical commit before reading the complete episode.
        runtime.wait_quiescent()
        if transition.strategy_target_outcome_hi is not None and transition.strategy_target_outcome_lo is not None:
            runtime.record_successful_trajectory(
                environment_id=int(environment),
                episode_id=int(transition.episode_id),
                target_outcome_uid=MemoryUid(
                    int(transition.strategy_target_outcome_hi),
                    int(transition.strategy_target_outcome_lo),
                ),
            )

    if (
        transition.strategy_terminal
        and transition.strategy_uid_hi is not None
        and transition.strategy_uid_lo is not None
    ):
        runtime.record_strategy_execution(
            MemoryUid(int(transition.strategy_uid_hi), int(transition.strategy_uid_lo)),
            success=bool(transition.task_success),
            realized_cost=max(1, int(transition.strategy_realized_cost)),
            primary_valence=int(transition.primary_valence),
        )

    runtime.record_curriculum_event(
        step=transition.curriculum_step,
        environment_family=identity.family,
        game_scenario=transition.game_scenario,
    )

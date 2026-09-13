from __future__ import annotations

from v9.environments.schemas import EnvironmentIdentity
from v9.memory.identity import EpisodeId, EventUid, stable_u64
from v9.modalities.contract import PassiveSymbolEvent, SYMBOL_MODALITY, TimelineIdentity
from v9.modalities.symbols import DeterministicSymbolCodec

from .multiprocess import EncodedTransition


def publish_encoded_transition(runtime, transition: EncodedTransition) -> None:
    identity = EnvironmentIdentity(*transition.environment_identity)
    environment = runtime.environments.register(identity).value
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
    episode_id = EpisodeId(int(transition.episode_id))
    runtime.submit(experience, episode_id=episode_id)

    if transition.symbols:
        codec = DeterministicSymbolCodec(f"{identity.family}-raw-symbols")
        runtime.symbol_codecs[codec.vocabulary_id.value] = codec
        observations = codec.encode_stream(
            transition.symbols,
            stream_name=f"{environment}:{episode_id.value}:{transition.producer_sequence}",
        )
        for index, row in enumerate(observations):
            event = PassiveSymbolEvent(
                TimelineIdentity(
                    EventUid.from_producer(
                        transition.actor_id + 1_000_000,
                        transition.producer_sequence * 10_000 + index,
                    ),
                    runtime.watermark + index + 1,
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

    runtime.unified_telemetry.record_curriculum_event(
        step=transition.curriculum_step,
        environment_family=identity.family,
        game_scenario=transition.game_scenario,
    )

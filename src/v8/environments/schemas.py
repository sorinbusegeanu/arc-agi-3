from __future__ import annotations

from dataclasses import dataclass

from v8.model import stable_u64
from v8.persistent_identity import environment_world_id


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    family: str
    environment_type: str
    config: str = "default"
    instance: str = "default"

    def __post_init__(self) -> None:
        # Runtime seeds and episode instances are intentionally not part of the
        # persisted environment identity.
        object.__setattr__(self, "instance", "default")

    @property
    def family_id(self) -> int:
        return stable_u64(self.family, person=b"v8.58-env-family")

    @property
    def type_id(self) -> int:
        return stable_u64(self.family, self.environment_type, person=b"v8.58-env-type")

    @property
    def config_id(self) -> int:
        return environment_world_id(self.family, self.environment_type, self.config)

    @property
    def instance_id(self) -> int:
        return self.config_id

    @property
    def source_hash(self) -> int:
        return self.instance_id


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    kind: str
    detail: str
    version: int = 1

    @property
    def schema_id(self) -> int:
        return stable_u64(
            self.kind,
            self.detail,
            int(self.version),
            person=b"v8.58-obs-schema",
        )


@dataclass(frozen=True, slots=True)
class ActionSchema:
    kind: str
    detail: str
    version: int = 1

    @property
    def schema_id(self) -> int:
        return stable_u64(
            self.kind,
            self.detail,
            int(self.version),
            person=b"v8.58-action-schema",
        )


class DiscreteObservationCodec:
    def __init__(self, size: int) -> None:
        if int(size) <= 0:
            raise ValueError("discrete observation size must be positive")
        self.size = int(size)
        self.schema = ObservationSchema("discrete", f"n={self.size}")

    def encode(self, observation) -> int:
        value = int(observation)
        if not 0 <= value < self.size:
            raise ValueError(f"observation {value} outside Discrete({self.size})")
        return value

    def signature(self, observation) -> int:
        return stable_u64(
            self.schema.schema_id,
            self.encode(observation),
            person=b"v8.58-discrete-observation",
        )


class DiscreteActionCodec:
    def __init__(self, size: int) -> None:
        if int(size) <= 0:
            raise ValueError("discrete action size must be positive")
        self.size = int(size)
        self.schema = ActionSchema("discrete", f"n={self.size}")

    def encode(self, action) -> int:
        value = int(action)
        if not 0 <= value < self.size:
            raise ValueError(f"action {value} outside Discrete({self.size})")
        return value

    def decode(self, token: int) -> int:
        return self.encode(token)

    def available_tokens(self) -> tuple[int, ...]:
        return tuple(range(self.size))

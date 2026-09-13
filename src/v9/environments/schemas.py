from __future__ import annotations

from dataclasses import dataclass

from v9.memory.identity import (
    EnvironmentConfigId, EnvironmentFamilyId, EnvironmentInstanceId,
    EnvironmentTypeId, stable_u64,
)


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    family: str
    environment_type: str
    config: str = "default"
    instance: str = "default"

    @property
    def family_id(self) -> EnvironmentFamilyId:
        return EnvironmentFamilyId(stable_u64(self.family, person=b"v9-env-family"))

    @property
    def type_id(self) -> EnvironmentTypeId:
        return EnvironmentTypeId(stable_u64(self.family, self.environment_type, person=b"v9-env-type"))

    @property
    def config_id(self) -> EnvironmentConfigId:
        return EnvironmentConfigId(stable_u64(self.family, self.environment_type, self.config, person=b"v9-env-config"))

    @property
    def instance_id(self) -> EnvironmentInstanceId:
        return EnvironmentInstanceId(stable_u64(self.config_id.value, self.instance, person=b"v9-env-instance"))


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    kind: str
    detail: str
    version: int = 1

    @property
    def schema_id(self) -> int:
        return stable_u64(self.kind, self.detail, self.version, person=b"v9-observation")


@dataclass(frozen=True, slots=True)
class ActionSchema:
    kind: str
    detail: str
    version: int = 1

    @property
    def schema_id(self) -> int:
        return stable_u64(self.kind, self.detail, self.version, person=b"v9-action")


class DiscreteObservationCodec:
    def __init__(self, size: int) -> None:
        if int(size) <= 0:
            raise ValueError("discrete observation size must be positive")
        self.size = int(size)
        self.schema = ObservationSchema("discrete", f"n={self.size}")

    def encode(self, observation: object) -> int:
        value = int(observation)
        if not 0 <= value < self.size:
            raise ValueError(f"observation {value} outside Discrete({self.size})")
        return value

    def signature(self, observation: object) -> int:
        return stable_u64(self.schema.schema_id, self.encode(observation), person=b"v9-discrete-obs")


class DiscreteActionCodec:
    def __init__(self, size: int) -> None:
        if int(size) <= 0:
            raise ValueError("discrete action size must be positive")
        self.size = int(size)
        self.schema = ActionSchema("discrete", f"n={self.size}")

    def encode(self, action: object) -> int:
        value = int(action)
        if not 0 <= value < self.size:
            raise ValueError(f"action {value} outside Discrete({self.size})")
        return value

    decode = encode

    def available_tokens(self) -> tuple[int, ...]:
        return tuple(range(self.size))


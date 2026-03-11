from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionId:
    value: str


@dataclass(frozen=True)
class RunId:
    value: str


@dataclass(frozen=True)
class EpisodeId:
    value: str


@dataclass(frozen=True)
class SnapshotHandle:
    value: str


@dataclass(frozen=True)
class PlanContextId:
    value: str


from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from v8.model import stable_u64


class ModalityId(IntEnum):
    WORLD = 1
    SYMBOL = 2


@dataclass(frozen=True, slots=True, order=True)
class SymbolVocabularyId:
    value: int


@dataclass(frozen=True, slots=True, order=True)
class SymbolStreamId:
    value: int


@dataclass(frozen=True, slots=True, order=True)
class SymbolId:
    value: int


@dataclass(frozen=True, slots=True)
class SymbolObservation:
    vocabulary_id: SymbolVocabularyId
    stream_id: SymbolStreamId
    symbol_id: SymbolId
    position: int

    def __post_init__(self) -> None:
        if int(self.position) < 0:
            raise ValueError("symbol position cannot be negative")


class DeterministicSymbolCodec:
    """Identity/order codec only; it intentionally exposes no semantic features."""

    def __init__(
        self,
        vocabulary_name: str,
        *,
        codec_name: str = "lexical-id",
        version: int = 1,
    ) -> None:
        if not vocabulary_name:
            raise ValueError("vocabulary_name must be non-empty")
        if not codec_name:
            raise ValueError("codec_name must be non-empty")
        if int(version) <= 0:
            raise ValueError("codec version must be positive")
        self.vocabulary_name = str(vocabulary_name)
        self.codec_name = str(codec_name)
        self.version = int(version)
        self.codec_id = stable_u64(self.codec_name, self.version, person=b"v9-symbol-codec")
        self.vocabulary_id = SymbolVocabularyId(
            stable_u64(self.codec_id, self.vocabulary_name, person=b"v9-symbol-vocab")
        )

    def symbol_id(self, token: str | bytes | int) -> SymbolId:
        if isinstance(token, bytes):
            raw = token
        elif isinstance(token, str):
            raw = token.encode("utf-8")
        else:
            raw = int(token).to_bytes(16, "little", signed=True)
        return SymbolId(
            stable_u64(self.vocabulary_id.value, raw, person=b"v9-symbol-id")
        )

    def stream_id(self, stream_name: str | int) -> SymbolStreamId:
        return SymbolStreamId(
            stable_u64(
                self.vocabulary_id.value,
                str(stream_name),
                person=b"v9-symbol-stream",
            )
        )

    def encode_stream(
        self,
        tokens: Iterable[str | bytes | int],
        *,
        stream_name: str | int,
        start_position: int = 0,
    ) -> tuple[SymbolObservation, ...]:
        start = int(start_position)
        if start < 0:
            raise ValueError("start_position cannot be negative")
        stream_id = self.stream_id(stream_name)
        return tuple(
            SymbolObservation(
                self.vocabulary_id,
                stream_id,
                self.symbol_id(token),
                start + offset,
            )
            for offset, token in enumerate(tokens)
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "codec_name": self.codec_name,
            "codec_version": self.version,
            "codec_id": int(self.codec_id),
            "vocabulary_name": self.vocabulary_name,
            "vocabulary_id": int(self.vocabulary_id.value),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "DeterministicSymbolCodec":
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported symbol codec state version")
        codec = cls(
            str(state["vocabulary_name"]),
            codec_name=str(state["codec_name"]),
            version=int(state["codec_version"]),
        )
        if int(state["codec_id"]) != codec.codec_id:
            raise ValueError("symbol codec identity mismatch")
        if int(state["vocabulary_id"]) != codec.vocabulary_id.value:
            raise ValueError("symbol vocabulary identity mismatch")
        return codec

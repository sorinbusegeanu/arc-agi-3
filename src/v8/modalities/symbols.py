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


@dataclass(frozen=True, slots=True, order=True)
class SymbolPosition:
    value: int


@dataclass(frozen=True, slots=True)
class SymbolObservation:
    vocabulary_id: SymbolVocabularyId
    stream_id: SymbolStreamId
    symbol_id: SymbolId
    position: SymbolPosition


class DeterministicSymbolCodec:
    """Identity/order codec only; deliberately exposes no lexical semantics."""

    STATE_VERSION = 1

    def __init__(self, vocabulary: str | int, *, codec_version: str = "v9-symbol-codec-1") -> None:
        self.codec_version = str(codec_version)
        self.vocabulary_name = str(vocabulary)
        self.vocabulary_id = SymbolVocabularyId(
            stable_u64(self.codec_version, self.vocabulary_name, person=b"v9-symbol-vocab")
        )

    def stream_id(self, stream: str | int) -> SymbolStreamId:
        return SymbolStreamId(
            stable_u64(self.vocabulary_id.value, str(stream), person=b"v9-symbol-stream")
        )

    def symbol_id(self, token: str | bytes | int) -> SymbolId:
        if isinstance(token, bytes):
            raw = token
        elif isinstance(token, str):
            raw = token.encode("utf-8")
        else:
            raw = int(token).to_bytes(16, "little", signed=True)
        return SymbolId(stable_u64(self.vocabulary_id.value, raw, person=b"v9-symbol-id"))

    def encode_stream(
        self,
        tokens: Iterable[str | bytes | int],
        *,
        stream_name: str | int,
        start_position: int = 0,
    ) -> tuple[SymbolObservation, ...]:
        stream = self.stream_id(stream_name)
        return tuple(
            SymbolObservation(
                self.vocabulary_id,
                stream,
                self.symbol_id(token),
                SymbolPosition(int(start_position) + offset),
            )
            for offset, token in enumerate(tokens)
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "codec_version": self.codec_version,
            "vocabulary_name": self.vocabulary_name,
            "vocabulary_id": self.vocabulary_id.value,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "DeterministicSymbolCodec":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported symbol codec state")
        codec = cls(str(state["vocabulary_name"]), codec_version=str(state["codec_version"]))
        if codec.vocabulary_id.value != int(state["vocabulary_id"]):
            raise ValueError("symbol vocabulary identity mismatch")
        return codec

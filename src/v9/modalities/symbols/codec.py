from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v9.memory.identity import SymbolId, SymbolStreamId, SymbolVocabularyId

from .vocabulary import SymbolPosition, SymbolVocabulary


@dataclass(frozen=True, slots=True)
class SymbolObservation:
    vocabulary_id: SymbolVocabularyId
    stream_id: SymbolStreamId
    symbol_id: SymbolId
    position: SymbolPosition


class DeterministicSymbolCodec:
    """Identity and ordering only; no lexical or pretrained semantic features."""

    def __init__(self, vocabulary_name: str, *, codec_name: str = "identity", version: int = 1) -> None:
        self.vocabulary = SymbolVocabulary(vocabulary_name, codec_name, version)

    @property
    def vocabulary_id(self) -> SymbolVocabularyId:
        return self.vocabulary.vocabulary_id

    def _raw(self, token: str | bytes | int) -> bytes:
        if isinstance(token, bytes):
            return token
        if isinstance(token, str):
            return token.encode("utf-8")
        return int(token).to_bytes(16, "little", signed=True)

    def symbol_id(self, token: str | bytes | int) -> SymbolId:
        return self.vocabulary.symbol_id(self._raw(token))

    def stream_id(self, name: str | int) -> SymbolStreamId:
        return self.vocabulary.stream_id(name)

    def encode_stream(self, tokens: Iterable[str | bytes | int], *, stream_name: str | int, start_position: int = 0) -> tuple[SymbolObservation, ...]:
        if start_position < 0:
            raise ValueError("start_position cannot be negative")
        stream_id = self.stream_id(stream_name)
        return tuple(SymbolObservation(self.vocabulary_id, stream_id, self.symbol_id(token), SymbolPosition(start_position + index)) for index, token in enumerate(tokens))

    def state_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "name": self.vocabulary.name, "codec_name": self.vocabulary.codec_name, "codec_version": self.vocabulary.version, "vocabulary_id": self.vocabulary_id.value}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "DeterministicSymbolCodec":
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError("unsupported symbol codec schema")
        result = cls(str(state["name"]), codec_name=str(state["codec_name"]), version=int(state["codec_version"]))
        if result.vocabulary_id.value != int(state["vocabulary_id"]):
            raise ValueError("symbol vocabulary identity mismatch")
        return result


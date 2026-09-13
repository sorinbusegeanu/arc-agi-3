from __future__ import annotations

from dataclasses import dataclass

from v9.memory.identity import SymbolId, SymbolStreamId, SymbolVocabularyId, stable_u64


@dataclass(frozen=True, slots=True)
class SymbolPosition:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("symbol position cannot be negative")


class SymbolVocabulary:
    def __init__(self, name: str, codec_name: str = "identity", version: int = 1) -> None:
        if not name or not codec_name or version <= 0:
            raise ValueError("symbol vocabulary identity fields must be valid")
        self.name = name
        self.codec_name = codec_name
        self.version = int(version)
        self.vocabulary_id = SymbolVocabularyId(stable_u64(codec_name, version, name, person=b"v9-symbol-vocab"))

    def symbol_id(self, raw: bytes) -> SymbolId:
        return SymbolId(stable_u64(self.vocabulary_id.value, raw, person=b"v9-symbol-id"))

    def stream_id(self, name: str | int) -> SymbolStreamId:
        return SymbolStreamId(stable_u64(self.vocabulary_id.value, str(name), person=b"v9-symbol-stream"))


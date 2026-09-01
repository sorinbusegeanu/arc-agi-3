from __future__ import annotations

from enum import IntEnum

from v8.model import stable_u64


class SymbolicPrimitive(IntEnum):
    SYMBOL_OCCURRED = 1
    SYMBOL_REPEATED = 2
    SYMBOL_PRECEDES_SYMBOL = 3
    SYMBOL_FOLLOWS_SYMBOL = 4
    SYMBOL_PRECEDES_ACTION = 5
    SYMBOL_FOLLOWS_ACTION = 6
    SYMBOL_PRECEDES_NORMALIZED_CHANGE = 7
    SYMBOL_FOLLOWS_NORMALIZED_CHANGE = 8
    SYMBOL_PRECEDES_BOUNDARY = 9
    SYMBOL_FOLLOWS_BOUNDARY = 10


# Keep the existing M1N marker and a low-byte NormalizedPrimitive value so old
# normalized-memory readers regard these as legitimate one-key contingencies. The
# v9 magic/primitive fields preserve exact symbolic structural identity without
# changing the v8 binary node schema.
_M1N_MARKER = 1 << 63
_V9_MAGIC = 0x2D
_LOW_BYTE_RELATION_APPEARED = 7


def symbol_normalized_token(primitive: SymbolicPrimitive, *structural_parts: object) -> int:
    payload = stable_u64(*structural_parts, person=b"v9-symbol-structure") & ((1 << 40) - 1)
    return int(
        _M1N_MARKER
        | (_V9_MAGIC << 56)
        | (payload << 16)
        | ((int(primitive) & 0xFF) << 8)
        | _LOW_BYTE_RELATION_APPEARED
    )


def is_symbol_normalized_token(value: int) -> bool:
    raw = int(value)
    return bool(raw & _M1N_MARKER and ((raw >> 56) & 0x7F) == _V9_MAGIC and (raw & 0xFF) == _LOW_BYTE_RELATION_APPEARED)


def symbol_normalized_primitive(value: int) -> SymbolicPrimitive:
    if not is_symbol_normalized_token(value):
        raise ValueError("not a v9 symbolic normalized token")
    return SymbolicPrimitive((int(value) >> 8) & 0xFF)


def symbol_family_key(value: int) -> tuple[int, int]:
    primitive = symbol_normalized_primitive(value)
    return (0x100 + int(primitive), 0)


def symbol_role_token(value: int) -> int:
    return stable_u64(int(symbol_normalized_primitive(value)), person=b"v9-symbol-role")

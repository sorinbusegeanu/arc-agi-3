from __future__ import annotations

import hashlib


def deterministic_game_seed(*, run_id: str, game_id: str, seed_base: int) -> int:
    payload = f"{run_id}|{game_id}|{int(seed_base)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")

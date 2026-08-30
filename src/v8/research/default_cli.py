from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from .researcher_packet import write_researcher_packet


def _requested_root(values: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default="runs/v8/continuous")
    parsed, _unknown = parser.parse_known_args(list(values))
    return Path(parsed.root)


def _is_normal_continuous_run(values: Sequence[str]) -> bool:
    values = tuple(str(value) for value in values)
    if "continuous-run" not in values:
        return False
    if "--show-best-trajectory" in values or "--save-best-trajectory" in values:
        return False
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--games", default=None)
    parsed, _unknown = parser.parse_known_args(list(values))
    return bool(parsed.games)


def run_with_default_research(
    main_func: Callable[[list[str] | None], int],
    argv: list[str] | None = None,
) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    result = int(main_func(values))
    if result != 0 or not _is_normal_continuous_run(values):
        return result

    root = _requested_root(values)
    try:
        packet = write_researcher_packet(root, argv=values)
    except BaseException as exc:
        research_root = root / "research"
        research_root.mkdir(parents=True, exist_ok=True)
        (research_root / "LLM_RESEARCH_PACKET_ERROR.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        print(
            f'[{time.strftime("%H:%M")}] LLM research packet failed: '
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return result

    print(
        f'[{time.strftime("%H:%M")}] LLM research packet ready: {packet}',
        flush=True,
    )
    return result

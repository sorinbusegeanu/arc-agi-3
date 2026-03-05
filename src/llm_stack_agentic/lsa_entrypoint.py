from __future__ import annotations

from .lsa_cli import main


def run() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(run())

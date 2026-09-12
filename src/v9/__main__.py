"""Compatibility entrypoint for the optional v9 package transition.

The existing `python -m v8 ...` command remains authoritative until runtime
stack consolidation is explicitly unlocked. This entrypoint deliberately
reuses the current CLI contract instead of introducing parameter drift.
"""

from v8.cli_v819 import main as _v8_main
from v8.runtime_observability_v836 import stdout_log_context


def main() -> int:
    with stdout_log_context():
        return int(_v8_main())


if __name__ == "__main__":
    raise SystemExit(main())

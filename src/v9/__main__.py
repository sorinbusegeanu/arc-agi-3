"""Compatibility entrypoint for the optional v9 package transition.

The existing `python -m v8 ...` command remains authoritative until runtime
stack consolidation is explicitly unlocked. This entrypoint deliberately
reuses the current CLI contract instead of introducing parameter drift.
"""

from v8.cli_v819 import main as _v8_main
from v8.research.default_cli import run_with_default_research
from v8.runtime_observability_v836 import stdout_log_context


def main() -> int:
    with stdout_log_context():
        return int(run_with_default_research(_v8_main))


if __name__ == "__main__":
    raise SystemExit(main())

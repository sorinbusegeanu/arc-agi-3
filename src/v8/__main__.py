from v8.cli_v819 import main
from v8.runtime_observability_v836 import stdout_log_context


def run(argv: list[str] | None = None) -> int:
    """Run the v8 CLI without imposing a research-experiment lifecycle."""
    with stdout_log_context(argv):
        return int(main(argv))


if __name__ == "__main__":
    raise SystemExit(run())

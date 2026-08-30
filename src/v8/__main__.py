from v8.cli_v819 import main
from v8.research.default_cli import run_with_default_research
from v8.runtime_observability_v836 import stdout_log_context


if __name__ == "__main__":
    with stdout_log_context():
        raise SystemExit(run_with_default_research(main))

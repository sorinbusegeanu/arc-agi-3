import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)


def _consume_reset_memory_flag() -> None:
    if "--reset-memory" not in sys.argv:
        return
    if "--transfer-validation" in sys.argv:
        raise SystemExit("--transfer-validation cannot be combined with --reset-memory")
    sys.argv[:] = [value for value in sys.argv if value != "--reset-memory"]
    os.environ["ARC_AGI3_V9_RESET_MEMORY"] = "1"


if __name__ == "__main__":
    _consume_reset_memory_flag()

    # Start the multiprocessing server before importing the full CLI and before
    # ContinuousMemoryRuntime restores a multi-gigabyte graph. Later actors and
    # pipeline workers are forked by this small server rather than started from
    # the restored parent process.
    from v9.runtime.multiprocess import ensure_process_server_ready

    ensure_process_server_ready()

    import v9.cli as cli
    from v9.environments.passive_capture import install_cli_adapter_factory

    install_cli_adapter_factory(cli)
    raise SystemExit(cli.main())

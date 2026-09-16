import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)


if __name__ == "__main__":
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

"""Local vLLM server startup for opt-in CLI runs."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def _local_target(host: str, port: int) -> bool:
    """Return whether connecting to host routes to an address on this machine."""
    try:
        targets = {
            item[4][0]
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
    except OSError:
        return False
    for target in targets:
        try:
            family = socket.AF_INET6 if ":" in target else socket.AF_INET
            with socket.socket(family, socket.SOCK_DGRAM) as probe:
                probe.connect((target, port))
                source = probe.getsockname()[0]
            if source == target or target in {"127.0.0.1", "::1"}:
                return True
        except OSError:
            continue
    return False


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _port_is_open(host: str, port: int) -> bool:
    for family, socktype, proto, _, address in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM,
    ):
        try:
            with socket.socket(family, socktype, proto) as connection:
                connection.settimeout(0.5)
                if connection.connect_ex(address) == 0:
                    return True
        except OSError:
            continue
    return False


def _read_models(base_url: str, api_key: str | None) -> list[str] | None:
    request = urllib.request.Request(_models_url(base_url), method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"vLLM endpoint at {base_url} requires different credentials"
            ) from exc
        if exc.code in {404, 502, 503, 504}:
            return None
        raise RuntimeError(f"vLLM endpoint returned HTTP {exc.code}: {exc.reason}") from exc
    except (OSError, urllib.error.URLError, TimeoutError):
        return None
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("vLLM endpoint returned invalid model metadata") from exc
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("vLLM endpoint returned no model list")
    return [str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id")]


def _tool_call_parser(model: str) -> str:
    model_id = model.casefold()
    if "qwen3-coder" in model_id:
        return "qwen3_xml"
    # Qwen3 and Qwen3-VL use the Hermes-style <tool_call> JSON format.
    if "qwen3" in model_id:
        return "hermes"
    return "hermes"


def _stop_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 5.0) -> None:
    """Stop the managed server and any engine workers it left behind."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=0.1)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass


def _server_command(
    model: str, host: str, port: int, max_model_len: int,
    tool_call_parser: str | None = None,
) -> list[str]:
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "serve", model, "--host", host, "--port", str(port),
        "--max-model-len", str(max_model_len),
    ]
    if "nvfp4" in model.casefold():
        command.extend(["--linear-backend", "flashinfer_b12x"])
    command.extend([
        "--enable-auto-tool-choice",
        "--tool-call-parser", tool_call_parser or _tool_call_parser(model),
    ])
    return command


def ensure_vllm_server(
    *,
    base_url: str,
    model: str,
    startup_timeout_seconds: float = 900.0,
    max_model_len: int = 65536,
    tool_call_parser: str | None = None,
) -> None:
    """Reuse a matching server or start vLLM locally and wait until it is ready."""
    if startup_timeout_seconds <= 0:
        raise ValueError("vLLM startup timeout must be positive")
    if max_model_len <= 0:
        raise ValueError("vLLM max model length must be positive")
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("vLLM auto-start requires an http:// local --base-url")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("vLLM auto-start requires a valid endpoint port") from exc
    if port is None:
        raise ValueError("vLLM auto-start requires an explicit port in --base-url")
    if not 1 <= port <= 65535:
        raise ValueError("vLLM endpoint port must be between 1 and 65535")
    if not _local_target(parsed.hostname, port):
        raise ValueError(
            f"cannot auto-start vLLM for remote endpoint {base_url}; "
            "use a local --base-url or start the remote server separately"
        )

    api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("STRATA_API_KEY")
    running_models = _read_models(base_url, api_key)
    if running_models is not None:
        if model in running_models:
            print(f"[v10] reusing vLLM server for {model} at {base_url}", file=sys.stderr, flush=True)
            return
        raise RuntimeError(
            f"vLLM endpoint at {base_url} is already serving {running_models}; "
            f"requested {model}"
        )

    # If another local process already owns the port but has not loaded its model
    # yet, wait for it instead of attempting a second server on the same port.
    if _port_is_open(parsed.hostname, port):
        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            running_models = _read_models(base_url, api_key)
            if running_models is not None:
                if model in running_models:
                    print(f"[v10] reusing vLLM server for {model} at {base_url}", file=sys.stderr, flush=True)
                    return
                raise RuntimeError(
                    f"vLLM endpoint at {base_url} is already serving {running_models}, not {model}"
                )
            time.sleep(1.0)
        raise RuntimeError(
            f"a local service already owns {parsed.hostname}:{port} but did not expose a ready "
            "vLLM model endpoint"
        )

    try:
        import vllm  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"vLLM is not installed in {sys.executable}; install it in this Python environment"
        ) from exc

    log_root = Path.home() / ".cache" / "v10" / "vllm"
    log_root.mkdir(parents=True, exist_ok=True)
    safe_model = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    log_path = log_root / f"{port}-{safe_model}.log"
    command = _server_command(model, parsed.hostname, port, max_model_len, tool_call_parser)
    with log_path.open("ab") as log_file:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"could not start vLLM; see {log_path}: {exc}") from exc
    print(
        f"[v10] started vLLM PID {process.pid} for {model}; waiting up to "
        f"{startup_timeout_seconds:g}s (log: {log_path})",
        file=sys.stderr,
        flush=True,
    )
    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        status = process.poll()
        if status is not None:
            _stop_process_group(process)
            raise RuntimeError(
                f"vLLM exited with status {status} before becoming ready; see {log_path}"
            )
        running_models = _read_models(base_url, api_key)
        if running_models is not None:
            if model not in running_models:
                _stop_process_group(process)
                raise RuntimeError(
                    f"vLLM became ready at {base_url} but serves {running_models}, not {model}; "
                    f"see {log_path}"
                )
            print(f"[v10] vLLM is ready for {model}", file=sys.stderr, flush=True)
            return
        time.sleep(1.0)
    _stop_process_group(process)
    raise RuntimeError(
        f"vLLM did not become ready within {startup_timeout_seconds:g}s; see {log_path}"
    )


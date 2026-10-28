from __future__ import annotations

from pathlib import Path
import sys
import signal
from types import ModuleType
from unittest.mock import Mock

import pytest

from v10 import vllm_server
from v10 import cli as cli_module
from v10.cli import build_parser


def test_autostart_reuses_matching_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_server, "_local_target", lambda host, port: True)
    monkeypatch.setattr(vllm_server, "_read_models", lambda base_url, api_key: ["org/model"])
    popen = Mock(side_effect=AssertionError("should reuse the running server"))
    monkeypatch.setattr(vllm_server.subprocess, "Popen", popen)

    vllm_server.ensure_vllm_server(base_url="http://127.0.0.1:8000", model="org/model")

    popen.assert_not_called()


def test_autostart_launches_model_and_waits_until_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setattr(vllm_server, "_local_target", lambda host, port: True)
    responses = iter((None, ["org/model"]))
    monkeypatch.setattr(vllm_server, "_read_models", lambda base_url, api_key: next(responses))
    monkeypatch.setattr(vllm_server, "_port_is_open", lambda host, port: False)
    monkeypatch.setattr(vllm_server.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(vllm_server.time, "sleep", lambda seconds: None)
    process = Mock()
    process.pid = 123
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(vllm_server.subprocess, "Popen", popen)

    vllm_server.ensure_vllm_server(
        base_url="http://127.0.0.1:8000", model="org/model", max_model_len=8192,
    )

    command = popen.call_args.args[0]
    assert command[:3] == [vllm_server.sys.executable, "-m", "vllm.entrypoints.cli.main"]
    assert command[3:5] == ["serve", "org/model"]
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert "--linear-backend" not in command
    assert popen.call_args.kwargs["start_new_session"] is True
    assert list((tmp_path / ".cache/v10/vllm").glob("8000-*.log"))


def test_autostart_cleans_process_group_after_early_server_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setattr(vllm_server, "_local_target", lambda host, port: True)
    monkeypatch.setattr(vllm_server, "_read_models", lambda base_url, api_key: None)
    monkeypatch.setattr(vllm_server, "_port_is_open", lambda host, port: False)
    monkeypatch.setattr(vllm_server.Path, "home", lambda: tmp_path)
    process = Mock()
    process.pid = 456
    process.poll.return_value = 1
    popen = Mock(return_value=process)
    monkeypatch.setattr(vllm_server.subprocess, "Popen", popen)
    killpg = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(vllm_server.os, "killpg", killpg)

    with pytest.raises(RuntimeError, match="exited with status 1"):
        vllm_server.ensure_vllm_server(
            base_url="http://127.0.0.1:8000", model="org/model",
        )

    killpg.assert_called_once_with(456, signal.SIGTERM)


def test_nvfp4_autostart_selects_sm120_backend() -> None:
    command = vllm_server._server_command(
        "kaitchup/Qwen3-VL-2B-Instruct-NVFP4", "192.168.0.51", 8000, 65536,
    )

    assert command[command.index("--linear-backend") + 1] == "flashinfer_b12x"
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "hermes"


def test_vllm_server_uses_qwen3_coder_parser() -> None:
    command = vllm_server._server_command("Qwen/Qwen3-Coder-30B-A3B-Instruct", "127.0.0.1", 8000, 65536)

    assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"


def test_vllm_server_allows_parser_override() -> None:
    command = vllm_server._server_command("org/model", "127.0.0.1", 8000, 65536, "llama3_json")

    assert command[command.index("--tool-call-parser") + 1] == "llama3_json"


def test_autostart_does_not_launch_for_remote_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_server, "_local_target", lambda host, port: False)
    with pytest.raises(ValueError, match="remote endpoint"):
        vllm_server.ensure_vllm_server(base_url="http://model-host:8000", model="org/model")


def test_autostart_rejects_another_model_on_the_local_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_server, "_local_target", lambda host, port: True)
    monkeypatch.setattr(vllm_server, "_read_models", lambda base_url, api_key: ["org/other"])

    with pytest.raises(RuntimeError, match="requested org/model"):
        vllm_server.ensure_vllm_server(base_url="http://127.0.0.1:8000", model="org/model")


def test_vllm_autostart_is_opt_in_and_runs_before_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = build_parser().parse_args(["analyze", "--run-dir", "run", "--vllm"])
    assert defaults.vllm_autostart is False

    called = []
    monkeypatch.setattr(cli_module, "ensure_vllm_server", lambda **kwargs: called.append(kwargs))
    monkeypatch.setattr(cli_module, "StrataClient", lambda **kwargs: object())
    monkeypatch.setattr(cli_module, "run_analysis", lambda *args, **kwargs: {"ok": True})

    result = cli_module.main([
        "analyze", "--run-dir", "run", "--vllm", "--vllm-autostart",
        "--model", "org/model",
    ])

    assert result == 0
    assert called == [{
        "base_url": "http://192.168.0.51:8000",
        "model": "org/model",
        "startup_timeout_seconds": 900.0,
        "max_model_len": 65536,
        "tool_call_parser": None,
    }]

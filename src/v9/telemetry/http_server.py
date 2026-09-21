from __future__ import annotations

from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable


DASHBOARD_REFRESH_SECONDS = 30.0
FULL_TELEMETRY_LOG_INTERVAL = 120


class MetricsHTTPServer:
    def __init__(
        self,
        metrics_provider: Callable[[], dict[str, Any]],
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        log_path: str | Path | None = None,
        refresh_seconds: float = DASHBOARD_REFRESH_SECONDS,
    ) -> None:
        self.metrics_provider = metrics_provider
        owner = getattr(metrics_provider, "__self__", None)
        dashboard_provider = getattr(owner, "dashboard_metrics", None)
        provider = dashboard_provider if callable(dashboard_provider) else metrics_provider
        self._dashboard_provider = provider
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        if log_path is None:
            config = getattr(owner, "config", None)
            root = getattr(config, "root", None)
            if root is not None:
                log_path = Path(root) / "telemetry" / "dashboard_metrics.jsonl"
        self.log_path = None if log_path is None else Path(log_path)
        self._stop_logging = Event()
        self._log_thread: Thread | None = None
        self._log_sequence = 0

        class Handler(BaseHTTPRequestHandler):
            def _write(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                if self.path in {"/api/metrics", "/metrics"}:
                    try:
                        snapshot = provider()
                    except Exception as exc:
                        raw = json.dumps(
                            {
                                "dashboard_metrics_error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            },
                            sort_keys=True,
                            default=str,
                        ).encode("utf-8")
                        self._write(503, "application/json; charset=utf-8", raw)
                        return
                    raw = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
                    self._write(200, "application/json; charset=utf-8", raw)
                    return
                if self.path in {"/", "/dashboard"}:
                    refresh_ms = int(round(self.server.refresh_seconds * 1000.0))
                    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hydra v9 Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
.card{{background:#1d1d1d;padding:12px;border-radius:8px}}.k{{color:#aaa;font-size:12px}}.v{{font-size:20px;margin-top:4px}}
pre{{background:#1d1d1d;padding:12px;overflow:auto}}
</style></head>
<body><h1>Hydra v9.7.6</h1><div id="grid" class="grid"></div>
<script>
async function refresh(){{
 const r=await fetch('/api/metrics',{{cache:'no-store'}}); const m=await r.json();
 const p=m.primary_dashboard||{{}}; const g=document.getElementById('grid'); g.innerHTML='';
 for(const [k,v] of Object.entries(p)){{const d=document.createElement('div');d.className='card';d.innerHTML='<div class="k">'+k+'</div><div class="v">'+v+'</div>';g.appendChild(d)}}
}}
refresh(); setInterval(refresh,{refresh_ms});
</script></body></html>""".encode("utf-8")
                    self._write(200, "text/html; charset=utf-8", html)
                    return
                self._write(404, "text/plain; charset=utf-8", b"not found")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((str(host), int(port)), Handler)
        self._server.refresh_seconds = self.refresh_seconds
        self.host = str(host)
        self.port = int(port)
        self._thread = Thread(target=self._server.serve_forever, name="v9-metrics-http", daemon=True)

    @staticmethod
    def _compact_log_snapshot(raw: dict[str, Any], *, full: bool) -> dict[str, Any]:
        if full:
            return dict(raw)
        result: dict[str, Any] = {}
        primary = raw.get("primary_dashboard")
        if isinstance(primary, dict):
            result["primary_dashboard"] = dict(primary)
        diagnostic = raw.get("telemetry_diagnostics")
        if isinstance(diagnostic, dict):
            result["telemetry_diagnostics"] = {
                str(key): value
                for key, value in diagnostic.items()
                if value is None or isinstance(value, (str, int, float, bool))
            }
        for key, value in raw.items():
            if key in {"primary_dashboard", "telemetry_diagnostics"}:
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                result[str(key)] = value
        return result

    def _rotate_existing_log(self) -> None:
        if self.log_path is None or not self.log_path.exists() or self.log_path.stat().st_size <= 0:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = self.log_path.with_name(
            f"{self.log_path.stem}.{stamp}{self.log_path.suffix}"
        )
        self.log_path.replace(archived)

    def _log_dashboard_metrics(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", buffering=1) as handle:
            while not self._stop_logging.is_set():
                timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    self._log_sequence += 1
                    full = self._log_sequence % FULL_TELEMETRY_LOG_INTERVAL == 0
                    raw = self._dashboard_provider()
                    snapshot = {
                        "timestamp_utc": timestamp,
                        "telemetry_log_full": full,
                        **self._compact_log_snapshot(raw, full=full),
                    }
                except Exception as exc:
                    snapshot = {
                        "timestamp_utc": timestamp,
                        "dashboard_metrics_error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                handle.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")
                if self._stop_logging.wait(self.refresh_seconds):
                    break

    def start(self) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_existing_log()
        self._thread.start()
        if self.log_path is not None:
            self._log_thread = Thread(target=self._log_dashboard_metrics, name="v9-dashboard-telemetry-log", daemon=True)
            self._log_thread.start()

    def close(self) -> None:
        self._stop_logging.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        if self._log_thread is not None:
            self._log_thread.join(timeout=5.0)

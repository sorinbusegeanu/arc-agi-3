from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Callable


DASHBOARD_REFRESH_SECONDS = 30.0
DASHBOARD_LIVE_REFRESH_SECONDS = 2.0


_COMPACT_LOG_KEYS = (
    "watermark",
    "graph_generation",
    "memories",
    "memory_levels",
    "M0_resident",
    "M1_grounded_resident",
    "resident_M0_limit",
    "resident_M1_grounded_limit",
    "compaction_backlog",
    "compaction_cycles",
    "compaction_seconds",
    "low_level_nodes_inserted",
    "low_level_nodes_deleted",
    "low_level_dedup_rate",
    "concrete_admission_retained_events",
    "concrete_admission_skipped_events",
    "concrete_admission_retention_rate",
    "concrete_nodes_avoided",
    "process_rss_bytes",
    "memory_governor_state",
)

_VISIBLE_FALLBACK_KEYS = (
    "run_phase",
    "sampled_steps",
    "actor_produced_steps",
    "causally_admitted_steps",
    "publication_drained_steps",
    "ingested_steps",
    "sampling_backlog",
    "publication_backlog",
    "canonical_ingest_backlog",
    "sampling_rate",
    "ingestion_rate",
    "derivation_rate",
    "canonical_batch_size",
    "canonical_apply_latency_ms",
    "memories",
    "edges",
    "watermark",
    "graph_generation",
    "process_rss_bytes",
    "memory_governor_state",
)


def _compact_log_snapshot(snapshot: dict[str, Any], *, timestamp: str) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "primary_dashboard": dict(snapshot.get("primary_dashboard", {})),
    }
    for key in _COMPACT_LOG_KEYS:
        if key in snapshot:
            compact[key] = snapshot[key]
    return compact


def _format_dashboard_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _visible_primary_dashboard(snapshot: dict[str, Any]) -> dict[str, Any]:
    primary = dict(snapshot.get("primary_dashboard", {}) or {})
    diagnostics = dict(snapshot.get("telemetry_diagnostics", {}) or {})
    for key in _VISIBLE_FALLBACK_KEYS:
        if key in primary:
            continue
        if key in snapshot:
            primary[key] = snapshot[key]
        elif key in diagnostics:
            primary[key] = diagnostics[key]
    levels = snapshot.get("memory_levels")
    if isinstance(levels, dict):
        for level in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"):
            if level in levels and level not in primary:
                primary[level] = levels[level]
    error = snapshot.get("dashboard_metrics_error")
    if isinstance(error, dict):
        primary["dashboard_metrics_error"] = f"{error.get('type', 'Error')}: {error.get('message', '')}"
    if not primary:
        primary["dashboard_status"] = "no metrics available yet"
    return primary


def _render_dashboard_cards(primary: dict[str, Any]) -> str:
    rows = []
    for key, value in primary.items():
        rows.append(
            '<div class="card"><div class="k">'
            + escape(str(key))
            + '</div><div class="v">'
            + escape(_format_dashboard_value(value))
            + "</div></div>"
        )
    return "".join(rows)


def _render_model_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No trained models yet"
    rows = []
    for row in history:
        rows.append(
            f"{row.get('model', '')} | levels={row.get('run_levels_solved', 0)} "
            f"| wins={float(row.get('run_wins', 0.0)):.4f} "
            f"| behavioral={float(row.get('behavioral_success_rate', 0.0)):.4f} "
            f"| VRAM={float(row.get('vram_gb', 0.0)):.2f} GB"
        )
    return "\n".join(rows)


class MetricsHTTPServer:
    def __init__(
        self,
        metrics_provider: Callable[[], dict[str, Any]],
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        log_path: str | Path | None = None,
        refresh_seconds: float = DASHBOARD_REFRESH_SECONDS,
        live_refresh_seconds: float = DASHBOARD_LIVE_REFRESH_SECONDS,
    ) -> None:
        self.metrics_provider = metrics_provider
        owner = getattr(metrics_provider, "__self__", None)
        dashboard_provider = getattr(owner, "dashboard_metrics", None)
        provider = dashboard_provider if callable(dashboard_provider) else metrics_provider
        self._dashboard_provider = provider
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        self.log_refresh_seconds = self.refresh_seconds
        self.live_refresh_seconds = max(0.1, float(live_refresh_seconds))
        if log_path is None:
            config = getattr(owner, "config", None)
            root = getattr(config, "root", None)
            if root is not None:
                log_path = Path(root) / "telemetry" / "dashboard_metrics.jsonl"
        self.log_path = None if log_path is None else Path(log_path)
        self._stop_logging = Event()
        self._log_thread: Thread | None = None
        self._stop_live = Event()
        self._live_thread: Thread | None = None
        self._live_cache_lock = RLock()
        self._live_cache: dict[str, Any] = {}
        self._live_cache_error: dict[str, str] | None = None
        self._serving = False
        self._model_history_lock = RLock()
        self._model_history: dict[str, dict[str, Any]] = {}
        server_owner = self

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
                    snapshot = server_owner._dashboard_snapshot()
                    raw = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
                    self._write(200, "application/json; charset=utf-8", raw)
                    return
                if self.path in {"/", "/dashboard"}:
                    initial_snapshot = server_owner._dashboard_snapshot()
                    initial_primary = _visible_primary_dashboard(initial_snapshot)
                    initial_cards = _render_dashboard_cards(initial_primary)
                    initial_history = escape(
                        _render_model_history(
                            list(initial_snapshot.get("model_history", []) or [])
                        )
                    )
                    refresh_ms = int(round(self.server.refresh_seconds * 1000.0))
                    fallback_keys = json.dumps(_VISIBLE_FALLBACK_KEYS)
                    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hydra v9 Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:20px}}h2{{font-size:16px;margin-top:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
.card{{background:#1d1d1d;padding:12px;border-radius:8px}}.k{{color:#aaa;font-size:12px}}.v{{font-size:20px;margin-top:4px;word-break:break-word}}
pre{{background:#1d1d1d;padding:12px;overflow:auto;line-height:1.5}}
</style></head>
<body><h1>Hydra v9.7.6</h1><div id="grid" class="grid">{initial_cards}</div>
<h2>Model history</h2><pre id="model-history">{initial_history}</pre>
<script>
const fallbackKeys = {fallback_keys};
function display(v, digits){{
 if(typeof v === 'number') return v.toFixed(digits);
 if(v && typeof v === 'object') return JSON.stringify(v);
 return String(v ?? '');
}}
function visiblePrimary(m){{
 const p = Object.assign({{}}, m.primary_dashboard || {{}});
 const d = m.telemetry_diagnostics || {{}};
 for(const k of fallbackKeys){{
  if(p[k] !== undefined) continue;
  if(m[k] !== undefined) p[k] = m[k];
  else if(d[k] !== undefined) p[k] = d[k];
 }}
 const levels = m.memory_levels || {{}};
 for(const k of ['M0','M1','M2','M3','M4','M5','M6','M7']){{
  if(p[k] === undefined && levels[k] !== undefined) p[k] = levels[k];
 }}
 if(m.dashboard_metrics_error) p.dashboard_metrics_error = display(m.dashboard_metrics_error.type,0)+': '+display(m.dashboard_metrics_error.message,0);
 if(!Object.keys(p).length) p.dashboard_status = 'no metrics available yet';
 return p;
}}
async function refresh(){{
 const g=document.getElementById('grid');
 try {{
  const r=await fetch('/api/metrics',{{cache:'no-store'}}); const m=await r.json();
  const p=visiblePrimary(m); g.innerHTML='';
  for(const [k,v] of Object.entries(p)){{const d=document.createElement('div');d.className='card';d.innerHTML='<div class="k">'+k+'</div><div class="v">'+display(v,2)+'</div>';g.appendChild(d)}}
  const history=m.model_history||[]; const h=document.getElementById('model-history');
  h.textContent=history.length ? history.map(x => x.model+' | levels='+x.run_levels_solved+' | wins='+display(x.run_wins,4)+' | behavioral='+display(x.behavioral_success_rate,4)+' | VRAM='+display(x.vram_gb,2)+' GB').join('\n') : 'No trained models yet';
 }} catch(e) {{
  g.innerHTML='<div class="card"><div class="k">dashboard_fetch_error</div><div class="v">'+String(e)+'</div></div>';
 }}
}}
refresh(); setInterval(refresh,{refresh_ms});
</script></body></html>""".encode("utf-8")
                    self._write(200, "text/html; charset=utf-8", html)
                    return
                self._write(404, "text/plain; charset=utf-8", b"not found")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((str(host), int(port)), Handler)
        self._server.refresh_seconds = self.live_refresh_seconds
        self.host = str(host)
        self.port = int(port)
        self._thread = Thread(target=self._server.serve_forever, name="v9-metrics-http", daemon=True)

    def _record_model_history(self, snapshot: dict[str, Any]) -> None:
        primary = dict(snapshot.get("primary_dashboard", {}))
        model = str(primary.get("ModelVersion", "untrained"))
        if not model or model == "untrained":
            return
        row = {
            "model": model,
            "run_levels_solved": int(primary.get("current_run_levels_solved", 0)),
            "run_wins": float(primary.get("current_run_wins", 0.0)),
            "behavioral_success_rate": float(primary.get("behavioral_success_rate", 0.0)),
            "vram_gb": float(primary.get("GPU_memory_GB", 0.0)),
        }
        with self._model_history_lock:
            self._model_history[model] = row

    def _model_history_snapshot(self) -> list[dict[str, Any]]:
        with self._model_history_lock:
            return [dict(row) for row in self._model_history.values()]

    def _publish_live_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        published = dict(snapshot)
        self._record_model_history(published)
        published["model_history"] = self._model_history_snapshot()
        with self._live_cache_lock:
            self._live_cache = published
            self._live_cache_error = None
        return published

    def _publish_live_error(self, exc: BaseException) -> None:
        with self._live_cache_lock:
            self._live_cache_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    def _capture_live_snapshot(self) -> dict[str, Any]:
        return self._publish_live_snapshot(dict(self._dashboard_provider()))

    def _dashboard_snapshot(self) -> dict[str, Any]:
        # Direct pre-start calls keep the established provider semantics used by
        # tests/tooling. Once serving, HTTP requests only read this server-owned
        # cache and therefore never block on the runtime lock.
        if not self._serving:
            return self._capture_live_snapshot()
        with self._live_cache_lock:
            snapshot = dict(self._live_cache)
            error = None if self._live_cache_error is None else dict(self._live_cache_error)
        if not snapshot:
            snapshot = {
                "primary_dashboard": {"dashboard_status": "starting"},
                "model_history": self._model_history_snapshot(),
            }
        if error is not None:
            snapshot["dashboard_metrics_error"] = error
        return snapshot

    def _live_dashboard_loop(self) -> None:
        # The logger performs the first immediate provider read. The faster live
        # sampler starts afterwards, avoiding races with the auditable first log row.
        if self._stop_live.wait(self.live_refresh_seconds):
            return
        while not self._stop_live.is_set():
            try:
                self._capture_live_snapshot()
            except Exception as exc:
                self._publish_live_error(exc)
            if self._stop_live.wait(self.live_refresh_seconds):
                break

    def _log_dashboard_metrics(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8", buffering=1) as handle:
            while not self._stop_logging.is_set():
                timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    full_snapshot = dict(self._dashboard_provider())
                    self._publish_live_snapshot(full_snapshot)
                    snapshot = _compact_log_snapshot(full_snapshot, timestamp=timestamp)
                except Exception as exc:
                    self._publish_live_error(exc)
                    snapshot = {
                        "timestamp_utc": timestamp,
                        "dashboard_metrics_error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                handle.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")
                if self._stop_logging.wait(self.log_refresh_seconds):
                    break

    def start(self) -> None:
        self._serving = True
        self._thread.start()
        if self.log_path is not None:
            self._log_thread = Thread(target=self._log_dashboard_metrics, name="v9-dashboard-telemetry-log", daemon=True)
            self._log_thread.start()
        else:
            try:
                self._capture_live_snapshot()
            except Exception as exc:
                self._publish_live_error(exc)
        self._live_thread = Thread(target=self._live_dashboard_loop, name="v9-dashboard-live", daemon=True)
        self._live_thread.start()

    def close(self) -> None:
        self._stop_logging.set()
        self._stop_live.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        if self._log_thread is not None:
            self._log_thread.join(timeout=5.0)
        if self._live_thread is not None:
            self._live_thread.join(timeout=5.0)
        self._serving = False

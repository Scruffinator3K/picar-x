from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from urllib.parse import urlparse, parse_qs
from typing import Callable, Dict, Any, Optional

from .logging_setup import init_logger


class TelemetryServer:
    def __init__(
        self,
        addr: str,
        port: int,
        get_status: Callable[[], Dict[str, Any]],
        on_heartbeat: Optional[Callable[[], None]] = None,
        on_set_mode: Optional[Callable[[str], None]] = None,
        on_manual: Optional[Callable[[int, int], None]] = None,
        get_snapshot: Optional[Callable[[], Optional[bytes]]] = None,
        logger_name: str = "picarx.telemetry",
    ) -> None:
        self.log = init_logger(logger_name)
        self._addr = addr
        self._port = int(port)
        self._get_status = get_status
        self._on_heartbeat = on_heartbeat
        self._on_set_mode = on_set_mode
        self._on_manual = on_manual
        self._get_snapshot = get_snapshot
        self._srv: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._srv is not None:
            return

        get_status = self._get_status
        on_beat = self._on_heartbeat
        on_mode = self._on_set_mode
        on_manual = self._on_manual
        get_snapshot = self._get_snapshot
        log = self.log

        class Handler(BaseHTTPRequestHandler):
            def _serve_index(self):
                html = (
                    "<html><head><meta name='viewport' content='width=device-width, initial-scale=1'>"
                    "<style>body{font-family:sans-serif;margin:1rem} button{margin:.25rem}</style>"
                    "</head><body>"
                    "<h2>Picar-X Dashboard</h2>"
                    "<div id='status'></div>"
                    "<div>Mode: <button onclick=fetch('/mode?m=auto')>Auto</button>"
                    "<button onclick=fetch('/mode?m=manual')>Manual</button></div>"
                    "<div>Manual: <button onclick=fetch('/manual?speed=50')>Fwd</button>"
                    "<button onclick=fetch('/manual?speed=-50')>Back</button>"
                    "<button onclick=fetch('/manual?speed=0')>Stop</button><br>"
                    "Steer: <button onclick=fetch('/manual?steer=-20')>Left</button>"
                    "<button onclick=fetch('/manual?steer=0')>Center</button>"
                    "<button onclick=fetch('/manual?steer=20')>Right</button></div>"
                    "<div><img id='snap' style='max-width:100%'></div>"
                    "<script>async function refresh(){const r=await fetch('/status');"
                    "const j=await r.json();document.getElementById('status').innerText=JSON.stringify(j);"
                    "const s=await fetch('/snapshot'); if(s.ok){const b=await s.blob();"
                    "document.getElementById('snap').src=URL.createObjectURL(b);} }"
                    "setInterval(refresh,1000); refresh();</script>"
                    "</body></html>"
                ).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type','text/html')
                self.send_header('Content-Length', str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    return self._serve_index()
                if self.path.startswith("/status"):
                    try:
                        data = get_status()
                    except Exception as e:
                        data = {"error": str(e)}
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/heartbeat"):
                    if on_beat:
                        try:
                            on_beat()
                        except Exception:
                            pass
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/mode"):
                    qs = parse_qs(urlparse(self.path).query)
                    m = (qs.get('m') or [''])[0]
                    if on_mode and m in ("auto","manual"):
                        try:
                            on_mode(m)
                        except Exception:
                            pass
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/manual"):
                    qs = parse_qs(urlparse(self.path).query)
                    sp = qs.get('speed'); st = qs.get('steer')
                    try:
                        speed = int(sp[0]) if sp else None
                        steer = int(st[0]) if st else None
                    except Exception:
                        speed = steer = None
                    if on_manual and (speed is not None or steer is not None):
                        try:
                            on_manual(speed or 0, steer or 0)
                        except Exception:
                            pass
                    self.send_response(204)
                    self.end_headers()
                elif self.path.startswith("/snapshot"):
                    if get_snapshot:
                        data = get_snapshot()
                    else:
                        data = None
                    if data:
                        self.send_response(200)
                        self.send_header('Content-Type','image/jpeg')
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        self.send_response(204)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                log.debug("HTTP: " + format % args)

        self._srv = HTTPServer((self._addr, self._port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        self.log.info(f"Telemetry server started on http://{self._addr}:{self._port}")

    def stop(self):
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.log.info("Telemetry server stopped")

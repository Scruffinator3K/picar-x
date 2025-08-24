from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from typing import Callable, Dict, Any, Optional

from .logging_setup import init_logger


class TelemetryServer:
    def __init__(
        self,
        addr: str,
        port: int,
        get_status: Callable[[], Dict[str, Any]],
        on_heartbeat: Optional[Callable[[], None]] = None,
        logger_name: str = "picarx.telemetry",
    ) -> None:
        self.log = init_logger(logger_name)
        self._addr = addr
        self._port = int(port)
        self._get_status = get_status
        self._on_heartbeat = on_heartbeat
        self._srv: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._srv is not None:
            return

        get_status = self._get_status
        on_beat = self._on_heartbeat
        log = self.log

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
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

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .upstream_detector_port import Detector
except ImportError:  # pragma: no cover - direct-file validation/import fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from upstream_detector_port import Detector

_DETECTOR = Detector()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _log(event: str, level: str = "INFO", **fields: Any) -> None:
    payload: dict[str, Any] = {
        "ts": time.time(),
        "level": level,
        "event": event,
        "skill": "neural_nlu_service_skill",
        "pid": os.getpid(),
    }
    payload.update(fields)
    try:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    except Exception:
        print(f"{level} {event} {fields!r}", flush=True)


def _text_preview(text: str, limit: int = 160) -> str:
    token = str(text or "").replace("\n", " ").strip()
    if len(token) <= limit:
        return token
    return token[: limit - 1] + "?"


class Handler(BaseHTTPRequestHandler):
    server_version = "AdaOSNeuralNLU/0.2"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: Any) -> None:  # noqa: A003
        if _env_flag("ADAOS_NEURAL_HTTP_LOG"):
            super().log_message(_format, *args)

    def do_GET(self) -> None:  # noqa: N802
        started = time.perf_counter()
        if self.path == "/health":
            health = _DETECTOR.health()
            if _env_flag("ADAOS_NEURAL_HEALTH_LOG"):
                _log(
                    "health",
                    status=200,
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    model_loaded=bool(health.get("model_loaded")),
                    examples_total=health.get("examples_total"),
                    artifact_root=health.get("artifact_root"),
                )
            self._json(200, health)
            return
        _log("http.not_found", level="WARNING", method="GET", path=self.path, status=404)
        self._json(404, {"ok": False, "error": "not_found"})

    def _read_payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except Exception:
            length = 0
        raw = self.rfile.read(max(length, 0)) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def do_POST(self) -> None:  # noqa: N802
        started = time.perf_counter()
        if self.path == "/reindex":
            payload = self._read_payload()
            try:
                result = _DETECTOR.reindex(purge_indexes=bool(payload.get("purge_indexes")))
            except Exception as exc:
                _log(
                    "reindex.error",
                    level="ERROR",
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    error=repr(exc),
                    traceback=traceback.format_exc(limit=12),
                )
                self._json(500, {"ok": False, "error": "reindex_failed", "message": str(exc)})
                return
            _log(
                "reindex",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                ok=bool(result.get("ok")),
                before=result.get("before"),
                after=result.get("after"),
            )
            self._json(200, result)
            return
        if self.path != "/parse":
            _log("http.not_found", level="WARNING", method="POST", path=self.path, status=404)
            self._json(404, {"ok": False, "error": "not_found"})
            return

        payload = self._read_payload()
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            _log("parse.invalid", level="WARNING", status=400, reason="text_required")
            self._json(400, {"ok": False, "error": "text_required"})
            return

        try:
            result = _DETECTOR.detect(
                text=text.strip(),
                webspace_id=payload.get("webspace_id") if isinstance(payload, dict) else None,
                locale=payload.get("locale") if isinstance(payload, dict) else None,
                canonicalized_text=payload.get("canonicalized_text") if isinstance(payload, dict) else None,
                entity_resolution=payload.get("entities") if isinstance(payload, dict) else None,
            )
        except Exception as exc:
            _log(
                "parse.error",
                level="ERROR",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                text_preview=_text_preview(text),
                error=repr(exc),
                traceback=traceback.format_exc(limit=12),
            )
            self._json(500, {"ok": False, "error": "parse_failed", "message": str(exc)})
            return

        evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
        _log(
            "parse",
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            webspace_id=payload.get("webspace_id"),
            locale=payload.get("locale"),
            text_preview=_text_preview(text),
            top_intent=result.get("top_intent") or result.get("intent") or "",
            confidence=result.get("confidence"),
            backend=evidence.get("backend"),
            reason=evidence.get("reason"),
            model_id=result.get("model_id"),
            slots=sorted((result.get("slots") or {}).keys()) if isinstance(result.get("slots"), dict) else [],
        )
        response = {"ok": True, **result, "result": result}
        self._json(200, response)


if __name__ == "__main__":
    host = os.getenv("ADAOS_SERVICE_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("ADAOS_SERVICE_PORT", "18091") or "18091")
    except Exception:
        port = 18091
    health = _DETECTOR.health()
    _log(
        "service.start",
        host=host,
        port=port,
        model_loaded=bool(health.get("model_loaded")),
        model_id=health.get("model_id"),
        examples_total=health.get("examples_total"),
        torch_available=health.get("torch_available"),
        faiss_available=health.get("faiss_available"),
        artifact_root=health.get("artifact_root"),
    )
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()

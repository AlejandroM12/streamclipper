"""Utilidades compartidas por los servidores locales (review y aplicación): respuestas JSON, archivos con Range y
verificación del encabezado Host contra DNS rebinding. Solo biblioteca estándar."""
import json
import re
from pathlib import Path

TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
         ".ass": "text/plain; charset=utf-8", ".srt": "text/plain; charset=utf-8"}


class ApiError(Exception):
    """Error de la API con código HTTP y mensaje apto para mostrar al usuario."""

    def __init__(self, status: int, message: str, **extra) -> None:
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def host_ok(handler) -> bool:
    """El Host debe ser 127.0.0.1 o localhost con el puerto del servidor (evita DNS rebinding)."""
    h = (handler.headers.get("Host") or "").lower()
    port = handler.server.server_port
    return h in (f"127.0.0.1:{port}", f"localhost:{port}")


def send_bytes(handler, data: bytes, ctype: str, status: int = 200, cache: str = "no-store") -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", cache)
    handler.end_headers()
    handler.wfile.write(data)


def send_json(handler, obj, status: int = 200) -> None:
    send_bytes(handler, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
               "application/json; charset=utf-8", status)


def _range_error(handler, size: int) -> None:
    handler.send_response(416)
    handler.send_header("Content-Range", f"bytes */{size}")
    handler.send_header("Content-Length", "0")
    handler.end_headers()


def send_file(handler, path: Path, download_name: str | None = None) -> None:
    """Sirve un archivo con soporte de un único rango (necesario para adelantar videos)."""
    size = path.stat().st_size
    start, end, status = 0, size - 1, 200
    rng = handler.headers.get("Range")
    if rng and size:
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        if not m or not (m.group(1) or m.group(2)):
            return _range_error(handler, size)
        if m.group(1) == "":
            start = max(0, size - int(m.group(2)))
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if start > end or start >= size:
            return _range_error(handler, size)
        status = 206
    handler.send_response(status)
    handler.send_header("Content-Type", TYPES.get(path.suffix.lower(), "application/octet-stream"))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(max(0, end - start + 1)))
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    if download_name:
        safe = re.sub(r'[^\w.\- ]', "_", download_name)
        handler.send_header("Content-Disposition", f'attachment; filename="{safe}"')
    handler.end_headers()
    try:
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        pass  # el navegador cerró la conexión (adelantó el video)


def read_json_body(handler, max_bytes: int = 200_000) -> dict:
    """Cuerpo JSON de una escritura. Exige Content-Type JSON (un formulario de otro sitio no puede enviarlo)."""
    if "application/json" not in (handler.headers.get("Content-Type") or ""):
        raise ApiError(415, "se espera application/json")
    try:
        n = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        raise ApiError(400, "Content-Length inválido") from None
    if not 0 < n <= max_bytes:
        raise ApiError(413, "cuerpo vacío o demasiado grande")
    try:
        data = json.loads(handler.rfile.read(n))
    except json.JSONDecodeError:
        raise ApiError(400, "JSON inválido") from None
    if not isinstance(data, dict):
        raise ApiError(400, "se espera un objeto JSON")
    return data

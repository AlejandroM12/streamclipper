"""Servidor local de la aplicación (solo 127.0.0.1). API JSON + archivos con Range + una página única.

No hay autenticación ni base de datos: es una herramienta personal que escucha solo en el equipo local.
Las escrituras exigen Content-Type JSON y el Host se valida contra DNS rebinding (ver streamclipper/webutil.py).
"""
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from streamclipper import __version__
from streamclipper.app.manager import DETECTORS, LANGUAGES, LAYOUTS, MODELS, JobManager
from streamclipper.app.runners import session_dir
from streamclipper.app.sessions import auto_detail, group_detail, list_sessions, rating_item, session_detail
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError, StreamClipperError
from streamclipper.doctor import model_cached, run_doctor
from streamclipper.edit import (
    PRESETS, EditParams, auto_cues, export_versions, merge_edits, resolve_version_files, srt_from_ass, validate_window,
)
from streamclipper.expand import load_proposal
from streamclipper.jobs import load_result
from streamclipper.media.download import DownloadLimits, inspect_media
from streamclipper.models import Transcript
from streamclipper.pipeline import settings_from_result
from streamclipper.selection.context_clip import DEFAULT_PRESET, PRESETS as CLIP_PRESETS
from streamclipper.webutil import ApiError, host_ok, read_json_body, send_bytes, send_file, send_json

log = logging.getLogger("streamclipper.app")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm")
_STATIC = Path(__file__).parent / "static" / "index.html"
# Versión del contrato entre la página y la API (/api/config, /api/inspect…). Súbela al cambiar el formato de lo que la página lee.
# La página y la API salen del MISMO proceso, y la página verifica que coincidan (ver loadConfig en index.html).
API_CONTRACT = 6


def pick_file_dialog() -> str:
    """Selector de archivos nativo, en un subproceso (tkinter no es seguro fuera del hilo principal). '' si se cancela."""
    code = ("import tkinter as t, tkinter.filedialog as f; r=t.Tk(); r.withdraw(); r.attributes('-topmost', True); "
            "print(f.askopenfilename(title='Elegí un video', filetypes=[('Videos','*.mp4 *.mkv *.mov *.webm'),('Todos','*.*')]) or '')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=900,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return proc.stdout.decode("utf-8", errors="replace").strip()


def open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606  (abre el Explorador; la ruta ya fue validada)
    else:
        subprocess.Popen(["xdg-open", str(path)])


class AppContext:
    def __init__(self, settings: Settings, manager: JobManager, picker=pick_file_dialog, opener=open_folder) -> None:
        self.settings, self.manager, self.picker, self.opener = settings, manager, picker, opener
        self._doctor: tuple[float, dict] | None = None
        self._lock = threading.Lock()
        # La página se carga UNA vez, al arrancar: así un proceso nunca sirve una página más nueva (o más vieja) que su propia API.
        self.page = _STATIC.read_text(encoding="utf-8").replace("__API_CONTRACT__", str(API_CONTRACT)).encode("utf-8")

    def doctor(self) -> dict:
        with self._lock:
            if self._doctor and time.time() - self._doctor[0] < 20:
                return self._doctor[1]
        res = run_doctor(self.settings)
        with self._lock:
            self._doctor = (time.time(), res)
        return res

    def active_sessions(self) -> set[str]:
        return ({j["session"] for j in self.manager.active() if j.get("session")}
                | {sg["session"] for j in self.manager.active() for sg in j.get("segments", []) if sg.get("session")}
                | {v for j in self.manager.active() for v in (j.get("result") or {}).get("rendered", {}).values()})

    def session(self, name: str) -> Path:
        try:
            return session_dir(self.settings, name)
        except InvalidInputError as e:
            raise ApiError(404, str(e)) from None


def make_handler(ctx: AppContext):
    class Handler(BaseHTTPRequestHandler):
        server_version = "StreamClipperApp/1"

        def log_message(self, *a):
            pass

        def _dispatch(self, method: str) -> None:
            if not host_ok(self):
                return send_json(self, {"error": "host no permitido"}, 403)
            url = urlsplit(self.path)
            path, query = unquote(url.path), parse_qs(url.query)
            try:
                self._route(method, path, query)
            except ApiError as e:
                send_json(self, {"error": e.message, **e.extra}, e.status)
            except StreamClipperError as e:
                send_json(self, {"error": str(e)}, 400)
            except FileNotFoundError:
                send_json(self, {"error": "no encontrado"}, 404)
            except Exception as e:  # nunca se devuelve un traceback al navegador
                log.exception("error en %s %s", method, path)
                send_json(self, {"error": f"Error interno: {type(e).__name__}: {e}"}, 500)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        # --- rutas ---
        def _route(self, method: str, path: str, query: dict) -> None:
            s = ctx.settings
            if method == "GET":
                if path == "/":
                    return send_bytes(self, ctx.page, "text/html; charset=utf-8")
                if path == "/favicon.ico":
                    self.send_response(204)
                    return self.end_headers()
                if path == "/api/health":
                    return send_json(self, {"ok": True, "version": __version__, "contract": API_CONTRACT})
                if path == "/api/doctor":
                    return send_json(self, ctx.doctor())
                if path == "/api/config":
                    return send_json(self, self._config())
                if path == "/api/evaluation":
                    from streamclipper.evaluation_smart import ratings_report
                    return send_json(self, ratings_report(s))
                if path == "/api/samples":
                    return send_json(self, self._samples())
                if path == "/api/jobs":
                    jobs = ctx.manager.jobs()
                    for j in jobs:
                        j["queue_position"] = ctx.manager.queue_position(j["id"]) if j["status"] == "queued" else 0
                    return send_json(self, {"jobs": jobs})
                if m := re.fullmatch(r"/api/jobs/([\w-]+)", path):
                    return send_json(self, ctx.manager.get(m.group(1)))
                if path == "/api/sessions":
                    return send_json(self, {"sessions": list_sessions(s, ctx.active_sessions())})
                if m := re.fullmatch(r"/api/auto/([\w-]+)", path):
                    return send_json(self, auto_detail(s, ctx.manager.get(m.group(1)), ctx.active_sessions()))
                if m := re.fullmatch(r"/api/groups/([\w-]+)", path):
                    return send_json(self, group_detail(s, m.group(1), ctx.active_sessions()))
                if m := re.fullmatch(r"/api/sessions/([\w.-]+)/context", path):
                    d = ctx.session(m.group(1))
                    return send_json(self, {"proposal": load_proposal(d, (query.get("clip") or [""])[0])})
                if m := re.fullmatch(r"/api/sessions/([\w.-]+)", path):
                    ctx.session(m.group(1))
                    return send_json(self, session_detail(s, m.group(1), ctx.active_sessions()))
                if m := re.fullmatch(r"/media/([\w.-]+)/(.+)", path):
                    return self._media(m.group(1), m.group(2))
                if m := re.fullmatch(r"/source/([\w.-]+)", path):
                    d = ctx.session(m.group(1))
                    f = Path(load_result(d).get("source", ""))
                    if f.is_file() and f.suffix.lower() in VIDEO_EXT:
                        return send_file(self, f)
                    raise ApiError(404, "El video original ya no está disponible.")
                if m := re.fullmatch(r"/download/([\w.-]+)/([\w-]+)/([\w-]+)/(mp4|srt)", path):
                    return self._download(*m.groups())
                raise ApiError(404, "no encontrado")
            # POST
            body = read_json_body(self)
            if path == "/api/jobs" and body.get("auto") is not None:
                job = ctx.manager.submit_auto(body.get("source", ""), body["auto"] or {}, body.get("params") or {}, bool(body.get("force")))
                return send_json(self, job.to_dict(), 201)
            if m := re.fullmatch(r"/api/auto/([\w-]+)/explore", path):
                return send_json(self, ctx.manager.extend_auto(m.group(1), int(body.get("count", 5))).to_dict(), 201)
            if path == "/api/jobs" and body.get("segments") is not None:
                job = ctx.manager.submit_vod(body.get("source", ""), body["segments"], body.get("params") or {}, bool(body.get("force")))
                return send_json(self, job.to_dict(), 201)
            if path == "/api/jobs":
                job = ctx.manager.submit_process(body.get("source", ""), body.get("params") or {}, bool(body.get("force")))
                return send_json(self, job.to_dict(), 201)
            if path == "/api/inspect":
                try:
                    return send_json(self, inspect_media(str(body.get("url", "")), DownloadLimits()))
                except InvalidInputError as e:
                    raise ApiError(400, str(e)) from None
            if path == "/api/pick-file":
                return send_json(self, {"path": ctx.picker()})
            if m := re.fullmatch(r"/api/jobs/([\w-]+)/(cancel|retry)", path):
                if m.group(2) == "cancel":
                    return send_json(self, ctx.manager.cancel(m.group(1)))
                return send_json(self, ctx.manager.retry(m.group(1)).to_dict(), 201)
            if m := re.fullmatch(r"/api/sessions/([\w.-]+)/delete", path):
                if body.get("confirm") is not True:
                    raise ApiError(400, "Falta la confirmación para eliminar la sesión.")
                ctx.session(m.group(1))
                res = ctx.manager.delete_session(m.group(1))
                return send_json(self, {"ok": True, **res})
            if m := re.fullmatch(r"/api/sessions/([\w.-]+)/context", path):
                ctx.session(m.group(1))
                return send_json(self, ctx.manager.submit_context(m.group(1), str(body.get("clip", "")), str(body.get("preset") or DEFAULT_PRESET)).to_dict(), 201)
            if m := re.fullmatch(r"/api/sessions/([\w.-]+)/try-smart", path):
                return send_json(self, self._try_smart(m.group(1), body), 201)
            if m := re.fullmatch(r"/api/sessions/([\w.-]+)/(subtitles|edit|rating|export)", path):
                return self._session_post(m.group(1), m.group(2), body)
            if path == "/api/open-folder":
                return self._open_folder(body)
            raise ApiError(404, "no encontrado")

        def _config(self) -> dict:
            dl = DownloadLimits()
            return {
                "version": __version__, "contract": API_CONTRACT, "defaults": {"model": "small", "language": "es", "layout": "blur-fit", "max_clips": 3, "detector": "classic", "clip_preset": DEFAULT_PRESET,
                                                     "platform": "universal"},
                "models": [{"id": m, "cached": model_cached(m), "label": {"small": "small (recomendado)", "base": "base (más rápido)"}[m]}
                           for m in MODELS],
                "detectors": list(DETECTORS), "languages": list(LANGUAGES), "layouts": list(LAYOUTS), "presets": list(PRESETS),
                "clip_presets": [{"id": k, "label": v["label"], "min": v["min"], "max": v["max"]} for k, v in CLIP_PRESETS.items()],
                "platforms": ["universal", "tiktok", "reels", "shorts"],
                "url_sources": ["YouTube", "Twitch", "Kick"], "url_limits": {"max_minutes": dl.max_duration_s / 60, "max_mb": dl.max_bytes // (1024 * 1024),
                                   "max_height": dl.max_height},
                "data_dir": str(ctx.settings.data_dir),
            }

        def _samples(self) -> dict:
            d = ctx.settings.data_dir / "samples"
            files = [{"path": str(p), "name": p.name, "mb": round(p.stat().st_size / 1e6, 1)}
                     for p in sorted(d.glob("*")) if p.is_file() and p.suffix.lower() in VIDEO_EXT] if d.is_dir() else []
            return {"folder": str(d), "files": files}

        def _media(self, session: str, rel: str) -> None:
            root = ctx.session(session).resolve()
            f = (root / rel).resolve()
            if f.is_relative_to(root) and f.suffix.lower() in (".mp4", ".srt", ".ass") and f.is_file():
                return send_file(self, f)
            raise ApiError(404, "no encontrado")

        def _download(self, session: str, clip: str, version: str, ext: str) -> None:
            d = ctx.session(session)
            try:
                files = resolve_version_files(d, clip, version)
            except InvalidInputError as e:
                raise ApiError(404, str(e)) from None
            name = f"{clip}_{version}.{ext}"
            if ext == "mp4":
                return send_file(self, files["mp4"], download_name=name)
            if files["srt"].is_file():
                return send_file(self, files["srt"], download_name=name)
            if files["ass"].is_file():   # versiones antiguas: se genera el SRT desde el ASS
                return send_bytes(self, srt_from_ass(files["ass"]).encode("utf-8"), "text/plain; charset=utf-8")
            raise ApiError(404, "Esta versión no tiene subtítulos.")

        def _session_post(self, name: str, action: str, body: dict) -> None:
            d = ctx.session(name)
            s = ctx.settings
            if action == "rating":
                try:
                    item, store, algo = rating_item(s, name, str(body.get("clip", "")), str(body.get("version", "original")))
                except KeyError:
                    raise ApiError(404, "Clip o versión inexistente.") from None
                store.set(item, body.get("rating"), str(body.get("note") or ""), algo)
                return send_json(self, {"ok": True, "rating": store.ratings.get(item["id"]), "summary": store.summary()})
            if action == "export":
                title = list_sessions_title(s, name)
                res = export_versions(d, body.get("selections") or [], s, title)
                return send_json(self, res)
            result = load_result(d)
            clip = str(body.get("clip", ""))
            win = next((c for c in result.get("clips", []) if c.get("name") == clip), None)
            if win is None:
                raise ApiError(404, "Clip inexistente.")
            use_ctx = bool(body.get("context"))
            if action == "subtitles":
                start, end = self._window(body, result, use_ctx, d)
                cfg = settings_from_result(result, data_dir=s.data_dir)
                tr = Transcript.from_dict(json.loads(((d / "context" if use_ctx else d) / "transcript.json").read_text(encoding="utf-8")))
                auto = auto_cues(tr, start, end, cfg)
                return send_json(self, {"cues": merge_edits(auto, body.get("edits") or []), "start": start, "end": end})
            # edit: se valida ahora (error inmediato y claro) y se encola; el render corre en el worker
            body = {k: v for k, v in body.items() if k not in ("clip", "note")}
            if body.get("from_context"):                     # el motivo de la construcción lo toma el servidor de la propuesta guardada
                prop = load_proposal(d, clip)
                if prop is None:
                    raise ApiError(409, "No hay una propuesta de contexto para este clip: pulsá «Ampliar con contexto» primero.")
                body["note"] = prop.get("context")
                body["context"] = prop.get("source") == "context"
            params = EditParams.from_dict(body)
            self._window(body, result, params.context, d)
            job = ctx.manager.submit_edit(name, clip, body)
            return send_json(self, job.to_dict(), 201)

        def _try_smart(self, name: str, body: dict) -> dict:
            """Procesa el mismo video con el otro detector, reutilizando la transcripción (no se retranscribe)."""
            d = ctx.session(name)
            result = load_result(d)
            src, cfg = Path(result.get("source", "")), result.get("settings", {})
            if not src.is_file():
                raise ApiError(404, "El video original ya no está disponible.")
            target = "classic" if cfg.get("detector") == "smart" else "smart"
            params = {"model": cfg.get("whisper_model", "small"), "language": cfg.get("language") or "auto", "layout": cfg.get("layout", "blur-fit"),
                      "max_clips": cfg.get("max_clips", 3), "platform": cfg.get("safe_platform", "universal"), "detector": target,
                      "smart_visual": bool(body.get("smart_visual", True))}
            return ctx.manager.submit_process(str(src), params, force=True).to_dict()

        def _window(self, body: dict, result: dict, use_ctx: bool = False, d: Path | None = None) -> tuple[float, float]:
            try:
                start, end = float(body["start"]), float(body["end"])
            except (KeyError, TypeError, ValueError):
                raise ApiError(400, "Indicá inicio y final del clip.") from None
            try:
                media = float(result["media"]["duration"])
                if use_ctx:
                    from streamclipper.media.ffmpeg import probe
                    media = probe(d / "context" / "source.mp4", ctx.settings.ffprobe).duration
                validate_window(start, end, media)
            except InvalidInputError as e:
                raise ApiError(400, str(e)) from None
            return start, end

        def _open_folder(self, body: dict) -> None:
            p = Path(str(body.get("path", ""))).resolve()
            allowed = [ctx.settings.data_dir.resolve() / "exports", ctx.settings.data_dir.resolve() / "jobs"]
            if not p.is_dir() or not any(p.is_relative_to(a) for a in allowed):
                raise ApiError(400, "Solo se pueden abrir las carpetas de exportación y de sesiones.")
            ctx.opener(p)
            send_json(self, {"ok": True})

    return Handler


def list_sessions_title(settings: Settings, name: str) -> str:
    return next((s["title"] for s in list_sessions(settings, include_derived=True) if s["id"] == name), name)


class AppServer(ThreadingHTTPServer):
    """En Windows, SO_REUSEADDR permite que DOS procesos escuchen en el mismo puerto y el más viejo sigue recibiendo las
    conexiones: una instancia nueva quedaba «abierta» sin atender nada y se veía la versión anterior de la API. Aquí el puerto
    es exclusivo; si está ocupado se prueba el siguiente."""
    allow_reuse_address = False

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(ctx: AppContext, host: str = "127.0.0.1", port: int = 8780) -> ThreadingHTTPServer:
    srv = AppServer((host, port), make_handler(ctx))
    srv.daemon_threads = True
    return srv


def serve_app(settings: Settings, port: int = 8780, open_browser: bool = True, host: str = "127.0.0.1") -> None:
    manager = JobManager(settings)
    ctx = AppContext(settings, manager)
    server = None
    for p in range(port, port + 20):
        try:
            server = make_server(ctx, host, p)
            break
        except OSError:
            continue
    if server is None:
        raise InvalidInputError(f"No hay puertos libres entre {port} y {port + 19}")
    url = f"http://127.0.0.1:{server.server_port}/"
    if server.server_port != port:
        print(f"AVISO: el puerto {port} está ocupado (¿hay otra ventana de StreamClipper abierta, quizá de una versión anterior?). "
              f"Se usa el {server.server_port}. Cerrá la otra instancia para evitar confusiones.", flush=True)
    print(f"StreamClipper está listo en {url}\nDatos: {settings.data_dir}\nCerrá esta ventana o presioná Ctrl+C para salir.", flush=True)
    if host != "127.0.0.1":
        print("AVISO: la aplicación no tiene autenticación; escuchar fuera de 127.0.0.1 expone tus archivos.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando…")
    finally:
        manager.shutdown()
        server.server_close()

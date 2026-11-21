"""Gestor de trabajos: cola con UN solo worker, cancelación, duplicados y recuperación tras un cierre.

Un trabajo pesado (descarga + Whisper + render) a la vez: con 16 GB de RAM compartidos con el navegador y el sistema,
dos transcripciones simultáneas no son seguras. Los demás trabajos esperan en la cola (límite configurable).
"""
import gc
import logging
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from streamclipper.app.state import (
    ACTIVE, EDIT_STAGES, PROCESS_STAGES, AppJob, JobStore, make_stages,
)
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, StreamClipperError
from streamclipper.core.sysinfo import available_ram_gb
from streamclipper.media.download import parse_youtube_url
from streamclipper.pipeline import resolve_input
from streamclipper.webutil import ApiError

log = logging.getLogger("streamclipper.app")

MODELS = ("small", "base")
LAYOUTS = ("blur-fit", "focus")
LANGUAGES = {"es": "es", "en": "en", "auto": None}
_THROTTLE_S = 0.5   # el avance se guarda como mucho cada 0.5 s


def friendly_error(text: str) -> tuple[str, str | None]:
    """(mensaje, sugerencia) comprensibles a partir del texto de un error del pipeline."""
    low = text.lower()
    rules = [
        (r"espacio libre|no space left", "Liberá espacio en el disco y reintentá."),
        (r"ffmpeg|ffprobe", "Revisá que FFmpeg funcione (`streamclipper doctor`) y probá con otro video."),
        (r"memoria|memoryerror|out of memory|alloc", "Cerrá otros programas (navegador, IDE) o usá el modelo «base», que consume menos."),
        (r"huggingface|hf_hub|model.*(download|not found)|snapshot", "No se pudo obtener el modelo Whisper: revisá tu conexión (la primera vez descarga unos 480 MB para «small»)."),
        (r"privado|no está disponible|edad|miembros|región|drm|anti-bot|directo|lista de reproducción|dominio", "Probá con otro video público de YouTube."),
        (r"conectar|conexi[oó]n|timed out|getaddrinfo", "Revisá tu conexión a internet y reintentá."),
        (r"no existe el archivo|extensión no soportada|no es un video válido", "Elegí un archivo de video válido (MP4, MKV, MOV o WEBM)."),
    ]
    hint = next((h for pat, h in rules if re.search(pat, low)), None)
    return text, hint


@dataclass
class JobControl:
    """Lo que un runner puede hacer con su trabajo: informar etapas, avisos y comprobar la cancelación."""
    job: AppJob
    manager: "JobManager"
    cancel: CancelToken

    def stage(self, key: str, status: str, fraction: float | None = None, detail: str = "") -> None:
        self.manager._set_stage(self.job, key, status, fraction, detail)

    def warn(self, msg: str) -> None:
        with self.manager._lock:
            if msg not in self.job.warnings:
                self.job.warnings.append(msg)
        self.manager._save(self.job, force=True)

    def set(self, **fields) -> None:
        with self.manager._lock:
            for k, v in fields.items():
                setattr(self.job, k, v)
        self.manager._save(self.job, force=True)

    def pipeline_progress(self, stage: str, frac: float, msg: str) -> None:
        """Mapea el callback del pipeline. Solo se muestra una fracción cuando el avance es real."""
        real = stage in ("transcribir", "renderizar")
        if not msg and frac == 0.0:
            self.stage(stage, "running")
        elif not msg and frac >= 1.0:
            self.stage(stage, "done", 1.0 if real else None)
        else:
            self.stage(stage, "running", frac if real else None, msg)


Runner = Callable[[AppJob, JobControl], None]


class JobManager:
    def __init__(self, settings: Settings, runners: dict[str, Runner] | None = None, max_queue: int = 8,
                 autostart: bool = True) -> None:
        self.settings = settings
        self.max_queue = max_queue
        self.store = JobStore(settings.data_dir / "app" / "jobs")
        self._lock = threading.RLock()
        self._jobs: dict[str, AppJob] = {j.id: j for j in self.store.load_all()}
        self._tokens: dict[str, CancelToken] = {}
        self._last_save: dict[str, float] = {}
        self._q: queue.Queue[str | None] = queue.Queue()
        self._stop = False
        if runners is None:
            from streamclipper.app.runners import default_runners
            runners = default_runners(settings)
        self.runners = runners
        self._recover()
        self._thread = threading.Thread(target=self._loop, name="streamclipper-worker", daemon=True)
        if autostart:
            self._thread.start()

    # --- recuperación -------------------------------------------------------------------------------
    def _recover(self) -> None:
        """Los trabajos que estaban en curso cuando la aplicación se cerró pasan a «interrumpido» (se pueden reintentar)."""
        for job in self._jobs.values():
            if job.status in ACTIVE:
                job.status, job.finished_at = "interrupted", time.time()
                job.message = "La aplicación se cerró mientras este trabajo estaba en curso. Podés reintentarlo."
                for s in job.stages:
                    if s["status"] == "running":
                        s["status"], s["detail"] = "cancelled", "interrumpido"
                self.store.save(job)
        cleanup_stale(self.settings)

    # --- altas --------------------------------------------------------------------------------------
    def _check_capacity(self) -> None:
        with self._lock:
            if sum(1 for j in self._jobs.values() if j.status in ACTIVE) >= self.max_queue:
                raise ApiError(429, f"Hay demasiados trabajos en cola (máximo {self.max_queue}). Esperá a que termine alguno.")

    def submit_process(self, source: str, params: dict | None = None, force: bool = False) -> AppJob:
        params = self._clean_params(params or {})
        source = (source or "").strip()
        if not source:
            raise ApiError(400, "Pegá una URL de YouTube o elegí un archivo de video.")
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", source):
            try:
                ref = parse_youtube_url(source)
            except InvalidInputError as e:
                raise ApiError(400, str(e)) from None
            kind, src_key, title = "url", f"yt:{ref.video_id}", f"YouTube {ref.video_id}"
            source = ref.canonical_url
        else:
            try:
                path = resolve_input(source)
            except InvalidInputError as e:
                raise ApiError(400, str(e)) from None
            st = path.stat()
            kind, src_key, title, source = "file", f"file:{path}:{st.st_size}:{st.st_mtime_ns}", path.name, str(path)
        dup_key = "|".join([src_key, params["model"], str(params["language"]), params["layout"], str(params["max_clips"])])
        self._check_duplicate(dup_key, force)
        self._check_capacity()
        job = AppJob(id=AppJob.new_id(), kind="process", title=title, input=source, input_kind=kind, params=params,
                     dup_key=dup_key, stages=make_stages((["descarga"] if kind == "url" else []) + PROCESS_STAGES))
        return self._enqueue(job)

    def submit_edit(self, session: str, clip: str, params: dict) -> AppJob:
        dup_key = f"edit|{session}|{clip}"
        self._check_duplicate(dup_key, force=True)   # solo se rechaza si hay otra edición del mismo clip en curso
        self._check_capacity()
        job = AppJob(id=AppJob.new_id(), kind="edit", title=f"Editar {clip}", input=session, input_kind="edit",
                     params={**params, "session": session, "clip": clip}, dup_key=dup_key, session=session,
                     stages=make_stages(EDIT_STAGES))
        return self._enqueue(job)

    def _check_duplicate(self, dup_key: str, force: bool) -> None:
        with self._lock:
            for j in self._jobs.values():
                if j.dup_key != dup_key:
                    continue
                if j.status in ACTIVE:
                    raise ApiError(409, "Este trabajo ya está en la cola o procesándose.", job=j.id, duplicate="active")
                if j.status == "completed" and not force and j.kind == "process" and j.session:
                    raise ApiError(409, "Este video ya fue procesado con la misma configuración. Podés abrir esa sesión "
                                        "o reprocesarlo.", job=j.id, duplicate="completed", existing_session=j.session)

    def _clean_params(self, p: dict) -> dict:
        model = p.get("model") or "small"
        lang = p.get("language", "es")
        layout = p.get("layout") or "blur-fit"
        try:
            max_clips = int(p.get("max_clips", 3))
        except (TypeError, ValueError):
            raise ApiError(400, "La cantidad de clips debe ser un número.") from None
        if model not in MODELS:
            raise ApiError(400, f"Modelo desconocido: {model}")
        if lang not in LANGUAGES:
            raise ApiError(400, "Idioma no soportado (es, en o auto).")
        if layout not in LAYOUTS:
            raise ApiError(400, f"Presentación desconocida: {layout}")
        if not 1 <= max_clips <= 6:
            raise ApiError(400, "La cantidad de clips debe estar entre 1 y 6.")
        platform = p.get("platform") or "universal"
        try:
            Settings(layout=layout, safe_platform=platform, max_clips=max_clips).validate()
        except InvalidInputError as e:
            raise ApiError(400, str(e)) from None
        return {"model": model, "language": lang, "layout": layout, "max_clips": max_clips, "platform": platform}

    def _enqueue(self, job: AppJob) -> AppJob:
        with self._lock:
            self._jobs[job.id] = job
            self._tokens[job.id] = CancelToken()
        self._save(job, force=True)
        self._q.put(job.id)
        return job

    # --- consulta -----------------------------------------------------------------------------------
    def get(self, job_id: str) -> dict:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                raise ApiError(404, "No existe ese trabajo.")
            return j.to_dict()

    def jobs(self) -> "list[dict]":
        with self._lock:
            return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def active(self) -> list[dict]:
        return [j for j in self.jobs() if j["status"] in ACTIVE]

    def queue_position(self, job_id: str) -> int:
        with self._lock:
            queued = sorted((j for j in self._jobs.values() if j.status == "queued"), key=lambda j: j.created_at)
            return next((i + 1 for i, j in enumerate(queued) if j.id == job_id), 0)

    # --- control ------------------------------------------------------------------------------------
    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, "No existe ese trabajo.")
            if job.status not in ACTIVE:
                raise ApiError(409, "El trabajo ya terminó.")
            tok = self._tokens.get(job_id)
            if tok:
                tok.cancel()
            if job.status == "queued":   # todavía no empezó: se descarta sin más
                job.status, job.finished_at, job.message = "cancelled", time.time(), "Cancelado antes de empezar."
            else:
                job.message = "Cancelando… (se detiene en cuanto termina el paso actual)"
        self._save(job, force=True)
        return self.get(job_id)

    def retry(self, job_id: str) -> AppJob:
        with self._lock:
            old = self._jobs.get(job_id)
            if old is None:
                raise ApiError(404, "No existe ese trabajo.")
            if old.status in ACTIVE or old.status == "completed":
                raise ApiError(409, "Solo se pueden reintentar trabajos fallidos, cancelados o interrumpidos.")
        if old.kind == "edit":
            return self.submit_edit(old.params["session"], old.params["clip"], {k: v for k, v in old.params.items()
                                                                                 if k not in ("session", "clip")})
        return self.submit_process(old.input, old.params, force=True)

    def shutdown(self, timeout: float = 20.0) -> None:
        """Cancela el trabajo en curso y detiene el worker (el estado queda guardado como «interrumpido»)."""
        self._stop = True
        with self._lock:
            for jid, tok in self._tokens.items():
                if self._jobs[jid].status in ACTIVE:
                    tok.cancel()
        self._q.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout)
        with self._lock:
            for j in self._jobs.values():
                if j.status in ACTIVE:
                    j.status, j.finished_at = "interrupted", time.time()
                    j.message = "La aplicación se cerró mientras este trabajo estaba en curso. Podés reintentarlo."
                    self.store.save(j)

    # --- persistencia y etapas ----------------------------------------------------------------------
    def _save(self, job: AppJob, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_save.get(job.id, 0) < _THROTTLE_S:
            return
        self._last_save[job.id] = now
        with self._lock:
            snapshot = AppJob.from_dict(job.to_dict())
        self.store.save(snapshot)

    def _set_stage(self, job: AppJob, key: str, status: str, fraction: float | None, detail: str) -> None:
        with self._lock:
            st = job.stage(key)
            if st is None:
                return
            if status == "running" and st["status"] != "running":
                st["started"] = time.time()
            if status in ("done", "failed", "cancelled") and st["started"] and st["seconds"] is None:
                st["seconds"] = round(time.time() - st["started"], 1)
            st["status"] = status
            if fraction is not None or status in ("done", "pending"):
                st["fraction"] = fraction
            if detail:
                st["detail"] = detail
            job.message = f"{st['label']}" + (f": {detail}" if detail else "")
        self._save(job, force=status != "running" or fraction is None)

    # --- worker -------------------------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            jid = self._q.get()
            if jid is None or self._stop:
                return
            with self._lock:
                job = self._jobs.get(jid)
                if job is None or job.status != "queued":
                    continue
                job.status, job.started_at = "running", time.time()
            self._save(job, force=True)
            self._run(job)
            gc.collect()   # libera el modelo Whisper y los búferes antes del siguiente trabajo

    def _run(self, job: AppJob) -> None:
        token = self._tokens.get(job.id) or CancelToken()
        ctl = JobControl(job, self, token)
        try:
            ram = available_ram_gb()
            if job.kind == "process" and ram is not None and ram < 1.0:
                ctl.warn(f"Solo hay {ram:.1f} GB de RAM libres. Si falla, cerrá otros programas o usá el modelo «base».")
            self.runners[job.kind](job, ctl)
            outcome, error = ("cancelled" if token.cancelled else "completed"), None
        except CancelledError:
            outcome, error = "cancelled", None
        except StreamClipperError as e:
            outcome, error = "failed", str(e)
        except Exception as e:  # error inesperado: se informa sin tumbar la aplicación
            log.exception("trabajo %s falló", job.id)
            outcome, error = "failed", f"Error inesperado: {type(e).__name__}: {e}"
        with self._lock:
            job.finished_at = time.time()
            if outcome == "failed":
                job.error, job.hint = friendly_error(error or "")
                job.error_detail = error
                for s in job.stages:
                    if s["status"] == "running":
                        s["status"] = "failed"
            elif outcome == "cancelled":
                for s in job.stages:
                    if s["status"] == "running":
                        s["status"] = "cancelled"
                job.message = "Cancelado."
            else:
                job.message = "Listo."
            job.status = outcome
        self._save(job, force=True)


def cleanup_stale(settings: Settings) -> list[str]:
    """Quita restos de ejecuciones interrumpidas: carpetas de trabajo temporales y descargas a medias."""
    removed = []
    for d in list(settings.jobs_dir.glob("*/work*")) + list((settings.data_dir / "samples").glob(".downloading-*")):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    for p in settings.jobs_dir.glob("*/out*/*.part"):
        p.unlink(missing_ok=True)
        removed.append(str(p))
    return removed


def source_dir(settings: Settings) -> Path:
    return settings.data_dir / "samples"

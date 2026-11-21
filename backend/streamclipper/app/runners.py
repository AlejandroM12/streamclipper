"""Runners reales de la aplicación. No reimplementan nada: llaman al pipeline, la descarga y la edición ya existentes."""
import re
import time
from pathlib import Path
from typing import Callable

from streamclipper.app.manager import LANGUAGES, JobControl, friendly_error
from streamclipper.app.sessions import write_segment_meta
from streamclipper.app.state import AppJob, Segment
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, StreamClipperError
from streamclipper.edit import EditParams, render_edit
from streamclipper.media.download import DownloadLimits, download_media, download_segment, format_hms, parse_media_url
from streamclipper.pipeline import run_pipeline


def session_dir(settings: Settings, name: str) -> Path:
    """Carpeta de una sesión (job). El nombre no puede contener separadores ni salir de data/jobs."""
    if not re.fullmatch(r"[\w][\w.-]*", name or ""):
        raise InvalidInputError("Nombre de sesión inválido.")
    d = settings.jobs_dir / name
    if d.resolve().parent != settings.jobs_dir.resolve():
        raise InvalidInputError("Nombre de sesión inválido.")
    if not (d / "result.json").is_file():
        raise InvalidInputError("La sesión no existe.")
    return d


def run_process(settings: Settings):
    def runner(job: AppJob, ctl: JobControl) -> None:
        p = job.params
        cfg = Settings(data_dir=settings.data_dir, whisper_model=p["model"], language=LANGUAGES[p["language"]],
                       max_clips=p["max_clips"], layout=p["layout"], safe_platform=p["platform"],
                       cpu_threads=settings.cpu_threads)
        source = job.input
        if job.input_kind == "url":
            ctl.stage("descarga", "running", None, "consultando metadata")

            def on_dl(msg: str) -> None:
                m = re.search(r"(\d+)%", msg)   # yt-dlp informa el avance real; si no hay porcentaje, no se inventa
                ctl.stage("descarga", "running", int(m.group(1)) / 100 if m else None, msg)
            res = download_media(source, cfg, DownloadLimits(), on_exists="reuse", cancel=ctl.cancel, progress=on_dl)
            ctl.stage("descarga", "done", 1.0, "ya estaba descargado" if res.reused else "")
            ctl.set(title=res.title or job.title)
            for w in res.warnings:
                ctl.warn(w)
            source = str(res.path)

        def on_dir(job_id: str, _d: Path) -> None:
            ctl.set(session=job_id)
        result = run_pipeline(source, cfg, ctl.cancel, ctl.pipeline_progress, label="app", on_job_dir=on_dir)
        ctl.set(session=result["job_id"], result={"session": result["job_id"], "clips": len(result["clips"]),
                                                  "candidates": len(result.get("candidates", []))})
        for w in result.get("warnings", [])[:12]:
            ctl.warn(w)
        if result["status"] == "cancelled":
            raise CancelledError("Cancelado")
        if result["status"] != "completed":
            raise StreamClipperError(result.get("error") or "El procesamiento falló.")
        if not result["clips"]:
            ctl.warn(result.get("note") or "No se encontraron momentos que superen el umbral: no se generaron clips.")
    return runner


def _process_settings(settings: Settings, p: dict) -> Settings:
    return Settings(data_dir=settings.data_dir, whisper_model=p["model"], language=LANGUAGES[p["language"]],
                    max_clips=p["max_clips"], layout=p["layout"], safe_platform=p["platform"], cpu_threads=settings.cpu_threads)


def real_segment(settings: Settings) -> Callable[[AppJob, Segment, JobControl], None]:
    """Un tramo de verdad: descarga SOLO ese rango del VOD y lo procesa con el pipeline existente (nueva sesión)."""
    def do(job: AppJob, seg: Segment, sctl: JobControl) -> None:
        cfg = _process_settings(settings, job.params)
        sctl.stage("descarga", "running", None, "consultando metadata")

        def on_dl(msg: str) -> None:
            m = re.search(r"(\d+)%", msg)   # avance real de FFmpeg; sin porcentaje no se inventa
            sctl.stage("descarga", "running", int(m.group(1)) / 100 if m else None, msg)
        res = download_segment(job.input, seg.start, seg.end, cfg, DownloadLimits(), on_exists="reuse", cancel=sctl.cancel, progress=on_dl)
        sctl.stage("descarga", "done", 1.0, "ya estaba descargado" if res.reused else f"{res.duration:.0f} s descargados")
        ref = parse_media_url(job.input)

        def on_dir(job_id: str, _d: Path) -> None:
            sctl.set(session=job_id)
            write_segment_meta(cfg, job_id, {
                "group": job.id, "url": job.input, "platform": ref.platform, "vod_id": ref.video_id, "vod_title": job.title,
                "index": seg.index, "count": len(job.segments), "start": seg.start, "end": seg.end,
                "start_hms": format_hms(seg.start), "end_hms": format_hms(seg.end), "measured_duration": round(res.duration or 0, 2)})
        result = run_pipeline(str(res.path), cfg, sctl.cancel, sctl.pipeline_progress, label="app", on_job_dir=on_dir)
        sctl.set(session=result["job_id"], result={"session": result["job_id"], "clips": len(result["clips"]),
                                                   "candidates": len(result.get("candidates", []))})
        for w in result.get("warnings", [])[:8]:
            sctl.warn(w)
        if result["status"] == "cancelled":
            raise CancelledError("Cancelado")
        if result["status"] != "completed":
            raise StreamClipperError(result.get("error") or "El procesamiento falló.")
        if not result["clips"]:
            sctl.warn(result.get("note") or "No se encontraron momentos que superen el umbral: no se generaron clips.")
    return do


def run_vod(settings: Settings, do_segment: Callable[[AppJob, Segment, JobControl], None] | None = None):
    """Procesa los tramos de un VOD uno tras otro dentro de UN solo trabajo pesado. Los tramos ya terminados se saltan
    (así se continúa tras un cierre o un fallo); un tramo fallido no impide procesar los siguientes."""
    do_segment = do_segment or real_segment(settings)

    def runner(job: AppJob, ctl: JobControl) -> None:
        failed = 0
        for seg in job.segments:
            if seg.status == "done":
                continue
            ctl.cancel.check()
            with ctl.manager._lock:
                seg.status, seg.started_at, seg.error, seg.hint = "running", time.time(), None, None
                job.message = f"Tramo {seg.index + 1} de {len(job.segments)}"
            ctl.manager._save(job, force=True)
            sctl = JobControl(seg, ctl.manager, ctl.cancel, root=job)
            try:
                do_segment(job, seg, sctl)
                outcome, err = "done", None
            except CancelledError:
                outcome, err = "cancelled", None
            except StreamClipperError as e:
                outcome, err = ("cancelled", None) if ctl.cancel.cancelled else ("failed", str(e))
            except Exception as e:  # error inesperado en un tramo: se informa y se sigue con los demás
                outcome, err = ("cancelled", None) if ctl.cancel.cancelled else ("failed", f"Error inesperado: {type(e).__name__}: {e}")
            with ctl.manager._lock:
                seg.status, seg.finished_at = outcome, time.time()
                if err:
                    seg.error, seg.hint = friendly_error(err)
                    failed += 1
                for s in seg.stages:
                    if s["status"] == "running":
                        s["status"] = "failed" if err else "cancelled"
            ctl.manager._save(job, force=True)
            if outcome == "cancelled":
                raise CancelledError("Cancelado")
        if failed:
            raise StreamClipperError(f"{failed} de {len(job.segments)} tramo(s) fallaron; los demás quedaron listos. "
                                     "Podés reintentar solo los fallidos.")
    return runner


def run_edit(settings: Settings):
    def runner(job: AppJob, ctl: JobControl) -> None:
        p = job.params
        d = session_dir(settings, p["session"])
        params = EditParams.from_dict({k: v for k, v in p.items() if k not in ("session", "clip")})
        meta = render_edit(d, p["clip"], params, settings, ctl.cancel, ctl.pipeline_progress)
        ctl.set(result={"session": p["session"], "clip": p["clip"], "version": meta["version"]})
        for w in meta["warnings"]:
            ctl.warn(w)
    return runner


def default_runners(settings: Settings) -> dict:
    return {"process": run_process(settings), "edit": run_edit(settings), "vod": run_vod(settings)}

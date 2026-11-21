"""Runners reales de la aplicación. No reimplementan nada: llaman al pipeline, la descarga y la edición ya existentes."""
import re
from pathlib import Path

from streamclipper.app.manager import LANGUAGES, JobControl
from streamclipper.app.state import AppJob
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, StreamClipperError
from streamclipper.edit import EditParams, render_edit
from streamclipper.media.download import DownloadLimits, download_youtube
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
            res = download_youtube(source, cfg, DownloadLimits(), on_exists="reuse", cancel=ctl.cancel, progress=on_dl)
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
    return {"process": run_process(settings), "edit": run_edit(settings)}

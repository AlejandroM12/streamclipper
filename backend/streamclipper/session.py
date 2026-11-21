"""Sesión de validación: procesa un video con varios modelos Whisper y deja todo listo para revisar.

Usa los parámetros por defecto (no se ajusta nada): el objetivo es observar los resultados originales. Por cada
modelo: job completo (top-3 renderizado + reporte), lista de hasta N candidatos alternativos; al final, la
comparación entre modelos y un informe por clip. La sesión queda en data/reviews/<etiqueta>/.
"""
import gc
import json
import re
import time
from pathlib import Path
from typing import Callable

from streamclipper.compare import compare_jobs, save_comparison
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.jobs import load_result, resolve_job
from streamclipper.media.download import DownloadLimits, download_youtube, looks_like_url
from streamclipper.pipeline import run_pipeline
from streamclipper.review import (
    algorithm_info, build_items, render_session_report, reviews_dir, write_alternatives,
)


def _label_from(source: Path) -> str:
    return re.sub(r"[^a-z0-9_-]", "", source.stem.lower().replace(" ", "-"))[:40] or "video"


def run_session(
    input_: str,
    models: list[str],
    settings: Settings,
    label: str | None = None,
    limits: DownloadLimits | None = None,
    n_alternatives: int = 10,
    cancel: CancelToken | None = None,
    progress: Callable[[str], None] | None = None,
    language: str | None = None,
) -> dict:
    cancel = cancel or CancelToken()
    progress = progress or (lambda _m: None)
    if not models:
        raise InvalidInputError("Indicá al menos un modelo")
    if looks_like_url(input_):
        res = download_youtube(input_, settings, limits or DownloadLimits(), on_exists="reuse", cancel=cancel,
                               progress=progress)
        source = str(res.path)
    else:
        source = input_
    label = re.sub(r"[^a-z0-9_-]", "", (label or _label_from(Path(source))).lower()) or "sesion"
    session_dir = reviews_dir(settings) / label
    if (session_dir / "session.json").exists():
        raise InvalidInputError(f"Ya existe la sesión '{label}'. Usá otra --label o borrá {session_dir}")

    jobs: list[dict] = []
    for model in models:
        progress(f"=== Whisper {model} ===")
        cfg = Settings(data_dir=settings.data_dir, whisper_model=model, language=language,
                       cpu_threads=settings.cpu_threads)  # resto de parámetros: valores por defecto
        r = run_pipeline(source, cfg, cancel, lambda s, f, m: progress(f"  [{s}] {m}") if m else None, label=label)
        jobs.append({"job_id": r["job_id"], "model": model, "status": r["status"], "error": r["error"],
                     "job_dir": r["job_dir"]})
        gc.collect()  # liberar el modelo antes del siguiente

    done = [j for j in jobs if j["status"] == "completed"]
    items_by_job: dict[str, list[dict]] = {}
    for j in done:
        progress(f"Preparando revisión de {j['job_id']}")
        d = Path(j["job_dir"])
        write_alternatives(d, settings, n_alternatives)
        items_by_job[j["job_id"]] = build_items(d, settings, n_alternatives, with_sync=True)

    compare_path = None
    if len(done) == 2:
        text = compare_jobs(done[0]["job_id"], done[1]["job_id"], settings)
        compare_path = str(save_comparison(text, done[0]["job_id"], done[1]["job_id"], settings))

    session_dir.mkdir(parents=True)
    first = load_result(Path(done[0]["job_dir"])) if done else {}
    meta = {"label": label, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
            "fingerprint": first.get("source_fingerprint"), "jobs": jobs, "compare": compare_path,
            "algorithm": algorithm_info(first) if first else None, "alternatives_per_job": n_alternatives,
            "note": "Parámetros por defecto, sin ajustes. Las evaluaciones humanas se guardan por huella de video."}
    (session_dir / "session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (session_dir / "report.md").write_text(render_session_report(label, jobs, items_by_job, compare_path),
                                           encoding="utf-8")
    meta["session_dir"] = str(session_dir)
    return meta


def resolve_review_target(targets: list[str], settings: Settings) -> list[Path]:
    """Un nombre de sesión, o una o más referencias a jobs."""
    if len(targets) == 1:
        sj = reviews_dir(settings) / targets[0] / "session.json"
        if sj.is_file():
            data = json.loads(sj.read_text(encoding="utf-8"))
            return [Path(j["job_dir"]) for j in data["jobs"] if j["status"] == "completed"]
    return [resolve_job(t, settings) for t in targets]

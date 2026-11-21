"""Re-puntúa un job existente con el algoritmo actual, SIN retranscribir ni volver a descargar.

Reutiliza transcript.json y el video original (solo se re-extrae el audio para la curva de volumen).
Crea un job nuevo («<id>-rescored») con los candidatos y la selección nuevos; el original no se toca, de modo
que se puede comparar antes/después con `evaluate` y exportar los clips con `export`.
"""
import copy
import json
import shutil
import time
from pathlib import Path

from streamclipper.analysis.audio import analyze_audio
from streamclipper.analysis.confidence import summarize
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.core.log import log_event, setup_logging
from streamclipper.jobs import load_result, resolve_job
from streamclipper.media.ffmpeg import extract_audio
from streamclipper.models import Transcript
from streamclipper.pipeline import fingerprint, settings_from_result
from streamclipper.report import write_reports
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.selector import find_clips, selection_report


def rescore_job(job_ref: str, settings: Settings, suffix: str = "rescored",
                cancel: CancelToken | None = None, **overrides) -> dict:
    cancel = cancel or CancelToken()
    old_dir = resolve_job(job_ref, settings)
    old = load_result(old_dir)
    t_path = old_dir / "transcript.json"
    if not t_path.is_file():
        raise InvalidInputError("El job no tiene transcript.json: no se puede re-puntuar sin retranscribir")
    src = Path(old["source"])
    if not src.is_file():
        raise InvalidInputError(f"El video original ya no existe: {src}")
    if fingerprint(src) != old.get("source_fingerprint"):
        raise InvalidInputError("El video original cambió desde que se procesó este job")

    cfg = settings_from_result(old, **overrides)
    cfg.validate()
    transcript = Transcript.from_dict(json.loads(t_path.read_text(encoding="utf-8")))

    base = f"{old_dir.name}-{suffix}"
    new_dir, n = old_dir.parent / base, 2
    while new_dir.exists():
        new_dir, n = old_dir.parent / f"{base}-{n}", n + 1
    work = new_dir / "work"
    (new_dir / "out").mkdir(parents=True)
    work.mkdir()
    setup_logging(new_dir / "job.log")
    t0 = time.monotonic()
    try:
        wav = work / "audio.wav"
        extract_audio(src, wav, cfg.ffmpeg, cancel)
        curve = analyze_audio(wav)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    ctx = AnalysisContext(transcript, curve, old["media"]["duration"], cfg)
    candidates, chosen = find_clips(ctx)
    for rank, c in enumerate(candidates, 1):
        c.rank = rank

    res = copy.deepcopy(old)
    for k in ("clips", "alt_clips", "export_runs", "resources", "error"):
        res.pop(k, None)
    selection = selection_report(candidates, chosen, ctx)
    warnings = [f"Re-puntuado desde {old_dir.name} con el algoritmo actual (sin retranscribir). "
                "Los clips no se renderizan: usá `export`."]
    if selection["fragile"]:
        warnings.append(selection["message"])
    if not chosen:
        warnings.append("No se encontraron momentos que superen el umbral; no se seleccionó ningún clip.")
    res.update({
        "job_id": new_dir.name, "status": "completed", "rescored_from": old_dir.name,
        "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {k: v for k, v in vars(cfg).items() if k != "data_dir"} | {"data_dir": str(cfg.data_dir)},
        "candidates": [c.to_dict() for c in candidates], "selected": [c.rank for c in chosen],
        "selection": selection, "clips": [], "alt_clips": [], "warnings": warnings, "error": None,
        "stages": {"extraer_audio+analizar_audio+seleccionar": {"seconds": round(time.monotonic() - t0, 2), "ok": True}},
        "total_seconds": round(time.monotonic() - t0, 2),
    })
    res.setdefault("transcript", {})["evidence"] = summarize(transcript)
    shutil.copy(t_path, new_dir / "transcript.json")
    (new_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_reports(new_dir, res)
    log_event("rescored", old=old_dir.name, new=new_dir.name, candidates=len(candidates), selected=len(chosen))
    res["job_dir"] = str(new_dir)
    return res

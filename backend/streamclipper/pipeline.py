"""Pipeline de extremo a extremo: validar → audio → transcribir → analizar → seleccionar → renderizar."""
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from streamclipper.analysis.audio import analyze_audio
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError, MediaError, StreamClipperError
from streamclipper.core.log import log_event, setup_logging
from streamclipper.core.sysinfo import available_ram_gb
from streamclipper.media.ffmpeg import extract_audio, probe
from streamclipper.media.validate import validate_clip
from streamclipper.models import Transcript
from streamclipper.render.ass import write_ass
from streamclipper.render.vertical import render_clip
from streamclipper.selection.base import AnalysisContext, CandidateScorer
from streamclipper.selection.selector import find_clips
from streamclipper.transcription.sentences import split_sentences
from streamclipper.transcription.whisper import transcribe

STAGES = ["validar", "extraer_audio", "transcribir", "analizar_audio", "seleccionar", "renderizar", "verificar"]

ProgressFn = Callable[[str, float, str], None]  # (etapa, fracción 0..1 de la etapa, mensaje)


def resolve_input(path_str: str) -> Path:
    """Ruta de entrada local: se resuelve a absoluta y debe ser un archivo regular, no UNC.

    (La confinación a un directorio base se aplica en la API, donde la ruta no la elige el usuario local.)
    """
    if path_str.startswith("\\\\"):
        raise InvalidInputError("No se aceptan rutas de red/UNC")
    p = Path(path_str).resolve()
    if not p.is_file():
        raise InvalidInputError(f"No existe el archivo: {p}")
    if p.suffix.lower() not in (".mp4", ".mkv", ".mov", ".webm"):
        raise InvalidInputError(f"Extensión no soportada: {p.suffix} (se espera MP4)")
    return p


def run_pipeline(
    input_path: str,
    settings: Settings,
    cancel: CancelToken | None = None,
    progress: ProgressFn | None = None,
    scorer: CandidateScorer | None = None,
    keep_temp: bool = False,
) -> dict:
    cancel = cancel or CancelToken()
    progress = progress or (lambda *_: None)
    settings.validate()
    src = resolve_input(input_path)

    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job_dir = settings.jobs_dir / job_id
    work, out_dir = job_dir / "work", job_dir / "out"
    out_dir.mkdir(parents=True)
    work.mkdir()
    setup_logging(job_dir / "job.log")

    result: dict = {"job_id": job_id, "source": str(src), "status": "running", "stages": {},
                    "settings": {k: v for k, v in vars(settings).items() if k != "data_dir"} | {"data_dir": str(settings.data_dir)},
                    "clips": [], "candidates": [], "error": None}

    def save() -> None:
        (job_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def stage(name: str):
        class _S:
            def __enter__(self_):
                cancel.check()
                self_.t = time.monotonic()
                progress(name, 0.0, "")
                log_event("stage_start", stage=name)
            def __exit__(self_, et, ev, tb):
                dt = round(time.monotonic() - self_.t, 2)
                result["stages"][name] = {"seconds": dt, "ok": et is None}
                log_event("stage_end", stage=name, seconds=dt, ok=et is None)
                if et is None:
                    progress(name, 1.0, "")
                save()
        return _S()

    try:
        free = shutil.disk_usage(settings.data_dir if settings.data_dir.exists() else job_dir).free / 1024**3
        if free < settings.min_free_gb:
            raise MediaError(f"Espacio libre insuficiente: {free:.1f} GB (mínimo {settings.min_free_gb} GB)")
        ram = available_ram_gb()
        if ram is not None and ram < 1.0:
            log_event("low_ram_warning", level=30, available_gb=round(ram, 2))
            progress("validar", 0.0, f"Aviso: solo {ram:.1f} GB de RAM libres; cerrá programas si falla la transcripción")

        with stage("validar"):
            info = probe(src, settings.ffprobe)
            result["media"] = vars(info)
            if not info.has_audio:
                raise InvalidInputError("El video no tiene pista de audio; no se puede analizar")
            if info.duration < settings.min_clip:
                raise InvalidInputError(
                    f"El video dura {info.duration:.1f}s, menos que el clip mínimo ({settings.min_clip:.0f}s)")

        wav = work / "audio.wav"
        with stage("extraer_audio"):
            extract_audio(src, wav, settings.ffmpeg, cancel)

        with stage("transcribir"):
            transcript = transcribe(wav, settings, cancel, info.duration,
                                    lambda f: progress("transcribir", f, f"{f * info.duration:.0f}s / {info.duration:.0f}s"))
            transcript = split_sentences(transcript)
            (job_dir / "transcript.json").write_text(
                json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            result["transcript"] = {"language": transcript.language, "segments": len(transcript.segments),
                                    "words": sum(len(s.words) for s in transcript.segments)}

        with stage("analizar_audio"):
            curve = analyze_audio(wav)

        with stage("seleccionar"):
            ctx = AnalysisContext(transcript, curve, info.duration, settings)
            candidates, chosen = find_clips(ctx, scorer)
            result["candidates"] = [c.to_dict() for c in candidates[:20]]
            result["selected"] = [c.to_dict() for c in chosen]
            result["note"] = None if chosen else (
                "No se encontraron momentos que superen el umbral; no se generaron clips.")
            progress("seleccionar", 1.0, f"{len(chosen)} clip(s) seleccionados de {len(candidates)} candidatos")

        with stage("renderizar"):
            for i, c in enumerate(chosen, 1):
                cancel.check()
                progress("renderizar", (i - 1) / len(chosen), f"clip {i}/{len(chosen)}")
                ass_work = work / f"clip_{i:02d}.ass"
                n_lines = write_ass(ass_work, transcript.segments, c.start, c.end)
                shutil.copy(ass_work, out_dir / ass_work.name)
                mp4 = out_dir / f"clip_{i:02d}.mp4"
                render_clip(src, c.start, c.end, ass_work, mp4, settings, cancel)
                result["clips"].append({"file": str(mp4), "start": round(c.start, 2), "end": round(c.end, 2),
                                        "score": round(c.score, 3), "subtitle_lines": n_lines,
                                        "explanation": c.explanation()})

        with stage("verificar"):
            for clip in result["clips"]:
                v = validate_clip(Path(clip["file"]), clip["end"] - clip["start"], settings.ffmpeg, settings.ffprobe)
                clip["validation"] = v
                if not v["ok"]:
                    raise MediaError(f"El clip {Path(clip['file']).name} no pasó la validación: {v['problems']}")

        result["status"] = "completed"
    except StreamClipperError as e:
        result["status"] = "cancelled" if type(e).__name__ == "CancelledError" else "failed"
        result["error"] = str(e)
        log_event("job_failed", level=40, status=result["status"], error=str(e))
        for f in out_dir.glob("*.part"):
            f.unlink(missing_ok=True)
    except Exception as e:  # error inesperado: se registra y se informa sin dejar procesos ni temporales
        result["status"] = "failed"
        result["error"] = f"Error inesperado: {type(e).__name__}: {e}"
        log_event("job_crashed", level=40, error=repr(e))
    finally:
        if not keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        save()
    result["job_dir"] = str(job_dir)
    return result

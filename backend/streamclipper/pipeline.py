"""Pipeline de extremo a extremo: validar → audio → transcribir → analizar → seleccionar → renderizar."""
import hashlib
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from streamclipper.analysis.audio import analyze_audio
from streamclipper.analysis.confidence import summarize as summarize_confidence
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, MediaError, StreamClipperError
from streamclipper.core.log import log_event, setup_logging
from streamclipper.core.memory import ResourceMonitor
from streamclipper.core.sysinfo import available_ram_gb
from streamclipper.models import Transcript
from streamclipper.media.ffmpeg import extract_audio, probe
from streamclipper.render.clips import render_set
from streamclipper.report import write_reports
from streamclipper.selection.base import AnalysisContext, CandidateScorer
from streamclipper.selection.context_clip import plan_windows
from streamclipper.selection.refine import refine_selected
from streamclipper.selection.selector import find_clips, is_eligible, selection_report
from streamclipper.selection.smart import build_smart_scorer
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


def fingerprint(path: Path) -> str:
    """Huella barata (tamaño + primer y último MB) para saber si dos jobs usaron el mismo video."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), 2)
            h.update(f.read())
    return h.hexdigest()[:16]


def new_job_dir(settings: Settings, model: str, label: str | None = None) -> tuple[str, Path]:
    """Cada ejecución tiene su carpeta; el modelo va en el nombre para comparar base vs small."""
    tag = re.sub(r"[^a-z0-9_.-]", "", model.lower())
    extra = "-" + re.sub(r"[^a-z0-9_-]", "", label.lower()) if label else ""
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}{extra}-{uuid.uuid4().hex[:4]}"
    return job_id, settings.jobs_dir / job_id


def settings_from_result(result: dict, **overrides) -> Settings:
    d = dict(result["settings"])
    d["data_dir"] = Path(d["data_dir"])
    d.update(overrides)
    return Settings(**d)


def find_reusable_transcript(settings: Settings, source_fingerprint: str):
    """(Transcript, job_id) de un job anterior COMPLETADO del mismo video, modelo e idioma con evidencia de Whisper
    completa (versión 2) y palabras; None si no hay. Evita retranscribir lo que ya se transcribió."""
    best = None
    for d in settings.jobs_dir.iterdir() if settings.jobs_dir.is_dir() else []:
        try:
            res = json.loads((d / "result.json").read_text(encoding="utf-8"))
            if (res.get("source_fingerprint") != source_fingerprint or res.get("status") != "completed" or res.get("rescored_from")
                    or res["settings"].get("whisper_model") != settings.whisper_model or res["settings"].get("language") != settings.language):
                continue
            tr = Transcript.from_dict(json.loads((d / "transcript.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            continue
        if tr.has_whisper_stats and any(s.words for s in tr.segments) and (best is None or d.name > best[1]):
            best = (tr, d.name)
    return best


def run_pipeline(
    input_path: str,
    settings: Settings,
    cancel: CancelToken | None = None,
    progress: ProgressFn | None = None,
    scorer: CandidateScorer | None = None,
    keep_temp: bool = False,
    label: str | None = None,
    on_job_dir: Callable[[str, Path], None] | None = None,
    analysis_only: bool = False,
    preset_transcript: Transcript | None = None,
    preset_candidates: list | None = None,
) -> dict:
    """analysis_only: analiza y guarda candidatos (y cuáles son «elegibles») pero NO renderiza (bloques del modo automático).
    preset_transcript / preset_candidates: usa esa transcripción y renderiza exactamente esos candidatos, sin transcribir ni
    detectar de nuevo (clips finales del modo automático)."""
    cancel = cancel or CancelToken()
    progress = progress or (lambda *_: None)
    settings.validate()
    src = resolve_input(input_path)

    job_id, job_dir = new_job_dir(settings, settings.whisper_model, label)
    work, out_dir = job_dir / "work", job_dir / "out"
    out_dir.mkdir(parents=True)
    work.mkdir()
    setup_logging(job_dir / "job.log")
    if on_job_dir:
        on_job_dir(job_id, job_dir)   # permite enlazar el trabajo con su carpeta desde el primer momento

    warnings: list[str] = []
    result: dict = {
        "job_id": job_id, "source": str(src), "label": label, "status": "running", "stages": {},
        "settings": {k: v for k, v in vars(settings).items() if k != "data_dir"} | {"data_dir": str(settings.data_dir)},
        "clips": [], "alt_clips": [], "candidates": [], "selected": [], "warnings": warnings, "error": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def save() -> None:
        (job_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def warn(msg: str) -> None:
        warnings.append(msg)
        log_event("warning", level=30, message=msg)

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

    t_job = time.monotonic()
    monitor = ResourceMonitor()
    try:
        with monitor:
            free = shutil.disk_usage(job_dir).free / 1024**3
            if free < settings.min_free_gb:
                raise MediaError(f"Espacio libre insuficiente: {free:.1f} GB (mínimo {settings.min_free_gb} GB)")
            ram = available_ram_gb()
            if ram is not None and ram < 1.0:
                warn(f"Solo {ram:.1f} GB de RAM libres al iniciar; cerrá programas si falla la transcripción")

            with stage("validar"):
                info = probe(src, settings.ffprobe)
                result["media"] = vars(info)
                result["source_fingerprint"] = fingerprint(src)
                if not info.has_audio:
                    raise InvalidInputError("El video no tiene pista de audio; no se puede analizar")
                if info.duration < settings.min_clip:
                    raise InvalidInputError(
                        f"El video dura {info.duration:.1f}s, menos que el clip mínimo ({settings.min_clip:.0f}s)")

            wav = work / "audio.wav"
            with stage("extraer_audio"):
                extract_audio(src, wav, settings.ffmpeg, cancel)

            with stage("transcribir"):
                reused = (find_reusable_transcript(settings, result["source_fingerprint"])
                          if settings.reuse_transcript and preset_transcript is None else None)
                if preset_transcript is not None:
                    transcript = preset_transcript
                    progress("transcribir", 1.0, "transcripción ya disponible (no se retranscribe)")
                elif reused:
                    transcript = reused[0]
                    result["transcript_reused_from"] = reused[1]
                    progress("transcribir", 1.0, f"transcripción reutilizada de {reused[1]} (no se retranscribe)")
                else:
                    transcript = transcribe(
                        wav, settings, cancel, info.duration,
                        lambda f: progress("transcribir", f, f"{f * info.duration:.0f}s / {info.duration:.0f}s"))
                    transcript = split_sentences(transcript)
                (job_dir / "transcript.json").write_text(
                    json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
                n_words = sum(len(s.words) for s in transcript.segments)
                result["transcript"] = {"language": transcript.language, "sentences": len(transcript.segments),
                                        "words": n_words, "evidence": summarize_confidence(transcript)}
                if n_words == 0:
                    warn("No se transcribió ninguna palabra: la selección depende solo del audio")

            with stage("analizar_audio"):
                curve = analyze_audio(wav)

            with stage("seleccionar"):
                ctx = AnalysisContext(transcript, curve, info.duration, settings)
                use_scorer = scorer
                if preset_candidates is not None:
                    candidates = chosen = list(preset_candidates)
                    result["detector"] = {"requested": settings.detector, "effective": "candidatos ya elegidos (modo automático)"}
                else:
                    if use_scorer is None and settings.detector == "smart":
                        use_scorer = build_smart_scorer(wav, curve, src, settings, cancel, warn)
                    candidates, chosen = find_clips(ctx, use_scorer)
                    result["detector"] = use_scorer.info() if hasattr(use_scorer, "info") else {
                        "requested": settings.detector, "effective": "clásico"}
                for rank, c in enumerate(candidates, 1):
                    c.rank = rank
                result["candidates"] = [c.to_dict() for c in candidates]
                result["selected"] = [c.rank for c in chosen]
                if analysis_only:
                    result["analysis_only"] = True
                    result["eligible"] = [c.rank for c in candidates if is_eligible(c, ctx, use_scorer)]
                if preset_candidates is None:
                    result["selection"] = selection_report(candidates, chosen, ctx)
                    if result["selection"]["fragile"]:
                        warn(result["selection"]["message"])
                if not chosen and not analysis_only:
                    result["note"] = "No se encontraron momentos que superen el umbral; no se generaron clips."
                    warn(result["note"])
                msg = (f"{len(result.get('eligible', []))} candidato(s) elegibles de {len(candidates)}" if analysis_only
                       else f"{len(chosen)} clip(s) seleccionados de {len(candidates)} candidatos")
                progress("seleccionar", 1.0, msg)

            if not analysis_only:
                with stage("renderizar"):
                    names = [f"clip_{i:02d}" for i in range(1, len(chosen) + 1)]
                    # los límites se ajustan DESPUÉS de elegir: el ranking y las puntuaciones no cambian
                    windows = dict(zip(names, plan_windows(chosen, ctx) if settings.clip_preset != "legacy" else refine_selected(chosen, ctx)))
                    result["clips"] = render_set(
                        src, list(zip(names, chosen)), transcript, settings, work, out_dir, cancel, progress, warnings,
                        refined=windows)

                with stage("verificar"):
                    pass  # la validación FFprobe/decodificación y los límites de frase ya corrieron en render_set

            result["status"] = "completed"
    except CancelledError as e:
        result["status"], result["error"] = "cancelled", str(e)
        log_event("job_cancelled", level=30)
    except StreamClipperError as e:
        result["status"], result["error"] = "failed", str(e)
        log_event("job_failed", level=40, error=str(e))
    except Exception as e:  # error inesperado: se registra y se informa sin dejar procesos ni temporales
        result["status"] = "failed"
        result["error"] = f"Error inesperado: {type(e).__name__}: {e}"
        log_event("job_crashed", level=40, error=repr(e))
    finally:
        for f in out_dir.glob("*.part"):
            f.unlink(missing_ok=True)
        if result["status"] != "completed":
            for f in out_dir.glob("clip_*"):  # no dejar salidas a medias de una tarea fallida/cancelada
                f.unlink(missing_ok=True)
            result["clips"] = []
        if not keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        result["total_seconds"] = round(time.monotonic() - t_job, 2)
        result["resources"] = monitor.summary()
        save()
        write_reports(job_dir, result)
    result["job_dir"] = str(job_dir)
    return result

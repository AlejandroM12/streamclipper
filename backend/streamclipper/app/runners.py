"""Runners reales de la aplicación. No reimplementan nada: llaman al pipeline, la descarga y la edición ya existentes."""
import json
import re
import shutil
import time
from pathlib import Path
from typing import Callable

from streamclipper.app.manager import LANGUAGES, JobControl, friendly_error
from streamclipper.app.sessions import write_segment_meta
from streamclipper.app.state import AppJob, Segment
from streamclipper.core.config import Settings
from streamclipper.auto import (
    ANALYSIS_HEIGHT, BLOCK_S, MARGIN_S, Block, block_scales, clip_range, max_download_s, pool_from_block, select_global, shift_candidate,
    shift_transcript,
)
from streamclipper.core.errors import CancelledError, InvalidInputError, MediaError, StreamClipperError
from streamclipper.edit import EditParams, render_edit
from streamclipper.media.download import DownloadLimits, download_media, download_segment, format_hms, parse_media_url
from streamclipper.models import Candidate, Transcript
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
        cfg = _process_settings(settings, p)
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
    # reuse_transcript: un video ya transcrito con el mismo modelo e idioma no se retranscribe
    return Settings(data_dir=settings.data_dir, whisper_model=p["model"], language=LANGUAGES[p["language"]],
                    max_clips=p["max_clips"], layout=p["layout"], safe_platform=p["platform"], cpu_threads=settings.cpu_threads,
                    detector=p.get("detector", "classic"), smart_visual=p.get("smart_visual", True), reuse_transcript=True)


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


def _run_segments(job: AppJob, ctl: JobControl, do_segment, noun: str = "Tramo") -> int:
    """Recorre los tramos/bloques pendientes de a UNO. Los terminados se saltan (así se continúa tras un cierre o un fallo); uno
    fallido no impide procesar los siguientes. Devuelve cuántos fallaron; propaga la cancelación."""
    failed = 0
    for seg in job.segments:
        if seg.status == "done":
            continue
        ctl.cancel.check()
        with ctl.manager._lock:
            seg.status, seg.started_at, seg.error, seg.hint = "running", time.time(), None, None
            job.message = f"{noun} {seg.index + 1} de {len(job.segments)}"
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
            for st in seg.stages:
                if st["status"] == "running":
                    st["status"] = "failed" if err else "cancelled"
        ctl.manager._save(job, force=True)
        if outcome == "cancelled":
            raise CancelledError("Cancelado")
    return failed


def run_vod(settings: Settings, do_segment: Callable[[AppJob, Segment, JobControl], None] | None = None):
    """Procesa los tramos de un VOD uno tras otro dentro de UN solo trabajo pesado."""
    do_segment = do_segment or real_segment(settings)

    def runner(job: AppJob, ctl: JobControl) -> None:
        failed = _run_segments(job, ctl, do_segment)
        if failed:
            raise StreamClipperError(f"{failed} de {len(job.segments)} tramo(s) fallaron; los demás quedaron listos. "
                                     "Podés reintentar solo los fallidos.")
    return runner


# --- transmisión completa (automático) ---------------------------------------------------------------------
def _drop(path: Path) -> None:
    """Borra un archivo de trabajo (bloque descargado) y su ficha de procedencia."""
    for p in (path, path.with_suffix(".source.json")):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _block_of(seg: Segment, total: int) -> Block:
    return Block(seg.index, seg.start, seg.end, seg.dl_start, seg.dl_end, seg.index == total - 1)


def real_block(settings: Settings, ydl_factory=None) -> Callable[[AppJob, Segment, JobControl], None]:
    """Un bloque: descarga SOLO su rango (con contexto) en baja resolución, lo analiza con el pipeline (sin renderizar) y borra el video."""
    def do(job: AppJob, seg: Segment, sctl: JobControl) -> None:
        p = job.params
        cfg = _process_settings(settings, p)
        cfg.reuse_transcript = False
        limits = DownloadLimits(max_duration_s=max_download_s(p.get("block_s", BLOCK_S), p.get("margin_s", MARGIN_S)),
                                max_height=p.get("analysis_height", ANALYSIS_HEIGHT))
        samples = settings.data_dir / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        need_gb = cfg.min_free_gb + (seg.dl_end - seg.dl_start) * 2e6 / 8 / 1e9
        free_gb = shutil.disk_usage(samples).free / 1e9
        if free_gb < need_gb:
            raise MediaError(f"Espacio libre insuficiente para descargar el bloque: {free_gb:.1f} GB libres, hacen falta {need_gb:.1f} GB")
        sctl.stage("descarga", "running", None, "consultando metadata")

        def on_dl(msg: str) -> None:
            m = re.search(r"(\d+)%", msg)
            sctl.stage("descarga", "running", int(m.group(1)) / 100 if m else None, msg)
        t0 = time.monotonic()
        res = download_segment(job.input, seg.dl_start, seg.dl_end, cfg, limits, on_exists="reuse", cancel=sctl.cancel, progress=on_dl,
                               ydl_factory=ydl_factory, auto=True)
        dl_s = time.monotonic() - t0
        sctl.stage("descarga", "done", 1.0, f"{(res.size_bytes or 0) / 1e6:.0f} MB en {dl_s:.0f} s")
        ref = parse_media_url(job.input)
        try:
            def on_dir(job_id: str, _d: Path) -> None:
                sctl.set(session=job_id)
                write_segment_meta(cfg, job_id, {
                    "group": job.id, "mode": "auto-block", "url": job.input, "platform": ref.platform, "vod_id": ref.video_id,
                    "vod_title": job.title, "index": seg.index, "count": len(job.segments), "start": seg.start, "end": seg.end,
                    "dl_start": seg.dl_start, "dl_end": seg.dl_end, "start_hms": format_hms(seg.start), "end_hms": format_hms(seg.end)})
            result = run_pipeline(str(res.path), cfg, sctl.cancel, sctl.pipeline_progress, label="auto", on_job_dir=on_dir, analysis_only=True)
        finally:
            _drop(res.path)                                   # libera el disco aunque falle o se cancele
        sctl.set(session=result["job_id"], result={
            "candidates": len(result.get("candidates", [])), "eligible": len(result.get("eligible", [])), "download_mb": round((res.size_bytes or 0) / 1e6),
            "download_s": round(dl_s, 1), "seconds": result.get("total_seconds"), "peak_ram_mb": (result.get("resources") or {}).get("total_peak_mb_approx"),
            "measured_s": round(res.duration or 0, 1)})
        for w in result.get("warnings", [])[:6]:
            sctl.warn(w)
        if result["status"] == "cancelled":
            raise CancelledError("Cancelado")
        if result["status"] != "completed":
            raise StreamClipperError(result.get("error") or "El análisis del bloque falló.")
    return do


def real_render(settings: Settings, ydl_factory=None) -> Callable[[AppJob, dict, JobControl], str]:
    """Render diferido de UN clip final: re-descarga solo su rango en alta calidad y lo renderiza con el pipeline existente,
    reutilizando la transcripción y el candidato del bloque (no se retranscribe ni se vuelve a detectar)."""
    def do(job: AppJob, item: dict, ctl: JobControl) -> str:
        p = job.params
        cfg = _process_settings(settings, p)
        cfg.reuse_transcript = False
        seg = job.segments[item["block"]]
        a, b = clip_range(item, p["vod_duration"])
        res = download_segment(job.input, a, b, cfg, DownloadLimits(), on_exists="reuse", cancel=ctl.cancel, ydl_factory=ydl_factory, auto=True)
        block_dir = settings.jobs_dir / item["session"]
        shift = a - seg.dl_start                                # eje de tiempo del bloque → eje del recorte
        tr = shift_transcript(Transcript.from_dict(json.loads((block_dir / "transcript.json").read_text(encoding="utf-8"))), shift, 0, b - a)
        raw = json.loads((block_dir / "result.json").read_text(encoding="utf-8"))["candidates"][item["rank"] - 1]
        cand = Candidate.from_dict(shift_candidate(raw, shift))
        ref = parse_media_url(job.input)

        def on_dir(job_id: str, _d: Path) -> None:
            write_segment_meta(cfg, job_id, {
                "group": job.id, "mode": "auto-clip", "url": job.input, "platform": ref.platform, "vod_id": ref.video_id, "vod_title": job.title,
                "index": item["global_rank"] - 1, "count": len(job.result.get("plan", [])), "start": item["start"], "end": item["end"],
                "start_hms": format_hms(item["start"]), "end_hms": format_hms(item["end"]), "global_rank": item["global_rank"],
                "kind": item["kind"], "score": item["score"], "block": item["block"], "block_session": item["session"], "cut": [a, b],
                "cut_shift_s": shift, "measured_cut_s": round(res.duration or 0, 2), "detector": p.get("detector", "classic"), "top": item.get("top", [])})
        result = run_pipeline(str(res.path), cfg, ctl.cancel, None, label="auto-clip", on_job_dir=on_dir,
                              preset_transcript=tr, preset_candidates=[cand])
        if result["status"] == "cancelled":
            raise CancelledError("Cancelado")
        if result["status"] != "completed" or not result["clips"]:
            raise StreamClipperError(result.get("error") or "No se pudo generar el clip.")
        return result["job_id"]
    return do


def run_auto(settings: Settings, do_block=None, do_render=None):
    """Transmisión completa: 1) analiza todos los bloques en serie, 2) elige los mejores momentos de TODA la transmisión,
    3) renderiza solo los elegidos. Cada fase es reanudable: lo terminado no se repite."""
    do_block = do_block or real_block(settings)
    do_render = do_render or real_render(settings)

    def runner(job: AppJob, ctl: JobControl) -> None:
        p = job.params
        failed = _run_segments(job, ctl, do_block, noun="Bloque")
        if failed:
            raise StreamClipperError(f"{failed} de {len(job.segments)} bloque(s) fallaron. No se eligen clips con la transmisión incompleta: "
                                     "reintentá los fallidos (los terminados no se repiten).")
        if "plan" not in job.result:
            ctl.stage("seleccion_global", "running", None, "comparando los candidatos de todos los bloques")
            pool = []
            for seg in job.segments:
                res = json.loads((settings.jobs_dir / seg.session / "result.json").read_text(encoding="utf-8"))
                pool += pool_from_block(_block_of(seg, len(job.segments)), seg.session, res)
            sel = select_global(pool, p["target"], p.get("explore", False))
            job.result.update(plan=sel["plan"], exploratory=sel["exploratory"], notes=sel["notes"],
                              summary={"recommended": sel["recommended"], "exploratory_available": len(sel["exploratory"]),
                                       "candidates_total": sel["candidates_total"], "duplicates_avoided": sel["duplicates_avoided"],
                                       "crowd_skipped": sel["crowd_skipped"], "target": p["target"]},
                              scales=block_scales(pool, len(job.segments)))
            ctl.set(result=job.result)
            for n in sel["notes"]:
                ctl.warn(n)
            ctl.stage("seleccion_global", "done", 1.0, f"{sel['recommended']} recomendados de {p['target']} pedidos; {sel['candidates_total']} candidatos")
        plan = job.result["plan"]
        rendered = job.result.setdefault("rendered", {})
        errors: dict = {}
        todo = [it for it in plan if it["id"] not in rendered]
        for i, item in enumerate(todo):
            ctl.cancel.check()
            ctl.stage("render_final", "running", i / len(todo), f"clip {i + 1} de {len(todo)} ({format_hms(item['start'])})")
            try:
                rendered[item["id"]] = do_render(job, item, ctl)
                ctl.set(result=job.result)
            except CancelledError:
                raise
            except StreamClipperError as e:
                if ctl.cancel.cancelled:
                    raise CancelledError("Cancelado") from e
                errors[item["id"]] = str(e)
                ctl.warn(f"No se pudo generar el clip de {format_hms(item['start'])}: {e}")
        ctl.stage("render_final", "done" if not errors else "failed", 1.0 if not errors else None,
                  f"{len(rendered)} clip(s) generados" + (f", {len(errors)} con error" if errors else ""))
        if errors:
            raise StreamClipperError(f"{len(errors)} clip(s) no se pudieron generar. Podés reintentar: los ya generados no se repiten.")
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
    vod, auto = run_vod(settings), run_auto(settings)

    def route(job: AppJob, ctl: JobControl) -> None:      # un mismo tipo de trabajo (vod) con dos modos: tramos manuales o automático
        (auto if job.params.get("mode") == "auto" else vod)(job, ctl)
    return {"process": run_process(settings), "edit": run_edit(settings), "vod": route}

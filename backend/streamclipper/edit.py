"""Edición mínima de un clip: ajustar inicio y final, editar subtítulos y elegir presentación, SIN retranscribir.

Cada regeneración crea una versión nueva en `<job>/edits/<clip>/vNNN/` (MP4 + SRT + ASS + meta.json). Nunca se escribe
sobre `out/`, `out_<tag>/`, `result.json` ni `transcript.json`: el original y las versiones anteriores se conservan.
"""
import json
import re
import shutil
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.context import load_context, load_context_dir
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError, MediaError
from streamclipper.jobs import load_result
from streamclipper.media.ffmpeg import probe
from streamclipper.media.validate import validate_clip
from streamclipper.models import Transcript
from streamclipper.pipeline import settings_from_result
from streamclipper.render.layout import compute_layout
from streamclipper.render.subtitles import (
    Cue, ass_document, build_cues, cue_metrics, layout_lines, srt_document,
)
from streamclipper.render.vertical import render_clip
from streamclipper.selection.refine import _fades

EDIT_MIN_S, EDIT_MAX_S = 3.0, 90.0
PRESETS = {  # nombre visible -> ajustes de Settings
    "seguro": {"layout": "blur-fit"},
    "gameplay": {"layout": "focus"},
    "pantalla": {"layout": "fill"},
    "manual": {"layout": "crop"},
}
ProgressFn = Callable[[str, float, str], None]


@dataclass
class EditParams:
    start: float
    end: float
    preset: str | None = None            # clave de PRESETS (None = el layout del job)
    fg_x: float | None = None
    fg_zoom: float | None = None
    keep_regions: list[str] | None = None
    platform: str | None = None
    cues: list[dict] | None = None       # [{start, end, text}] en segundos del video original; None = automáticos
    fades: bool = True
    context: bool = False                # el intervalo está en el eje de `<sesión>/context/source.mp4` (ver expand.py)
    note: dict | None = None             # motivo de la construcción contextual (lo pone el servidor, nunca el cliente)

    @classmethod
    def from_dict(cls, d: dict) -> "EditParams":
        try:
            p = cls(start=float(d["start"]), end=float(d["end"]), preset=d.get("preset") or None,
                    fg_x=None if d.get("fg_x") in (None, "") else float(d["fg_x"]),
                    fg_zoom=None if d.get("fg_zoom") in (None, "") else float(d["fg_zoom"]),
                    keep_regions=[str(r) for r in d["keep_regions"]] if d.get("keep_regions") else None,
                    platform=d.get("platform") or None, cues=d.get("cues"), fades=bool(d.get("fades", True)),
                    context=bool(d.get("context", False)), note=d.get("note") if isinstance(d.get("note"), dict) else None)
        except (KeyError, TypeError, ValueError):
            raise InvalidInputError("Parámetros de edición inválidos: se necesitan inicio y final numéricos.") from None
        if p.preset and p.preset not in PRESETS:
            raise InvalidInputError(f"Presentación desconocida: {p.preset}")
        if p.cues is not None:
            if not isinstance(p.cues, list) or len(p.cues) > 500:
                raise InvalidInputError("Subtítulos inválidos.")
            for c in p.cues:
                try:
                    float(c["start"]), float(c["end"])
                    if not isinstance(c.get("text", ""), str) or len(c.get("text", "")) > 400:
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    raise InvalidInputError("Cada subtítulo necesita inicio, final y un texto de hasta 400 caracteres.") from None
        return p


def validate_window(start: float, end: float, media_duration: float) -> None:
    if not (0 <= start < end <= media_duration + 1e-6):
        raise InvalidInputError(f"El clip debe estar dentro del video (0 – {media_duration:.1f} s) y terminar después de empezar.")
    if end - start < EDIT_MIN_S:
        raise InvalidInputError(f"El clip debe durar al menos {EDIT_MIN_S:g} s.")
    if end - start > EDIT_MAX_S:
        raise InvalidInputError(f"El clip no puede durar más de {EDIT_MAX_S:g} s.")


# --- subtítulos -----------------------------------------------------------------------------------
def auto_cues(transcript: Transcript, start: float, end: float, cfg: Settings, width: int = 1920, height: int = 1080) -> list[dict]:
    """Subtítulos automáticos (desde la transcripción guardada) con tiempos ABSOLUTOS del video."""
    lay = compute_layout(width, height, cfg)
    return [{"start": round(start + c.start, 2), "end": round(start + c.end, 2), "text": c.text, "low_confidence": c.low_conf,
             "edited": False} for c in build_cues(transcript.segments, start, end, cfg, lay.sub_line_chars)]


def merge_edits(auto: list[dict], edits: list[dict]) -> list[dict]:
    """Aplica ediciones de texto a subtítulos recién generados (p. ej. tras cambiar los límites): un subtítulo automático
    toma el texto de la edición que cubre su centro; un texto vacío es una eliminación."""
    out = []
    for a in auto:
        mid = (a["start"] + a["end"]) / 2
        hit = next((e for e in edits if float(e["start"]) - 1e-6 <= mid <= float(e["end"]) + 1e-6), None)
        if hit is None:
            out.append(a)
        else:
            out.append({**a, "text": hit.get("text", ""), "edited": hit.get("text", "") != a["text"]})
    return out


def _lines_for(text: str, line_chars: int, max_lines: int) -> tuple[list[str], bool]:
    words = text.split()
    lines = layout_lines(words, line_chars, max_lines)
    if lines is not None:
        return lines, True
    wrapped = textwrap.wrap(" ".join(words), width=line_chars, break_long_words=True) or [""]
    return wrapped, len(wrapped) <= max_lines


def cues_from_edits(cues: list[dict], start: float, end: float, cfg: Settings, line_chars: int) -> tuple[list[Cue], list[str]]:
    """Subtítulos editados -> Cue relativos al clip. Devuelve (cues, avisos); no cambia el texto escrito por la persona."""
    warnings, out = [], []
    dur = end - start
    for c in sorted(cues, key=lambda c: float(c["start"])):
        text = re.sub(r"\s+", " ", str(c.get("text", ""))).strip()
        if not text:
            continue   # eliminado
        a, b = max(0.0, float(c["start"]) - start), min(dur, float(c["end"]) - start)
        if b - a < 0.2:
            warnings.append(f"El subtítulo «{text[:25]}» queda fuera del clip o dura menos de 0.2 s y se omitió.")
            continue
        lines, fits = _lines_for(text, line_chars, cfg.sub_max_lines)
        if not fits:
            warnings.append(f"El subtítulo «{text[:25]}…» es muy largo y puede salirse de los márgenes: dividilo en dos.")
        out.append(Cue(a, b, lines))
    for prev, cur in zip(out, out[1:]):
        if prev.end > cur.start - 0.04:
            prev.end = max(prev.start + 0.2, cur.start - 0.04)
            if prev.end > cur.start:
                warnings.append(f"Los subtítulos «{prev.text[:20]}» y «{cur.text[:20]}» se superponen.")
    return out, warnings


# --- versiones ------------------------------------------------------------------------------------
def edits_dir(job_dir: Path, clip: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", clip):
        raise InvalidInputError("Nombre de clip inválido.")
    return job_dir / "edits" / clip


def list_edits(job_dir: Path, clip: str) -> list[dict]:
    root = edits_dir(job_dir, clip)
    out = []
    for p in sorted(root.glob("v*/meta.json")) if root.is_dir() else []:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def resolve_version_files(job_dir: Path, clip: str, version: str) -> dict:
    """Rutas del MP4 / SRT / ASS de una versión: 'original', una etiqueta de rerender o 'edit-vNNN'."""
    if version == "original":
        base, name = job_dir / "out", clip
    elif version.startswith("edit-v"):
        base, name = edits_dir(job_dir, clip) / version.removeprefix("edit-"), "clip"
    elif re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,30}", version):
        base, name = job_dir / f"out_{version}", clip
    else:
        raise InvalidInputError("Versión inválida.")
    files = {"mp4": base / f"{name}.mp4", "srt": base / f"{name}.srt", "ass": base / f"{name}.ass"}
    if not files["mp4"].is_file():
        raise InvalidInputError(f"No existe la versión «{version}» del clip {clip}.")
    return files


def srt_from_ass(ass: Path) -> str:
    """SRT a partir de un ASS (para versiones antiguas que no guardaron el .srt)."""
    from streamclipper.review import read_ass

    def ts(t: float) -> str:
        ms = round(max(0.0, t) * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    rows = read_ass(ass)
    return "".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t.replace(chr(92) + 'N', chr(10))}\n\n" for i, (a, b, t) in enumerate(rows, 1))


# --- render ---------------------------------------------------------------------------------------
def render_edit(job_dir: Path, clip: str, params: EditParams, settings: Settings, cancel: CancelToken | None = None,
                progress: ProgressFn | None = None) -> dict:
    cancel = cancel or CancelToken()
    progress = progress or (lambda *_: None)
    result = load_result(job_dir)
    if not any(c["name"] == clip for c in result.get("clips", [])):
        raise InvalidInputError(f"El clip {clip} no existe en esta sesión.")
    if not params.context:
        validate_window(params.start, params.end, float(result["media"]["duration"]))
    overrides: dict = {}
    if params.preset:
        overrides.update(PRESETS[params.preset])
    if params.fg_x is not None:
        overrides["fg_x"] = params.fg_x
    if params.fg_zoom is not None:
        overrides["fg_zoom"] = params.fg_zoom
    if params.keep_regions is not None:
        overrides["keep_regions"] = params.keep_regions
    if params.platform:
        overrides["safe_platform"] = params.platform
    cfg = settings_from_result(result, data_dir=settings.data_dir, **overrides)
    cfg.validate()

    root = edits_dir(job_dir, clip)
    root.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name[1:]) for p in root.glob("v[0-9]*") if p.name[1:].isdigit()] or [0])
    vdir = root / f"v{n:03d}"
    work = root / f".work-{n:03d}"
    t0 = time.monotonic()
    try:
        progress("preparar", 0.0, "")
        ctx, src = load_context_dir(job_dir, cfg, work, cancel) if params.context else load_context(job_dir, result, cfg, work, cancel)
        info = probe(src, cfg.ffprobe)
        if params.context:
            validate_window(params.start, params.end, info.duration)
        layout = compute_layout(info.width, info.height, cfg)
        if params.cues is None:
            cues = build_cues(ctx.transcript.segments, params.start, params.end, cfg, layout.sub_line_chars)
            cue_warnings: list[str] = []
        else:
            cues, cue_warnings = cues_from_edits(params.cues, params.start, params.end, cfg, layout.sub_line_chars)
        fi, fo = _fades(params.start, params.end, ctx.transcript.segments, ctx.curve, cfg) if params.fades else (0.0, 0.0)
        vdir.mkdir()
        (vdir / "clip.ass").write_text(ass_document(cues, layout, cfg), encoding="utf-8")
        (vdir / "clip.srt").write_text(srt_document(cues), encoding="utf-8")
        progress("preparar", 1.0, "")
        cancel.check()

        progress("renderizar", 0.0, "")
        t1 = time.monotonic()
        render_clip(src, params.start, params.end, vdir / "clip.ass", vdir / "clip.mp4", cfg, cancel, layout, fi, fo)
        render_s = round(time.monotonic() - t1, 1)
        progress("renderizar", 1.0, "")

        progress("verificar", 0.0, "")
        v = validate_clip(vdir / "clip.mp4", params.end - params.start, cfg.ffmpeg, cfg.ffprobe)
        if not v["ok"]:
            raise MediaError(f"El clip editado no pasó la validación: {v['problems']}")
        b = check_boundaries(ctx.transcript, params.start, params.end)
        warnings = cue_warnings + [f"Límite de frase: {m}" for m in b["errors"] + b["warnings"]]
        if not 8 <= params.end - params.start <= 90:
            warnings.append(f"La duración ({params.end - params.start:.1f} s) está fuera del rango habitual de 8–90 s para Shorts, Reels y TikTok.")
        meta = {
            "version": f"edit-v{n:03d}", "clip": clip, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "start": round(params.start, 2), "end": round(params.end, 2), "duration": round(params.end - params.start, 2),
            "preset": params.preset or cfg.layout, "layout": cfg.layout, "framing": layout.framing,
            "fade_in": fi, "fade_out": fo, "render_seconds": render_s, "total_seconds": round(time.monotonic() - t0, 1),
            "subtitles": "editados" if params.cues is not None else "automáticos", "cue_count": len(cues),
            "metrics": cue_metrics([(c.start, c.end, "\\N".join(c.lines)) for c in cues], params.end - params.start,
                                   cfg.sub_max_cps, cfg.sub_min_duration, layout.sub_line_chars),
            "warnings": warnings, "validation": v, "context": params.context, "context_note": params.note,
            "params": {"start": params.start, "end": params.end, "preset": params.preset, "fg_x": params.fg_x,
                       "fg_zoom": params.fg_zoom, "keep_regions": params.keep_regions, "platform": params.platform,
                       "fades": params.fades, "cues": params.cues},
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        progress("verificar", 1.0, "")
        return meta
    except BaseException:
        shutil.rmtree(vdir, ignore_errors=True)   # nunca queda una versión a medias
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- exportación ----------------------------------------------------------------------------------
def _slug(text: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", text, flags=re.U).strip().replace(" ", "-")
    return re.sub(r"-+", "-", s)[:40] or "clip"


def export_versions(job_dir: Path, selections: list[dict], settings: Settings, title: str = "") -> dict:
    """Copia el MP4 y el SRT de cada versión elegida a data/exports/<sesión>/ con nombres claros. Nunca sobrescribe:
    si el archivo existe, agrega un sufijo numérico."""
    result = load_result(job_dir)
    if not selections:
        raise InvalidInputError("Elegí al menos un clip para exportar.")
    dest = settings.data_dir / "exports" / job_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    stem = _slug(title or job_dir.name)
    known = {c["name"] for c in result.get("clips", [])}
    written = []
    for sel in selections:
        clip, version = str(sel.get("clip", "")), str(sel.get("version", "original"))
        if clip not in known:
            raise InvalidInputError(f"El clip {clip} no existe en esta sesión.")
        files = resolve_version_files(job_dir, clip, version)
        base = f"{stem}_{clip}_{version}"
        n = 0
        while (dest / f"{base}{'' if n == 0 else f'_{n}'}.mp4").exists():
            n += 1
        name = f"{base}{'' if n == 0 else f'_{n}'}"
        shutil.copy2(files["mp4"], dest / f"{name}.mp4")
        if files["srt"].is_file():
            shutil.copy2(files["srt"], dest / f"{name}.srt")
        elif files["ass"].is_file():
            (dest / f"{name}.srt").write_text(srt_from_ass(files["ass"]), encoding="utf-8")
        written.append({"clip": clip, "version": version, "mp4": str(dest / f"{name}.mp4"),
                        "srt": str(dest / f"{name}.srt")})
    return {"folder": str(dest), "files": written}

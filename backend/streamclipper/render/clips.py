"""Renderiza un conjunto de candidatos y verifica cada MP4 resultante."""
import shutil
import time
from pathlib import Path
from typing import Callable

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import MediaError
from streamclipper.core.log import log_event
from streamclipper.media.ffmpeg import probe
from streamclipper.media.validate import validate_clip
from streamclipper.models import Candidate, Transcript
from streamclipper.render.layout import compute_layout
from streamclipper.render.subtitles import write_subtitles
from streamclipper.render.vertical import render_clip
from streamclipper.selection.refine import Refined


def render_set(
    src: Path,
    items: list[tuple[str, Candidate]],
    transcript: Transcript,
    settings: Settings,
    work: Path,
    out_dir: Path,
    cancel: CancelToken,
    progress: Callable[[str, float, str], None],
    warnings: list[str],
    refined: dict[str, Refined] | None = None,
) -> list[dict]:
    """`items` = [(nombre_base, candidato)]. Falla si un MP4 no pasa FFprobe/decodificación.

    `refined` (opcional) da la ventana de render de cada nombre; el candidato conserva su ventana original.
    Los problemas de límites de frase no abortan: se informan en `warnings` y en el clip.
    """
    info = probe(src, settings.ffprobe)
    layout = compute_layout(info.width, info.height, settings)
    refined = refined or {}
    clips: list[dict] = []
    for i, (name, c) in enumerate(items, 1):
        cancel.check()
        progress("renderizar", (i - 1) / len(items), f"{name} ({i}/{len(items)})")
        r = refined.get(name)
        start, end = (r.start, r.end) if r else (c.start, c.end)
        ass_work = work / f"{name}.ass"
        srt_work = work / f"{name}.srt"
        subs = write_subtitles(ass_work, srt_work, transcript.segments, start, end, settings, layout)
        shutil.copy(ass_work, out_dir / ass_work.name)
        shutil.copy(srt_work, out_dir / srt_work.name)
        mp4 = out_dir / f"{name}.mp4"
        t0 = time.monotonic()
        fi, fo = (r.fade_in, r.fade_out) if r else (0.0, 0.0)
        render_clip(src, start, end, ass_work, mp4, settings, cancel, layout, fi, fo)
        clips.append({
            "name": name, "rank": c.rank, "file": str(mp4), "start": round(start, 2), "end": round(end, 2),
            "duration": round(end - start, 2), "score": round(c.score, 3), "subtitle_lines": len(subs.cues),
            "subtitle_metrics": subs.metrics, "low_confidence_subtitles": subs.low_confidence,
            "selected_window": [round(c.start, 2), round(c.end, 2)], "refine_notes": r.notes if r else [],
            "render_seconds": round(time.monotonic() - t0, 2), "layout": settings.layout, "framing": layout.framing,
            "fade_in": fi, "fade_out": fo, "subtitle_problems": subs.problems, "duration_bounds": list(r.bounds) if r and r.bounds else None,
            "explanation": c.explanation(), "triggers": c.triggers, "context": r.context if r else None,
        })

    for clip in clips:
        cancel.check()
        v = validate_clip(Path(clip["file"]), clip["end"] - clip["start"], settings.ffmpeg, settings.ffprobe)
        clip["validation"] = v
        if not v["ok"]:
            raise MediaError(f"El clip {clip['name']} no pasó la validación: {v['problems']}")
        b = check_boundaries(transcript, clip["start"], clip["end"])
        clip["boundaries"] = b
        for msg in b["errors"]:
            warnings.append(f"{clip['name']}: {msg}")
        for msg in b["warnings"]:
            warnings.append(f"{clip['name']}: {msg}")
        for msg in clip["subtitle_problems"]:
            warnings.append(f"{clip['name']}: subtítulos: {msg}")
        if clip["low_confidence_subtitles"]:
            warnings.append(f"{clip['name']}: {len(clip['low_confidence_subtitles'])} subtítulo(s) con muchas palabras "
                            "de baja confianza (el texto puede estar mal; no se corrige automáticamente): "
                            + "; ".join(f"«{t[:35]}»" for t in clip["low_confidence_subtitles"][:3]))
        if b["errors"]:
            log_event("clip_boundary_error", level=30, clip=clip["name"], errors=b["errors"])
    return clips

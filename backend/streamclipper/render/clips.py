"""Renderiza un conjunto de candidatos y verifica cada MP4 resultante."""
import shutil
from pathlib import Path
from typing import Callable

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import MediaError
from streamclipper.core.log import log_event
from streamclipper.media.validate import validate_clip
from streamclipper.models import Candidate, Transcript
from streamclipper.render.ass import write_ass
from streamclipper.render.vertical import render_clip


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
) -> list[dict]:
    """`items` = [(nombre_base, candidato)]. Falla si un MP4 no pasa FFprobe/decodificación.

    Los problemas de límites de frase no abortan: se informan en `warnings` y en el clip.
    """
    clips: list[dict] = []
    for i, (name, c) in enumerate(items, 1):
        cancel.check()
        progress("renderizar", (i - 1) / len(items), f"{name} ({i}/{len(items)})")
        ass_work = work / f"{name}.ass"
        n_lines = write_ass(ass_work, transcript.segments, c.start, c.end)
        shutil.copy(ass_work, out_dir / ass_work.name)
        mp4 = out_dir / f"{name}.mp4"
        render_clip(src, c.start, c.end, ass_work, mp4, settings, cancel)
        clips.append({
            "name": name, "rank": c.rank, "file": str(mp4), "start": round(c.start, 2), "end": round(c.end, 2),
            "duration": round(c.duration, 2), "score": round(c.score, 3), "subtitle_lines": n_lines,
            "explanation": c.explanation(), "triggers": c.triggers,
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
        if b["errors"]:
            log_event("clip_boundary_error", level=30, clip=clip["name"], errors=b["errors"])
    return clips

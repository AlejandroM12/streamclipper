"""Compatibilidad: API previa de subtítulos ASS, ahora respaldada por render/subtitles.py (motor nuevo)."""
from pathlib import Path

from streamclipper.core.config import Settings
from streamclipper.models import Segment
from streamclipper.render.layout import compute_layout
from streamclipper.render.subtitles import _ts, build_cues, write_subtitles  # noqa: F401  (_ts se reexporta)


def subtitle_lines(segments: list[Segment], start: float, end: float, cfg: Settings | None = None
                   ) -> list[tuple[float, float, str]]:
    """Subtítulos [(inicio, fin, texto)] relativos al clip, con las líneas unidas por un espacio."""
    cfg = cfg or Settings()
    layout = compute_layout(1920, 1080, cfg)
    return [(c.start, c.end, c.text) for c in build_cues(segments, start, end, cfg, layout.sub_line_chars)]


def write_ass(path: Path, segments: list[Segment], start: float, end: float, cfg: Settings | None = None) -> int:
    cfg = cfg or Settings()
    return len(write_subtitles(path, None, segments, start, end, cfg, compute_layout(1920, 1080, cfg)).cues)

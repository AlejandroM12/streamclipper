import numpy as np

from streamclipper.analysis.audio import HOP, curve_from_db
from streamclipper.core.config import Settings
from streamclipper.models import Segment, Transcript, Word
from streamclipper.selection.base import AnalysisContext


def sentence(text: str, start: float, dur: float, prob: float = 0.95) -> Segment:
    toks = text.split()
    step = dur / len(toks)
    words = [Word(start + i * step, start + (i + 1) * step - 0.02, t, prob) for i, t in enumerate(toks)]
    return Segment(start, start + dur, text, words)


def make_ctx(duration: float, loud: list[tuple[float, float]], sentences: list[Segment],
             settings: Settings | None = None) -> AnalysisContext:
    """Audio plano a -35 dB con tramos fuertes (-15 dB) en `loud` = [(inicio, fin), ...]."""
    n = int(duration / HOP)
    db = np.full(n, -35.0)
    rng = np.random.default_rng(0)
    db += rng.normal(0, 0.5, n)
    for a, b in loud:
        db[int(a / HOP):int(b / HOP)] = -15.0
    return AnalysisContext(Transcript("en", sentences), curve_from_db(db), duration, settings or Settings())

from dataclasses import dataclass, field
from typing import Protocol

from streamclipper.analysis.audio import AudioCurve
from streamclipper.analysis.text import TextSignal, analyze_segment
from streamclipper.core.config import Settings
from streamclipper.models import Candidate, Transcript


@dataclass
class AnalysisContext:
    transcript: Transcript
    curve: AudioCurve
    duration: float
    settings: Settings
    signals: list[TextSignal] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.signals:
            self.signals = [analyze_segment(s) for s in self.transcript.segments]


class CandidateScorer(Protocol):
    """Punto de extensión: puntúa candidatos y explica el resultado.

    HeuristicScorer es la implementación por defecto. Una implementación basada en Ollama
    (con límite de memoria y timeout, y fallback al heurístico) puede enchufarse aquí.
    """

    name: str

    def score(self, candidates: list[Candidate], ctx: AnalysisContext) -> list[Candidate]: ...

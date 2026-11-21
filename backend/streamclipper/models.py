from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Segment:
    """Frase (o segmento de Whisper). Las estadísticas de Whisper son None cuando no están disponibles:
    nunca se rellenan con valores inventados. En una frase armada a partir de varios segmentos de Whisper
    se conserva el peor caso (menor avg_logprob, mayores no_speech_prob y compression_ratio)."""
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


# 1: sin estadísticas de Whisper (jobs anteriores); 2: avg_logprob/no_speech_prob/compression_ratio persistidos
EVIDENCE_VERSION = 2


@dataclass
class Transcript:
    language: str
    segments: list[Segment]
    whisper_segments: list[dict] = field(default_factory=list)  # salida cruda de Whisper por segmento
    evidence_version: int = EVIDENCE_VERSION

    @property
    def has_whisper_stats(self) -> bool:
        return self.evidence_version >= 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        version = d.get("evidence_version", 1)
        segs = []
        for s in d["segments"]:
            seg = Segment(
                start=s["start"], end=s["end"], text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            if version >= 2:
                seg.no_speech_prob = s.get("no_speech_prob")
                seg.avg_logprob = s.get("avg_logprob")
                seg.compression_ratio = s.get("compression_ratio")
            # En la versión 1 'no_speech_prob' valía 0.0 por omisión (nunca lo midió Whisper): se ignora.
            segs.append(seg)
        return cls(language=d["language"], segments=segs, whisper_segments=d.get("whisper_segments", []),
                   evidence_version=version)


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    has_audio: bool
    size_bytes: int


@dataclass
class ScoreComponent:
    name: str
    value: float  # 0..1
    weight: float
    note: str = ""
    window: tuple[float, float] | None = None  # ventana temporal EXACTA usada para calcular el valor
    evidence: list[str] = field(default_factory=list)  # '+ ...' suma al valor, '- ...' fue descartado

    @property
    def contribution(self) -> float:
        return self.value * self.weight


@dataclass
class Candidate:
    start: float
    end: float
    anchor: float
    triggers: list[str]
    components: list[ScoreComponent] = field(default_factory=list)
    score: float = 0.0
    text: str = ""
    rank: int = 0  # posición por puntuación dentro del video (1 = mejor)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(
            start=d["start"], end=d["end"], anchor=d["anchor"], triggers=d["triggers"],
            components=[ScoreComponent(c["name"], c["value"], c["weight"], c.get("note", ""),
                                       tuple(c["window"]) if c.get("window") else None, c.get("evidence", []))
                        for c in d["components"]],
            score=d["score"], text=d.get("text", ""), rank=d.get("rank", 0),
        )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def explanation(self) -> str:
        parts = [
            f"{c.name} {c.value:.2f}x{c.weight:.2f}" + (f" ({c.note})" if c.note else "")
            for c in self.components if c.weight
        ]
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "anchor": round(self.anchor, 2),
            "score": round(self.score, 3),
            "triggers": self.triggers,
            "explanation": self.explanation(),
            "components": [
                {"name": c.name, "value": round(c.value, 3), "weight": c.weight,
                 "contribution": round(c.contribution, 3), "note": c.note,
                 "window": [round(c.window[0], 2), round(c.window[1], 2)] if c.window else None,
                 "evidence": c.evidence}
                for c in self.components
            ],
            "text": self.text,
        }

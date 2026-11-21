from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float = 0.0


@dataclass
class Transcript:
    language: str
    segments: list[Segment]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segs = [
            Segment(
                start=s["start"], end=s["end"], text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
                no_speech_prob=s.get("no_speech_prob", 0.0),
            )
            for s in d["segments"]
        ]
        return cls(language=d["language"], segments=segs)


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
            components=[ScoreComponent(c["name"], c["value"], c["weight"], c.get("note", ""))
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
                 "contribution": round(c.contribution, 3), "note": c.note}
                for c in self.components
            ],
            "text": self.text,
        }

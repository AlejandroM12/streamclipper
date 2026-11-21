"""Señales textuales: reacciones verbales y calidad de transcripción. Léxico es/en."""
import re
import unicodedata
from dataclasses import dataclass

from streamclipper.models import Segment

# (patrón, peso). Se evalúan sobre texto en minúsculas y sin acentos.
_STRONG = [
    r"oh my god", r"omg", r"no way", r"holy (shit|crap|cow|moly)?", r"what the (fuck|hell)", r"wtf",
    r"let'?s go+", r"are you kidding", r"i can'?t believe", r"that'?s insane", r"so close",
    r"no puede ser", r"dios mio", r"no lo puedo creer", r"que locura", r"vamo+s", r"la puta madre",
    r"que haces", r"increible", r"mierda", r"joder", r"hijo de", r"ganamos", r"gane",
]
_MILD = [
    r"insane", r"crazy", r"unreal", r"clutch", r"nice", r"wow", r"bro", r"dude", r"dead", r"ace",
    r"got (him|them)", r"killed", r"win", r"lose", r"lost", r"no no", r"yes yes", r"oh no",
    r"loco", r"brutal", r"tremendo", r"boludo", r"nooo+", r"siii+", r"dale", r"mato", r"morir",
]
_STRONG_RE = [re.compile(rf"\b{p}\b") for p in _STRONG]
_MILD_RE = [re.compile(rf"\b{p}\b") for p in _MILD]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


@dataclass
class TextSignal:
    reaction: float      # 0..1
    interest: float      # 0..1 (exclamaciones, preguntas, repeticiones)
    hits: list[str]


def analyze_segment(seg: Segment) -> TextSignal:
    t = normalize(seg.text)
    hits: list[str] = []
    score = 0.0
    for rx in _STRONG_RE:
        m = rx.search(t)
        if m:
            hits.append(m.group(0))
            score += 1.0
    for rx in _MILD_RE:
        m = rx.search(t)
        if m:
            hits.append(m.group(0))
            score += 0.5
    words = re.findall(r"[a-z']+", t)
    repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
    excl = seg.text.count("!")
    quest = seg.text.count("?")
    interest = min(1.0, (excl + 0.5 * quest + repeats) / 3)
    return TextSignal(reaction=min(1.0, score / 2), interest=interest, hits=hits)


def quality(segments: list[Segment]) -> tuple[float, int]:
    """Probabilidad media por palabra y cantidad de palabras."""
    probs = [w.prob for s in segments for w in s.words]
    if not probs:
        return 0.0, 0
    return sum(probs) / len(probs), len(probs)

"""Contexto de transcripción para Smart Highlights: planteamiento → desenlace, habla en el clip y calidad del texto.

Solo usa la transcripción ya existente (nunca retranscribe) y las señales de texto del ranking clásico. Los marcadores de
«planteamiento» (anticipación) son una lista corta y explícita; un planteamiento NO es evidencia de interés por sí solo:
solo cuenta junto a un desenlace (reacción o interés) que ocurre DESPUÉS.
"""
import re
from dataclasses import dataclass, field

from streamclipper.analysis.confidence import segment_flags
from streamclipper.analysis.text import normalize
from streamclipper.models import Candidate, Segment
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.refine import _HARD, _dependent, _incomplete

SETUP_WINDOW = 10.0       # s antes del ancla donde se busca el planteamiento
PAYOFF_BEFORE, PAYOFF_AFTER = 1.0, 6.0
_SETUP = (
    "a ver", "mira", "mirad", "espera", "esperen", "cuidado", "ojo", "vamos a", "voy a", "intentemos", "probemos",
    "que pasa si", "si hago", "preparate", "ahi viene", "ahi va", "casi", "mira esto", "no te", "no se si",
    "look", "watch", "wait", "let's", "here we go", "what if", "about to", "gonna", "careful", "check this",
)
_SETUP_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in _SETUP) + r")\b")


@dataclass
class NarrativeSignals:
    available: bool                       # hay transcripción con palabras en el video
    setup: float = 0.0                    # 0, 0.5 (hay habla previa) o 1 (marcador de planteamiento o pregunta)
    payoff: float = 0.0                   # 0..1 reacción/interés después del planteamiento
    arc: float = 0.0                      # setup × payoff: planteamiento Y desenlace
    speech_cover: float = 0.0             # fracción del clip con habla transcrita
    flagged: float = 0.0                  # 0..1 transcripción dudosa (banderas de Whisper, bucles, baja confianza)
    incomplete: float = 0.0               # 0..1 cortes a mitad de frase en los bordes
    evidence: dict[str, list[str]] = field(default_factory=dict)


def _overlap(segs: list[Segment], a: float, b: float) -> list[tuple[int, Segment]]:
    return [(i, s) for i, s in enumerate(segs) if s.end > a and s.start < b]


def analyze_narrative(c: Candidate, ctx: AnalysisContext) -> NarrativeSignals:
    segs = ctx.transcript.segments
    if not any(s.words or s.text.strip() for s in segs):
        return NarrativeSignals(available=False)
    sigs = ctx.signals
    out = NarrativeSignals(available=True, evidence={"planteamiento": [], "desenlace": [], "transcripcion": [], "cortes": []})

    # planteamiento: pregunta o marcador de anticipación en los segundos previos al ancla
    before = _overlap(segs, c.anchor - SETUP_WINDOW, c.anchor - 0.5)
    for i, s in before:
        m = _SETUP_RE.search(normalize(s.text))
        if m or "?" in s.text:
            out.setup = 1.0
            out.evidence["planteamiento"].append(f"«{s.text.strip()[:50]}» ({'«' + m.group(1) + '»' if m else 'pregunta'})")
            break
    if out.setup == 0.0 and sum(min(s.end, c.anchor) - max(s.start, c.anchor - SETUP_WINDOW) for _, s in before) >= 1.5:
        out.setup = 0.5
        out.evidence["planteamiento"].append("habla previa al momento (sin marcador de anticipación)")

    # desenlace: reacción o interés aceptados por el análisis de texto, DESPUÉS del planteamiento
    after = _overlap(segs, c.anchor - PAYOFF_BEFORE, c.anchor + PAYOFF_AFTER)
    best, best_s = 0.0, None
    for i, s in after:
        v = max(sigs[i].reaction, 0.6 * sigs[i].interest)
        if v > best:
            best, best_s = v, s
    out.payoff = best
    if best_s is not None and best > 0:
        out.evidence["desenlace"].append(f"«{best_s.text.strip()[:50]}» (reacción/interés {best:.2f})")
    out.arc = out.setup * out.payoff

    # habla en el clip
    inside = _overlap(segs, c.start, c.end)
    spoken = sum(min(s.end, c.end) - max(s.start, c.start) for _, s in inside)
    out.speech_cover = max(0.0, min(1.0, spoken / max(c.duration, 1e-6)))

    # transcripción dudosa: banderas de Whisper, palabras poco confiables y bucles (la misma frase 3+ veces seguidas)
    if inside:
        flagged = [bool(segment_flags(s)) for _, s in inside]
        low_words = [w for _, s in inside for w in s.words if w.prob < ctx.settings.min_word_prob]
        n_words = max(1, sum(len(s.words) for _, s in inside))
        loop = 0
        texts = [normalize(s.text).strip() for _, s in inside]
        for k in range(len(texts) - 2):
            if texts[k] and texts[k] == texts[k + 1] == texts[k + 2]:
                loop = 1
        out.flagged = min(1.0, sum(flagged) / len(flagged) + 0.5 * (len(low_words) / n_words) + 0.5 * loop)
        if out.flagged > 0:
            out.evidence["transcripcion"].append(
                f"{sum(flagged)}/{len(flagged)} frases con banderas de Whisper, {len(low_words)}/{n_words} palabras poco confiables"
                + (", frase repetida en bucle" if loop else ""))
        # cortes a mitad de frase: el clip empieza continuando una oración o termina sin cerrarla
        idx = [i for i, _ in inside]
        first, last = segs[idx[0]], segs[idx[-1]]
        prev = segs[idx[0] - 1] if idx[0] > 0 else None
        nxt = segs[idx[-1] + 1] if idx[-1] + 1 < len(segs) else None
        if _dependent(first, prev):
            out.incomplete += 0.5
            out.evidence["cortes"].append(f"empieza a mitad de oración: «{first.text.strip()[:40]}»")
        if not last.text.rstrip().endswith(_HARD) and _incomplete(last, nxt):   # una frase con puntuación final ya está cerrada
            out.incomplete += 0.5
            out.evidence["cortes"].append(f"termina sin cerrar la frase: «{last.text.strip()[-40:]}»")
    return out

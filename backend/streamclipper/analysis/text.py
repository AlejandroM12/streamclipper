"""Señales textuales: reacciones verbales, interés de la frase y calidad de transcripción.

Una reacción solo cuenta si la transcripción es confiable (probabilidad de Whisper de las palabras implicadas)
y, para palabras ambiguas como «vamos», si hay evidencia de énfasis. Los signos de exclamación suman, pero
nunca son la única evidencia (Whisper puede omitirlos). Reglas: docs/lexicon.md.
"""
import re
import unicodedata
from dataclasses import dataclass, field

from streamclipper.analysis.confidence import segment_flags
from streamclipper.analysis.lexicon import AmbiguousRule, Lexicon
from streamclipper.core.config import Settings
from streamclipper.models import Segment

_STRONG_W, _MILD_W = 1.0, 0.5


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


@dataclass
class Hit:
    term: str
    weight: float          # aporte a la reacción si se acepta
    accepted: bool
    reason: str            # por qué se aceptó o descartó, con la evidencia usada
    kind: str = "strong"   # strong | mild | ambiguous

    def label(self) -> str:
        return f"{'+' if self.accepted else '-'} «{self.term}» ({self.reason})"


@dataclass
class TextSignal:
    reaction: float                       # 0..1, solo con aciertos aceptados
    interest: float                       # 0..1
    hits: list[str]                       # términos aceptados
    details: list[Hit] = field(default_factory=list)          # aceptados y descartados, con razón
    exclamations: int = 0
    questions: int = 0
    repetitions_counted: int = 0
    repetitions_discarded: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)            # banderas de confianza de Whisper


@dataclass
class _Tok:
    n: str                # normalizado
    raw: str
    prob: float | None
    start: float | None
    end: float | None


def _tokens(seg: Segment) -> list[_Tok]:
    out = []
    if seg.words:
        for w in seg.words:
            n = re.sub(r"[^a-z0-9']", "", normalize(w.text))
            if n:
                out.append(_Tok(n, w.text, w.prob, w.start, w.end))
    else:  # sin marcas por palabra: no hay probabilidades ni tiempos (se dejan como desconocidos)
        for raw in seg.text.split():
            n = re.sub(r"[^a-z0-9']", "", normalize(raw))
            if n:
                out.append(_Tok(n, raw, None, None, None))
    return out


def _mean_prob(toks: list[_Tok]) -> float | None:
    probs = [t.prob for t in toks if t.prob is not None]
    return sum(probs) / len(probs) if probs else None


def _runs(toks: list[_Tok]) -> list[tuple[int, int]]:
    """Corridas de la misma palabra repetida (índices [i, j) con j-i >= 2)."""
    runs, i = [], 0
    while i < len(toks):
        j = i + 1
        while j < len(toks) and toks[j].n == toks[i].n:
            j += 1
        if j - i >= 2:
            runs.append((i, j))
        i = j
    return runs


def _loud(curve, tok: _Tok, threshold: float) -> float | None:
    """Exceso de volumen (dB sobre el nivel habitual) alrededor de la palabra; None si no hay datos."""
    if curve is None or tok.start is None:
        return None
    return float(curve.window(curve.excess, tok.start - 0.2, tok.end + 0.2).max())


def _evaluate_ambiguous(rule: AmbiguousRule, pat, elong, toks: list[_Tok], seg: Segment, curve,
                        cfg: Settings, repetitive: bool) -> list[Hit]:
    hits: list[Hit] = []
    matches = [i for i, t in enumerate(toks) if pat.match(t.n)]
    for i in matches:
        t = toks[i]
        if t.prob is not None and t.prob < cfg.min_word_prob:
            hits.append(Hit(rule.name, 0.0, False, f"baja confianza de Whisper (prob {t.prob:.2f} < {cfg.min_word_prob})",
                            "ambiguous"))
            continue
        nxt = toks[i + 1].n if i + 1 < len(toks) else None
        mundane = nxt in rule.mundane_next
        elongated = bool(elong and elong.match(t.n))
        run = next(((a, b) for a, b in _runs(toks) if a <= i < b and all(pat.match(x.n) for x in toks[a:b])), None)
        repeated = False
        if run:
            mp = _mean_prob(toks[run[0]:run[1]])
            repeated = (mp is None or mp >= cfg.repeat_min_prob) and not repetitive
        loud = _loud(curve, t, cfg.audio_peak_db)
        is_loud = loud is not None and loud >= cfg.audio_peak_db
        exclaimed = "!" in seg.text or "¡" in seg.text  # 1 punto: nunca alcanza por sí sola (Whisper la omite a veces)
        isolated = len(toks) <= 3 and not mundane

        signals = []
        points = 0
        for ok, pts, label in ((elongated, 2, "alargada"), (repeated, 2, "repetición confiable"),
                               (is_loud, 2, f"volumen +{loud:.1f} dB" if loud is not None else ""),
                               (exclaimed, 1, "exclamación"), (isolated, 1, "frase corta")):
            if ok:
                points += pts
                signals.append(label)
        if mundane and not (elongated or points >= rule.strong_points):
            hits.append(Hit(rule.name, 0.0, False, f"uso cotidiano: «{t.n} {nxt} …»", "ambiguous"))
            continue
        if points >= rule.strong_points and not mundane:
            hits.append(Hit(rule.name, _STRONG_W, True, "énfasis: " + ", ".join(signals), "ambiguous"))
        elif points >= rule.mild_points:
            hits.append(Hit(rule.name, _MILD_W, True, "énfasis moderado: " + ", ".join(signals), "ambiguous"))
        else:
            why = "sin evidencia de énfasis (tono neutro)" if curve is not None else \
                  "sin evidencia de énfasis (sin análisis acústico)"
            hits.append(Hit(rule.name, 0.0, False, why, "ambiguous"))
    return hits


def analyze_segment(seg: Segment, curve=None, cfg: Settings | None = None,
                    lexicon: Lexicon | None = None) -> TextSignal:
    cfg = cfg or Settings()
    lexicon = lexicon or Lexicon()
    toks = _tokens(seg)
    flags = segment_flags(seg)
    repetitive = "repetitiva" in flags
    no_voice = "posible_no_voz" in flags

    joined = " ".join(t.n for t in toks)
    offsets, pos = [], 0
    for t in toks:
        offsets.append((pos, pos + len(t.n)))
        pos += len(t.n) + 1

    details: list[Hit] = []
    for kind, weight, patterns in (("strong", _STRONG_W, lexicon.strong_re), ("mild", _MILD_W, lexicon.mild_re)):
        for rx in patterns:
            m = rx.search(joined)
            if not m:
                continue
            covered = [t for t, (a, b) in zip(toks, offsets) if a < m.end() and b > m.start()]
            mp = _mean_prob(covered)
            term = m.group(0)
            if no_voice:
                details.append(Hit(term, 0.0, False, "Whisper marcó la frase como posible no-voz", kind))
            elif mp is not None and mp < cfg.min_word_prob:
                details.append(Hit(term, 0.0, False,
                                   f"baja confianza de Whisper (prob {mp:.2f} < {cfg.min_word_prob})", kind))
            else:
                details.append(Hit(term, weight, True, kind + (f", prob {mp:.2f}" if mp is not None else ""), kind))
    for rule, pat, elong in lexicon.ambiguous_re:
        if no_voice:
            continue
        found = _evaluate_ambiguous(rule, pat, elong, toks, seg, curve, cfg, repetitive)
        accepted = [h for h in found if h.accepted]
        # una palabra ambigua suma una vez por frase (la mejor evidencia); los descartes se conservan para explicar
        if accepted:
            details.append(max(accepted, key=lambda h: h.weight))
        details.extend([h for h in found if not h.accepted][:2])

    reaction = min(1.0, sum(h.weight for h in details if h.accepted) / 2)

    # Repeticiones: solo cuentan si son confiables. «¡no, no, no!» dicho con claridad suma; un bucle de baja
    # confianza (típico artefacto de Whisper) no.
    counted, discarded = 0, []
    for a, b in _runs(toks):
        mp = _mean_prob(toks[a:b])
        word = toks[a].n
        if repetitive:
            discarded.append(f"«{word}» ×{b - a} (transcripción repetitiva según Whisper)")
        elif mp is not None and mp < cfg.repeat_min_prob:
            discarded.append(f"«{word}» ×{b - a} (confianza media {mp:.2f} < {cfg.repeat_min_prob})")
        else:
            counted += b - a - 1
    excl, quest = seg.text.count("!"), seg.text.count("?")
    interest = min(1.0, (excl + 0.5 * quest + counted) / 3)
    return TextSignal(reaction=reaction, interest=interest, hits=sorted({h.term for h in details if h.accepted}),
                      details=details, exclamations=excl, questions=quest, repetitions_counted=counted,
                      repetitions_discarded=discarded, flags=flags)


def quality(segments: list[Segment]) -> tuple[float, int]:
    """Probabilidad media por palabra y cantidad de palabras."""
    probs = [w.prob for s in segments for w in s.words]
    if not probs:
        return 0.0, 0
    return sum(probs) / len(probs), len(probs)

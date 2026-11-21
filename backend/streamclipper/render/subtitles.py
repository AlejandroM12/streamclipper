"""Subtítulos legibles a partir de los timestamps por palabra de Whisper.

Principios:
- Nunca se inventan ni se corrigen palabras: el texto es el transcrito. Las palabras de baja confianza se MARCAN
  (aviso en el informe y, opcionalmente, color en el video) pero no se reemplazan.
- El audio no se toca: los tiempos de aparición salen de los timestamps de las palabras (un subtítulo no aparece antes
  de que se diga) y solo se ajusta cuánto tiempo permanece en pantalla.
- Segmentación en frases (puntuación y pausas) y, dentro de cada frase, el corte óptimo por programación dinámica:
  prefiere cortar en comas y pausas, evita terminar con artículos/preposiciones y respeta los límites configurables
  de palabras, caracteres por línea, líneas y velocidad de lectura.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from streamclipper.models import Segment, Word

_MIN_GAP = 0.04        # separación mínima entre subtítulos consecutivos (s)
_MAX_WORD_DUR = 1.2    # una palabra no puede "durar" más que esto (marcas de Whisper a veces abarcan pausas)
_FINAL_HOLD = 0.5      # el último subtítulo permanece un poco más
_MIN_VISIBLE = 0.08    # una palabra con menos tiempo que este dentro del clip no se puede leer: no se muestra
_ABS_MIN_CUE = 0.7     # por debajo de esto un subtítulo es ilegible: se une al vecino aunque exceda el máximo de palabras
_IDEAL_WORDS = 5
_COMMA_BONUS = 120   # cortar una línea tras una coma gana a equilibrar unas letras de más o de menos
_WEAK_END = {  # terminar un subtítulo con estas palabras deja la frase colgada
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en", "con", "por", "para", "que",
    "y", "o", "pero", "porque", "se", "me", "te", "le", "lo", "mi", "tu", "su", "sin", "como", "si", "ni", "no",
    "the", "a", "an", "of", "to", "in", "and", "or", "but", "that", "is", "for",
}
_HARD_END = (".", "?", "!", "…")
_SOFT_END = (",", ";", ":")


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]
    words: list[Word] = field(default_factory=list)
    low_conf: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.lines)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def duration(self) -> float:
        return self.end - self.start


# --- palabras del clip ---------------------------------------------------------------------------
def clip_words(segments: list[Segment], start: float, end: float) -> list[Word]:
    """Palabras dentro del clip, con tiempos relativos al clip, monótonas y con duración acotada."""
    out: list[Word] = []
    for s in segments:
        if s.end <= start or s.start >= end:
            continue
        ws = s.words
        if not ws:  # sin marcas por palabra: reparto uniforme dentro del segmento
            toks = s.text.split()
            step = (s.end - s.start) / max(1, len(toks))
            ws = [Word(s.start + i * step, s.start + (i + 1) * step, t) for i, t in enumerate(toks)]
        for w in ws:
            text = w.text.strip()
            if text and w.end > start + _MIN_VISIBLE and w.start < end - _MIN_VISIBLE:
                a = max(w.start, start) - start
                b = min(w.end, end) - start
                out.append(Word(a, min(b, a + _MAX_WORD_DUR) if b > a else a + 0.05, text, w.prob))
    out.sort(key=lambda w: w.start)
    for prev, cur in zip(out, out[1:]):  # sin solapes entre palabras
        if cur.start < prev.end:
            prev.end = max(prev.start + 0.02, cur.start)
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^\wáéíóúüñ']", "", text.lower())


# --- distribución en líneas ------------------------------------------------------------------------
def layout_lines(words: list[str], line_chars: int, max_lines: int) -> list[str] | None:
    """Reparte las palabras en hasta max_lines líneas de <= line_chars, lo más parejas posible. None si no cabe."""
    n = len(words)
    lens = [len(w) for w in words]

    def width(i: int, j: int) -> int:
        return sum(lens[i:j]) + (j - i - 1)

    if width(0, n) <= line_chars:
        return [" ".join(words)]
    INF = float("inf")
    # best[k][j]: costo mínimo de repartir las primeras j palabras en k líneas. Cada línea cuesta el cuadrado de
    # su holgura (líneas parejas) y cortar justo después de una coma resta costo.
    best = [[INF] * (n + 1) for _ in range(max_lines + 1)]
    prev = [[0] * (n + 1) for _ in range(max_lines + 1)]
    best[0][0] = 0
    for k in range(1, max_lines + 1):
        for j in range(1, n + 1):
            for i in range(j):
                w = width(i, j)
                if w > line_chars or best[k - 1][i] == INF:
                    continue
                cost = best[k - 1][i] + (line_chars - w) ** 2
                if j < n and words[j - 1].endswith(_SOFT_END):
                    cost -= _COMMA_BONUS
                if cost < best[k][j]:
                    best[k][j], prev[k][j] = cost, i
    # solo cuentan las repartijas de 2 o más líneas (la de una línea ya se descartó arriba por no caber)
    k_best = min(range(2, max_lines + 1), key=lambda k: (best[k][n], k), default=None)
    if k_best is None or best[k_best][n] == INF:
        return None
    out, j = [], n
    for k in range(k_best, 0, -1):
        i = prev[k][j]
        out.append(" ".join(words[i:j]))
        j = i
    return out[::-1]


# --- segmentación en subtítulos ---------------------------------------------------------------------
def _phrases(words: list[Word], gap: float) -> list[list[Word]]:
    phrases, cur = [], []
    for w in words:
        if cur and w.start - cur[-1].end > gap:
            phrases.append(cur)
            cur = []
        cur.append(w)
        if w.text.endswith(_HARD_END):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    return phrases


def _cue_cost(ws: list[Word], is_phrase_end: bool, phrase_len: int, cfg, line_chars: int) -> float:
    k = len(ws)
    if k > cfg.sub_max_words:
        return float("inf")
    lines = layout_lines([w.text for w in ws], line_chars, cfg.sub_max_lines)
    if lines is None:
        return float("inf")
    cost = 1.0 + 0.15 * abs(k - _IDEAL_WORDS)
    last = ws[-1]
    if not is_phrase_end:
        if _norm(last.text) in _WEAK_END:
            cost += 2.0
        if last.text.endswith(_SOFT_END):
            cost -= 0.8
    if k == 1 and phrase_len > 1:
        cost += 1.5   # palabra huérfana
    chars = len(" ".join(w.text for w in ws))
    avail = last.end - ws[0].start + cfg.sub_tail_hold
    required = max(cfg.sub_min_duration, chars / cfg.sub_max_cps)
    if avail < required:
        cost += 1.5 * (required - avail) / required
    cost += 0.5 * sum(1 for a, b in zip(ws, ws[1:]) if b.start - a.end > 0.4)  # pausas dentro del subtítulo
    return cost


def _split_phrase(ph: list[Word], cfg, line_chars: int) -> list[list[Word]]:
    n = len(ph)
    INF = float("inf")
    best, prev = [INF] * (n + 1), [0] * (n + 1)
    best[0] = 0
    for j in range(1, n + 1):
        for i in range(max(0, j - cfg.sub_max_words), j):
            c = _cue_cost(ph[i:j], j == n, n, cfg, line_chars)
            if best[i] + c < best[j]:
                best[j], prev[j] = best[i] + c, i
    if best[n] == INF:  # una palabra gigante (más larga que la línea): se emite sola
        return [[w] for w in ph]
    out, j = [], n
    while j > 0:
        out.append(ph[prev[j]:j])
        j = prev[j]
    return out[::-1]


def build_cues(segments: list[Segment], start: float, end: float, cfg, line_chars: int | None = None) -> list[Cue]:
    """Subtítulos del clip [start, end] con tiempos relativos al inicio del clip."""
    duration = end - start
    words = clip_words(segments, start, end)
    if not words:
        return []
    lc = line_chars or cfg.sub_line_chars or 25
    groups: list[list[Word]] = []
    for ph in _phrases(words, cfg.sub_phrase_gap):
        groups += _split_phrase(ph, cfg, lc)

    cues = [Cue(g[0].start, g[-1].end, layout_lines([w.text for w in g], lc, cfg.sub_max_lines) or [" ".join(w.text for w in g)], g)
            for g in groups]
    _time_cues(cues, duration, cfg, lc)
    for c in cues:
        low = sum(w.prob < cfg.sub_low_prob for w in c.words)
        c.low_conf = bool(c.words) and low / len(c.words) >= 0.4
    return cues


def _required(c: Cue, cfg) -> float:
    return max(cfg.sub_min_duration, c.chars / cfg.sub_max_cps)


def _time_cues(cues: list[Cue], duration: float, cfg, lc: int) -> None:
    """Fija cuánto permanece cada subtítulo: cola tras la última palabra, huecos pequeños rellenados y lectura mínima."""
    def assign() -> None:
        for i, c in enumerate(cues):
            last_end = c.words[-1].end
            nxt = cues[i + 1].words[0].start if i + 1 < len(cues) else None
            if nxt is None:
                end = min(duration, last_end + max(cfg.sub_tail_hold, _FINAL_HOLD))
            elif nxt - last_end < cfg.sub_bridge_gap:
                end = nxt - _MIN_GAP                       # habla continua: sin parpadeo entre subtítulos
            else:
                end = min(last_end + cfg.sub_tail_hold, nxt - _MIN_GAP)
            c.start = c.words[0].start
            c.end = min(duration, max(c.start + 0.1, end))   # nunca más allá de la duración del clip
            need = c.start + _required(c, cfg)
            limit = (cues[i + 1].words[0].start - _MIN_GAP) if i + 1 < len(cues) else duration
            if c.end < need:
                c.end = max(c.end, min(need, limit))

    assign()
    # un subtítulo que sigue siendo demasiado rápido se une al siguiente si el resultado respeta los límites
    i = 0
    while i < len(cues) - 1:
        c, nx = cues[i], cues[i + 1]
        if c.duration < 0.85 * _required(c, cfg) and nx.words[0].start - c.words[-1].end < cfg.sub_bridge_gap:
            merged = c.words + nx.words
            lines = layout_lines([w.text for w in merged], lc, cfg.sub_max_lines)
            # un subtítulo inviable (< 0.7 s) en habla muy rápida puede superar el máximo de palabras en hasta 3
            limit = cfg.sub_max_words + (3 if c.duration < _ABS_MIN_CUE else 0)
            if len(merged) <= limit and lines is not None:
                cues[i:i + 2] = [Cue(merged[0].start, merged[-1].end, lines, merged)]
                assign()
                continue
        i += 1


def speech_rate_cps(segments: list[Segment], start: float, end: float) -> float | None:
    """Caracteres por segundo MIENTRAS SE HABLA en [start, end] (une pausas de hasta 0.3 s). Un subtítulo no puede ir más
    lento que el habla: si este ritmo supera el cps cómodo, parte del exceso no es defecto de los subtítulos."""
    words = clip_words(segments, start, end)
    if not words:
        return None
    chars = sum(len(w.text) for w in words) + len(words) - 1
    active, a, b = 0.0, words[0].start, words[0].end
    for w in words[1:]:
        if w.start - b <= 0.3:
            b = max(b, w.end)
        else:
            active += b - a
            a, b = w.start, w.end
    active += b - a
    return round(chars / active, 1) if active > 0 else None


# --- métricas ---------------------------------------------------------------------------------------
def cue_metrics(cues: list[tuple[float, float, str]], duration: float, max_cps: float = 17.0,
                min_duration: float = 1.0, line_chars: int | None = None) -> dict:
    """Métricas de legibilidad de una lista de (inicio, fin, texto); el texto puede traer \\N entre líneas."""
    if not cues:
        return {"cues": 0, "coverage_pct": 0.0}
    durs = [max(1e-6, b - a) for a, b, _ in cues]
    flat = [re.sub(r"\\N|\n", " ", t) for _, _, t in cues]
    cps = sorted(len(t) / d for t, d in zip(flat, durs))
    covered = 0.0
    last_end = 0.0
    for a, b, _ in sorted(cues):
        covered += max(0.0, b - max(a, last_end))
        last_end = max(last_end, b)
    lines = [re.split(r"\\N|\n", t) for _, _, t in cues]
    comfortable = sum(1 for t, d, ln in zip(flat, durs, lines)
                      if len(t) / d <= max_cps and d >= 0.9 * min_duration and (line_chars is None or all(len(x) <= line_chars for x in ln)))
    return {
        "cues": len(cues), "coverage_pct": round(100 * covered / duration, 1) if duration else 0.0,
        "cps_mean": round(sum(cps) / len(cps), 1), "cps_p90": round(cps[int(0.9 * (len(cps) - 1))], 1),
        "cps_max": round(cps[-1], 1), "pct_over_max_cps": round(100 * sum(c > max_cps for c in cps) / len(cps), 1),
        "min_cue_duration_s": round(min(durs), 2), "cues_shorter_than_min": sum(d < 0.9 * min_duration for d in durs),
        "max_line_chars": max(len(x) for ln in lines for x in ln), "max_lines": max(len(ln) for ln in lines),
        "words_per_cue_mean": round(sum(len(t.split()) for t in flat) / len(flat), 1),
        "gaps_over_1s": sum(1 for (_, b, _), (a2, _, _) in zip(sorted(cues), sorted(cues)[1:]) if a2 - b > 1.0),
        "last_cue_end_s": round(max(b for _, b, _ in cues), 2),
        "comfortable_pct": round(100 * comfortable / len(cues), 1),
    }


# --- escritura -------------------------------------------------------------------------------------
def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = round(t * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _srt_ts(t: float) -> str:
    ms = round(max(0.0, t) * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},2,{align},{ml},{mr},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
_LOW_CONF_TAG = r"{\c&H0080FF&}"   # naranja (BGR) para palabras de baja confianza


def _ass_text(c: Cue, mark: bool) -> str:
    if not mark:
        return r"\N".join(_escape(x) for x in c.lines)
    low = {id(w) for w in c.words if w.prob < 0.4}
    out_lines = []
    wi = 0
    for ln in c.lines:
        parts = []
        for tok in ln.split(" "):
            w = c.words[wi] if wi < len(c.words) else None
            wi += 1
            t = _escape(tok)
            parts.append(f"{_LOW_CONF_TAG}{t}{{\\r}}" if w is not None and id(w) in low else t)
        out_lines.append(" ".join(parts))
    return r"\N".join(out_lines)


def ass_document(cues: list[Cue], layout, cfg) -> str:
    header = _ASS_HEADER.format(size=cfg.sub_font_size, outline=max(4, cfg.sub_font_size // 11), align=layout.sub_alignment,
                                ml=layout.margin_l, mr=layout.margin_r, mv=layout.sub_margin_v)
    body = "".join(f"Dialogue: 0,{_ts(c.start)},{_ts(c.end)},Default,,0,0,0,,{_ass_text(c, cfg.sub_mark_low_conf)}\n"
                   for c in cues)
    return header + body


def srt_document(cues: list[Cue]) -> str:
    return "".join(f"{i}\n{_srt_ts(c.start)} --> {_srt_ts(c.end)}\n" + "\n".join(c.lines) + "\n\n"
                   for i, c in enumerate(cues, 1))


def integrity_problems(cues: list[Cue], segments: list[Segment], start: float, end: float, layout, cfg) -> list[str]:
    """Defectos DEMOSTRABLES de los subtítulos de un clip (lista vacía = ninguno). No juzga la calidad del texto.

    - fuera de la duración del clip, tiempos invertidos o superposición de subtítulos;
    - líneas o cantidad de líneas por encima de los límites (texto que se saldría de los márgenes);
    - palabras transcritas del clip omitidas, repetidas o cambiadas (frases truncadas por los límites);
    - huecos sin subtítulo mientras hay palabras que mostrar (huecos injustificados);
    - estilo ASS con márgenes distintos de los del layout."""
    out: list[str] = []
    dur = end - start
    for i, c in enumerate(cues):
        if c.start < -1e-6 or c.end > dur + 1e-6:
            out.append(f"el subtítulo {i + 1} («{c.text[:30]}») queda fuera de la duración del clip ({c.start:.2f}–{c.end:.2f} s de {dur:.2f} s)")
        if c.end <= c.start:
            out.append(f"el subtítulo {i + 1} tiene duración no positiva")
        if len(c.lines) > cfg.sub_max_lines:
            out.append(f"el subtítulo {i + 1} tiene {len(c.lines)} líneas (máximo {cfg.sub_max_lines})")
        too_long = [ln for ln in c.lines if len(ln) > layout.sub_line_chars and " " in ln]
        if too_long:
            out.append(f"el subtítulo {i + 1} tiene una línea de {len(too_long[0])} caracteres (máximo {layout.sub_line_chars}): «{too_long[0][:30]}»")
    for i, (a, b) in enumerate(zip(cues, cues[1:]), 1):
        if a.end > b.start + 1e-6:
            out.append(f"los subtítulos {i} y {i + 1} se superponen ({a.end:.2f} s > {b.start:.2f} s)")
    expected = [w.text for w in clip_words(segments, start, end)]
    shown = [w.text for c in cues for w in c.words]
    if shown != expected:
        missing = [w for w in expected if w not in shown]
        out.append(f"el texto mostrado no coincide con la transcripción del clip ({len(shown)} de {len(expected)} palabras"
                   + (f"; faltan: {', '.join(missing[:5])}" if missing else "") + ")")
    words = clip_words(segments, start, end)
    for a, b in zip(cues, cues[1:]):
        inside = [w for w in words if a.end - 1e-6 <= w.start < b.start - 1e-6]
        if inside:
            out.append(f"hueco injustificado entre {a.end:.2f} s y {b.start:.2f} s: hay palabras («{inside[0].text}») sin subtítulo")
    if cues and words and cues[0].start > words[0].start + 1e-6:
        out.append("el primer subtítulo aparece después de la primera palabra")
    return out


def style_problems(ass_text: str, layout) -> list[str]:
    """El estilo ASS escrito coincide con los márgenes y la alineación del layout."""
    style = next((ln for ln in ass_text.splitlines() if ln.startswith("Style:")), "")
    f = style.split(",")
    if len(f) < 22:
        return ["no se encontró el estilo ASS"]
    got = (int(f[18]), int(f[19]), int(f[20]), int(f[21]))
    want = (layout.sub_alignment, layout.margin_l, layout.margin_r, layout.sub_margin_v)
    return [] if got == want else [f"márgenes/alineación del estilo ASS {got} distintos de los del layout {want}"]


@dataclass
class SubtitleResult:
    cues: list[Cue]
    metrics: dict
    low_confidence: list[str]   # textos de los subtítulos con muchas palabras de baja confianza
    problems: list[str] = field(default_factory=list)   # defectos demostrables (ver integrity_problems)


def write_subtitles(ass_path: Path, srt_path: Path | None, segments: list[Segment], start: float, end: float,
                    cfg, layout) -> SubtitleResult:
    cues = build_cues(segments, start, end, cfg, layout.sub_line_chars)
    doc = ass_document(cues, layout, cfg)
    ass_path.write_text(doc, encoding="utf-8")
    if srt_path is not None:
        srt_path.write_text(srt_document(cues), encoding="utf-8")
    metrics = cue_metrics([(c.start, c.end, "\\N".join(c.lines)) for c in cues], end - start, cfg.sub_max_cps,
                          cfg.sub_min_duration, layout.sub_line_chars)
    problems = integrity_problems(cues, segments, start, end, layout, cfg) + style_problems(doc, layout)
    return SubtitleResult(cues, metrics, [c.text for c in cues if c.low_conf], problems)

"""Subtítulos ASS sincronizados con las marcas por palabra de Whisper (tiempos relativos al clip)."""
from pathlib import Path

from streamclipper.models import Segment, Word

_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = round(t * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _clip_words(segments: list[Segment], start: float, end: float) -> list[Word]:
    words: list[Word] = []
    for s in segments:
        if s.end <= start or s.start >= end:
            continue
        ws = s.words
        if not ws:  # sin marcas por palabra: reparto uniforme dentro del segmento
            toks = s.text.split()
            step = (s.end - s.start) / max(1, len(toks))
            ws = [Word(s.start + i * step, s.start + (i + 1) * step, t) for i, t in enumerate(toks)]
        for w in ws:
            if w.end > start and w.start < end:
                words.append(Word(max(w.start, start) - start, min(w.end, end) - start, w.text, w.prob))
    return words


def subtitle_lines(segments: list[Segment], start: float, end: float,
                   max_words: int = 5, max_chars: int = 28, gap: float = 0.6) -> list[tuple[float, float, str]]:
    words = _clip_words(segments, start, end)
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        chars = sum(len(x.text) + 1 for x in cur) + len(w.text)
        if cur and (len(cur) >= max_words or chars > max_chars or w.start - cur[-1].end > gap
                    or cur[-1].text.endswith((".", "?", "!"))):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    lines = []
    limit = end - start
    for i, g in enumerate(groups):
        t0 = g[0].start
        t1 = min(g[-1].end + 0.2, limit)
        if i + 1 < len(groups):
            t1 = min(t1, groups[i + 1][0].start)
        t1 = min(max(t1, t0 + 0.3), limit)
        lines.append((t0, t1, " ".join(w.text for w in g)))
    return lines


def write_ass(path: Path, segments: list[Segment], start: float, end: float) -> int:
    lines = subtitle_lines(segments, start, end)
    body = "".join(
        f"Dialogue: 0,{_ts(a)},{_ts(b)},Default,,0,0,0,,{_escape(t)}\n" for a, b, t in lines
    )
    path.write_text(_HEADER + body, encoding="utf-8")
    return len(lines)

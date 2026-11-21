"""Re-segmenta la transcripción en oraciones usando las marcas por palabra.

Whisper puede devolver segmentos de 20-30 s; para no cortar frases sin arrastrar charla irrelevante
se trabaja con unidades más finas: fin de oración (. ? !), pausas largas o un máximo de duración.
"""
from streamclipper.models import Segment, Transcript, Word

_END = (".", "?", "!", "…")


def split_sentences(transcript: Transcript, max_pause: float = 0.8, max_len: float = 12.0) -> Transcript:
    out: list[Segment] = []
    for seg in transcript.segments:
        if not seg.words:
            out.append(seg)
            continue
        cur: list[Word] = []
        for w in seg.words:
            if cur and (w.start - cur[-1].end > max_pause or w.end - cur[0].start > max_len):
                out.append(_make(cur, seg))
                cur = []
            cur.append(w)
            if w.text.endswith(_END):
                out.append(_make(cur, seg))
                cur = []
        if cur:
            out.append(_make(cur, seg))
    return Transcript(language=transcript.language, segments=out)


def _make(words: list[Word], parent: Segment) -> Segment:
    return Segment(start=words[0].start, end=words[-1].end, text=" ".join(w.text for w in words),
                   words=list(words), no_speech_prob=parent.no_speech_prob)

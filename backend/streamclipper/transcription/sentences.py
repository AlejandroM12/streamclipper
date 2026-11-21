"""Re-segmenta la transcripción en oraciones usando las marcas por palabra.

Whisper puede devolver segmentos de 20-30 s, o cortar a mitad de frase (sobre todo si no pone puntuación).
Se recorren las palabras de todos los segmentos juntas y se corta solo en: fin de oración (. ? !),
pausas largas o un máximo de duración. El borde entre segmentos de Whisper, por sí solo, no corta.
"""
from streamclipper.models import Segment, Transcript, Word

_END = (".", "?", "!", "…")


def split_sentences(transcript: Transcript, max_pause: float = 0.8, max_len: float = 12.0) -> Transcript:
    out: list[Segment] = []
    cur: list[Word] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            out.append(Segment(start=cur[0].start, end=cur[-1].end, text=" ".join(w.text for w in cur),
                               words=list(cur)))
            cur = []

    for seg in transcript.segments:
        if not seg.words:  # sin marcas por palabra no se puede afinar: se conserva el segmento
            flush()
            out.append(seg)
            continue
        for w in seg.words:
            if cur and (w.start - cur[-1].end > max_pause or w.end - cur[0].start > max_len):
                flush()
            cur.append(w)
            if w.text.endswith(_END):
                flush()
    flush()
    return Transcript(language=transcript.language, segments=out)

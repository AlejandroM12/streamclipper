"""Re-segmenta la transcripción en oraciones usando las marcas por palabra.

Whisper puede devolver segmentos de 20-30 s, o cortar a mitad de frase (sobre todo si no pone puntuación).
Se recorren las palabras de todos los segmentos juntas y se corta solo en: fin de oración (. ? !),
pausas largas o un máximo de duración. El borde entre segmentos de Whisper, por sí solo, no corta.

Cada oración conserva las estadísticas de Whisper de los segmentos de los que salieron sus palabras
(peor caso), para poder detectar transcripciones de baja confianza o repetitivas.
"""
from streamclipper.models import Segment, Transcript, Word

_END = (".", "?", "!", "…")


def _worst(values: list[float | None], pick) -> float | None:
    known = [v for v in values if v is not None]
    return pick(known) if known else None


def split_sentences(transcript: Transcript, max_pause: float = 0.8, max_len: float = 12.0) -> Transcript:
    out: list[Segment] = []
    cur: list[tuple[Word, Segment]] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        words = [w for w, _ in cur]
        parents = list({id(p): p for _, p in cur}.values())
        out.append(Segment(
            start=words[0].start, end=words[-1].end, text=" ".join(w.text for w in words), words=words,
            no_speech_prob=_worst([p.no_speech_prob for p in parents], max),
            avg_logprob=_worst([p.avg_logprob for p in parents], min),
            compression_ratio=_worst([p.compression_ratio for p in parents], max),
        ))
        cur = []

    for seg in transcript.segments:
        if not seg.words:  # sin marcas por palabra no se puede afinar: se conserva el segmento
            flush()
            out.append(seg)
            continue
        for w in seg.words:
            if cur and (w.start - cur[-1][0].end > max_pause or w.end - cur[0][0].start > max_len):
                flush()
            cur.append((w, seg))
            if w.text.endswith(_END):
                flush()
    flush()
    return Transcript(language=transcript.language, segments=out, whisper_segments=transcript.whisper_segments,
                      evidence_version=transcript.evidence_version)

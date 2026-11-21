"""Confianza de la transcripción: banderas a partir de las estadísticas de Whisper y de las palabras.

Solo se usan valores que Whisper proporcionó. Los umbrales de Whisper son los propios de su heurística
interna de reintento (log_prob_threshold=-1.0, compression_ratio_threshold=2.4, no_speech_threshold=0.6).
"""
from streamclipper.models import Segment, Transcript

LOGPROB_MIN = -1.0
COMPRESSION_MAX = 2.4
NO_SPEECH_MIN = 0.6


def segment_flags(seg: Segment) -> list[str]:
    flags = []
    low = seg.avg_logprob is not None and seg.avg_logprob < LOGPROB_MIN
    if low:
        flags.append("baja_confianza_whisper")
    if seg.compression_ratio is not None and seg.compression_ratio > COMPRESSION_MAX:
        flags.append("repetitiva")
    # la regla de silencio de Whisper exige AMBAS condiciones; sin avg_logprob no se infiere
    if seg.no_speech_prob is not None and seg.no_speech_prob > NO_SPEECH_MIN and low:
        flags.append("posible_no_voz")
    return flags


def mean_prob(seg: Segment) -> float | None:
    probs = [w.prob for w in seg.words]
    return sum(probs) / len(probs) if probs else None


def summarize(transcript: Transcript) -> dict:
    """Resumen para el reporte. Si el job no tiene estadísticas de Whisper, lo indica en vez de inventarlas."""
    segs = transcript.segments
    words = [w for s in segs for w in s.words]
    out: dict = {
        "whisper_stats_available": transcript.has_whisper_stats,
        "sentences": len(segs),
        "words": len(words),
        "words_below_0_4_pct": round(100 * sum(w.prob < 0.4 for w in words) / len(words), 1) if words else None,
    }
    if transcript.has_whisper_stats:
        flagged = [segment_flags(s) for s in segs]
        out["sentences_low_confidence"] = sum("baja_confianza_whisper" in f for f in flagged)
        out["sentences_repetitive"] = sum("repetitiva" in f for f in flagged)
        out["sentences_possible_no_voice"] = sum("posible_no_voz" in f for f in flagged)
        lp = [s.avg_logprob for s in segs if s.avg_logprob is not None]
        out["mean_avg_logprob"] = round(sum(lp) / len(lp), 3) if lp else None
    return out

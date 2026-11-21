import os
from pathlib import Path

# la caché de Hugging Face avisa por falta de symlinks en Windows; funciona igual (solo ocupa más)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.log import log_event
from streamclipper.models import Segment, Transcript, Word


def _opt(obj, name: str) -> float | None:
    """Valor de Whisper si existe y es numérico; None si no (no se inventan valores)."""
    v = getattr(obj, name, None)
    return float(v) if isinstance(v, (int, float)) else None


def transcribe(wav: Path, settings: Settings, cancel: CancelToken, duration: float,
               on_progress=None) -> Transcript:
    """Transcribe con faster-whisper (CPU). Conserva marcas por palabra y las estadísticas de Whisper
    por segmento (avg_logprob, no_speech_prob, compression_ratio, temperature).

    El progreso es real: segundos de audio ya transcritos / duración total.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.whisper_model, device="cpu", compute_type=settings.whisper_compute_type,
        cpu_threads=settings.cpu_threads,
    )
    try:
        seg_iter, info = model.transcribe(
            str(wav), language=settings.language, word_timestamps=True,
            beam_size=settings.whisper_beam_size, vad_filter=True,
            condition_on_previous_text=False,
        )
        segments: list[Segment] = []
        raw: list[dict] = []
        for i, s in enumerate(seg_iter):  # generador: permite cancelar entre segmentos
            cancel.check()
            words = [Word(w.start, w.end, w.word.strip(), float(w.probability)) for w in (s.words or [])
                     if w.word.strip()]
            stats = {"avg_logprob": _opt(s, "avg_logprob"), "no_speech_prob": _opt(s, "no_speech_prob"),
                     "compression_ratio": _opt(s, "compression_ratio")}
            segments.append(Segment(s.start, s.end, s.text.strip(), words, **stats))
            raw.append({"index": i, "start": round(s.start, 3), "end": round(s.end, 3),
                        "temperature": _opt(s, "temperature"), **stats})
            if on_progress and duration > 0:
                on_progress(min(1.0, s.end / duration))
        log_event("transcribed", language=info.language, segments=len(segments),
                  words=sum(len(s.words) for s in segments))
        return Transcript(language=info.language, segments=segments, whisper_segments=raw)
    finally:
        del model

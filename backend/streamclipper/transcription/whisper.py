from pathlib import Path

from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.log import log_event
from streamclipper.models import Segment, Transcript, Word


def transcribe(wav: Path, settings: Settings, cancel: CancelToken, duration: float,
               on_progress=None) -> Transcript:
    """Transcribe con faster-whisper (CPU). Conserva marcas por palabra.

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
        for s in seg_iter:  # generador: permite cancelar entre segmentos
            cancel.check()
            words = [Word(w.start, w.end, w.word.strip(), float(w.probability)) for w in (s.words or [])
                     if w.word.strip()]
            segments.append(Segment(s.start, s.end, s.text.strip(), words, float(s.no_speech_prob)))
            if on_progress and duration > 0:
                on_progress(min(1.0, s.end / duration))
        log_event("transcribed", language=info.language, segments=len(segments),
                  words=sum(len(s.words) for s in segments))
        return Transcript(language=info.language, segments=segments)
    finally:
        del model

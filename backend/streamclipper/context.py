"""Reconstruye el contexto de análisis de un job existente (transcripción guardada + audio re-extraído)."""
import json
import shutil
from pathlib import Path

from streamclipper.analysis.audio import analyze_audio
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.media.ffmpeg import extract_audio
from streamclipper.models import Transcript
from streamclipper.pipeline import fingerprint
from streamclipper.selection.base import AnalysisContext


def load_context(job_dir: Path, result: dict, cfg: Settings, work: Path,
                 cancel: CancelToken | None = None) -> tuple[AnalysisContext, Path]:
    """Devuelve (contexto, ruta del video). No retranscribe ni descarga: solo extrae el audio para la curva de volumen."""
    t_path = job_dir / "transcript.json"
    if not t_path.is_file():
        raise InvalidInputError("El job no tiene transcript.json: no se puede reutilizar sin retranscribir")
    src = Path(result["source"])
    if not src.is_file():
        raise InvalidInputError(f"El video original ya no existe: {src}")
    if fingerprint(src) != result.get("source_fingerprint"):
        raise InvalidInputError("El video original cambió desde que se procesó este job; reprocesalo")
    transcript = Transcript.from_dict(json.loads(t_path.read_text(encoding="utf-8")))
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio.wav"
    try:
        extract_audio(src, wav, cfg.ffmpeg, cancel)
        curve = analyze_audio(wav)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return AnalysisContext(transcript, curve, result["media"]["duration"], cfg), src

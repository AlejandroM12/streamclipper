import shutil
from dataclasses import dataclass, field
from pathlib import Path

from streamclipper.core.errors import InvalidInputError, ToolNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[3]


def _find_tool(name: str) -> Path:
    local = REPO_ROOT / "tools" / "ffmpeg" / f"{name}.exe"
    if local.is_file():
        return local
    found = shutil.which(name)
    if found:
        return Path(found)
    raise ToolNotFoundError(
        f"No se encontró {name}. Colocá el build portable en tools/ffmpeg/ (ver README)."
    )


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    # transcripción
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    language: str | None = None
    cpu_threads: int = 4
    # clips
    min_clip: float = 15.0
    max_clip: float = 60.0
    pre_context: float = 6.0
    post_context: float = 6.0
    max_clips: int = 3
    min_score: float = 0.35
    max_overlap: float = 0.15
    # análisis
    audio_peak_db: float = 6.0  # dB sobre el nivel habitual para considerar un pico de audio
    text_trigger: float = 0.5
    # léxico y confianza (ver docs/lexicon.md)
    lexicon_path: str | None = None   # JSON que amplía/reemplaza reglas del léxico por defecto
    min_word_prob: float = 0.4        # una palabra de reacción con menor confianza de Whisper no cuenta
    repeat_min_prob: float = 0.6      # una repetición solo cuenta si su confianza media es al menos esta
    # render
    layout: str = "blur-fit"  # blur-fit | fill | crop
    crop_x: float = 0.5
    crop_y: float = 0.5
    zoom: float = 1.0
    output_fps: int = 30
    x264_preset: str = "veryfast"
    crf: int = 21
    # recursos
    min_free_gb: float = 5.0

    @property
    def ffmpeg(self) -> Path:
        return _find_tool("ffmpeg")

    @property
    def ffprobe(self) -> Path:
        return _find_tool("ffprobe")

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def validate(self) -> None:
        if not 5 <= self.min_clip <= self.max_clip <= 180:
            raise InvalidInputError("Duración de clip inválida: se requiere 5 <= min <= max <= 180 s")
        if not 1 <= self.max_clips <= 10:
            raise InvalidInputError("max_clips debe estar entre 1 y 10")
        if self.layout not in ("blur-fit", "fill", "crop"):
            raise InvalidInputError(f"Layout desconocido: {self.layout}")
        if not (0 <= self.crop_x <= 1 and 0 <= self.crop_y <= 1 and 1 <= self.zoom <= 4):
            raise InvalidInputError("crop_x/crop_y en [0,1] y zoom en [1,4]")
        if self.pre_context < 0 or self.post_context < 0:
            raise InvalidInputError("El contexto no puede ser negativo")
        if not (0 <= self.min_word_prob <= 1 and 0 <= self.repeat_min_prob <= 1):
            raise InvalidInputError("min_word_prob y repeat_min_prob deben estar en [0, 1]")
        if self.lexicon_path and not Path(self.lexicon_path).is_file():
            raise InvalidInputError(f"No existe el archivo de léxico: {self.lexicon_path}")

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
    # construcción del clip: «legacy» = recorte ajustado a la reacción (8–30 s, etapa 3.1); «cortos», «tiktok» y «extendido» construyen
    # el contexto narrativo alrededor del momento (ver docs/context-clips.md). La aplicación usa «tiktok» por defecto.
    clip_preset: str = "legacy"
    # detección: «classic» (ranking original, línea base) o «smart» (Smart Highlights); ver docs/smart-highlights.md
    detector: str = "classic"
    smart_visual: bool = True              # smart: muestreo visual ligero y opcional (se degrada solo si falla)
    reuse_transcript: bool = False         # reutiliza la transcripción válida de un job anterior del mismo video, modelo e idioma
    # análisis
    audio_peak_db: float = 6.0  # dB sobre el nivel habitual para considerar un pico de audio
    text_trigger: float = 0.5
    # léxico y confianza (ver docs/lexicon.md)
    lexicon_path: str | None = None   # JSON que amplía/reemplaza reglas del léxico por defecto
    min_word_prob: float = 0.4        # una palabra de reacción con menor confianza de Whisper no cuenta
    repeat_min_prob: float = 0.6      # una repetición solo cuenta si su confianza media es al menos esta
    # ajuste de límites del clip (después de elegir el momento; no cambia el ranking)
    refine_bounds: bool = True
    # Duración del clip FINAL. min_clip/max_clip definen la ventana candidata que se puntúa (y por tanto el ranking);
    # estos definen cuánto puede durar el clip ya ajustado. None = usar min_clip/max_clip (comportamiento anterior).
    refine_min_clip: float | None = 8.0
    refine_max_clip: float | None = 30.0
    refine_fades: bool = True               # pequeña transición de audio en los bordes cuando el corte cae sobre sonido
    refine_fade_in: float = 0.12            # s máximos de fade-in (nunca invade el habla)
    refine_fade_out: float = 0.30           # s máximos de fade-out (nunca invade el habla)
    refine_lead: float = 1.0                # s de margen antes de la frase que contiene el momento
    refine_tail: float = 0.6                # s de margen después de que termina la reacción
    refine_min_post: float = 2.5            # s mínimos que se conservan después del momento principal
    refine_reaction_drop_db: float = 6.0    # la reacción termina cuando el volumen cae esta cantidad bajo su pico
    refine_calm_hold: float = 1.0           # s que debe sostenerse la caída para considerarla el final
    refine_max_extend: float = 4.0          # s máximos que se puede alargar un clip para completar una frase
    refine_max_post: float = 8.0            # s máximos tras el pico que la reacción puede extenderse (evita colas largas)
    refine_burst_gap: float = 1.5           # pausa máxima entre frases de una misma reacción (el volumen cae entre palabras)
    # subtítulos (ver docs/subtitles.md)
    sub_max_words: int = 7
    sub_line_chars: int | None = None       # None: se deriva del tamaño de fuente y los márgenes seguros
    sub_max_lines: int = 2
    sub_min_duration: float = 1.0           # duración mínima de lectura de un subtítulo (s)
    sub_max_cps: float = 17.0               # caracteres por segundo cómodos de leer
    sub_bridge_gap: float = 0.6             # huecos menores se rellenan (evita parpadeos)
    sub_tail_hold: float = 0.3              # s que se mantiene un subtítulo tras la última palabra
    sub_phrase_gap: float = 0.7             # pausa que separa frases
    sub_low_prob: float = 0.4               # palabra de baja confianza si su probabilidad es menor
    sub_font_size: int = 68
    sub_mark_low_conf: bool = False         # colorear palabras de baja confianza (útil para revisar, no para publicar)
    # render
    layout: str = "blur-fit"  # blur-fit | focus | fill | crop | stack
    composition: dict | None = None         # stack: regiones por rol (render/compose.py)
    crop_x: float = 0.5
    crop_y: float = 0.5
    zoom: float = 1.0
    fg_zoom: float = 1.2                    # focus: cuánto se acerca el gameplay como máximo (recorta los costados)
    fg_x: float = 0.5                       # focus: posición horizontal del recorte (0 = izquierda, 1 = derecha)
    # Regiones que NO deben recortarse, indicadas a mano como "x,y,w,h" normalizados (0..1) del cuadro original, p. ej. la
    # facecam o un HUD. No hay detección visual: el sistema solo respeta lo que se declara aquí.
    keep_regions: list[str] = field(default_factory=list)
    bg_blur: int = 30                       # radio del desenfoque del fondo (0 = sin desenfoque)
    bg_dim: float = 0.08                    # oscurecimiento del fondo (0..0.5) para que se lean los subtítulos
    safe_platform: str = "universal"        # universal | tiktok | reels | shorts (márgenes seguros aproximados)
    output_fps: int = 30
    x264_preset: str = "veryfast"
    crf: int = 21
    # recursos
    min_free_gb: float = 5.0

    def clip_bounds(self) -> tuple[float, float]:
        """(mínimo, máximo) de duración del clip final. Nunca supera max_clip; el mínimo nunca supera el máximo."""
        hi = min(self.refine_max_clip if self.refine_max_clip is not None else self.max_clip, self.max_clip)
        lo = self.refine_min_clip if self.refine_min_clip is not None else self.min_clip
        return min(lo, hi), hi

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
        if self.clip_preset not in ("legacy", "cortos", "tiktok", "extendido"):
            raise InvalidInputError(f"Preset de duración desconocido: {self.clip_preset}")
        if self.detector not in ("classic", "smart"):
            raise InvalidInputError(f"Detector desconocido: {self.detector} (classic o smart)")
        if self.layout not in ("blur-fit", "focus", "fill", "crop", "stack"):
            raise InvalidInputError(f"Layout desconocido: {self.layout}")
        if not (0 <= self.crop_x <= 1 and 0 <= self.crop_y <= 1 and 1 <= self.zoom <= 4):
            raise InvalidInputError("crop_x/crop_y en [0,1] y zoom en [1,4]")
        if not (1 <= self.fg_zoom <= 2.5 and 0 <= self.fg_x <= 1 and 0 <= self.bg_blur <= 80 and 0 <= self.bg_dim <= 0.5):
            raise InvalidInputError("fg_zoom en [1,2.5], fg_x en [0,1], bg_blur en [0,80] y bg_dim en [0,0.5]")
        if self.safe_platform not in ("universal", "tiktok", "reels", "shorts"):
            raise InvalidInputError(f"Plataforma desconocida: {self.safe_platform}")
        if not (1 <= self.sub_max_words <= 15 and 1 <= self.sub_max_lines <= 3 and 40 <= self.sub_font_size <= 110
                and 0.3 <= self.sub_min_duration <= 3 and 8 <= self.sub_max_cps <= 30
                and (self.sub_line_chars is None or 10 <= self.sub_line_chars <= 40)):
            raise InvalidInputError("Parámetros de subtítulos fuera de rango (ver docs/subtitles.md)")
        if not (0 <= self.refine_lead <= 5 and 0 <= self.refine_tail <= 5 and 0 <= self.refine_min_post <= 15
                and 1 <= self.refine_reaction_drop_db <= 20 and 0.2 <= self.refine_calm_hold <= 5
                and 0 <= self.refine_max_extend <= 15 and 0 <= self.refine_burst_gap <= 5
                and 1 <= self.refine_max_post <= 30):
            raise InvalidInputError("Parámetros de ajuste de límites fuera de rango")
        for name in ("refine_min_clip", "refine_max_clip"):
            v = getattr(self, name)
            if v is not None and not 8 <= v <= 30:
                raise InvalidInputError(f"{name} debe estar entre 8 y 30 s (o ser None para usar min_clip/max_clip)")
        if self.refine_min_clip is not None and self.refine_max_clip is not None and self.refine_min_clip > self.refine_max_clip:
            raise InvalidInputError("refine_min_clip no puede superar refine_max_clip")
        if not (0 <= self.refine_fade_in <= 1 and 0 <= self.refine_fade_out <= 2):
            raise InvalidInputError("refine_fade_in en [0,1] y refine_fade_out en [0,2]")
        for r in self.keep_regions:
            parse_region(r)
        if self.pre_context < 0 or self.post_context < 0:
            raise InvalidInputError("El contexto no puede ser negativo")
        if not (0 <= self.min_word_prob <= 1 and 0 <= self.repeat_min_prob <= 1):
            raise InvalidInputError("min_word_prob y repeat_min_prob deben estar en [0, 1]")
        if self.lexicon_path and not Path(self.lexicon_path).is_file():
            raise InvalidInputError(f"No existe el archivo de léxico: {self.lexicon_path}")


def recommended_model(language: str | None) -> str:
    """Modelo Whisper recomendado: small para español (mediciones en gameplay: transcripción claramente mejor que
    base) y cuando no se indica idioma; base para otros idiomas, donde todavía no hay mediciones propias."""
    return "small" if language in (None, "es") else "base"


def parse_region(text: str) -> tuple[float, float, float, float]:
    """Región «x,y,w,h» normalizada (0..1) del cuadro original. Lanza InvalidInputError si no es válida."""
    try:
        x, y, w, h = (float(v) for v in str(text).split(","))
    except ValueError:
        raise InvalidInputError(f"Región inválida '{text}': se espera x,y,w,h con números entre 0 y 1") from None
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.0001 and y + h <= 1.0001):
        raise InvalidInputError(f"Región fuera del cuadro '{text}': x,y,w,h deben estar en [0,1] y x+w, y+h <= 1")
    return x, y, w, h

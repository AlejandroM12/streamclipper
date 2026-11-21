"""Recorte y composición vertical 1080x1920 (H.264 + AAC) con subtítulos ASS quemados."""
from pathlib import Path

from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import MediaError
from streamclipper.core.log import log_event
from streamclipper.media.ffmpeg import run_cancellable

W, H = 1080, 1920


def build_filter(settings: Settings, ass_name: str) -> str:
    """Grafo de filtros. `ass_name` es relativo al cwd de FFmpeg (evita escapes de rutas de Windows)."""
    fps = f"fps={settings.output_fps}"
    tail = f"setsar=1,{fps},ass={ass_name},format=yuv420p"
    if settings.layout == "blur-fit":
        return (
            "[0:v]split=2[a][b];"
            f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:4[bg];"
            f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{tail}[v]"
        )
    # fill / crop: cubre el lienzo y recorta con posición (crop_x, crop_y) y zoom configurables
    zoom = 1.0 if settings.layout == "fill" else settings.zoom
    cx, cy = (0.5, 0.5) if settings.layout == "fill" else (settings.crop_x, settings.crop_y)
    sw, sh = int(W * zoom) // 2 * 2, int(H * zoom) // 2 * 2
    return (
        f"[0:v]scale={sw}:{sh}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H}:(iw-{W})*{cx}:(ih-{H})*{cy},{tail}[v]"
    )


def render_clip(src: Path, start: float, end: float, ass_file: Path, out: Path,
                settings: Settings, cancel: CancelToken) -> None:
    part = out.with_name(out.name + ".part")
    cmd = [
        settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", src.resolve(), "-t", f"{end - start:.3f}",
        "-filter_complex", build_filter(settings, ass_file.name),
        "-map", "[v]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", settings.x264_preset, "-crf", str(settings.crf),
        "-profile:v", "high", "-threads", str(settings.cpu_threads),
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-f", "mp4", part.resolve(),
    ]
    try:
        code, _, err = run_cancellable(cmd, cancel=cancel, cwd=ass_file.parent, timeout=3600)
        if code != 0:
            raise MediaError(f"FFmpeg falló al renderizar el clip: {err.strip()[-400:]}")
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)  # no deja .part ante error o cancelación
    log_event("clip_rendered", out=str(out), start=round(start, 2), end=round(end, 2))

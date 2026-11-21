"""Envoltorios de FFmpeg/FFprobe. Nunca se usa shell: los comandos son listas de argumentos."""
import json
import subprocess
from pathlib import Path

from streamclipper.core.cancel import CancelToken
from streamclipper.core.errors import InvalidInputError, MediaError
from streamclipper.core.log import log_event
from streamclipper.core.memory import register_child, unregister_child
from streamclipper.models import MediaInfo

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_cancellable(
    cmd: list[str],
    cancel: CancelToken | None = None,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """Ejecuta un proceso; si se cancela, vence el timeout o hay una excepción, se mata el proceso."""
    import time

    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
    )
    started = time.monotonic()
    register_child(proc.pid)
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=0.25)
                return proc.returncode, out, err
            except subprocess.TimeoutExpired:
                if cancel and cancel.cancelled:
                    cancel.check()
                if timeout and time.monotonic() - started > timeout:
                    raise MediaError(f"Tiempo de espera agotado ({timeout:.0f}s) ejecutando {Path(cmd[0]).name}")
    finally:
        unregister_child(proc.pid)
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def probe(path: Path, ffprobe: Path) -> MediaInfo:
    code, out, err = run_cancellable(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        timeout=60,
    )
    if code != 0:
        raise InvalidInputError(f"El archivo no es un video válido: {err.strip()[:300]}")
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and s.get("disposition", {}).get("attached_pic") != 1), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video:
        raise InvalidInputError("El archivo no contiene pista de video")
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise InvalidInputError("No se pudo determinar la duración del video")
    num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return MediaInfo(
        path=str(path), duration=duration, width=int(video["width"]), height=int(video["height"]),
        fps=fps, video_codec=video.get("codec_name", "?"),
        audio_codec=audio.get("codec_name") if audio else None, has_audio=audio is not None,
        size_bytes=int(data.get("format", {}).get("size") or path.stat().st_size),
    )


def extract_audio(src: Path, dst: Path, ffmpeg: Path, cancel: CancelToken | None = None) -> None:
    """Audio mono 16 kHz PCM, usado para transcripción y análisis."""
    code, _, err = run_cancellable(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
        cancel=cancel,
    )
    if code != 0:
        raise MediaError(f"Falló la extracción de audio: {err.strip()[-300:]}")
    log_event("audio_extracted", path=str(dst), bytes=dst.stat().st_size)

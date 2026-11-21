"""Validación del MP4 final con FFprobe y decodificación completa con FFmpeg."""
import json
from pathlib import Path

from streamclipper.media.ffmpeg import run_cancellable


def validate_clip(path: Path, expected_duration: float, ffmpeg: Path, ffprobe: Path,
                  tolerance: float = 0.75) -> dict:
    """Devuelve {'ok': bool, 'problems': [...], 'info': {...}}."""
    problems: list[str] = []
    info: dict = {}
    code, out, err = run_cancellable(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], timeout=60)
    if code != 0:
        return {"ok": False, "problems": [f"ffprobe no puede abrir el archivo: {err.strip()[:200]}"], "info": {}}
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    duration = float(data["format"].get("duration", 0))
    info = {
        "duration": round(duration, 3),
        "video": {k: v.get(k) for k in ("codec_name", "width", "height", "pix_fmt", "avg_frame_rate")} if v else None,
        "audio": {k: a.get(k) for k in ("codec_name", "sample_rate", "channels")} if a else None,
        "size_bytes": int(data["format"].get("size", 0)),
    }
    if not v or v["codec_name"] != "h264":
        problems.append(f"códec de video esperado h264, obtenido {v and v['codec_name']}")
    if v and (v["width"], v["height"]) != (1080, 1920):
        problems.append(f"resolución esperada 1080x1920, obtenida {v['width']}x{v['height']}")
    if v and v.get("pix_fmt") != "yuv420p":
        problems.append(f"pix_fmt esperado yuv420p, obtenido {v.get('pix_fmt')}")
    if not a or a["codec_name"] != "aac":
        problems.append(f"códec de audio esperado aac, obtenido {a and a['codec_name']}")
    if abs(duration - expected_duration) > tolerance:
        problems.append(f"duración {duration:.2f}s difiere de la esperada {expected_duration:.2f}s")

    code, _, err = run_cancellable(
        [ffmpeg, "-v", "error", "-xerror", "-i", path, "-f", "null", "-"], timeout=600)
    if code != 0 or err.strip():
        problems.append(f"errores al decodificar: {err.strip()[:200]}")
    return {"ok": not problems, "problems": problems, "info": info}

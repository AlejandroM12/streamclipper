"""Chequeo de requisitos: lo que StreamClipper necesita para funcionar en esta máquina (`streamclipper doctor` y el panel
«Estado del sistema» de la aplicación)."""
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

from streamclipper.core.config import Settings
from streamclipper.core.errors import ToolNotFoundError
from streamclipper.core.sysinfo import available_ram_gb


def _check(cid: str, label: str, ok: bool, detail: str, level: str | None = None) -> dict:
    return {"id": cid, "label": label, "ok": ok, "detail": detail, "level": level or ("ok" if ok else "error")}


def _hf_cache() -> Path:
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def model_cached(model: str) -> bool:
    d = _hf_cache() / f"models--Systran--faster-whisper-{model}" / "snapshots"
    return d.is_dir() and any(d.rglob("model.bin"))


def _ffmpeg_features(ffmpeg: Path) -> tuple[bool, bool]:
    def run(*a):
        return subprocess.run([str(ffmpeg), "-hide_banner", *a], capture_output=True, text=True, timeout=30).stdout
    try:
        return " libx264 " in run("-encoders"), "ass " in run("-filters")
    except (OSError, subprocess.SubprocessError):
        return False, False


def run_doctor(settings: Settings) -> dict:
    checks = []
    v = sys.version_info
    checks.append(_check("python", "Python 3.11 o superior", v >= (3, 11), f"{v.major}.{v.minor}.{v.micro}"))
    for mod, label in (("faster_whisper", "faster-whisper"), ("yt_dlp", "yt-dlp"), ("numpy", "numpy"), ("av", "PyAV")):
        try:
            m = importlib.import_module(mod)
            checks.append(_check(mod, label, True, getattr(m, "__version__", None) or getattr(getattr(m, "version", None), "__version__", "instalado") or "instalado"))
        except Exception as e:
            checks.append(_check(mod, label, False, f"no se pudo importar ({type(e).__name__}); ejecutá instalar.bat"))
    try:
        ff, _ = settings.ffmpeg, settings.ffprobe   # ambos deben existir
        x264, ass = _ffmpeg_features(ff)
        checks.append(_check("ffmpeg", "FFmpeg y FFprobe", True, str(ff.parent)))
        checks.append(_check("x264", "FFmpeg con H.264 (libx264)", x264, "disponible" if x264 else "falta libx264: usá el build «essentials» de gyan.dev"))
        checks.append(_check("libass", "FFmpeg con subtítulos (libass)", ass, "disponible" if ass else "falta libass: usá el build «essentials» de gyan.dev"))
    except ToolNotFoundError as e:
        checks.append(_check("ffmpeg", "FFmpeg y FFprobe", False, str(e)))
    for model in ("small", "base"):
        cached = model_cached(model)
        checks.append(_check(f"model_{model}", f"Modelo Whisper «{model}»", cached,
                             "descargado" if cached else "se descargará la primera vez que se use (small ≈ 480 MB, base ≈ 150 MB)",
                             "ok" if cached else "warn"))
    ram = available_ram_gb()
    if ram is not None:
        checks.append(_check("ram", "Memoria RAM libre", ram >= 1.5, f"{ram:.1f} GB libres (small necesita ≈ 1 GB; cerrá otros programas si hay poca)",
                             "ok" if ram >= 1.5 else "warn"))
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        free = shutil.disk_usage(settings.data_dir).free / 1024**3
        checks.append(_check("disk", "Espacio en disco", free >= settings.min_free_gb, f"{free:.0f} GB libres en {settings.data_dir}"))
    except OSError as e:
        checks.append(_check("disk", "Carpeta de datos escribible", False, str(e)))
    try:
        import tkinter  # noqa: F401
        checks.append(_check("tk", "Selector de archivos (tkinter)", True, "disponible", "ok"))
    except ImportError:
        checks.append(_check("tk", "Selector de archivos (tkinter)", False, "no disponible: pegá la ruta del archivo a mano", "warn"))
    return {"ok": all(c["ok"] or c["level"] == "warn" for c in checks), "checks": checks}

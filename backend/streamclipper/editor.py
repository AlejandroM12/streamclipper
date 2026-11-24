"""Datos para el editor manual de clips (1.5): un único eje de tiempo y todo lo que el editor dibuja sobre él.

El editor trabaja sobre el video más amplio que haya de la sesión, en SU eje de tiempo («eje del editor»):
  · `context`: `<sesión>/context/source.mp4` (tramo del VOD ya recuperado, ver expand.py), con su transcripción reunida.
  · `source`:  el video de la sesión (un archivo completo, o el recorte del modo automático).
Los intervalos guardados en otros ejes (el clip original está en el eje de la fuente de la sesión) se convierten con `shift`.
Si hay tiempo absoluto del VOD (`offset_abs`) el editor lo muestra aparte: nunca se mezcla con los tiempos del clip.
Nada de esto escribe sobre el clip original, sus versiones ni la transcripción.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.edit import list_edits
from streamclipper.expand import load_proposal
from streamclipper.jobs import load_result
from streamclipper.media.ffmpeg import probe

WAVE_HOP = 0.5            # s por punto de la forma de onda
MAX_SENTENCES = 800
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def editor_axis(job_dir: Path, result: dict) -> dict:
    """(archivo, eje, desplazamiento del eje de la fuente al del editor, tiempo absoluto del VOD en t=0, ruta de la transcripción)."""
    seg = _load_json(job_dir / "segment.json") or {}
    cut0 = float(seg["cut"][0]) if seg.get("cut") else None          # tiempo absoluto del VOD en t=0 de la fuente de la sesión
    cm = _load_json(job_dir / "context" / "meta.json")
    if cm and (job_dir / "context" / cm.get("source", "source.mp4")).is_file() and (job_dir / "context" / "transcript.json").is_file():
        off = float(cm["offset"])
        return {"axis": "context", "path": job_dir / "context" / cm["source"], "shift": (cut0 - off) if cut0 is not None else 0.0,
                "offset_abs": off, "transcript": job_dir / "context" / "transcript.json", "url": cm.get("url")}
    src = Path(result.get("source", ""))
    return {"axis": "source", "path": src if src.is_file() else None, "shift": 0.0, "offset_abs": cut0,
            "transcript": job_dir / "transcript.json", "url": seg.get("url")}


def waveform(path: Path, cache: Path, ffmpeg, hop: float = WAVE_HOP) -> list[float]:
    """Envolvente de volumen 0..1 (RMS en dB normalizado) cada `hop` segundos. Se guarda en caché junto a la sesión."""
    key = f"{path.stat().st_size}-{int(path.stat().st_mtime)}-{hop}"
    c = _load_json(cache)
    if c and c.get("key") == key:
        return c["wave"]
    sr = 8000
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    x = np.frombuffer(raw[: len(raw) // 2 * 2], np.int16).astype(np.float32) / 32768.0
    n = int(sr * hop)
    if len(x) < n:
        return []
    rms = np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    wave = np.clip((db + 60.0) / 60.0, 0, 1)                         # -60 dB..0 dB -> 0..1
    out = [round(float(v), 3) for v in wave]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "wave": out}), encoding="utf-8")
    return out


def _sentences(tr_path: Path) -> list[list]:
    d = _load_json(tr_path) or {}
    return [[round(float(s["start"]), 2), round(float(s["end"]), 2), str(s.get("text", ""))[:80]] for s in (d.get("segments") or [])][:MAX_SENTENCES]


def editor_data(job_dir: Path, clip: str, settings: Settings, with_wave: bool = True) -> dict:
    result = load_result(job_dir)
    entry = next((c for c in result.get("clips", []) if c.get("name") == clip), None)
    if entry is None:
        raise InvalidInputError(f"El clip {clip} no existe en esta sesión.")
    ax = editor_axis(job_dir, result)
    shift, path = ax["shift"], ax["path"]
    if path is None:
        return {"available": False, "reason": "El video original ya no está disponible, así que no se puede editar sobre él.",
                "axis": ax["axis"]}
    info = probe(path, settings.ffprobe)
    cand = next((c for c in result.get("candidates", []) if c.get("rank") == entry.get("rank")), None)
    def on(v):                                    # eje de la fuente de la sesión -> eje del editor
        return None if v is None else round(float(v) + (shift if ax["axis"] == "context" else 0.0), 2)
    ctx = entry.get("context") or {}
    prop = load_proposal(job_dir, clip)
    versions = []
    for m in list_edits(job_dir, clip):
        if bool(m.get("context")) == (ax["axis"] == "context"):
            a, b = m["start"], m["end"]
        elif ax["axis"] == "context":                 # versión hecha sobre la fuente de la sesión: se pasa al eje del editor
            a, b = on(m["start"]), on(m["end"])
        else:
            continue
        versions.append({"id": m["version"], "start": a, "end": b, "context": bool(m.get("context_note")),
                         "composition": (m.get("composition") or {}).get("preset")})
    event = ctx.get("event")
    original = [on(entry["start"]), on(entry["end"])]
    return {
        "available": True, "axis": ax["axis"], "duration": round(info.duration, 2), "width": info.width, "height": info.height,
        "has_audio": info.has_audio, "media": f"/media/{job_dir.name}/context/source.mp4" if ax["axis"] == "context" else f"/source/{job_dir.name}",
        "offset_abs": ax["offset_abs"] if ax["offset_abs"] is not None else None,
        "can_fetch_more": ax["axis"] == "context" and bool(ax["url"]) and ax["offset_abs"] is not None,
        "needs_vod_context": ax["axis"] == "source" and bool(ax["url"]) and (_load_json(job_dir / "segment.json") or {}).get("mode") == "auto-clip",
        "original": original, "candidate": [on(cand["start"]), on(cand["end"])] if cand else None, "anchor": on(cand["anchor"]) if cand else None,
        "event": [on(event[0]), on(event[1])] if event else original,
        "contextual": [prop["start"], prop["end"]] if prop and (ax["axis"] == "context" or prop.get("source") == "own") else None,
        "context_info": prop.get("context") if prop else None, "built_context": ctx or None,
        "versions": versions, "sentences": _sentences(ax["transcript"]),
        "waveform": waveform(path, job_dir / "editor" / "wave.json", settings.ffmpeg) if with_wave else [], "wave_hop": WAVE_HOP,
    }

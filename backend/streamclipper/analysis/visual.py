"""Análisis visual ligero y OPCIONAL para Smart Highlights.

Se muestrean pocos fotogramas diminutos (96×54, escala de grises) solo en las ventanas de los candidatos y con un tope de
segundos de video: primero solo fotogramas clave (casi gratis de decodificar) y, si hay muy pocos, 1 fps. No hay detección de
objetos ni modelos. Mide cambios de escena, «picos» de cambio visual respecto del contexto y cuán distinto es el momento de
los segundos previos. El movimiento intenso NO se premia por sí solo: lo que cuenta es el cambio respecto del contexto, y
solo suma si hay corroboración de audio o texto (esa regla vive en selection/smart.py).
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from streamclipper.core.cancel import CancelToken
from streamclipper.core.memory import register_child, unregister_child
from streamclipper.models import Candidate

W, H = 96, 54
PRE_ROLL = 8.0            # s de contexto visual previo a cada candidato
MAX_SAMPLED_S = 900.0     # tope de segundos de video muestreados por análisis
MAX_WINDOWS = 20          # solo los mejores candidatos (por el ranking clásico) se analizan
CUT_DIFF = 0.30           # diferencia media de píxeles (0..1) que se considera un corte de escena
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class VisualSignals:
    cuts: int = 0                      # cortes de escena cerca del momento
    spike: float = 0.0                 # 0..1: cambio visual del momento frente a su contexto
    novelty: float = 0.0               # 0..1: percentil entre los candidatos analizados de cuán distinto es el cuadro del momento
    novelty_raw: float = 0.0           # diferencia media de píxeles (0..1) entre el momento y los segundos previos
    steady_motion: bool = False        # mucho cambio visual, pero constante (no distinto del contexto)
    frames: int = 0
    note: str = ""


@dataclass
class VisualData:
    available: bool
    reason: str = ""
    sampled_seconds: float = 0.0
    seconds_spent: float = 0.0
    per_candidate: dict[int, VisualSignals] = field(default_factory=dict)   # id(candidato) -> señales


def _frames(src: Path, ffmpeg: Path, t0: float, dur: float, vf: str, extra: list[str], cancel: CancelToken) -> np.ndarray:
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", *extra, "-ss", f"{t0:.2f}", "-t", f"{dur:.2f}",
           "-i", str(src), "-an", "-sn", "-vf", vf, "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    register_child(proc.pid)
    try:
        while True:
            try:
                out, _ = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                cancel.check()
    finally:
        unregister_child(proc.pid)
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    if proc.returncode != 0:
        return np.empty((0, H, W), np.uint8)
    n = len(out) // (W * H)
    return np.frombuffer(out[: n * W * H], np.uint8).reshape(n, H, W)


def sample_window(src: Path, ffmpeg: Path, t0: float, t1: float, cancel: CancelToken) -> tuple[np.ndarray, np.ndarray]:
    """(fotogramas, tiempos). Fotogramas clave primero; si son muy pocos para la ventana, 1 fps."""
    dur = max(0.5, t1 - t0)
    scale = f"scale={W}:{H}:flags=area"
    fr = _frames(src, ffmpeg, t0, dur, scale, ["-skip_frame", "nokey"], cancel)
    if len(fr) >= max(4, dur / 4):
        return fr, t0 + np.linspace(0, dur, len(fr), endpoint=False)
    fr = _frames(src, ffmpeg, t0, dur, f"fps=1,{scale}", [], cancel)
    return fr, t0 + np.arange(len(fr)) * 1.0


def _signals(fr: np.ndarray, ts: np.ndarray, anchor: float) -> VisualSignals:
    if len(fr) < 4:
        return VisualSignals(frames=len(fr), note="muy pocos fotogramas muestreados")
    f = fr.astype(np.float32) / 255.0
    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))             # cambio entre muestras consecutivas
    dt = ts[1:]
    near = (dt >= anchor - 3) & (dt <= anchor + 3)
    before = dt < anchor - 3
    base = float(np.median(diff[before])) if before.sum() >= 2 else float(np.median(diff))
    peak = float(diff[near].max()) if near.any() else 0.0
    cuts = int(((diff >= CUT_DIFF) & near).sum())
    spike = float(np.clip((peak - base) / max(0.25, 1e-6 + 0.4 - base), 0, 1)) if near.any() else 0.0
    pre = f[(ts >= anchor - PRE_ROLL) & (ts < anchor - 2)]
    now = f[(ts >= anchor - 1) & (ts <= anchor + 2)]
    novelty = float(np.abs(now.mean(0) - pre.mean(0)).mean()) if len(pre) and len(now) else 0.0   # crudo; se normaliza entre candidatos
    steady = base >= 0.12 and spike < 0.25
    note = (f"{len(fr)} fotogramas; cambio visual del momento {peak:.2f} vs {base:.2f} del contexto; {cuts} corte(s) de escena"
            + ("; movimiento constante (no distinto del contexto)" if steady else ""))
    return VisualSignals(cuts=cuts, spike=spike, novelty_raw=novelty, steady_motion=steady, frames=len(fr), note=note)


def analyze_visual(src: Path, ffmpeg: Path, candidates: list[Candidate], cancel: CancelToken | None = None,
                   max_seconds: float = MAX_SAMPLED_S, max_windows: int = MAX_WINDOWS) -> VisualData:
    """Muestrea los mejores candidatos (orden recibido) hasta el tope de segundos. Nunca recorre el video completo."""
    import time
    cancel = cancel or CancelToken()
    if not src.is_file():
        return VisualData(False, "no se encuentra el video original")
    data, t0, budget = VisualData(True), time.monotonic(), max_seconds
    try:
        for c in candidates[:max_windows]:
            a, b = max(0.0, c.anchor - PRE_ROLL), c.anchor + 4.0
            if b - a > budget:
                break
            fr, ts = sample_window(src, ffmpeg, a, b, cancel)
            budget -= b - a
            data.sampled_seconds += b - a
            data.per_candidate[id(c)] = _signals(fr, ts, c.anchor)
    except OSError as e:
        return VisualData(False, f"no se pudo ejecutar FFmpeg: {e}")
    data.seconds_spent = time.monotonic() - t0
    raws = sorted(v.novelty_raw for v in data.per_candidate.values() if v.frames >= 4)
    if len(raws) >= 4:   # la novedad es relativa al propio video: percentil entre los candidatos muestreados
        for v in data.per_candidate.values():
            v.novelty = float(np.searchsorted(raws, v.novelty_raw, side="right") - 1) / (len(raws) - 1) if v.frames >= 4 else 0.0
    if not data.per_candidate:
        data.available, data.reason = False, "no se muestreó ningún candidato"
    return data

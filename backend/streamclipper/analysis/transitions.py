"""Transiciones visuales entre situaciones (1.5, etapa 3): muestreo ligero de fotogramas, sin OCR ni modelos.

Se decodifican fotogramas diminutos (160×90, gris) a 2 por segundo y se mide cuánto cambia la imagen entre muestras consecutivas.
Un cambio MUY por encima de lo habitual de ese mismo video (≥ 6 veces la mediana y ≥ 0,03) es una transición: otra pregunta, otra
pantalla, otro escenario. Es una señal de límite, no una comprensión de la escena:

  · En una trivia / presentación con paneles estáticos la mediana es ≈ 0 y cada cambio de pregunta destaca (medido: ver
    docs/context-clips.md). Con una cámara que se mueve poco, los cambios de escena también.
  · En gameplay con movimiento constante la mediana es alta: el umbral relativo casi nunca se supera y la señal se descarta
    (`noisy`), de modo que NO inventa límites donde la imagen cambia todo el tiempo.
Coste medido: ≈ 4 s de CPU por cada 90 s de video (≈ 20× tiempo real).
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from streamclipper.core.cancel import CancelToken
from streamclipper.core.memory import register_child, unregister_child

W, H, FPS = 160, 90, 2
MIN_DIFF = 0.03           # cambio medio de píxeles (0..1) mínimo para llamarlo transición
REL_MEDIAN = 6.0          # y al menos esta cantidad de veces la mediana del video
NOISY_MEDIAN = 0.03       # con una mediana así, la imagen cambia siempre: no hay señal fiable
GROUP_S = 1.5             # cambios seguidos (p. ej. una animación) cuentan como una sola transición
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Transitions:
    times: list[float] = field(default_factory=list)        # instante (s, eje del video analizado) en que empieza cada transición
    strengths: list[float] = field(default_factory=list)
    median: float = 0.0
    noisy: bool = False
    frames: int = 0
    seconds: float = 0.0                                    # tiempo de CPU/pared usado en el análisis

    def to_dict(self) -> dict:
        return {"times": [round(t, 2) for t in self.times], "strengths": [round(s, 3) for s in self.strengths], "median": round(self.median, 4),
                "noisy": self.noisy, "frames": self.frames, "seconds": round(self.seconds, 1)}


def find_transitions(frames: np.ndarray, t0: float = 0.0, fps: float = FPS, region: tuple[float, float, float, float] | None = None) -> Transitions:
    """A partir de fotogramas (n, H, W) uint8. `region` = (x, y, w, h) en fracciones: mide solo esa zona (p. ej. el panel de preguntas)."""
    if len(frames) < 6:
        return Transitions(frames=len(frames))
    f = frames.astype(np.float32) / 255.0
    if region is not None:
        x, y, w, h = region
        f = f[:, int(y * f.shape[1]):max(int(y * f.shape[1]) + 1, int((y + h) * f.shape[1])), int(x * f.shape[2]):max(int(x * f.shape[2]) + 1, int((x + w) * f.shape[2]))]
    d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    med = float(np.median(d))
    if med >= NOISY_MEDIAN:
        return Transitions(median=med, noisy=True, frames=len(frames))
    thr = max(MIN_DIFF, REL_MEDIAN * med)
    times, strengths, last = [], [], -1e9
    for i, v in enumerate(d):
        t = t0 + (i + 1) / fps
        if v < thr:
            continue
        if t - last <= GROUP_S:
            last = t
            continue
        times.append(t)
        strengths.append(float(v))
        last = t
    return Transitions(times, strengths, med, False, len(frames))


def sample_gray(src: Path, ffmpeg, a: float, b: float, cancel: CancelToken | None = None) -> np.ndarray:
    cancel = cancel or CancelToken()
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", str(src), "-an", "-sn",
           "-vf", f"fps={FPS},scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
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


def detect_transitions(src: Path, ffmpeg, a: float, b: float, cancel: CancelToken | None = None,
                       region: tuple[float, float, float, float] | None = None) -> Transitions:
    """Transiciones del tramo [a, b] del video, en el eje del video. Nunca falla: sin FFmpeg o sin fotogramas devuelve una señal vacía."""
    import time
    t0 = time.monotonic()
    try:
        fr = sample_gray(src, ffmpeg, max(0.0, a), b, cancel)
    except OSError:
        return Transitions()
    tr = find_transitions(fr, max(0.0, a), FPS, region)
    tr.seconds = time.monotonic() - t0
    return tr

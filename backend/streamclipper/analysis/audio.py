import wave
from dataclasses import dataclass

import numpy as np

HOP = 0.25  # segundos por ventana de energía
_FLOOR_DB = -90.0


@dataclass
class AudioCurve:
    """Curva de intensidad con una muestra cada HOP segundos."""
    db: np.ndarray       # nivel RMS en dBFS
    excess: np.ndarray   # dB por encima del nivel "habitual" del video (percentil 75)
    rise: np.ndarray     # subida respecto de la mediana de los ~8 s previos (dB)
    hop: float = HOP

    def index(self, t: float) -> int:
        return int(np.clip(round(t / self.hop), 0, len(self.db) - 1))

    def window(self, arr: np.ndarray, t0: float, t1: float) -> np.ndarray:
        a, b = self.index(t0), self.index(t1) + 1
        return arr[a:max(b, a + 1)]


def _hop_energies(path, hop: float) -> np.ndarray:
    """Energía media cuadrática por hop, leyendo el WAV por bloques (apto para videos largos)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise ValueError("Se esperaba WAV mono PCM 16-bit")
        rate = wf.getframerate()
        hop_n = int(rate * hop)
        energies: list[float] = []
        rest = np.empty(0, dtype=np.int16)
        while True:
            raw = wf.readframes(hop_n * 240)
            if not raw:
                break
            buf = np.concatenate([rest, np.frombuffer(raw, dtype=np.int16)])
            n = len(buf) // hop_n
            if n:
                blocks = buf[: n * hop_n].reshape(n, hop_n).astype(np.float64) / 32768.0
                energies.extend((blocks**2).mean(axis=1))
            rest = buf[n * hop_n:]
        if len(rest) > hop_n // 2:
            energies.append(float(((rest.astype(np.float64) / 32768.0) ** 2).mean()))
    return np.asarray(energies)


def curve_from_db(db: np.ndarray, hop: float = HOP) -> AudioCurve:
    # padding con el valor del borde: el relleno con ceros de mode="same" equivale a 0 dBFS
    # y crearía un pico falso en el primer y último hop
    smooth = np.convolve(np.pad(db, (1, 0), mode="edge"), np.ones(2) / 2, mode="valid")
    # Referencia = percentil 75: el nivel de habla/juego habitual. Una medida basada en la mediana
    # trataría toda la voz como "pico" cuando el audio es bimodal (silencio vs habla).
    ref = float(np.percentile(smooth, 75))
    k = int(8 / hop)
    rise = np.zeros_like(smooth)
    for i in range(len(smooth)):
        prev = smooth[max(0, i - k):i]
        rise[i] = smooth[i] - (np.median(prev) if len(prev) else ref)
    return AudioCurve(db=smooth, excess=smooth - ref, rise=rise, hop=hop)


def analyze_audio(path, hop: float = HOP) -> AudioCurve:
    energy = _hop_energies(path, hop)
    if len(energy) == 0:
        raise ValueError("Audio vacío")
    return curve_from_db(np.maximum(10 * np.log10(energy + 1e-10), _FLOOR_DB), hop)

"""Señales de audio ligeras para Smart Highlights (solo numpy, sobre el mismo WAV mono de 16 kHz del pipeline).

Todas las curvas usan la misma rejilla que AudioCurve (una muestra cada HOP s). Un volumen alto, por sí solo, NO es una señal
de highlight: aquí se separa la voz (banda 300–3400 Hz) del ruido/música (espectro plano), se mide la sorpresa respecto del
contexto previo, y se detectan los patrones que suelen producir falsos positivos (volumen constante, picos repetitivos,
silencio). La «risa» es una evidencia débil (ráfagas vocales regulares de 3–8 Hz): se informa como «posible», nunca como
una probabilidad.
"""
import wave
from dataclasses import dataclass

import numpy as np

from streamclipper.analysis.audio import AudioCurve

FRAME = 512
VOICE_LO, VOICE_HI = 300.0, 3400.0
ENV_FRAME_S = 0.025          # envolvente de amplitud a 40 Hz para medir la modulación (ráfagas de risa ≈ 3–8 Hz)
CONTEXT_S = 30.0             # contexto previo para medir cambios de energía
QUIET_DB = -8.0              # «silencio relativo»: 8 dB por debajo del nivel habitual (percentil 75)


@dataclass
class AudioFeatures:
    hop: float
    voice: np.ndarray           # fracción de la energía en la banda de voz (0..1)
    flatness: np.ndarray        # planitud espectral (0..1): ruido/música/explosiones ≈ alta, voz ≈ baja
    centroid: np.ndarray        # centroide espectral (Hz)
    laugh: np.ndarray           # 0..1: evidencia de ráfagas vocales regulares (3–8 Hz), «posible risa»
    context_delta: np.ndarray   # dB sobre la mediana de los 30 s previos (sin el último segundo)
    constancy: np.ndarray       # 0..1: volumen casi constante en ±10 s con nivel alto (música/ambiente)
    periodic: np.ndarray        # 0..1: picos de volumen a intervalos regulares en ±15 s (explosiones/disparos repetitivos)
    quiet: np.ndarray           # bool: hop claramente por debajo del nivel habitual

    def index(self, t: float) -> int:
        return int(np.clip(round(t / self.hop), 0, len(self.voice) - 1))

    def window(self, arr: np.ndarray, t0: float, t1: float) -> np.ndarray:
        a, b = self.index(t0), self.index(t1) + 1
        return arr[a:max(b, a + 1)]


def _spectral(path, hop: float, n_hops: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """voice, flatness, centroid y envolvente (25 ms RMS) por bloques, sin cargar el WAV entero."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        hop_n = int(rate * hop)
        env_n = int(rate * ENV_FRAME_S)
        per_hop = hop_n // env_n
        freqs = np.fft.rfftfreq(FRAME, 1 / rate)
        win = np.hanning(FRAME)
        band_v = (freqs >= VOICE_LO) & (freqs <= VOICE_HI)
        band_all = freqs >= 80
        band_f = (freqs >= 100) & (freqs <= 6000)
        frames = hop_n // FRAME
        voice, flat, cent, env = [], [], [], []
        rest = np.empty(0, dtype=np.int16)
        while True:
            raw = wf.readframes(hop_n * 240)
            if not raw:
                break
            buf = np.concatenate([rest, np.frombuffer(raw, dtype=np.int16)])
            n = len(buf) // hop_n
            rest = buf[n * hop_n:]
            if not n:
                continue
            x = buf[: n * hop_n].reshape(n, hop_n).astype(np.float32) / 32768.0
            fr = x[:, : frames * FRAME].reshape(n, frames, FRAME) * win
            p = (np.abs(np.fft.rfft(fr, axis=-1)) ** 2).mean(axis=1) + 1e-12
            voice.append(p[:, band_v].sum(1) / p[:, band_all].sum(1))
            pf = p[:, band_f]
            flat.append(np.exp(np.log(pf).mean(1)) / pf.mean(1))
            cent.append((p[:, band_all] * freqs[band_all]).sum(1) / p[:, band_all].sum(1))
            e = x[:, : per_hop * env_n].reshape(n, per_hop, env_n)
            env.append(np.sqrt((e ** 2).mean(-1)).reshape(-1))
    voice, flat, cent = (np.concatenate(a) if a else np.zeros(0) for a in (voice, flat, cent))
    env = np.concatenate(env) if env else np.zeros(0)
    pad = n_hops - len(voice)
    if pad > 0:   # la curva de volumen puede tener un hop más (cola parcial): se repite el último valor
        voice, flat, cent = (np.pad(a, (0, pad), mode="edge") for a in (voice, flat, cent))
    return voice[:n_hops], flat[:n_hops], cent[:n_hops], env


def _laugh(env: np.ndarray, hop: float, excess: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Concentración de la modulación de amplitud en 3–8 Hz en ventanas de 2 s (ráfagas regulares), con planitud baja y sonoridad."""
    n = len(excess)
    out = np.zeros(n)
    per_hop = int(round(hop / ENV_FRAME_S))
    w = int(round(2.0 / ENV_FRAME_S))
    if len(env) < w + per_hop:
        return out
    rate = 1.0 / ENV_FRAME_S
    mf = np.fft.rfftfreq(128, 1 / rate)
    band = (mf >= 3.0) & (mf <= 8.0)
    full = (mf >= 0.8) & (mf <= 20.0)
    wins = np.lib.stride_tricks.sliding_window_view(env, w)[::per_hop]
    x = wins - wins.mean(axis=1, keepdims=True)
    p = np.abs(np.fft.rfft(x * np.hanning(w), n=128, axis=1)) ** 2
    narrow = p[:, band].max(axis=1) / (p[:, full].sum(axis=1) + 1e-12)   # un pico estrecho = ráfagas regulares
    score = np.clip((narrow - 0.20) / 0.35, 0, 1)
    half = w // per_hop // 2
    idx = np.arange(len(score)) + half                                   # la ventana se centra en su hop
    ok = idx < n
    idx, score = idx[ok], score[ok]
    gate = np.clip((0.55 - flat[idx]) / 0.3, 0, 1) * np.clip((excess[idx] - 2.0) / 4.0, 0, 1)   # no ruido plano; sonoro
    out[idx] = score * gate
    return out


def _context_delta(db: np.ndarray, hop: float) -> np.ndarray:
    k, skip = int(CONTEXT_S / hop), int(1.0 / hop)
    out = np.zeros_like(db)
    for i in range(len(db)):
        prev = db[max(0, i - k): max(0, i - skip)]
        out[i] = db[i] - np.median(prev) if len(prev) >= int(4 / hop) else 0.0
    return out


def _constancy(db: np.ndarray, excess: np.ndarray, hop: float) -> np.ndarray:
    h = int(10 / hop)
    c1, c2 = np.cumsum(np.pad(db, (1, 0))), np.cumsum(np.pad(db ** 2, (1, 0)))
    lo, hi = np.clip(np.arange(len(db)) - h, 0, len(db)), np.clip(np.arange(len(db)) + h + 1, 0, len(db))
    n = np.maximum(hi - lo, 1)
    mean = (c1[hi] - c1[lo]) / n
    std = np.sqrt(np.maximum((c2[hi] - c2[lo]) / n - mean ** 2, 0))
    ref = float(np.mean(db - excess))                    # nivel habitual (percentil 75) de la curva
    loud = np.clip((mean - ref + 3.0) / 3.0, 0, 1)       # el volumen medio local no está por debajo de lo habitual
    return np.clip(1 - std / 6.0, 0, 1) * loud


def _periodic(excess: np.ndarray, hop: float) -> np.ndarray:
    """Picos (≥ 5 dB sobre lo habitual) a intervalos casi iguales: típico de disparos o explosiones repetitivas."""
    out = np.zeros_like(excess)
    w = int(1.0 / hop)
    peaks = [i for i in range(len(excess)) if excess[i] >= 5.0 and excess[i] == excess[max(0, i - w): i + w + 1].max()]
    if len(peaks) < 4:
        return out
    pk = np.asarray(peaks) * hop
    r = 15.0
    for i in range(len(excess)):
        t = i * hop
        a, b = np.searchsorted(pk, t - r), np.searchsorted(pk, t + r)
        if b - a >= 4:
            gaps = np.diff(pk[a:b])
            cv = gaps.std() / max(gaps.mean(), 1e-6)
            out[i] = float(np.clip(1 - cv / 0.4, 0, 1) * min(1.0, (b - a - 3) / 3))
    return out


def analyze_features(path, curve: AudioCurve) -> AudioFeatures:
    n = len(curve.db)
    voice, flat, cent, env = _spectral(path, curve.hop, n)
    return AudioFeatures(
        hop=curve.hop, voice=voice, flatness=flat, centroid=cent,
        laugh=_laugh(env, curve.hop, curve.excess, flat),
        context_delta=_context_delta(curve.db, curve.hop),
        constancy=_constancy(curve.db, curve.excess, curve.hop),
        periodic=_periodic(curve.excess, curve.hop),
        quiet=curve.excess < QUIET_DB,
    )

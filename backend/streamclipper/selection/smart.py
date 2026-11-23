"""Smart Highlights: ranking que combina las señales del clásico con audio contextual, transcripción y (opcional) imagen.

Principios (ver docs/smart-highlights.md):
- El ranking clásico (HeuristicScorer) sigue existiendo, intacto, como línea base: aquí se reutilizan sus componentes.
- Los pesos son PRIORES fijados a priori, NO ajustados con valoraciones humanas (hay muy pocas): ver evaluation_smart.py.
- Cada componente registra su valor medido, su ventana y su evidencia; las penalizaciones son componentes de peso negativo.
- No hay «confianza»: la puntuación es un ranking relativo dentro del video, igual que en el clásico.
- Si no hay ninguna señal nueva disponible, el resultado es EXACTAMENTE el del clásico (degradación controlada). Si falta
  solo alguna (p. ej. la imagen), los pesos de las disponibles se renormalizan y se informa qué faltó.
- Volumen alto NO es highlight: se descuenta cuando es ruido plano, constante, repetitivo o sin habla que lo acompañe.
"""
from pathlib import Path

import numpy as np

from streamclipper.analysis.audio_features import AudioFeatures
from streamclipper.analysis.narrative import NarrativeSignals, analyze_narrative
from streamclipper.analysis.visual import VisualData, analyze_visual
from streamclipper.core.cancel import CancelToken
from streamclipper.models import Candidate, ScoreComponent
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.candidates import _audio_anchors, _merge_anchors, _text_anchors, build_window
from streamclipper.selection.heuristic import HeuristicScorer
from streamclipper.selection.selector import has_trigger, overlap_ratio, text_similarity

# Pesos de los componentes que suman (suman 1.0) y de las penalizaciones (se restan). Priores, no ajustados.
POSITIVE = {
    "intensidad_audio": 0.16, "dinamica_audio": 0.06, "energia_contexto": 0.10, "reaccion_vocal": 0.12, "risa_probable": 0.04,
    "reaccion_verbal": 0.18, "frases": 0.06, "arco_narrativo": 0.08, "contexto": 0.05, "duracion": 0.02,
    "calidad_transcripcion": 0.08, "visual": 0.05,
}
PENALTY = {"musica_o_ruido": 0.20, "audio_repetitivo": 0.15, "aire_muerto": 0.20, "transcripcion_dudosa": 0.12, "frase_incompleta": 0.08}
CLASSIC = ("intensidad_audio", "dinamica_audio", "reaccion_verbal", "frases", "contexto", "duracion", "calidad_transcripcion")
MIN_ANCHOR_GAP = 20.0     # s: dos clips con el ancla más cerca se consideran el mismo momento
WIN = 2.5                 # s a cada lado del ancla para las señales de audio


def _speech_mask(ctx: AnalysisContext, n: int, hop: float) -> np.ndarray:
    m = np.zeros(n, bool)
    for s in ctx.transcript.segments:
        if s.words:
            for w in s.words:
                m[int(np.clip(w.start / hop, 0, n - 1)): int(np.clip(w.end / hop, 0, n - 1)) + 1] = True
        elif s.text.strip():
            m[int(np.clip(s.start / hop, 0, n - 1)): int(np.clip(s.end / hop, 0, n - 1)) + 1] = True
    return m


def _vocal_curve(f: AudioFeatures, excess: np.ndarray, speech: np.ndarray, has_speech: bool) -> np.ndarray:
    """Reacción vocal por hop: sonoro, sorprendente frente a los 30 s previos, con voz transcrita y sin espectro plano."""
    loud = np.clip(excess / 12.0, 0, 1)
    surprise = np.clip(0.4 + f.context_delta / 20.0, 0, 1)
    tonal = np.clip((0.6 - f.flatness) / 0.4, 0, 1)
    voiced = np.where(speech, 1.0, 0.3) if has_speech else np.full(len(excess), 0.5)
    return loud * surprise * tonal * voiced


def build_smart_scorer(wav: Path, curve, src: Path | None, settings, cancel: CancelToken | None = None, warn=None) -> "SmartScorer":
    """Calcula las señales de audio y arma el scorer. Si el análisis falla, Smart se degrada y lo avisa (no aborta el trabajo)."""
    from streamclipper.analysis.audio_features import analyze_features
    features = None
    try:
        features = analyze_features(wav, curve)
    except Exception as e:  # noqa: BLE001  el audio contextual es opcional: se informa y se degrada
        if warn:
            warn(f"Smart Highlights: no se pudo calcular el audio contextual ({type(e).__name__}: {e}); se usan las demás señales")
    return SmartScorer(features, src, settings.ffmpeg, settings.smart_visual, cancel)


class SmartScorer:
    name = "smart"

    def __init__(self, features: AudioFeatures | None = None, src: Path | None = None, ffmpeg: Path | None = None,
                 use_visual: bool = True, cancel: CancelToken | None = None) -> None:
        self.features, self.src, self.ffmpeg, self.use_visual = features, src, ffmpeg, use_visual
        self.cancel = cancel or CancelToken()
        self.visual: VisualData | None = None
        self.duplicates_avoided = 0
        self._info: dict = {"requested": "smart", "effective": "smart"}
        self._speech: np.ndarray | None = None
        self._vocal: np.ndarray | None = None
        self._has_speech = False

    # --- candidatos: anclas del clásico + reacciones vocales y risas que el clásico no veía ---------------
    def _prepare(self, ctx: AnalysisContext) -> None:
        if self.features is None or self._vocal is not None:
            return
        n = len(ctx.curve.db)
        self._speech = _speech_mask(ctx, n, ctx.curve.hop)
        self._has_speech = bool(self._speech.any())
        self._vocal = _vocal_curve(self.features, ctx.curve.excess, self._speech, self._has_speech)

    def _extra_anchors(self, ctx: AnalysisContext) -> list[tuple[float, str]]:
        if self.features is None:
            return []
        self._prepare(ctx)
        hop, w = ctx.curve.hop, int(4 / ctx.curve.hop)
        out = []
        for kind, arr, thr in (("vocal", self._vocal, 0.55), ("risa", self.features.laugh, 0.5)):
            for i in range(len(arr)):
                if arr[i] >= thr and arr[i] == arr[max(0, i - w): i + w + 1].max():
                    out.append((arr[i], i * hop, kind))
        out.sort(reverse=True)
        return [(t, k) for _, t, k in out[:20]]

    def generate(self, ctx: AnalysisContext) -> list[Candidate]:
        raw = _audio_anchors(ctx) + _text_anchors(ctx) + self._extra_anchors(ctx)
        out = []
        for t, triggers in _merge_anchors(ctx, raw):
            start, end = build_window(t, ctx)
            if end - start >= ctx.settings.min_clip - 0.5:
                out.append(Candidate(start=start, end=end, anchor=t, triggers=triggers))
        return out

    # --- puntuación ---------------------------------------------------------------------------------------
    def score(self, candidates: list[Candidate], ctx: AnalysisContext) -> list[Candidate]:
        HeuristicScorer().score(candidates, ctx)                 # componentes clásicos (la línea base no cambia)
        classic = {id(c): c.score for c in candidates}
        narratives = {id(c): analyze_narrative(c, ctx) for c in candidates}
        narrative_ok = any(n.available for n in narratives.values())
        missing: list[str] = []
        if self.features is None:
            missing.append("audio contextual (no hubo WAV o falló el análisis)")
        if not narrative_ok:
            missing.append("contexto de transcripción (no hay palabras transcritas)")
        self.visual = None
        if self.use_visual and self.src is not None and self.ffmpeg is not None:
            ranked = sorted(candidates, key=lambda c: classic[id(c)], reverse=True)
            self.visual = analyze_visual(self.src, self.ffmpeg, ranked, self.cancel)
            if not self.visual.available:
                missing.append(f"imagen ({self.visual.reason})")
        elif self.use_visual:
            missing.append("imagen (sin video disponible)")
        else:
            missing.append("imagen (desactivada)")
        has_visual = bool(self.visual and self.visual.available)
        new_available = self.features is not None or narrative_ok or has_visual
        self._info = {"requested": "smart", "effective": "smart" if new_available else "clásico (degradado: sin señales nuevas)",
                      "signals": {"audio_contextual": self.features is not None, "transcripcion": narrative_ok, "imagen": has_visual},
                      "missing": missing, "weights": "priores fijados a priori; no ajustados con valoraciones humanas"}
        if self.visual and self.visual.available:
            self._info["visual"] = {"sampled_seconds": round(self.visual.sampled_seconds), "seconds_spent": round(self.visual.seconds_spent, 1),
                                    "windows": len(self.visual.per_candidate)}
        if not new_available:        # degradación exacta: mismos componentes y puntuación que el ranking clásico
            return candidates
        self._prepare(ctx)
        for c in candidates:
            self._score_one(c, ctx, narratives[id(c)], has_visual)
        return candidates

    def _score_one(self, c: Candidate, ctx: AnalysisContext, nar: NarrativeSignals, has_visual: bool) -> None:
        f, curve = self.features, ctx.curve
        classic = {x.name: x for x in c.components}
        vals: dict[str, tuple[float, str, tuple | None, list[str]]] = {n: (classic[n].value, classic[n].note, classic[n].window, classic[n].evidence)
                                                                     for n in CLASSIC}
        pen: dict[str, tuple[float, str, tuple | None, list[str]]] = {}
        iw = (c.anchor - WIN, c.anchor + WIN)
        if f is not None:
            ex = curve.window(curve.excess, *iw)
            p = int(np.argmax(ex)) + curve.index(iw[0])           # hop del pico dentro de la ventana
            p = int(np.clip(p, 0, len(curve.db) - 1))
            cd = float(f.window(f.context_delta, *iw).max())
            vals["energia_contexto"] = (float(np.clip(cd / 15.0, 0, 1)), f"ventana {iw[0]:.1f}–{iw[1]:.1f}s: +{cd:.1f} dB sobre los 30 s previos", iw, [])
            vc = float(self._vocal[curve.index(iw[0]): curve.index(iw[1]) + 1].max())
            spoken = bool(self._speech[p])
            vals["reaccion_vocal"] = (vc, f"ventana {iw[0]:.1f}–{iw[1]:.1f}s: pico +{ex.max():.1f} dB, planitud espectral {f.flatness[p]:.2f}, "
                                      + ("con habla transcrita" if spoken else "sin habla transcrita en el pico" if self._has_speech else "sin transcripción"),
                                      iw, [])
            lw = (c.anchor - 3.0, c.anchor + 5.0)
            lg = float(f.window(f.laugh, *lw).max())
            vals["risa_probable"] = ((lg if lg >= 0.4 else 0.0), (f"ráfagas regulares 3–8 Hz {lg:.2f}: posible risa, sin verificar" if lg >= 0.4
                                                                  else f"sin evidencia suficiente ({lg:.2f} < 0.40)"), lw, [])
            loud = float(np.clip(ex.max() / 10.0, 0, 1))
            const = float(f.window(f.constancy, c.anchor - 8, c.anchor + 8).mean())
            noisy = loud * float(np.clip((f.flatness[p] - 0.3) / 0.4, 0, 1))
            nonvoice = loud * 0.6 if (self._has_speech and not self._speech[max(0, p - 4): p + 5].any()) else 0.0
            parts = {"ruido plano": noisy, "volumen constante": loud * const, "fuerte sin habla": nonvoice}
            top = max(parts, key=parts.get)
            pen["musica_o_ruido"] = (parts[top], f"pico +{ex.max():.1f} dB; mayor causa: {top} ({parts[top]:.2f})", iw, [])
            per = float(f.window(f.periodic, c.anchor - 4, c.anchor + 4).mean())
            pen["audio_repetitivo"] = (loud * per, f"picos a intervalos regulares {per:.2f} × sonoridad {loud:.2f}", (c.anchor - 4, c.anchor + 4), [])
            quiet = float(f.window(f.quiet, c.start, c.end).mean())
            dead = float(np.clip((quiet - 0.35) / 0.4, 0, 1)) if nar.speech_cover < 0.25 else 0.0
            pen["aire_muerto"] = (dead, f"{quiet * 100:.0f}% del clip por debajo del nivel habitual; habla en {nar.speech_cover * 100:.0f}%",
                                  (c.start, c.end), [])
        if nar.available:
            ev = nar.evidence
            vals["arco_narrativo"] = (nar.arc, f"planteamiento {nar.setup:.1f} → desenlace {nar.payoff:.2f}", (c.start, c.end),
                                      [f"+ {e}" for k in ("planteamiento", "desenlace") for e in ev.get(k, [])])
            pen["transcripcion_dudosa"] = (nar.flagged, "; ".join(ev["transcripcion"]) or "sin banderas", (c.start, c.end), [])
            pen["frase_incompleta"] = (min(1.0, nar.incomplete), "; ".join(ev["cortes"]) or "bordes en frases completas", (c.start, c.end), [])
        if has_visual and (vs := self.visual.per_candidate.get(id(c))) is not None and vs.frames >= 4:
            corroborated = vals["intensidad_audio"][0] >= 0.35 or vals["reaccion_verbal"][0] > 0 or vals.get("reaccion_vocal", (0,))[0] >= 0.4
            v = 0.0 if (vs.steady_motion or not corroborated) else 0.5 * vs.spike + 0.5 * vs.novelty
            why = ("; no suma: movimiento constante" if vs.steady_motion else "; no suma sin corroboración de audio o texto" if not corroborated else "")
            vals["visual"] = (v, vs.note + why, None, [])
        # pesos efectivos: solo las señales disponibles, renormalizadas para sumar 1
        total = sum(POSITIVE[n] for n in vals)
        comps = [ScoreComponent(n, vals[n][0], POSITIVE[n] / total, vals[n][1], vals[n][2], vals[n][3]) for n in POSITIVE if n in vals]
        comps += [ScoreComponent(n, pen[n][0], -PENALTY[n], pen[n][1], pen[n][2], pen[n][3]) for n in PENALTY if n in pen]
        c.components = comps
        c.score = float(max(0.0, sum(x.contribution for x in comps)))

    def eligible(self, c: Candidate, ctx: AnalysisContext) -> bool:
        if self._info.get("effective", "").startswith("clásico"):
            return c.score >= ctx.settings.min_score and has_trigger(c, ctx)
        by = {x.name: x.value for x in c.components}
        return c.score >= ctx.settings.min_score and bool(has_trigger(c, ctx) or by.get("reaccion_vocal", 0) >= 0.5 or by.get("risa_probable", 0) >= 0.5)

    # --- selección: sin duplicados prácticos --------------------------------------------------------------
    def select(self, candidates: list[Candidate], ctx: AnalysisContext) -> list[Candidate]:
        cfg, chosen = ctx.settings, []
        self.duplicates_avoided = 0
        if self._info.get("effective", "").startswith("clásico"):
            from streamclipper.selection.selector import select_candidates
            return select_candidates(candidates, ctx)
        for c in sorted(candidates, key=lambda c: c.score, reverse=True):
            if len(chosen) >= cfg.max_clips:
                break
            by = {x.name: x.value for x in c.components}
            if c.score < cfg.min_score or not (has_trigger(c, ctx) or by.get("reaccion_vocal", 0) >= 0.5 or by.get("risa_probable", 0) >= 0.5):
                continue
            if any(overlap_ratio(c, o) > cfg.max_overlap or text_similarity(c, o) > 0.6 or abs(c.anchor - o.anchor) < MIN_ANCHOR_GAP for o in chosen):
                self.duplicates_avoided += 1
                continue
            chosen.append(c)
        return sorted(chosen, key=lambda c: c.start)

    def info(self) -> dict:
        return {**self._info, "duplicates_avoided": self.duplicates_avoided}

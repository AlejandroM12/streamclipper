"""Puntuación heurística y explicable. NO es una probabilidad de viralización:
es un ranking relativo dentro de este video, útil solo para ordenar y filtrar candidatos."""
import numpy as np

from streamclipper.analysis.text import quality
from streamclipper.models import Candidate, ScoreComponent
from streamclipper.selection.base import AnalysisContext

WEIGHTS = {
    "intensidad_audio": 0.30,
    "dinamica_audio": 0.10,
    "reaccion_verbal": 0.25,
    "frases": 0.10,
    "contexto": 0.10,
    "duracion": 0.05,
    "calidad_transcripcion": 0.10,
}


class HeuristicScorer:
    name = "heuristic"

    def score(self, candidates: list[Candidate], ctx: AnalysisContext) -> list[Candidate]:
        for c in candidates:
            self._score_one(c, ctx)
        return candidates

    def _score_one(self, c: Candidate, ctx: AnalysisContext) -> None:
        cfg, curve = ctx.settings, ctx.curve
        segs = [(s, sig) for s, sig in zip(ctx.transcript.segments, ctx.signals)
                if s.end > c.start and s.start < c.end]
        c.text = " ".join(s.text for s, _ in segs)

        # La intensidad se mide cerca del ancla, no en toda la ventana: así un estallido que cae
        # justo en el borde del clip (otro momento) no infla la puntuación de este.
        near = curve.window(curve.excess, c.anchor - 2.5, c.anchor + 2.5)
        peak_db = float(near.max())
        sustained = float(np.sort(near)[-max(1, len(near) // 4):].mean())  # favorece momentos sostenidos
        intensity = float(np.clip((0.7 * peak_db + 0.3 * sustained) / 15.0, 0, 1))
        rise = curve.window(curve.rise, c.anchor - 2, c.anchor + 2)
        dynamics = float(np.clip(rise.max() / 15.0, 0, 1))

        # reacción verbal: solo frases cercanas al ancla (no cualquiera dentro de la ventana)
        near_segs = [(s, sig) for s, sig in segs if s.end >= c.anchor - 4 and s.start <= c.anchor + 4]
        reaction = max((sig.reaction for _, sig in near_segs), default=0.0)
        hits = sorted({h for _, sig in segs for h in sig.hits})
        interest = max((sig.interest for _, sig in segs), default=0.0)

        pre, post = c.anchor - c.start, c.end - c.anchor
        context = float(np.clip(min(pre, post) / max(1.0, min(cfg.pre_context, cfg.post_context)), 0, 1))

        d = c.duration
        duration_fit = 1.0 if 20 <= d <= 45 else float(np.clip(1 - (20 - d) / 20 if d < 20 else 1 - (d - 45) / 30, 0.3, 1))

        q, n_words = quality([s for s, _ in segs])
        if n_words == 0:
            q_val, q_note = 0.0, "sin habla transcrita"
        else:
            q_val, q_note = float(np.clip((q - 0.4) / 0.5, 0, 1)), f"{n_words} palabras, prob. media {q:.2f}"

        c.components = [
            ScoreComponent("intensidad_audio", intensity, WEIGHTS["intensidad_audio"], f"pico +{peak_db:.1f} dB sobre el nivel habitual"),
            ScoreComponent("dinamica_audio", dynamics, WEIGHTS["dinamica_audio"], f"subida {rise.max():.1f} dB"),
            ScoreComponent("reaccion_verbal", reaction, WEIGHTS["reaccion_verbal"], ", ".join(hits[:4]) or "sin reacción detectada"),
            ScoreComponent("frases", interest, WEIGHTS["frases"], "exclamaciones/preguntas/repeticiones"),
            ScoreComponent("contexto", context, WEIGHTS["contexto"], f"{pre:.0f}s antes / {post:.0f}s después"),
            ScoreComponent("duracion", duration_fit, WEIGHTS["duracion"], f"{d:.0f}s"),
            ScoreComponent("calidad_transcripcion", q_val, WEIGHTS["calidad_transcripcion"], q_note),
        ]
        c.score = float(sum(x.contribution for x in c.components))

"""Puntuación heurística y explicable. NO es una probabilidad de viralización:
es un ranking relativo dentro de este video, útil solo para ordenar y filtrar candidatos.

Cada componente registra la ventana temporal exacta con la que se calculó y la evidencia usada ('+') o
descartada ('-'), de modo que la explicación coincide con el valor."""
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

INTENSITY_WINDOW = 2.5   # s a cada lado del ancla
DYNAMICS_WINDOW = 2.0
REACTION_WINDOW = 4.0


def _win(a: float, b: float) -> str:
    return f"{a:.1f}–{b:.1f}s"


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
        clip_win = (c.start, c.end)

        # La intensidad se mide cerca del ancla, no en toda la ventana: así un estallido que cae
        # justo en el borde del clip (otro momento) no infla la puntuación de este.
        iw = (c.anchor - INTENSITY_WINDOW, c.anchor + INTENSITY_WINDOW)
        near = curve.window(curve.excess, *iw)
        peak_db = float(near.max())
        sustained = float(np.sort(near)[-max(1, len(near) // 4):].mean())  # favorece momentos sostenidos
        intensity = float(np.clip((0.7 * peak_db + 0.3 * sustained) / 15.0, 0, 1))
        dw = (c.anchor - DYNAMICS_WINDOW, c.anchor + DYNAMICS_WINDOW)
        rise = curve.window(curve.rise, *dw)
        dynamics = float(np.clip(rise.max() / 15.0, 0, 1))

        # reacción verbal: solo frases cercanas al ancla; el valor es el de la frase con mayor reacción
        rw = (c.anchor - REACTION_WINDOW, c.anchor + REACTION_WINDOW)
        near_segs = [(s, sig) for s, sig in segs if s.end >= rw[0] and s.start <= rw[1]]
        reaction, r_evidence = 0.0, []
        if near_segs:
            best_i = max(range(len(near_segs)), key=lambda i: near_segs[i][1].reaction)
            reaction = near_segs[best_i][1].reaction
            for i, (s, sig) in enumerate(near_segs):
                for h in sig.details:
                    if not h.accepted:
                        r_evidence.append(h.label())
                    elif i == best_i and reaction > 0:
                        r_evidence.append(h.label())
                    else:
                        r_evidence.append(f"~ «{h.term}» ({h.reason}) en otra frase de la ventana: no suma, "
                                          "se toma la frase de mayor reacción")
        used = [e for e in r_evidence if e.startswith("+")]
        dropped = [e for e in r_evidence if e.startswith("-")]
        r_note = (f"ventana {_win(*rw)} (±{REACTION_WINDOW:.0f} s del ancla): "
                  + (", ".join(e[2:].split(" (")[0] for e in used) if used else "sin reacción aceptada")
                  + (f"; descartadas: {len(dropped)}" if dropped else ""))

        # frases: la frase con mayor interés dentro de TODO el clip (exclamaciones, preguntas, repeticiones confiables)
        interest, f_evidence = 0.0, []
        f_stats = "sin exclamaciones, preguntas ni repeticiones confiables"
        if segs:
            best = max(segs, key=lambda p: p[1].interest)
            interest = best[1].interest
            if interest > 0:
                f_stats = (f"{best[1].exclamations} «!», {best[1].questions} «?», "
                           f"{best[1].repetitions_counted} repeticiones confiables")
                f_evidence.append(f"+ frase «{best[0].text[:40]}»: {f_stats}")
            for s, sig in segs:
                f_evidence += [f"- repetición {d}" for d in sig.repetitions_discarded]
        n_disc = sum(1 for e in f_evidence if e.startswith("-"))
        f_note = (f"ventana del clip {_win(*clip_win)}: {f_stats}"
                  + (f"; repeticiones descartadas: {n_disc}" if n_disc else ""))

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
            ScoreComponent("intensidad_audio", intensity, WEIGHTS["intensidad_audio"],
                           f"ventana {_win(*iw)}: pico +{peak_db:.1f} dB sobre el nivel habitual", iw),
            ScoreComponent("dinamica_audio", dynamics, WEIGHTS["dinamica_audio"],
                           f"ventana {_win(*dw)}: subida {rise.max():.1f} dB", dw),
            ScoreComponent("reaccion_verbal", reaction, WEIGHTS["reaccion_verbal"], r_note, rw, r_evidence),
            ScoreComponent("frases", interest, WEIGHTS["frases"], f_note, clip_win, f_evidence),
            ScoreComponent("contexto", context, WEIGHTS["contexto"], f"{pre:.0f}s antes / {post:.0f}s después",
                           clip_win),
            ScoreComponent("duracion", duration_fit, WEIGHTS["duracion"], f"{d:.0f}s", clip_win),
            ScoreComponent("calidad_transcripcion", q_val, WEIGHTS["calidad_transcripcion"],
                           f"ventana del clip {_win(*clip_win)}: {q_note}", clip_win),
        ]
        c.score = float(sum(x.contribution for x in c.components))

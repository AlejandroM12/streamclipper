import re

from streamclipper.models import Candidate
from streamclipper.selection.base import AnalysisContext, CandidateScorer
from streamclipper.selection.candidates import generate_candidates
from streamclipper.selection.heuristic import HeuristicScorer


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    """Fracción del clip más corto que queda cubierta por el otro."""
    inter = min(a.end, b.end) - max(a.start, b.start)
    return max(0.0, inter) / min(a.duration, b.duration)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w{4,}", text.lower()))


def text_similarity(a: Candidate, b: Candidate) -> float:
    ta, tb = _tokens(a.text), _tokens(b.text)
    if len(ta) < 5 or len(tb) < 5:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def has_trigger(c: Candidate, ctx: AnalysisContext) -> bool:
    """Un candidato necesita una señal real (audio intenso o reacción verbal)."""
    by_name = {x.name: x for x in c.components}
    return (by_name["intensidad_audio"].value >= 0.35 or by_name["reaccion_verbal"].value >= ctx.settings.text_trigger)


def is_eligible(c: Candidate, ctx: AnalysisContext, scorer: CandidateScorer | None = None) -> bool:
    """Cumple los criterios normales de calidad (umbral y señal propia), sin mirar solapes con otros candidatos.
    Es lo que separa los «recomendados» de los «exploratorios» en el modo automático."""
    fn = getattr(scorer, "eligible", None)
    if fn is not None:
        return bool(fn(c, ctx))
    return c.score >= ctx.settings.min_score and has_trigger(c, ctx)


def select_candidates(candidates: list[Candidate], ctx: AnalysisContext) -> list[Candidate]:
    """Selección voraz por puntuación, sin solapes excesivos ni textos repetidos.

    Si ningún candidato supera min_score, devuelve una lista vacía: no se fabrican clips.
    """
    cfg = ctx.settings
    chosen: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        if len(chosen) >= cfg.max_clips:
            break
        if c.score < cfg.min_score or not has_trigger(c, ctx):
            continue
        if any(overlap_ratio(c, o) > cfg.max_overlap or text_similarity(c, o) > 0.6 for o in chosen):
            continue
        chosen.append(c)
    return sorted(chosen, key=lambda c: c.start)


TIE_MARGIN = 0.05  # diferencia de puntuación por debajo de la cual el corte entre elegidos y no elegidos es frágil


def selection_report(candidates: list[Candidate], chosen: list[Candidate], ctx: AnalysisContext) -> dict:
    """Mide cuán estable es la selección: margen entre el último elegido y el siguiente candidato elegible.

    «Elegible» = supera min_score, tiene señal propia y no se solapa ni repite lo ya elegido; los vecinos del
    mismo momento no cuentan como competidores. Un margen chico significa que un cambio mínimo de
    transcripción (≈0.02) puede cambiar qué clips salen: se informa, no se corrige.
    """
    cfg = ctx.settings
    out: dict = {"tie_margin": TIE_MARGIN, "margin_to_next": None, "next_rank": None, "top_spread": None,
                 "fragile": False, "message": None}
    if not chosen:
        return out
    scores = [c.score for c in chosen]
    out["top_spread"] = round(max(scores) - min(scores), 3)
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        if c in chosen or c.score < cfg.min_score or not has_trigger(c, ctx):
            continue
        if any(overlap_ratio(c, o) > cfg.max_overlap or text_similarity(c, o) > 0.6 for o in chosen):
            continue
        out["margin_to_next"] = round(min(scores) - c.score, 3)
        out["next_rank"] = c.rank
        break
    gaps = [out["margin_to_next"]] if out["margin_to_next"] is not None else []
    if len(chosen) > 1:
        gaps.append(out["top_spread"])
    if any(g < TIE_MARGIN for g in gaps):
        out["fragile"] = True
        parts = []
        if out["margin_to_next"] is not None and out["margin_to_next"] < TIE_MARGIN:
            parts.append(f"el siguiente candidato elegible (#{out['next_rank']}) quedó a solo "
                         f"{out['margin_to_next']:.3f} del último elegido")
        if len(chosen) > 1 and out["top_spread"] < TIE_MARGIN:
            parts.append(f"los {len(chosen)} elegidos se diferencian por solo {out['top_spread']:.3f}")
        out["message"] = ("Selección frágil: " + "; ".join(parts) + f" (margen < {TIE_MARGIN}). "
                          "Pequeños cambios de transcripción pueden alterar los clips elegidos; "
                          "conviene revisar los candidatos cercanos (export --alternatives).")
    return out


def find_clips(ctx: AnalysisContext, scorer: CandidateScorer | None = None) -> tuple[list[Candidate], list[Candidate]]:
    """Devuelve (todos los candidatos puntuados, seleccionados)."""
    scorer = scorer or HeuristicScorer()
    generate = getattr(scorer, "generate", generate_candidates)      # Smart Highlights agrega anclas y su propia selección
    select = getattr(scorer, "select", select_candidates)
    candidates = scorer.score(generate(ctx), ctx)
    return sorted(candidates, key=lambda c: c.score, reverse=True), select(candidates, ctx)

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


def find_clips(ctx: AnalysisContext, scorer: CandidateScorer | None = None) -> tuple[list[Candidate], list[Candidate]]:
    """Devuelve (todos los candidatos puntuados, seleccionados)."""
    scorer = scorer or HeuristicScorer()
    candidates = scorer.score(generate_candidates(ctx), ctx)
    return sorted(candidates, key=lambda c: c.score, reverse=True), select_candidates(candidates, ctx)

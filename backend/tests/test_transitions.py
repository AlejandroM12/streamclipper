"""Transiciones visuales (1.5, etapa 3): detección ligera y límites del contexto respaldados por evidencia."""
import numpy as np

from streamclipper.analysis.transitions import find_transitions
from streamclipper.core.config import Settings
from streamclipper.models import Candidate
from streamclipper.selection.context_clip import build_window
from tests.helpers import make_ctx, sentence as S


def frames(n, changes, noise=0.002, seed=1):
    """n fotogramas 90x160 casi estáticos que cambian por completo en los índices de `changes`."""
    rng = np.random.default_rng(seed)
    out, base = [], rng.integers(0, 255, (90, 160)).astype(np.float32)
    for i in range(n):
        if i in changes:
            base = rng.integers(0, 255, (90, 160)).astype(np.float32)
        out.append(np.clip(base + rng.normal(0, noise * 255, base.shape), 0, 255))
    return np.array(out, np.uint8)


def test_static_panels_with_a_few_changes_give_clean_transitions():
    tr = find_transitions(frames(180, {55, 66, 160}), t0=0.0, fps=2)
    assert not tr.noisy and tr.median < 0.01
    assert [round(t, 1) for t in tr.times] == [27.5, 33.0, 80.0]               # el instante en que cambia la imagen (índice / fps)
    assert all(s >= 0.03 for s in tr.strengths)


def test_consecutive_changes_of_an_animation_count_once():
    tr = find_transitions(frames(100, {40, 41, 42, 43}), t0=10.0, fps=2)
    assert len(tr.times) == 1 and 29.0 < tr.times[0] < 32.0                       # el eje del video: t0 + índice / fps


def test_constant_motion_is_discarded_instead_of_inventing_boundaries():
    tr = find_transitions(frames(120, set(range(120))), t0=0.0, fps=2)       # la imagen cambia casi en cada muestra (gameplay)
    assert tr.noisy and tr.times == []


def test_region_mask_ignores_motion_outside_the_panel():
    f = frames(120, set())
    rng = np.random.default_rng(3)
    f[:, :45, :80] = rng.integers(0, 255, (120, 45, 80))                           # una «cámara» que cambia todo el tiempo en la esquina
    whole = find_transitions(f, fps=2)
    panel = find_transitions(f, fps=2, region=(0.55, 0.55, 0.4, 0.4))
    assert whole.noisy and not panel.noisy and panel.times == []


def test_too_few_frames_gives_an_empty_signal():
    assert find_transitions(frames(3, set())).times == []


# --- el contexto no cruza una transición hacia otra situación -------------------------------------------------------------------
def chat_scene():
    """Conversación continua (sin pausas ≥ 2,5 s) de 40 a 130 s: sin evidencia visual el contexto llega hasta el tope."""
    sents = [S(f"frase continua número {i} de la charla", 40 + i * 3.3, 3.0) for i in range(28)]
    sents[14] = S("¡No puede ser! ¡increíble!", 40 + 14 * 3.3, 3.0)               # el evento ≈ 86–89 s
    return sents


def build(transitions):
    cfg = Settings(clip_preset="tiktok")
    ctx = make_ctx(200, [(87, 91)], chat_scene(), cfg)
    c = Candidate(start=80, end=98, anchor=88, triggers=["audio"])
    return build_window(c, ctx.transcript.segments, ctx.curve, 200, cfg, "tiktok", event=(86.2, 90.2), transitions=transitions)


def test_a_visual_transition_before_the_event_stops_the_context_there():
    free = build(None)
    cut = build([72.0])                                                          # otra pregunta/situación empezó a los 72 s
    assert free.start < 72.0 - 3                                                  # sin la señal, el contexto cruzaba esa frontera
    assert cut.start >= 71.7 and cut.start < 86.2                                  # con ella: empieza donde empieza la nueva situación
    x = cut.context
    assert x["start_kind"] == "transicion" and x["boundary_evidence"]["start"] == "respaldado" and x["signals"]["barrier_start"] == 72.0
    assert "cambios visuales" in x["method"] and "otra situación" in x["start_reason"]
    assert cut.end > 90.2 and cut.start < 86.2                                     # el momento principal sigue completo dentro del clip


def test_a_transition_after_the_event_limits_the_end():
    free, cut = build(None), build([100.0])
    assert free.end > 101 and cut.end <= 100.5
    assert cut.context["end_kind"] == "transicion" and cut.context["signals"]["barrier_end"] == 100.0


def test_transitions_inside_or_next_to_the_event_never_cut_it():
    r = build([83.0, 87.5, 89.0, 92.5])                                          # dentro del evento o a menos de 4 s: parte de la misma situación, no es barrera
    assert r.start <= 86.2 and r.end >= 90.2 and r.context["signals"]["barrier_start"] is None and r.context["signals"]["barrier_end"] is None


def test_without_evidence_boundaries_are_flagged_as_estimated():
    x = build(None).context
    assert x["start_kind"] == "limite" and x["boundary_evidence"] == {"start": "estimado", "end": "estimado"} and x["signals"]["visual_transitions"] == []

"""Construcción narrativa del clip: de un momento detectado a un clip que se entiende solo.

Dos etapas separadas (ver docs/context-clips.md):
  A. DETECCIÓN: el ranking existente (Clásico o Smart) elige el momento. Aquí no se toca.
  B. CONSTRUCCIÓN: se decide qué fragmento necesita el espectador. El pico no es el clip: `refine_window` solo se usa para hallar el
     EVENTO (la ráfaga de reacción), y alrededor se construye la situación.

Heurística sencilla y medible (sin modelos nuevos; no «entiende» la escena, solo sigue la continuidad de lo que se dice y se oye):
  · INICIO: desde el evento hacia atrás se encadenan las frases separadas por pausas < CHAIN_GAP (2,5 s) = la misma conversación o
    jugada. Una pausa ≥ CHAIN_GAP es el comienzo natural de la situación. Si la cadena llega al tope de contexto sin pausa, se elige
    el mejor inicio de frase (pausa previa larga, no dependiente de la frase anterior, cerca de lo que el preset pide).
  · FINAL: igual hacia adelante, y además se espera a que el audio vuelva a la calma (la reacción terminó / se vio el resultado).
  · Nunca se corta a mitad de frase. No se agrega tiempo sin actividad (silencio muerto) solo para llegar a una duración.
  · Sin evidencia de un límite (sin habla, cadena cortada por un tope) se aplica una expansión conservadora y se deja CONSTANCIA de que
    es una estimación y de su nivel de confianza.
"""
from dataclasses import dataclass

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.models import Candidate, Segment
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.candidates import _TOL, _inside
from streamclipper.selection.refine import Refined, _dependent, _fades, _incomplete, _pad_after, _pad_before, _valley, refine_window

# mínimo preferido / objetivo / máximo preferido / máximo excepcional (s). El preset ORIENTA los límites, no fuerza una duración.
PRESETS = {
    "cortos": {"label": "Cortos (≈15–30 s)", "min": 15.0, "target": 24.0, "max": 30.0, "exc": 30.0},
    "tiktok": {"label": "TikTok / Reels (≈30–60 s)", "min": 30.0, "target": 42.0, "max": 60.0, "exc": 90.0},
    "extendido": {"label": "Extendidos (≈45–90 s)", "min": 45.0, "target": 62.0, "max": 90.0, "exc": 90.0},
}
DEFAULT_PRESET = "tiktok"
CHAIN_GAP = 2.5                   # pausa que separa dos situaciones distintas
PRE_MAX, POST_MAX = 30.0, 30.0    # contexto máximo que se examina antes y después del evento
MIN_PRE, MIN_POST = 3.0, 3.0      # lo mínimo de situación que se conserva a cada lado cuando hay que recortar
BACKUP_PRE = {"cortos": 4.0, "tiktok": 8.0, "extendido": 12.0}     # expansión de respaldo (estimación) sin límite narrativo
BACKUP_POST = {"cortos": 3.0, "tiktok": 5.0, "extendido": 7.0}
RESULT_TAIL = {"cortos": 2.0, "tiktok": 3.0, "extendido": 4.0}      # tras la calma, para mostrar el resultado (solo si hay actividad)
EDGE = 2.0                                                          # a menos de esto del borde del material se considera el borde
CALM_DB, CALM_HOLD = 1.0, 1.5     # el audio «volvió a la calma»: bajo +1 dB sobre lo habitual durante 1,5 s
QUIET_DB = -8.0                   # silencio relativo (misma referencia que Smart Highlights)
EXC_MIN_WORDS_PER_S = 1.2         # una situación de más de 60 s se justifica solo si es una conversación continua
TR_MARGIN = 4.0                   # una transición visual limita el contexto solo si queda a más de esto del evento: más cerca suele ser parte
                                  # de la misma situación (la pantalla cambia al reaccionar o al mostrarse el resultado) y no se la trata como límite
NATURAL = ("pausa", "material", "transicion")   # tipos de límite fuertes (respaldados por evidencia)
BACKED = NATURAL + ("calma",)     # límites respaldados por evidencia; el resto (tope de contexto, estimación, sin habla) son estimados


@dataclass
class _Opt:
    t: float            # instante del límite
    kind: str           # pausa | material | limite | calma | estimacion
    reason: str
    gap: float          # pausa junto al límite (s)
    bad: bool = False   # inicio dependiente de la frase anterior / final incompleto


def _sorted(segs: list[Segment]) -> list[Segment]:
    return sorted(segs, key=lambda s: s.start)


def start_options(segs: list[Segment], e0: float, e1: float, dur: float) -> tuple[list[_Opt], float]:
    """Posibles inicios, del más cercano al evento hacia atrás, mientras la conversación sea continua. Devuelve (opciones, ref)."""
    first = next((s for s in segs if s.end > e0 + 0.05 and s.start < e1), None)
    ref = min(e0, first.start) if first is not None else e0
    prev0 = max((s for s in segs if s.end <= ref + _TOL and s.start < ref - 0.05), key=lambda s: s.end, default=None)
    gap0 = ref - prev0.end if prev0 is not None else float("inf")
    opts = [_Opt(ref, "pausa" if gap0 >= CHAIN_GAP else "limite", "", gap0)]
    cur = ref
    while True:
        p = max((s for s in segs if s.end <= cur + _TOL and s.start < cur - 0.05), key=lambda s: s.end, default=None)
        if p is None:
            if ref <= EDGE:
                opts[-1].kind, opts[-1].reason = "material", "es el comienzo del material"
            else:
                opts[-1].kind, opts[-1].reason = "sinhabla", "sin habla previa: no hay evidencia de dónde empieza la situación"
            break
        gap = cur - p.end
        if gap >= CHAIN_GAP:
            break                                         # la situación empieza después de esta pausa
        if ref - p.start > PRE_MAX:
            opts[-1].kind = "limite"
            break
        prev = max((s for s in segs if s.end <= p.start + _TOL and s is not p), key=lambda s: s.end, default=None)
        opts.append(_Opt(p.start, "limite", "", p.start - prev.end if prev is not None else float("inf"), _dependent(p, prev)))
        cur = p.start
    for o in opts:
        if o.gap >= CHAIN_GAP and o.kind not in ("material", "sinhabla"):
            o.kind, o.reason = "pausa", f"pausa natural de {min(o.gap, 99):.1f} s antes: ahí empieza la situación"
        elif o.kind == "limite":
            o.reason = ("tope de contexto previo sin pausa natural: se inicia en una frase completa" if o.t < ref - 0.05
                        else "sin pausa natural previa dentro del tope: se inicia donde empieza la reacción")
    return opts, ref


def end_options(segs: list[Segment], e0: float, e1: float, dur: float) -> tuple[list[_Opt], float]:
    last = max((s for s in segs if s.start < e1 - 0.05 and s.end > e0 + 0.05), key=lambda s: s.end, default=None)
    ref = max(e1, last.end) if last is not None else e1
    nxt0 = min((s for s in segs if s.start >= ref - _TOL and s.end > ref + 0.05), key=lambda s: s.start, default=None)
    gap0 = nxt0.start - ref if nxt0 is not None else float("inf")
    opts = [_Opt(ref, "pausa" if gap0 >= CHAIN_GAP else "limite", "", gap0, bool(last and nxt0 and _incomplete(last, nxt0)))]
    cur = ref
    while True:
        n = min((s for s in segs if s.start >= cur - _TOL and s.end > cur + 0.05), key=lambda s: s.start, default=None)
        if n is None:
            if ref >= dur - EDGE:
                opts[-1].kind, opts[-1].reason = "material", "es el final del material"
            else:
                opts[-1].kind, opts[-1].reason = "sinhabla", "sin habla posterior: no hay evidencia de dónde termina la situación"
            break
        gap = n.start - cur
        if gap >= CHAIN_GAP:
            break
        if n.end - e1 > POST_MAX:
            opts[-1].kind = "limite"
            break
        after = min((s for s in segs if s.start >= n.end - _TOL and s is not n and s.end > n.end + 0.05), key=lambda s: s.start, default=None)
        opts.append(_Opt(n.end, "limite", "", after.start - n.end if after is not None else float("inf"),
                         bool(after is not None and _incomplete(n, after) and after.start - n.end < 1.0)))
        cur = n.end
    for o in opts:
        if o.gap >= CHAIN_GAP and o.kind not in ("material", "sinhabla"):
            o.kind, o.reason = "pausa", f"pausa natural de {min(o.gap, 99):.1f} s después: la situación terminó"
        elif o.kind == "limite":
            o.reason = "tope de contexto posterior sin pausa natural: se termina en una frase completa"
    return opts, ref


def _score(opt: _Opt, length: float, target: float) -> float:
    pause = 1.0 if opt.gap == float("inf") else min(opt.gap / CHAIN_GAP, 1.0)
    close = max(0.0, 1.0 - abs(length - target) / max(target, 1.0))
    return pause + 0.8 * close + (0.5 if opt.kind in NATURAL else 0.0) - (0.3 if opt.bad else 0.0)


def calm_after(curve, t0: float, limit: float) -> float | None:
    """Primer instante desde t0 en que el audio vuelve a la calma durante CALM_HOLD s (la reacción terminó). None si no ocurre."""
    if curve is None:
        return None
    x, hop = curve.excess, curve.hop
    hold = max(1, int(CALM_HOLD / hop))
    for i in range(curve.index(t0), min(len(x) - hold, curve.index(limit)) + 1):
        if bool((x[i:i + hold] < CALM_DB).all()):
            return i * hop
    return None


def _active_fraction(curve, a: float, b: float) -> float:
    if curve is None or b <= a:
        return 1.0
    return float((curve.window(curve.excess, a, b) > QUIET_DB).mean())


def _adjust_end(e: _Opt, curve, e1: float, preset: str, dur: float) -> _Opt:
    """Final por audio: si el final elegido no tiene evidencia (sin habla) o la reacción sigue sonando después de la última frase, se
    espera a la calma del audio y se agrega un breve margen para ver el resultado (solo si hay actividad, nunca silencio muerto)."""
    if curve is None:
        return e
    start_at = e.t if e.kind == "sinhabla" else e1
    calm = calm_after(curve, start_at, min(dur, start_at + POST_MAX))
    if calm is None or not (e.kind == "sinhabla" or (e.kind in NATURAL and 0.5 < calm + 0.5 - e.t <= 6.0)):
        return e
    tail = RESULT_TAIL[preset] if _active_fraction(curve, calm, calm + RESULT_TAIL[preset]) >= 0.5 else 0.0
    return _Opt(min(dur, max(e.t, calm + 0.5 + tail)), "calma",
                "el audio volvió a la calma: terminó la reacción" + (f" (+{tail:g} s para ver el resultado)" if tail else ""), e.gap)


def _pick(opts: list[_Opt], ref: float, sign: int, cap: float, target: float) -> _Opt:
    pool = [o for o in opts if sign * (ref - o.t) <= cap + 1e-6]
    pool = pool or opts[:1]
    return max(pool, key=lambda o: (_score(o, sign * (ref - o.t), target), sign * (ref - o.t)))


def _select(s_opts, s_ref, e_opts, e_ref, e0, e1, budget, pre_t, post_t) -> tuple[_Opt, _Opt]:
    """Elige inicio y final con un presupuesto total de contexto `budget` (s): si cabe todo lo disponible, se toma; si no, se reparte."""
    avail_pre = max(s_ref - o.t for o in s_opts)
    avail_post = max(o.t - e_ref for o in e_opts)
    if avail_pre + avail_post <= budget:
        pre_cap, post_cap = avail_pre, avail_post
    else:
        pre_cap = min(avail_pre, max(0.55 * budget, budget - avail_post))
        post_cap = budget - pre_cap
    return _pick(s_opts, s_ref, 1, pre_cap, pre_t), _pick(e_opts, e_ref, -1, post_cap, post_t)


def _dense(segs: list[Segment], a: float, b: float) -> float:
    words = sum(1 for s in segs for w in s.words if w.end > a and w.start < b)
    return words / max(b - a, 1.0)


def build_window(c: Candidate, segs: list[Segment], curve, dur: float, cfg, preset: str, event: tuple[float, float] | None = None,
                 core: Refined | None = None, transitions: list[float] | None = None) -> Refined:
    """Ventana contextual de un candidato. `event` (opcional) fija el evento; si no, se halla con `refine_window` (necesita `core`).
    `transitions` (opcional): instantes de cambios visuales significativos (analysis/transitions.py). Una transición antes/después del
    evento marca donde empieza/termina OTRA situación: el contexto no la cruza."""
    P = PRESETS[preset]
    segs = _sorted(segs)
    if event is None:
        e0, e1 = core.start, core.end
    else:
        e0, e1 = event
    L = e1 - e0
    remaining = max(0.0, P["target"] - L)
    pre_t, post_t = 0.55 * remaining, 0.45 * remaining
    s_opts, s_ref = start_options(segs, e0, e1, dur)
    e_opts, e_ref = end_options(segs, e0, e1, dur)
    sig: dict = {"visual_transitions": [round(t, 2) for t in (transitions or [])], "barrier_start": None, "barrier_end": None}
    end_barrier = None
    if transitions:
        tr_pre = max((t for t in transitions if e0 - PRE_MAX <= t <= e0 - TR_MARGIN), default=None)
        tr_post = min((t for t in transitions if e1 + TR_MARGIN <= t <= e1 + POST_MAX), default=None)
        keep = [o for o in s_opts if tr_pre is not None and o.t >= tr_pre - 0.3]
        if keep and len(keep) < len(s_opts):
            keep[-1].kind, keep[-1].reason = "transicion", f"cambio visual significativo (≈{tr_pre:.0f} s del video): lo anterior parece otra situación"
            s_opts, sig["barrier_start"] = keep, round(tr_pre, 2)
        keep = [o for o in e_opts if tr_post is not None and o.t <= tr_post + 0.3]
        if keep and len(keep) < len(e_opts):
            keep[-1].kind, keep[-1].reason = "transicion", f"cambio visual significativo (≈{tr_post:.0f} s del video): lo siguiente parece otra situación"
            e_opts, sig["barrier_end"], end_barrier = keep, round(tr_post, 2), keep[-1]
    budget_max = max(P["max"] - L, 0.0)
    sel = _select(s_opts, s_ref, e_opts, e_ref, e0, e1, budget_max, pre_t, post_t)
    justified = False
    if P["exc"] > P["max"]:
        s2, e2 = _select(s_opts, s_ref, e_opts, e_ref, e0, e1, max(P["exc"] - L, 0.0), pre_t, post_t)
        total2 = e2.t - s2.t
        if (total2 > P["max"] and s2.kind in NATURAL and e2.kind in NATURAL
                and _dense(segs, s2.t, e2.t) >= EXC_MIN_WORDS_PER_S):
            sel, justified = (s2, e2), True
    s, e = sel
    e = _adjust_end(e, curve, e1, preset, dur)       # el audio decide cuándo terminó la reacción y se vio el resultado
    if end_barrier is not None and e.t > end_barrier.t:
        e = end_barrier                              # ...pero sin cruzar un cambio visual hacia otra situación
    notes: list[str] = []
    s_t, e_t = s.t, e.t
    s_kind, e_kind, s_reason, e_reason = s.kind, e.kind, s.reason, e.reason

    # ---- respaldo conservador: sin evidencia de límite y clip demasiado corto -------------------------------
    total = e_t - s_t
    if total < P["min"] and (s_kind == "sinhabla" or e_kind == "sinhabla"):
        deficit = P["min"] - total
        can_pre = s_kind == "sinhabla" and s_t > 0          # solo donde NO hay evidencia (habla ni calma de audio)
        can_post = e_kind == "sinhabla" and e_t < dur
        add_pre = min(BACKUP_PRE[preset], deficit * 0.6 if can_post else deficit, s_t) if can_pre else 0.0
        add_post = min(BACKUP_POST[preset], deficit - add_pre, dur - e_t) if can_post else 0.0
        if add_pre > 0.5 and _active_fraction(curve, s_t - add_pre, s_t) >= 0.5:    # no se agrega silencio muerto
            t = s_t - add_pre
            inside = _inside(t, segs)
            s_t = inside.start if inside is not None else t
            s_kind, s_reason = "estimacion", f"expansión de respaldo de {s_ref - s_t:.1f} s (no se halló un límite narrativo)"
        if add_post > 0.5 and _active_fraction(curve, e_t, e_t + add_post) >= 0.5:
            t = e_t + add_post
            inside = _inside(t, segs)
            e_t = inside.end if inside is not None else t
            e_kind, e_reason = "estimacion", f"expansión de respaldo de {e_t - e_ref:.1f} s (no se halló un límite narrativo)"
    if P["min"] > e_t - s_t and s_kind in NATURAL and e_kind in NATURAL:
        notes.append(f"situación completa de {total:.0f} s (< {P['min']:g} s): empieza y termina en pausas naturales, no se agrega relleno")

    # ---- márgenes y cortes limpios --------------------------------------------------------------------------------
    start0 = max(0.0, s_t)
    end0 = min(dur, e_t)
    start = _pad_before(start0, segs, 0.4)
    end = _pad_after(end0, segs, cfg.refine_tail, dur)
    if curve is not None:
        start = max(0.0, _valley(start, curve, segs, start - 0.35, start + 0.2))
        end = min(dur, _valley(end, curve, segs, end - 0.2, end + 0.35))
    start, end = max(0.0, start), min(dur, end)
    if justified:
        notes.append(f"duración excepcional ({end - start:.0f} s): conversación continua con pausas naturales en ambos extremos")
    bad = check_boundaries(_T(segs), start, end)["errors"]
    if bad or end <= start:                               # un corte dentro de una frase invalida la construcción: respaldo al evento
        fi, fo = _fades(e0, e1, segs, curve, cfg) if curve is not None else (0.0, 0.0)
        ctxinfo = _info(preset, c, (e0, e1), (e0, e1), (e0, e1), "estimacion", "estimacion", "la construcción cortaba una frase: se conserva el evento",
                        "la construcción cortaba una frase: se conserva el evento", notes + ["respaldo: evento sin contexto"], False, sig)
        return Refined(e0, e1, ["la construcción contextual cortaba una frase: se conserva el evento"], None, False, None, fi, fo, (P["min"], P["max"]), ctxinfo)
    fi, fo = _fades(start, end, segs, curve, cfg) if curve is not None else (0.0, 0.0)
    pre, post = e0 - start, end - e1
    notes.append(f"contexto: {max(pre, 0):.0f} s antes y {max(post, 0):.0f} s después del momento ({L:.0f} s)")
    info = _info(preset, c, (e0, e1), (start0, end0), (start, end), s_kind, e_kind, s_reason, e_reason, notes, justified, sig)
    return Refined(start, end, [f"inicio: {s_reason}", f"fin: {e_reason}"] + notes, None, True, None, fi, fo, (P["min"], P["max"]), info)


class _T:                         # adaptador mínimo para reutilizar check_boundaries con una lista de frases
    def __init__(self, segs: list[Segment]) -> None:
        self.segments = segs


def _info(preset, c, event, contextual, final, s_kind, e_kind, s_reason, e_reason, notes, exceptional, signals=None) -> dict:
    kinds = {s_kind, e_kind}
    conf = "baja" if kinds & {"estimacion", "sinhabla"} else "alta" if kinds <= set(NATURAL) else "media"
    visual = "transicion" in kinds
    return {
        "preset": preset, "original": [round(c.start, 2), round(c.end, 2)], "event": [round(event[0], 2), round(event[1], 2)],
        "contextual": [round(contextual[0], 2), round(contextual[1], 2)], "final": [round(final[0], 2), round(final[1], 2)],
        "start_kind": s_kind, "end_kind": e_kind, "start_reason": s_reason, "end_reason": e_reason, "confidence": conf,
        "estimated": bool(kinds & {"estimacion", "sinhabla"}), "exceptional": exceptional, "notes": notes,
        "method": "estimación conservadora" if kinds <= {"estimacion", "sinhabla"} else "transcripción + audio" + (" + cambios visuales" if visual else ""),
        "boundary_evidence": {"start": "respaldado" if s_kind in BACKED else "estimado", "end": "respaldado" if e_kind in BACKED else "estimado"},
        "signals": signals or {"visual_transitions": [], "barrier_start": None, "barrier_end": None},
    }


def plan_windows(chosen: list[Candidate], ctx: AnalysisContext) -> list[Refined]:
    """Ventanas finales de los clips elegidos. El ranking, los candidatos y las puntuaciones no cambian."""
    cfg = ctx.settings
    out = []
    for c in chosen:
        core = refine_window(c, ctx)                   # el EVENTO: la ráfaga de reacción
        out.append(build_window(c, ctx.transcript.segments, ctx.curve, ctx.duration, cfg, cfg.clip_preset, core=core))
    return out


def is_context_preset(name: str) -> bool:
    return name in PRESETS

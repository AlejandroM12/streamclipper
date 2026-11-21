"""Ajuste de los límites de un clip YA elegido. Es un paso separado de la selección del momento:

- No toca candidatos, puntuaciones ni ranking (se aplica después de `find_clips`).
- Parte de la ventana del candidato y devuelve otra [inicio, fin] con las razones de cada cambio. La ventana del
  candidato (min_clip/max_clip) es la que se puntúa; la duración del clip FINAL la fijan refine_min_clip/refine_max_clip.

Reglas (todas explicables y configurables, ver Settings.refine_*):
 FIN    1. La reacción termina cuando el volumen cae `refine_reaction_drop_db` bajo su pico y se sostiene
           `refine_calm_hold` s. Si no hay una caída clara, se conserva el fin original.
        2. El clip termina al final de la frase que contiene ese instante (nunca a mitad de frase), con un
           margen de `refine_tail` s, y conserva al menos `refine_min_post` s después del momento principal.
        3. Si esa frase quedó incompleta (termina en artículo, preposición o conjunción, o continúa sin pausa en
           la frase siguiente), se completa con la siguiente (hasta `refine_max_extend` s más allá del original).
 INICIO 4. Empieza en la primera frase de la reacción (que puede arrancar antes de su pico) o, si es posterior, en la
           frase que contiene el momento, con `refine_lead` s de margen; si esa frase depende de la
           anterior (empieza con «y», «pero», «porque»… o es muy corta) se incluye también la anterior (contexto previo
           solo cuando hace falta).
 AMBOS  5. Los cortes se desplazan, dentro de ±0.35 s, al instante más silencioso que no caiga dentro de una frase.
        6. Duración del clip final: `Settings.clip_bounds()` (refine_min_clip/refine_max_clip, por defecto 8–30 s, sin superar
           max_clip). Solo se alarga hasta el mínimo exacto; nunca para llegar a 15 s por costumbre.
        7. Fades de audio mínimos en los bordes si el corte cae sobre sonido; se limitan al hueco previo a la primera
           palabra / posterior a la última, así la voz no se modifica.
        8. Si el resultado cortara una frase o una palabra, se conserva la ventana original (recortada al máximo).
"""
from dataclasses import dataclass, field

import numpy as np

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.models import Candidate, Segment
from streamclipper.render.subtitles import _WEAK_END, _norm
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.candidates import _TOL, _inside

_CONNECTIVE_START = {"y", "pero", "porque", "que", "o", "entonces", "pues", "sino", "aunque", "así", "asi", "cuando",
                     "si", "como", "ni", "and", "but", "so", "because", "then", "or"}
_REFERENTIAL_START = {"eso", "esto", "esa", "ese", "esos", "esas", "aquello", "ahí", "ahi", "aquí", "aqui", "allí",
                      "allá", "también", "tampoco", "this", "that", "it", "there"}
_HARD = (".", "?", "!", "…")
_VALLEY = 0.35
_NATURAL_PAUSE = 0.25   # una pausa menor no es un final natural
_NATURAL_EXTRA = 1.5    # como mucho se alarga tanto el clip para terminar en una pausa natural


@dataclass
class Refined:
    start: float
    end: float
    notes: list[str] = field(default_factory=list)
    reaction_end: float | None = None
    changed: bool = False
    reaction_start: float | None = None
    fade_in: float = 0.0               # s de fade-in del audio (0 = sin fade)
    fade_out: float = 0.0
    bounds: tuple[float, float] | None = None   # (mínimo, máximo) de duración aplicados


def _containing(segs: list[Segment], t: float) -> Segment | None:
    return next((s for s in segs if s.start - _TOL <= t <= s.end + _TOL), None)


def _last_word(s: Segment) -> str:
    return _norm(s.words[-1].text) if s.words else _norm(s.text.split()[-1]) if s.text.split() else ""


def _first_word(s: Segment) -> str:
    return _norm(s.words[0].text) if s.words else _norm(s.text.split()[0]) if s.text.split() else ""


def _dependent(s: Segment, prev: Segment | None) -> bool:
    """La frase necesita la anterior para entenderse: empieza con un conector o una referencia («eso», «ahí»), continúa
    una oración partida sin pausa o responde a una pregunta inmediata. Una exclamación corta («¡Mierda, mierda!») NO es
    dependiente."""
    first = _first_word(s)
    if first in _CONNECTIVE_START or first in _REFERENTIAL_START:
        return True
    if prev is None:
        return False
    if not prev.text.rstrip().endswith(_HARD) and s.start - prev.end <= 0.3:
        return True
    n = len(s.words) or len(s.text.split())
    return prev.text.rstrip().endswith("?") and s.start - prev.end <= 1.5 and n <= 5


def _incomplete(s: Segment, nxt: Segment | None) -> bool:
    if _last_word(s) in _WEAK_END:
        return True
    # sin puntuación final y la frase siguiente arranca sin pausa: es la misma oración partida
    return bool(nxt) and not s.text.rstrip().endswith(_HARD) and nxt.start - s.end <= 0.3


def _pad_before(t: float, segs: list[Segment], pad: float) -> float:
    base = t - pad
    prev_end = max((s.end for s in segs if s.end <= t + _TOL), default=0.0)
    if base < prev_end:
        base = min(t, (prev_end + t) / 2)
    return max(0.0, base)


def _pad_after(t: float, segs: list[Segment], pad: float, dur: float) -> float:
    base = t + pad
    nxt = min((s.start for s in segs if s.start >= t - _TOL), default=dur)
    if base > nxt:
        base = max(t, (t + nxt) / 2)
    return min(dur, base)


def _peak(c: Candidate, ctx: AnalysisContext) -> tuple[int, float]:
    """Índice del pico de la reacción (cerca del ancla) y umbral = pico − refine_reaction_drop_db."""
    curve = ctx.curve
    lo, hi = curve.index(max(c.start, c.anchor - 1.5)), curve.index(min(ctx.duration, c.anchor + 3.0))
    pk_i = lo + int(np.argmax(curve.excess[lo:hi + 1]))
    return pk_i, float(curve.excess[pk_i]) - ctx.settings.refine_reaction_drop_db


def _speech_burst(c: Candidate, ctx: AnalysisContext, thr: float, pk_t: float) -> tuple[float, float] | None:
    """Frases de la reacción: desde la del pico, se encadenan hacia atrás y hacia adelante las frases separadas por
    pausas de hasta `refine_burst_gap` s cuyo volumen alcanza el umbral. El volumen cae entre palabra y palabra, así que
    la reacción no puede definirse solo por una caída sostenida del volumen."""
    cfg, segs, curve = ctx.settings, ctx.transcript.segments, ctx.curve
    s0 = _containing(segs, pk_t) or min((s for s in segs if abs(s.start - pk_t) <= 1.5 or abs(s.end - pk_t) <= 1.5),
                                         key=lambda s: min(abs(s.start - pk_t), abs(s.end - pk_t)), default=None)
    if s0 is None:
        return None
    floor_t = c.anchor - (cfg.pre_context + cfg.refine_max_extend)
    ceil_t = min(c.end + cfg.refine_max_extend, pk_t + cfg.refine_max_post)  # no se arrastra una tirada larga

    def loud(s: Segment) -> bool:
        return float(curve.window(curve.excess, s.start, s.end).max()) >= thr

    start, end, cur = s0.start, s0.end, s0
    while True:
        prev = max((s for s in segs if s.end <= cur.start + _TOL and s is not cur), key=lambda s: s.end, default=None)
        if prev is not None and cur.start - prev.end <= cfg.refine_burst_gap and loud(prev) and prev.start >= floor_t:
            start, cur = prev.start, prev
        else:
            break
    cur = s0
    while True:
        nxt = min((s for s in segs if s.start >= cur.end - _TOL and s is not cur), key=lambda s: s.start, default=None)
        if nxt is not None and nxt.start - cur.end <= cfg.refine_burst_gap and loud(nxt) and nxt.end <= ceil_t:
            end, cur = nxt.end, nxt
        else:
            break
    return start, end


def reaction_end(c: Candidate, ctx: AnalysisContext) -> float | None:
    """Instante en que termina la reacción: lo que ocurra más tarde entre la caída sostenida del volumen
    (`refine_calm_hold` s bajo el umbral) y el final de la última frase encadenada de la ráfaga de habla."""
    cfg, curve = ctx.settings, ctx.curve
    x, hop = curve.excess, curve.hop
    pk_i, thr = _peak(c, ctx)
    hold = max(1, int(cfg.refine_calm_hold / hop))
    cap_t = min(ctx.duration, pk_i * hop + cfg.refine_max_post)      # la reacción no se extiende más allá de esto
    limit = curve.index(min(ctx.duration, c.end + cfg.refine_max_extend, cap_t))
    drop = next((i * hop for i in range(pk_i, limit - hold + 1) if bool((x[i:i + hold] < thr).all())), None)
    if drop is None and limit * hop >= cap_t - hop:
        drop = cap_t                                                 # sigue fuerte: se corta en el tope
    burst = _speech_burst(c, ctx, thr, pk_i * hop)
    ends = [t for t in (drop, burst[1] if burst else None) if t is not None]
    return min(max(ends), cap_t) if ends else None


def reaction_start(c: Candidate, ctx: AnalysisContext) -> float | None:
    """Instante en que empieza la reacción: lo que ocurra antes entre el inicio de la ráfaga de habla encadenada y el
    punto hasta donde el volumen se mantiene cerca del pico. La reacción puede abarcar varias frases ANTES del pico
    («Oh my god. No way. …»): el clip no puede empezar después de que empezó."""
    cfg, curve = ctx.settings, ctx.curve
    x, hop = curve.excess, curve.hop
    pk_i, thr = _peak(c, ctx)
    hold = max(1, int(cfg.refine_calm_hold / hop))
    floor = curve.index(max(0.0, c.anchor - (cfg.pre_context + cfg.refine_max_extend)))
    i, onset = pk_i, None
    while i > floor:
        if i - hold >= 0 and bool((x[i - hold:i] < thr).all()):
            onset = i * hop
            break
        i -= 1
    if onset is None:
        onset = floor * hop
    burst = _speech_burst(c, ctx, thr, pk_i * hop)
    return min(onset, burst[0]) if burst else onset


def _valley(t: float, curve, segs: list[Segment], lo: float, hi: float) -> float:
    """Instante más silencioso en [lo, hi] que no caiga dentro de una frase; si no hay, t."""
    hop = curve.hop
    best, best_db = t, None
    for k in range(int(round(lo / hop)), int(round(hi / hop)) + 1):
        tt = k * hop
        if tt < 0 or _inside(tt, segs) is not None:
            continue
        db = float(curve.db[min(k, len(curve.db) - 1)])
        if best_db is None or db < best_db:
            best, best_db = tt, db
    return best


def _fit_duration(start: float, end: float, anchor: float, segs: list[Segment], dur: float, cfg, lo: float, hi: float,
                  notes: list[str]) -> tuple[float, float]:
    """Lleva [start, end] a [lo, hi]. Solo se alarga hasta el mínimo EXACTO (no hasta 15 s «por costumbre»); al recortar se
    conserva el pico. Sin habla transcrita se recorta por tiempo alrededor del ancla."""
    if end - start < lo:
        # Se alarga hasta el instante exacto que da la duración mínima (no frase por frase, que puede pasarse mucho
        # si hay un silencio largo): primero el contexto previo y, si no alcanza, el final.
        target = end - lo
        if target > 0 and (anchor - target) <= cfg.pre_context + cfg.refine_max_extend:
            s_in = _inside(target, segs)
            start = _pad_before(s_in.start if s_in else target, segs, 0.0 if not s_in else cfg.refine_lead)
            notes.append(f"inicio: se amplía el contexto previo solo hasta la duración mínima ({lo:g} s)")
        if end - start < lo and end < dur:
            target = start + lo
            s_in = _inside(target, segs)
            end = _pad_after(s_in.end if s_in else target, segs, cfg.refine_tail if s_in else 0.0, dur)
            notes.append(f"fin: se amplía solo hasta la duración mínima ({lo:g} s)")
    for _ in range(30):
        if end - start <= hi:
            break
        first = next((s for s in segs if s.start >= start - _TOL and s.end <= end + _TOL), None)
        if first is not None and anchor - first.end > 1.0:
            nxt_start = min((s.start for s in segs if s.start > first.start + 0.05), default=None)
            if nxt_start is None:
                break
            start = _pad_before(nxt_start, segs, cfg.refine_lead)
        else:
            # el final incluye el margen: se parte del final de la última frase REAL del clip para quitar esa y no repetirla
            cur_last = max((s.end for s in segs if s.end <= end + _TOL), default=end)
            last = max((s.end for s in segs if s.end < cur_last - 0.05), default=None)
            if last is None or last <= anchor + 1.0:
                break
            end = _pad_after(last, segs, cfg.refine_tail, dur)
    if end - start > hi:  # sin frases que quitar (p. ej. sin voz): recorte por tiempo conservando el ancla
        start = min(max(start, anchor - 0.4 * hi), max(0.0, dur - hi))
        end = min(dur, start + hi)
        notes.append(f"se recorta por tiempo a la duración máxima ({hi:g} s) conservando el momento")
    return max(0.0, start), min(dur, end)


def _fades(start: float, end: float, segs: list[Segment], curve, cfg) -> tuple[float, float]:
    """(fade_in, fade_out) en s. Solo si el corte cae sobre sonido (no sobre silencio) y NUNCA sobre el habla: el fade
    se limita al hueco entre el borde del clip y la primera/última palabra, así la voz queda intacta."""
    if not cfg.refine_fades:
        return 0.0, 0.0
    words = [w for s in segs for w in s.words if w.end > start and w.start < end]

    def audible(t: float) -> bool:
        return float(curve.excess[curve.index(t)]) > -15.0   # por encima de ~silencio

    fi = fo = 0.0
    if audible(start):
        room = (min(w.start for w in words) - start - 0.03) if words else cfg.refine_fade_in
        fi = max(0.0, min(cfg.refine_fade_in, room))
    if audible(end):
        room = (end - max(w.end for w in words) - 0.03) if words else cfg.refine_fade_out
        fo = max(0.0, min(cfg.refine_fade_out, room))
    return round(fi, 2), round(fo, 2)


def refine_window(c: Candidate, ctx: AnalysisContext) -> Refined:
    cfg, segs, dur, curve = ctx.settings, ctx.transcript.segments, ctx.duration, ctx.curve
    lo, hi = cfg.clip_bounds()
    notes: list[str] = []
    anchor = c.anchor

    # ---- fin --------------------------------------------------------------------------------------
    r_end = reaction_end(c, ctx)
    if r_end is None:
        end0, why_end = c.end, "sin caída clara del volumen: se conserva el fin original"
        e = _inside(end0, segs)
        if e:
            end0 = e.end
    else:
        s_end = _containing(segs, r_end) or max((s for s in segs if s.end <= r_end and s.end > anchor),
                                                key=lambda s: s.end, default=None)
        end0 = s_end.end if s_end else r_end
        why_end = f"la reacción termina hacia {r_end:.1f}s (volumen −{cfg.refine_reaction_drop_db:g} dB bajo su pico)"
    end0 = max(end0, anchor + cfg.refine_min_post)
    if (e := _inside(end0, segs)):
        end0 = e.end
    cap = c.end + cfg.refine_max_extend
    for _ in range(2):  # frase incompleta: se completa con la siguiente
        last = next((s for s in segs if abs(s.end - end0) <= _TOL), None)
        nxt = next((s for s in segs if s.start >= end0 - _TOL), None)
        if last and nxt and _incomplete(last, nxt) and nxt.start - end0 <= 1.0 and nxt.end <= cap:
            end0 = nxt.end
            notes.append(f"fin: se completa la frase «…{last.text[-25:]}» con la siguiente")
        else:
            break
    # final natural: el corte preferido es una pausa; si la frase se pega a la siguiente (sin pausa) y esta es corta, se
    # cierra con ella en vez de cortar en medio de una tirada de habla
    nxt = next((s for s in segs if s.start >= end0 - _TOL), None)
    if nxt is not None and nxt.start - end0 < _NATURAL_PAUSE and nxt.end - end0 <= _NATURAL_EXTRA and nxt.end <= cap:
        end0 = nxt.end
        notes.append("fin: se cierra con la frase siguiente para terminar en una pausa natural")
    end = _pad_after(end0, segs, cfg.refine_tail, dur)

    # ---- inicio -----------------------------------------------------------------------------------
    cur = _containing(segs, anchor) or min(
        (s for s in segs if abs(s.start - anchor) <= 2.0 or abs(s.end - anchor) <= 2.0),
        key=lambda s: min(abs(s.start - anchor), abs(s.end - anchor)), default=None)
    why_start = "frase que contiene el momento"
    r_start = reaction_start(c, ctx)
    if r_start is not None:  # la reacción puede empezar varias frases antes del pico: el clip la incluye completa
        onset = _containing(segs, r_start) or next((s for s in segs if r_start <= s.start <= r_start + 1.0), None)
        if onset is not None and (cur is None or onset.start < cur.start - _TOL):
            cur, why_start = onset, f"la reacción empieza hacia {r_start:.1f}s (antes de su pico)"
    if cur is None:
        start0, why_start = c.start, "sin frase en el momento: se conserva el inicio original"
    else:
        start0 = cur.start
        for _ in range(2):  # el contexto previo se agrega solo si la frase depende de la anterior
            prev = max((s for s in segs if s.end <= cur.start + _TOL and s is not cur), key=lambda s: s.end, default=None)
            if (prev is not None and _dependent(cur, prev) and cur.start - prev.end <= 2.5
                    and anchor - prev.start <= cfg.pre_context + cfg.refine_max_extend):
                start0, cur = prev.start, prev
                why_start = "incluye la frase anterior de la que depende el momento"
            else:
                break
    start = _pad_before(start0, segs, cfg.refine_lead)

    # ---- cortes en valles de energía --------------------------------------------------------------
    start = _valley(start, curve, segs, start - _VALLEY, start + 0.2)
    end = _valley(end, curve, segs, end - 0.2, end + _VALLEY)
    start, end = max(0.0, start), min(dur, end)

    # ---- duración y validación --------------------------------------------------------------------
    start, end = _fit_duration(start, end, anchor, segs, dur, cfg, lo, hi, notes)
    if check_boundaries(ctx.transcript, start, end)["errors"] or end <= start:
        # respaldo: la ventana original, recortada al máximo si hace falta (sin alargarla)
        fb_notes: list[str] = []
        s0, e0 = _fit_duration(c.start, c.end, anchor, segs, dur, cfg, 0.0, hi, fb_notes)
        if check_boundaries(ctx.transcript, s0, e0)["errors"]:
            if c.end - c.start <= hi + 1e-6:
                s0, e0 = c.start, c.end          # la ventana original ya cabe y es segura
            else:                                # el máximo es un límite duro: se prefiere recortar por tiempo
                fb_notes.append("el recorte por tiempo a la duración máxima puede cortar una frase")
        fi, fo = _fades(s0, e0, segs, curve, cfg)
        return Refined(s0, e0, ["el ajuste cortaba una frase o palabra: se conserva la ventana original"] + fb_notes,
                       r_end, False, r_start, fi, fo, (lo, hi))
    if abs(start - c.start) > 0.05:
        notes.insert(0, f"inicio {c.start:.2f}→{start:.2f}s: {why_start}")
    if abs(end - c.end) > 0.05:
        notes.insert(1 if abs(start - c.start) > 0.05 else 0, f"fin {c.end:.2f}→{end:.2f}s: {why_end}")
    fi, fo = _fades(start, end, segs, curve, cfg)
    return Refined(start, end, notes, r_end, abs(start - c.start) > 0.05 or abs(end - c.end) > 0.05, r_start, fi, fo,
                   (lo, hi))


def refine_selected(chosen: list[Candidate], ctx: AnalysisContext) -> list[Refined]:
    """Ajusta los límites de cada clip elegido; el orden y los candidatos no se modifican."""
    if not ctx.settings.refine_bounds:
        return [Refined(c.start, c.end, [], None, False) for c in chosen]
    return [refine_window(c, ctx) for c in chosen]

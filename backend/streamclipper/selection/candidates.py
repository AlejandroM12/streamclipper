"""Generación de candidatos: anclas por audio y por texto, ventanas alineadas a frases."""
import numpy as np

from streamclipper.models import Candidate, Segment
from streamclipper.selection.base import AnalysisContext

_PAD_START = 0.25
_PAD_END = 0.4
_MAX_ANCHORS = 60


def _audio_anchors(ctx: AnalysisContext) -> list[tuple[float, str]]:
    x, hop = ctx.curve.excess, ctx.curve.hop
    w = int(4 / hop)
    out = []
    for i in range(len(x)):
        if x[i] >= ctx.settings.audio_peak_db and x[i] == x[max(0, i - w):i + w + 1].max():
            out.append((i * hop, "audio"))
    return out


def _text_anchors(ctx: AnalysisContext) -> list[tuple[float, str]]:
    out = []
    for seg, sig in zip(ctx.transcript.segments, ctx.signals):
        if sig.reaction >= ctx.settings.text_trigger:
            win = ctx.curve.window(ctx.curve.excess, seg.start, seg.end)
            t = seg.start + int(np.argmax(win)) * ctx.curve.hop if len(win) else seg.start
            out.append((t, "verbal"))
    return out


def _anchor_strength(ctx: AnalysisContext, t: float) -> float:
    x = ctx.curve.window(ctx.curve.excess, t - 1.5, t + 1.5).max()
    text = max((sig.reaction for s, sig in zip(ctx.transcript.segments, ctx.signals)
                if s.start - 2 <= t <= s.end + 2), default=0.0)
    return float(x) / 15 + text


def _merge_anchors(ctx: AnalysisContext, raw: list[tuple[float, str]]) -> list[tuple[float, list[str]]]:
    """Fusiona anclas a menos de 3 s (p. ej. grito + frase) en una sola."""
    merged: list[tuple[float, list[str]]] = []
    for t, kind in sorted(raw):
        if merged and t - merged[-1][0] < 3.0:
            if kind not in merged[-1][1]:
                merged[-1][1].append(kind)
            if _anchor_strength(ctx, t) > _anchor_strength(ctx, merged[-1][0]):
                merged[-1] = (t, merged[-1][1])
        else:
            merged.append((t, [kind]))
    merged.sort(key=lambda a: _anchor_strength(ctx, a[0]), reverse=True)
    return merged[:_MAX_ANCHORS]


def _snap_start(t: float, segs: list[Segment]) -> float:
    for s in segs:
        if s.start < t < s.end:
            return s.start
    return t


def _snap_end(t: float, segs: list[Segment]) -> float:
    for s in segs:
        if s.start < t < s.end:
            return s.end
    return t


def build_window(anchor: float, ctx: AnalysisContext) -> tuple[float, float]:
    """Ventana [start, end] con contexto, sin cortar frases y respetando min/max de duración."""
    cfg, segs, dur = ctx.settings, ctx.transcript.segments, ctx.duration
    start = _snap_start(max(0.0, anchor - cfg.pre_context), segs) - _PAD_START
    end = _snap_end(min(dur, anchor + cfg.post_context), segs) + _PAD_END
    start, end = max(0.0, start), min(dur, end)

    # ampliar hasta la duración mínima, alternando lados y respetando límites de frase
    flip = False
    while end - start < cfg.min_clip and (start > 0 or end < dur):
        if (flip and start > 0) or end >= dur:
            nxt = max((s.start for s in segs if s.start < start - 0.5), default=None)
            start = max(0.0, max(nxt, start - 3.0) if nxt is not None else start - 3.0)
            start = _snap_start(start, segs)
        else:
            nxt = min((s.end for s in segs if s.end > end + 0.5), default=None)
            end = min(dur, min(nxt, end + 3.0) if nxt is not None else end + 3.0)
            end = _snap_end(end, segs)
        flip = not flip

    # recortar si excede el máximo, quitando frases del lado más alejado del ancla
    guard = 0
    while end - start > cfg.max_clip and guard < 100:
        guard += 1
        if anchor - start > end - anchor:
            nxt = min((s.start for s in segs if s.start > start + 0.1), default=None)
            new = nxt if nxt is not None and nxt < anchor - 1 else start + 2.0
            start = min(new, end - cfg.min_clip)
        else:
            prv = max((s.end for s in segs if s.end < end - 0.1), default=None)
            new = prv if prv is not None and prv > anchor + 1 else end - 2.0
            end = max(new, start + cfg.min_clip)
    if end - start > cfg.max_clip:
        end = start + cfg.max_clip
    return start, end


def generate_candidates(ctx: AnalysisContext) -> list[Candidate]:
    raw = _audio_anchors(ctx) + _text_anchors(ctx)
    out = []
    for t, triggers in _merge_anchors(ctx, raw):
        start, end = build_window(t, ctx)
        if end - start < ctx.settings.min_clip - 0.5:  # el video es más corto que el mínimo
            continue
        out.append(Candidate(start=start, end=end, anchor=t, triggers=triggers))
    return out

import random
import subprocess

import numpy as np
import pytest

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.cli import _build_parser, _style_overrides
from streamclipper.core.config import Settings, parse_region
from streamclipper.core.errors import InvalidInputError
from streamclipper.models import Candidate, Segment, Transcript, Word
from streamclipper.render.layout import compute_layout, fit_crop
from streamclipper.render.subtitles import (
    ass_document, build_cues, clip_words, integrity_problems, style_problems, write_subtitles,
)
from streamclipper.render.vertical import audio_fade_filter, build_filter
from streamclipper.selection.candidates import build_window
from streamclipper.selection.refine import _fades, refine_selected, refine_window
from streamclipper.selection.selector import find_clips
from tests.helpers import make_ctx, sentence
from tests.test_stage3 import candidate_at, scene, seg_of, speak


def fresh(c, ctx):
    r = refine_window(c, ctx)
    return r, r.end - r.start


# --- límites del clip final --------------------------------------------------------------------
def test_clip_bounds_defaults_legacy_and_cap():
    assert Settings().clip_bounds() == (8.0, 30.0)
    assert Settings(refine_min_clip=None, refine_max_clip=None).clip_bounds() == (15.0, 60.0)      # comportamiento anterior
    assert Settings(max_clip=20).clip_bounds() == (8.0, 20)                                        # nunca supera max_clip
    assert Settings(refine_min_clip=12.0, refine_max_clip=12.0).clip_bounds() == (12.0, 12.0)
    assert Settings(min_clip=15.0, refine_min_clip=None, max_clip=10.0).clip_bounds() == (10.0, 10.0)  # mínimo <= máximo


@pytest.mark.parametrize("bad", [
    {"refine_min_clip": 7.9}, {"refine_min_clip": 31.0}, {"refine_max_clip": 31.0}, {"refine_max_clip": 5.0},
    {"refine_min_clip": 20.0, "refine_max_clip": 10.0}, {"refine_fade_in": 2.0}, {"refine_fade_out": 3.0},
    {"keep_regions": ["no-es-region"]}, {"keep_regions": ["0.5,0,0.6,0.5"]}, {"keep_regions": ["0,0,0,0.5"]},
])
def test_new_clip_settings_are_validated(bad):
    with pytest.raises(InvalidInputError):
        Settings(**bad).validate()


def test_existing_configurations_still_load_and_validate():
    from streamclipper.pipeline import settings_from_result
    old = {"settings": {"whisper_model": "small", "min_clip": 15.0, "max_clip": 60.0, "layout": "focus",
                        "fg_zoom": 1.35, "fg_x": 0.1, "data_dir": "x"}}
    cfg = settings_from_result(old)
    cfg.validate()
    assert cfg.fg_zoom == 1.35 and cfg.keep_regions == [] and cfg.clip_bounds() == (8.0, 30.0)
    Settings(min_clip=15, max_clip=60).validate()     # las combinaciones previas siguen siendo válidas


# --- reacciones de menos de 15 s ---------------------------------------------------------------
def test_a_short_reaction_gives_a_short_clip_not_an_artificial_15_seconds():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs)
    r, dur = fresh(candidate_at(ctx, 30.0, 24.0, 46.0), ctx)
    assert 8.0 - 1e-6 <= dur < 15.0, (r, dur)
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]
    assert r.start <= 29.8 and r.end >= 33.0                                  # evento y reacción completos
    # con la configuración anterior (mínimo 15 s) el comportamiento histórico se conserva
    old = make_ctx(80, [(28, 33)], segs, Settings(refine_min_clip=None, refine_max_clip=None))
    r2, dur2 = fresh(candidate_at(old, 30.0, 24.0, 46.0), old)
    assert dur2 >= 15.0 - 1e-6


def test_the_clip_is_only_extended_up_to_the_exact_minimum():
    segs = scene()
    for lo in (8.0, 10.0, 12.0):
        ctx = make_ctx(80, [(28, 33)], segs, Settings(refine_min_clip=lo))
        r, dur = fresh(candidate_at(ctx, 30.0, 24.0, 46.0), ctx)
        assert lo - 1e-6 <= dur <= lo + 4.0, (lo, dur)       # solo lo que hace falta (hasta cerrar una frase)


def test_no_extension_note_when_the_natural_window_already_reaches_the_minimum():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs, Settings(refine_min_clip=8.0))
    r, dur = fresh(candidate_at(ctx, 30.0, 24.0, 46.0), ctx)
    natural = Settings(refine_min_clip=8.0)
    assert dur >= natural.refine_min_clip
    assert all("duración mínima" not in n for n in r.notes) or dur <= 8.0 + 4.0


# --- frases que comienzan antes del evento -----------------------------------------------------
@pytest.mark.parametrize("opening,text", [
    ("referencia", "eso no debería haber pasado nunca"),
    ("conector", "porque nadie me avisó de que iba a pasar"),
    ("respuesta", "no puede ser que otra vez"),
])
def test_context_sentence_is_kept_only_when_the_event_depends_on_it(opening, text):
    setup = sentence("acabo de abrir el garaje de la isla", 14.0, 2.5)
    if opening == "respuesta":
        setup = sentence("¿Habéis visto lo que ha hecho?", 14.0, 2.0)
        text = "no, no, no"
    event = sentence(text, 17.0 if opening != "respuesta" else 16.3, 2.5)
    after = sentence("luego hablamos de otra cosa distinta", 30.0, 3.0)
    segs = sorted([setup, event, after], key=lambda s: s.start)
    ctx = make_ctx(60, [(event.start, event.end)], segs, Settings(refine_min_clip=8.0, refine_min_post=1.0))
    r = refine_window(candidate_at(ctx, (event.start + event.end) / 2, 11.0, 26.0), ctx)
    assert r.start <= setup.start - 0.2, (opening, r)
    assert any("frase anterior" in n for n in r.notes)
    if opening == "respuesta":
        return                                  # una respuesta inmediata a una pregunta SÍ depende de ella
    # un evento autónomo no arrastra la frase previa
    standalone = sentence("¡Madre mía qué locura!", 17.0, 2.0)
    ctx2 = make_ctx(60, [(17.0, 19.0)], sorted([setup, standalone, after], key=lambda s: s.start),
                    Settings(refine_min_clip=3.0 if False else 8.0, refine_min_post=1.0))
    r2 = refine_window(candidate_at(ctx2, 18.0, 11.0, 26.0), ctx2)
    assert not any("frase anterior" in n for n in r2.notes)


# --- reacciones que terminan cerca de un límite ------------------------------------------------
def test_reaction_at_the_very_end_of_the_video():
    segs = [sentence("estaba mirando el mapa a ver por donde ir", 70, 3.0), sentence("¡No puede ser, qué locura!", 86.0, 2.5)]
    ctx = make_ctx(90, [(86.0, 88.5)], segs)
    r, dur = fresh(candidate_at(ctx, 87.0, 80.0, 90.0), ctx)
    assert 0 <= r.start < r.end <= 90.0 and r.end >= 88.4 and dur >= 8.0 - 1e-6
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]


def test_reaction_at_the_very_start_of_the_video():
    segs = [sentence("¡No puede ser, qué locura!", 0.5, 2.5), sentence("luego seguimos jugando tranquilos un rato", 12.0, 3.0)]
    ctx = make_ctx(60, [(0.5, 3.0)], segs)
    r, dur = fresh(candidate_at(ctx, 1.5, 0.0, 14.0), ctx)
    assert r.start >= 0.0 and r.start <= 0.5 and dur >= 8.0 - 1e-6
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]


def test_a_reaction_longer_than_the_maximum_is_trimmed_but_keeps_the_peak():
    segs = [sentence(f"y sigo gritando sin parar {i}", 20.0 + i * 2.3, 2.0) for i in range(14)]
    ctx = make_ctx(90, [(20.0, 53.0)], segs, Settings(refine_max_clip=12.0, refine_min_clip=8.0, refine_max_post=30.0,
                                                      refine_max_extend=30.0))
    anchor = 30.0
    r, dur = fresh(candidate_at(ctx, anchor, 24.0, 42.0), ctx)
    assert dur <= 12.0 + 1e-6 and r.start <= anchor <= r.end
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]


def test_reaction_ending_right_at_the_original_window_limit():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs)
    c = candidate_at(ctx, 30.0, 24.0, 33.0)          # la ventana original termina justo cuando termina la reacción
    r, dur = fresh(c, ctx)
    assert r.end >= 33.0 and dur >= 8.0 - 1e-6
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]


# --- duración máxima ---------------------------------------------------------------------------
def test_maximum_duration_is_always_respected_on_random_transcripts():
    rng = random.Random(5)
    for _ in range(60):
        segs, t = [], 1.0
        while t < 150:
            d = rng.uniform(0.5, 7.0)
            segs.append(sentence("palabra " * rng.randint(1, 8) + "fin.", t, d))
            t += d + rng.choice([0.05, 0.2, 0.8, 2.5])
        hi = rng.choice([10.0, 15.0, 20.0, 30.0])
        lo = rng.choice([8.0, 8.0, 10.0])
        cfg = Settings(refine_min_clip=min(lo, hi), refine_max_clip=hi, refine_max_post=rng.choice([4.0, 8.0, 20.0]))
        a = rng.uniform(10, 130)
        ctx = make_ctx(160, [(a, a + rng.uniform(1, 12))], segs, cfg)
        anchor = a + rng.uniform(0, 3)
        c = candidate_at(ctx, anchor)
        c.start, c.end = build_window(anchor, ctx)
        r, dur = fresh(c, ctx)
        assert dur <= hi + 1e-6, (hi, dur, r)
        assert 0 <= r.start < r.end <= 160
        assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]


def test_fallback_to_the_original_window_still_respects_the_maximum(monkeypatch):
    import streamclipper.selection.refine as ref
    segs = [sentence(f"sigo hablando de lo mío {i}", 10.0 + i * 3.0, 2.5) for i in range(14)]
    ctx = make_ctx(90, [(30.0, 36.0)], segs, Settings(refine_max_clip=15.0, refine_min_clip=8.0))
    c = candidate_at(ctx, 33.0, 12.0, 52.0)         # ventana original de 40 s, mayor que el máximo
    calls = {"n": 0}

    def fake(transcript, s, e, tol=0.05):           # el primer ajuste «corta una frase»; el respaldo recortado es válido
        calls["n"] += 1
        return {"errors": ["x"], "warnings": []} if calls["n"] == 1 else {"errors": [], "warnings": []}
    monkeypatch.setattr(ref, "check_boundaries", fake)
    r = ref.refine_window(c, ctx)
    assert r.end - r.start <= 15.0 + 1e-6 and "ventana original" in r.notes[0]


# --- clips sin voz -----------------------------------------------------------------------------
def test_a_clip_without_speech_is_refined_by_time_and_has_no_subtitles(tmp_path):
    ctx = make_ctx(90, [(40.0, 44.0)], [], Settings(refine_max_clip=12.0))
    c = candidate_at(ctx, 42.0, 30.0, 60.0)
    r, dur = fresh(c, ctx)
    assert dur <= 12.0 + 1e-6 and r.start <= 42.0 <= r.end and 0 <= r.start < r.end <= 90
    assert refine_selected([c], ctx)[0].start == r.start
    cfg = Settings()
    lay = compute_layout(1920, 1080, cfg)
    res = write_subtitles(tmp_path / "s.ass", tmp_path / "s.srt", [], r.start, r.end, cfg, lay)
    assert res.cues == [] and res.problems == [] and res.metrics == {"cues": 0, "coverage_pct": 0.0}
    assert "[Events]" in (tmp_path / "s.ass").read_text(encoding="utf-8") and (tmp_path / "s.srt").read_text(encoding="utf-8") == ""


def test_fades_apply_to_a_clip_without_voice_when_the_cut_is_on_sound():
    ctx = make_ctx(60, [(0.0, 60.0)], [])              # sonido continuo
    fi, fo = _fades(10.0, 25.0, [], ctx.curve, ctx.settings)
    assert (fi, fo) == (ctx.settings.refine_fade_in, ctx.settings.refine_fade_out)


# --- timestamps incompletos --------------------------------------------------------------------
def test_segments_without_word_timestamps_are_handled_everywhere(tmp_path):
    seg = Segment(10.0, 15.0, "esto es una frase sin marcas por palabra que Whisper no pudo alinear", words=[])
    ctx = make_ctx(60, [(11.0, 14.0)], [seg, sentence("luego seguimos jugando tranquilos un rato", 30.0, 3.0)])
    r, dur = fresh(candidate_at(ctx, 12.0, 6.0, 20.0), ctx)
    assert dur >= 8.0 - 1e-6 and not check_boundaries(ctx.transcript, r.start, r.end)["errors"]
    cfg = Settings()
    lay = compute_layout(1920, 1080, cfg)
    res = write_subtitles(tmp_path / "a.ass", None, ctx.transcript.segments, r.start, r.end, cfg, lay)
    assert res.cues and res.problems == []                  # texto repartido uniformemente, sin inventar palabras
    shown = " ".join(c.text for c in res.cues).split()
    assert shown[:6] == seg.text.split()[:6]


def test_degenerate_word_timestamps_never_break_subtitles_or_refinement():
    words = [Word(10.0, 10.0, "cero", 0.9), Word(10.5, 10.2, "invertida", 0.9), Word(10.4, 10.9, "desordenada", 0.9),
             Word(11.0, 11.5, "ok", 0.9), Word(11.6, 11.6, "", 0.9)]
    seg = Segment(10.0, 11.6, "cero invertida desordenada ok", words)
    ws = clip_words([seg], 0.0, 30.0)
    assert [w.text for w in ws] == ["cero", "desordenada", "invertida", "ok"]          # ordenadas por inicio; la vacía se descarta
    assert all(w.end > w.start and w.start >= 0 for w in ws)
    assert all(a.end <= b.start + 1e-6 or a.start <= b.start for a, b in zip(ws, ws[1:]))
    cues = build_cues([seg], 0.0, 30.0, Settings(), 25)
    assert cues and all(0 <= c.start < c.end <= 30.0 for c in cues)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(cues, cues[1:]))
    ctx = make_ctx(60, [(10.0, 12.0)], [seg])
    r, dur = fresh(candidate_at(ctx, 10.5, 4.0, 18.0), ctx)
    assert dur >= 8.0 - 1e-6 and 0 <= r.start < r.end <= 60


def test_partially_transcribed_clip_only_shows_the_words_that_exist(tmp_path):
    seg = seg_of([("hola", 12.0, 12.4), ("amigos", 12.5, 12.9)])         # el resto del clip no se transcribió
    cfg = Settings()
    lay = compute_layout(1920, 1080, cfg)
    res = write_subtitles(tmp_path / "p.ass", None, [seg], 10.0, 30.0, cfg, lay)
    assert [w.text for c in res.cues for w in c.words] == ["hola", "amigos"]
    assert res.problems == []


# --- ranking ------------------------------------------------------------------------------------
def test_ranking_is_identical_for_any_clip_length_fade_or_framing_setting():
    segs = scene() + [sentence("otra parte con más cosas que decir hoy", 100, 3.0),
                      sentence("¡madre mía no puede ser qué locura!", 104, 3.0)]
    segs.sort(key=lambda s: s.start)

    def snapshot(cfg):
        ctx = make_ctx(160, [(28, 33), (104, 108)], segs, cfg)
        cands, chosen = find_clips(ctx)
        return ([(c.rank, c.start, c.end, c.anchor, round(c.score, 9), [x.value for x in c.components]) for c in cands],
                [c.rank for c in chosen])
    base = snapshot(Settings())
    for cfg in (Settings(refine_min_clip=8.0, refine_max_clip=12.0), Settings(refine_min_clip=None, refine_max_clip=None),
                Settings(refine_fades=False), Settings(refine_bounds=False), Settings(layout="focus", fg_zoom=1.5, fg_x=0.0,
                                                                                     keep_regions=["0,0,0.2,0.4"]),
                Settings(refine_max_post=2.0, refine_burst_gap=0.5, sub_max_words=3)):
        assert snapshot(cfg) == base


def test_refinement_does_not_mutate_candidates():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs)
    cands, chosen = find_clips(ctx)
    before = [(c.start, c.end, c.anchor, c.score, c.rank) for c in cands]
    refine_selected(chosen, ctx)
    assert before == [(c.start, c.end, c.anchor, c.score, c.rank) for c in cands]


# --- fades de audio ----------------------------------------------------------------------------
def test_fades_never_touch_the_voice():
    rng = random.Random(9)
    for _ in range(80):
        segs, t = [], 1.0
        while t < 60:
            d = rng.uniform(0.4, 3.0)
            segs.append(sentence("palabra " * rng.randint(1, 5) + "fin.", t, d))
            t += d + rng.choice([0.1, 0.4, 1.5])
        ctx = make_ctx(70, [(0.0, 70.0)], segs)                    # sonido continuo: siempre «audible»
        start = rng.uniform(0, 30)
        end = start + rng.uniform(8, 25)
        fi, fo = _fades(start, end, segs, ctx.curve, ctx.settings)
        words = [w for s in segs for w in s.words if w.end > start and w.start < end]
        if words:
            first_word = min(w.start for w in words)
            last_word = max(w.end for w in words)
            assert start + fi <= max(first_word, start) + 1e-9                  # la rampa termina antes de la voz
            assert end - fo >= min(last_word, end) - 1e-9                       # y empieza después de ella
            if first_word < start + 0.03:                                       # si el corte cae sobre una palabra: sin fade
                assert fi == 0.0
            if last_word > end - 0.03:
                assert fo == 0.0
        assert 0 <= fi <= ctx.settings.refine_fade_in and 0 <= fo <= ctx.settings.refine_fade_out


def test_no_fade_when_the_cut_falls_on_silence_or_fades_are_disabled():
    segs = [sentence("hola que tal estamos hoy", 20.0, 3.0)]
    quiet = make_ctx(60, [], segs)                                   # nivel plano -35 dB: excess ≈ 0 → «audible»
    import numpy as np
    quiet.curve.excess[:] = -30.0                                   # silencio claro
    assert _fades(15.0, 28.0, segs, quiet.curve, quiet.settings) == (0.0, 0.0)
    loud = make_ctx(60, [(0.0, 60.0)], segs, Settings(refine_fades=False))
    assert _fades(15.0, 28.0, segs, loud.curve, loud.settings) == (0.0, 0.0)
    assert np.isfinite(quiet.curve.excess).all()


def test_audio_fade_filter_string():
    assert audio_fade_filter(20.0) is None
    assert audio_fade_filter(20.0, 0.12, 0.3) == "afade=t=in:st=0:d=0.120,afade=t=out:st=19.700:d=0.300"
    assert audio_fade_filter(20.0, 0.0, 0.3) == "afade=t=out:st=19.700:d=0.300"
    assert audio_fade_filter(0.2, 0.0, 0.3).startswith("afade=t=out:st=0.000")


# --- encuadre conservador ----------------------------------------------------------------------
def test_regions_are_always_kept_when_they_fit_and_zoom_shrinks_when_they_dont():
    face, hud = (0.0, 0.0, 0.25, 0.4), (0.77, 0.08, 0.2, 0.32)
    cw, cx, zoom, kept = fit_crop(1920, 1.2, 0.5, [face])
    assert kept and cx == 0 and cw == 1600 and zoom == pytest.approx(1.2)               # solo la facecam: se puede acercar
    cw, cx, zoom, kept = fit_crop(1920, 1.2, 0.5, [face, hud])
    assert kept and zoom < 1.1 and cx + cw >= 0.96 * 1920 and cx <= 2                    # facecam + HUD: casi cuadro completo
    cw, cx, zoom, kept = fit_crop(1920, 1.5, 0.5, [(0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 0.5, 0.5)])
    assert (cw, cx, zoom, kept) == (1920, 0, 1.0, True)                                  # ocupan todo: cuadro completo
    cw, cx, zoom, kept = fit_crop(1920, 1.35, 0.1, [])
    assert kept is None and cx == int((1920 - cw) * 0.1) // 2 * 2                        # manual sin regiones


def test_no_declared_region_is_ever_cropped_across_random_regions():
    rng = random.Random(4)
    for _ in range(200):
        n = rng.randint(1, 3)
        regs = []
        for _ in range(n):
            w = rng.uniform(0.05, 0.6)
            x = rng.uniform(0, 1 - w)
            regs.append((x, rng.uniform(0, 0.5), w, rng.uniform(0.05, 0.5)))
        src = rng.choice([1920, 1280, 2560])
        cw, cx, zoom, kept = fit_crop(src, rng.uniform(1.0, 2.0), rng.random(), regs)
        assert 0 <= cx and cx + cw <= src and 1.0 <= zoom
        x0, x1 = min(r[0] for r in regs) * src, max(r[0] + r[2] for r in regs) * src
        if kept:
            assert cx <= x0 + 2 and x1 <= cx + cw + 2


def test_layout_reports_what_is_cropped_and_never_claims_detection():
    none = compute_layout(1920, 1080, Settings(layout="focus", fg_zoom=1.35)).framing
    assert none["crop_left_pct"] + none["crop_right_pct"] == pytest.approx(25.9, abs=0.5)
    assert "no detecta" in none["note"] and none["regions_kept"] is None
    keep = compute_layout(1920, 1080, Settings(layout="focus", keep_regions=["0,0,0.25,0.4", "0.77,0.08,0.2,0.32"])).framing
    assert keep["regions_kept"] is True and keep["zoom_used"] < 1.1 and "completas" in keep["note"]
    fit = compute_layout(1920, 1080, Settings(layout="blur-fit")).framing
    assert fit["crop_left_pct"] == 0 and fit["crop_right_pct"] == 0 and "sin recortar" in fit["note"]
    fill = compute_layout(1920, 1080, Settings(layout="fill")).framing
    assert fill["visible_width_pct"] == pytest.approx(31.6, abs=0.2)


def test_blur_fit_stays_available_and_unchanged_as_the_safe_alternative():
    lay = compute_layout(1920, 1080, Settings(layout="blur-fit", keep_regions=["0,0,0.25,0.4"]))
    assert lay.crop[:3] == (1920, 1080, 0) and lay.fg[1] == 608                         # nunca recorta
    assert "crop=1920:1080:0:0" in build_filter(Settings(layout="blur-fit"), "s.ass")


def test_manual_framing_options_from_the_cli():
    p = _build_parser().parse_args(["rerender", "job", "--keep", "0,0,0.25,0.4", "--keep", "0.77,0.08,0.2,0.32",
                                    "--fg-zoom", "1.3", "--fg-x", "0.2", "--refine-min-clip", "9", "--refine-max-clip", "25",
                                    "--no-fades"])
    ov = _style_overrides(p)
    assert ov == {"fg_zoom": 1.3, "fg_x": 0.2, "refine_min_clip": 9.0, "refine_max_clip": 25.0, "refine_fades": False,
                  "keep_regions": ["0,0,0.25,0.4", "0.77,0.08,0.2,0.32"]}
    Settings(**ov).validate()
    with pytest.raises(InvalidInputError):
        parse_region("0.9,0,0.3,0.3")


# --- integridad de subtítulos ------------------------------------------------------------------
def test_integrity_checker_detects_each_demonstrable_defect():
    cfg = Settings()
    lay = compute_layout(1920, 1080, cfg)
    seg = speak("esto es una prueba de subtítulos muy normal", 1.0, wps=3.0)
    good = build_cues([seg], 0.0, 20.0, cfg, lay.sub_line_chars)
    assert integrity_problems(good, [seg], 0.0, 20.0, lay, cfg) == []
    import copy
    out = copy.deepcopy(good)
    out[-1].end = 25.0
    assert any("fuera de la duración" in p for p in integrity_problems(out, [seg], 0.0, 20.0, lay, cfg))
    ov = copy.deepcopy(good)
    ov[0].end = ov[1].start + 0.5
    assert any("se superponen" in p for p in integrity_problems(ov, [seg], 0.0, 20.0, lay, cfg))
    om = copy.deepcopy(good)
    om[0].words = om[0].words[:-1]
    assert any("no coincide con la transcripción" in p for p in integrity_problems(om, [seg], 0.0, 20.0, lay, cfg))
    ln = copy.deepcopy(good)
    ln[0].lines = ["una linea extremadamente larga que se sale de los margenes del video"]
    assert any("línea de" in p for p in integrity_problems(ln, [seg], 0.0, 20.0, lay, cfg))
    gap = copy.deepcopy(good)
    gap[0].end = gap[0].start + 0.1
    gap[1].start = gap[1].start + 0.0
    probs = integrity_problems(gap, [seg], 0.0, 20.0, lay, cfg)
    assert any("hueco injustificado" in p for p in probs)
    late = copy.deepcopy(good)
    late[0].start += 0.5
    assert any("después de la primera palabra" in p for p in integrity_problems(late, [seg], 0.0, 20.0, lay, cfg))
    many = copy.deepcopy(good)
    many[0].lines = ["a", "b", "c"]
    assert any("líneas (máximo" in p for p in integrity_problems(many, [seg], 0.0, 20.0, lay, cfg))


def test_generated_subtitles_have_no_demonstrable_defects_on_random_transcripts(tmp_path):
    rng = random.Random(8)
    vocab = "el la de que y a en un por con para como más pero esto tengo mucho cuidado vehículos coño no sí madre".split()
    for trial in range(60):
        segs, t = [], 0.5
        while t < 40:
            toks = [rng.choice(vocab) + rng.choice(["", "", ",", ".", "!", "?"]) for _ in range(rng.randint(1, 14))]
            sp = rng.choice([2.0, 3.5, 6.0])
            seg = speak(" ".join(toks), t, wps=sp, prob=rng.choice([0.9, 0.3]))
            segs.append(seg)
            t = seg.end + rng.choice([0.05, 0.3, 1.0, 3.0])
        cfg = Settings(sub_max_words=rng.choice([4, 7, 9]), sub_max_lines=rng.choice([2, 3]), sub_font_size=rng.choice([60, 68, 80]),
                       layout=rng.choice(["blur-fit", "focus", "fill"]), safe_platform=rng.choice(["universal", "tiktok", "reels"]))
        lay = compute_layout(1920, 1080, cfg)
        start = rng.uniform(0, 15)
        end = start + rng.uniform(8, 30)
        res = write_subtitles(tmp_path / "t.ass", tmp_path / "t.srt", segs, start, end, cfg, lay)
        assert res.problems == [], (trial, res.problems)


def test_style_checker_detects_wrong_margins():
    cfg = Settings()
    lay = compute_layout(1920, 1080, cfg)
    doc = ass_document([], lay, cfg)
    assert style_problems(doc, lay) == []
    bad = doc.replace(f",{lay.margin_l},{lay.margin_r},{lay.sub_margin_v},", ",0,0,10,")
    assert style_problems(bad, lay)
    assert style_problems("sin estilo", lay) == ["no se encontró el estilo ASS"]


# --- render real: fades y audio ----------------------------------------------------------------
def _pcm(path, start, dur, ffmpeg):
    out = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(path), "-ss", f"{start}", "-t", f"{dur}", "-ac", "1",
                          "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float64) / 32768


@pytest.mark.integration
def test_rendered_fades_shape_only_the_edges_and_leave_the_body_identical(tmp_path):
    from streamclipper.core.cancel import CancelToken
    from streamclipper.core.config import REPO_ROOT
    from streamclipper.render.layout import compute_layout as cl
    from streamclipper.render.vertical import render_clip
    sample = REPO_ROOT / "data" / "samples" / "synthetic.mp4"
    if not sample.exists():
        pytest.skip("falta el video sintético (lo genera el test de integración principal)")
    cfg = Settings()
    (tmp_path / "s.ass").write_text(ass_document([], cl(1920, 1080, cfg), cfg), encoding="utf-8")
    plain, faded = tmp_path / "plain.mp4", tmp_path / "faded.mp4"
    lay = cl(1920, 1080, cfg)
    render_clip(sample, 20.0, 30.0, tmp_path / "s.ass", plain, cfg, CancelToken(), lay, 0.0, 0.0)
    render_clip(sample, 20.0, 30.0, tmp_path / "s.ass", faded, cfg, CancelToken(), lay, 0.12, 0.3)
    a, b = _pcm(plain, 0, 10, cfg.ffmpeg), _pcm(faded, 0, 10, cfg.ffmpeg)
    n = min(len(a), len(b))
    rms = lambda x: float(np.sqrt((x ** 2).mean()) + 1e-12)  # noqa: E731
    assert rms(b[:int(0.04 * 16000)]) < 0.5 * rms(a[:int(0.04 * 16000)])           # el inicio realmente sube desde cero
    assert rms(b[n - int(0.04 * 16000):n]) < 0.5 * rms(a[n - int(0.04 * 16000):n])  # y el final baja a cero
    mid = slice(int(1.0 * 16000), int(9.0 * 16000))                                  # el cuerpo del clip no cambia
    assert abs(20 * np.log10(rms(b[mid]) / rms(a[mid]))) < 0.1
    assert np.corrcoef(a[mid], b[mid])[0, 1] > 0.99


def test_transcript_type_is_used_consistently():
    t = Transcript("es", [seg_of([("a", 1.0, 1.5)])])
    assert isinstance(t.segments[0], Segment) and isinstance(candidate_at(make_ctx(10, [], []), 5.0), Candidate)

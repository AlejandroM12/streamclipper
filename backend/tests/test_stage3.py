import random
import re

import pytest

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.core.config import Settings, recommended_model
from streamclipper.core.errors import InvalidInputError
from streamclipper.models import Candidate, Segment, Word
from streamclipper.render.ass import subtitle_lines, write_ass
from streamclipper.render.layout import H, SAFE_MARGINS, W, compute_layout, line_chars
from streamclipper.render.subtitles import (
    _ts, ass_document, build_cues, clip_words, cue_metrics, layout_lines, speech_rate_cps, srt_document,
    write_subtitles,
)
from streamclipper.render.vertical import build_filter
from streamclipper.selection.refine import reaction_end, reaction_start, refine_selected, refine_window
from streamclipper.selection.selector import find_clips
from tests.helpers import make_ctx, sentence


def words_at(spec, prob=0.9):
    """spec = [(texto, inicio, fin)]."""
    return [Word(a, b, t, prob if not isinstance(prob, list) else prob[i]) for i, (t, a, b) in enumerate(spec)]


def seg_of(spec, prob=0.9) -> Segment:
    ws = words_at(spec, prob)
    return Segment(ws[0].start, ws[-1].end, " ".join(w.text for w in ws), ws)


def speak(text, start, wps=3.0, prob=0.9):
    """Frase con `wps` palabras por segundo."""
    toks = text.split()
    step = 1 / wps
    return seg_of([(t, start + i * step, start + i * step + step * 0.85) for i, t in enumerate(toks)], prob)


CFG = Settings()
LC = compute_layout(1920, 1080, CFG).sub_line_chars


# --- distribución en líneas --------------------------------------------------------------------
def test_layout_lines_fit_split_and_limits():
    assert layout_lines(["hola", "mundo"], 25, 2) == ["hola mundo"]
    two = layout_lines("esto es una frase larga para dos líneas".split(), 25, 2)
    assert len(two) == 2 and all(len(x) <= 25 for x in two)
    assert layout_lines("esto es una frase larga para dos líneas".split(), 25, 1) is None
    assert layout_lines("esto es una frase bastante larga para dos líneas".split(), 25, 2) is None  # 49 caracteres no caben
    assert layout_lines(["x" * 30], 25, 2) is None                      # una palabra más ancha que la línea
    assert layout_lines(("palabra " * 12).split(), 25, 2) is None        # no cabe en dos líneas


def test_layout_lines_prefers_breaking_after_a_comma():
    lines = layout_lines("ya lo sé, pero no me importa nada".split(), 25, 2)
    assert lines == ["ya lo sé,", "pero no me importa nada"]
    assert layout_lines("uno dos tres cuatro cinco seis siete".split(), 25, 2) is not None  # sin comas: lo más parejo


# --- segmentación y tiempos --------------------------------------------------------------------
def all_words(cues):
    return [w.text for c in cues for w in c.words]


def test_text_is_never_invented_or_removed():
    seg = speak("¿Qué cojones está pasando, gente? ¿Es explosivo? No, no, no, no. Para explotar los NPCs.", 1.0, wps=3.5)
    cues = build_cues([seg], 0.0, 20.0, CFG, LC)
    assert all_words(cues) == [w.text for w in seg.words]
    assert " ".join(c.text for c in cues).split() == seg.text.split()


def test_cues_respect_configured_limits():
    rng = random.Random(3)
    vocab = "el la de que y a en un por con para como más pero esto aquello tengo mucho cuidado vehículos".split()
    for trial in range(20):
        toks = [rng.choice(vocab) + rng.choice(["", "", ",", "."]) for _ in range(rng.randint(5, 40))]
        seg = speak(" ".join(toks), 0.5, wps=rng.choice([2.0, 3.5, 5.0]))
        cfg = Settings(sub_max_words=rng.choice([4, 6, 7]), sub_max_lines=2)
        lc = compute_layout(1920, 1080, cfg).sub_line_chars
        cues = build_cues([seg], 0.0, 60.0, cfg, lc)
        assert all_words(cues) == [w.text for w in seg.words]
        for c in cues:
            assert len(c.lines) <= 2 and all(len(x) <= lc for x in c.lines), c.lines
        # una unión por velocidad solo puede excederse en habla muy rápida y nunca más de 3 palabras
        assert all(len(c.words) <= cfg.sub_max_words + 3 for c in cues)
        for a, b in zip(cues, cues[1:]):
            assert a.end <= b.start + 1e-9 and a.start < a.end


def test_subtitle_never_appears_before_the_word_is_spoken():
    seg = speak("esto aparece justo cuando se dice nada antes", 2.0)
    cues = build_cues([seg], 0.0, 20.0, CFG, LC)
    assert cues[0].start == pytest.approx(seg.words[0].start)
    for c in cues:
        assert c.start == pytest.approx(c.words[0].start)


def test_subtitles_are_not_cut_before_the_dialogue_ends():
    seg = speak("y entonces pasó lo que tenía que pasar", 1.0, wps=3.0)
    cues = build_cues([seg], 0.0, 30.0, CFG, LC)
    last_word_end = seg.words[-1].end
    assert cues[-1].end >= last_word_end + 0.3 - 1e-9                    # cola tras la última palabra
    for a, b in zip(cues, cues[1:]):
        assert a.end >= a.words[-1].end - 1e-9                          # nunca antes de terminar de hablar


def test_short_gaps_are_bridged_but_real_pauses_are_kept():
    a = seg_of([("primera", 1.0, 1.5), ("frase.", 1.6, 2.1)])
    b = seg_of([("segunda", 2.4, 2.9), ("frase.", 3.0, 3.5)])         # hueco de 0.3 s: habla continua
    c = seg_of([("tercera", 7.0, 7.5), ("frase.", 7.6, 8.1)])         # pausa real de 3.5 s
    cues = build_cues([a, b, c], 0.0, 20.0, CFG, LC)
    assert len(cues) == 3
    assert cues[1].start - cues[0].end == pytest.approx(0.04, abs=1e-6)   # sin parpadeo entre frases pegadas
    assert cues[1].end - cues[1].words[-1].end == pytest.approx(0.3, abs=0.05)  # tras una pausa solo la cola
    assert cues[2].start - cues[1].end > 3.0


def test_minimum_reading_time_is_granted_when_there_is_room():
    quick = seg_of([("sí", 5.0, 5.2), ("claro", 5.25, 5.5)])           # se dice en 0.5 s, luego silencio
    cue = build_cues([quick], 0.0, 30.0, CFG, LC)[0]
    assert cue.duration >= CFG.sub_min_duration - 1e-9


def test_fast_phrases_are_merged_instead_of_flashing():
    # dos frases cortas dichas muy rápido y pegadas: se unen en vez de dejar un subtítulo de 0.3 s
    a = seg_of([("¡No!", 1.0, 1.2)])
    b = seg_of([("¡No!", 1.25, 1.45), ("¡No!", 1.5, 1.7)])
    cues = build_cues([a, b], 0.0, 20.0, CFG, LC)
    assert len(cues) == 1 and cues[0].duration >= 1.0 - 1e-9


def test_avoids_ending_a_subtitle_on_an_article_or_preposition():
    seg = speak("vamos a ver qué pasa con el coche de la gente que está en la calle ahora", 1.0, wps=3.0)
    cues = build_cues([seg], 0.0, 30.0, Settings(sub_max_words=5), LC)
    weak = {"el", "la", "de", "que", "a", "con", "en", "y"}
    ends = [c.words[-1].text.lower().strip(",.") for c in cues[:-1]]
    assert sum(e in weak for e in ends) <= 1  # puede ser inevitable una vez, no la norma


def test_configurable_word_limit_and_hard_punctuation_split():
    seg = speak("uno dos tres cuatro cinco seis. siete ocho nueve", 1.0)
    cues = build_cues([seg], 0.0, 20.0, Settings(sub_max_words=3), LC)
    assert all(len(c.words) <= 3 for c in cues) and len(cues) >= 3
    assert build_cues([seg], 0.0, 20.0, CFG, LC)[0].words[-1].text == "seis."  # el punto corta el subtítulo


def test_absurdly_long_word_timestamps_do_not_make_a_subtitle_linger():
    seg = seg_of([("ahí", 2.0, 5.5), ("sigue", 5.6, 6.0)])             # Whisper estiró «ahí» sobre una pausa
    ws = clip_words([seg], 0.0, 20.0)
    assert ws[0].end - ws[0].start <= 1.2 + 1e-9
    assert build_cues([seg], 0.0, 20.0, CFG, LC)[0].end < 4.5


def test_clip_window_selects_and_rebases_words():
    seg = speak("antes del clip y después del clip también", 8.0, wps=2.0)
    ws = clip_words([seg], 10.0, 13.0)
    assert ws and all(0 <= w.start < 3.0 for w in ws)
    assert [w.text for w in ws] == [w.text for w in seg.words if 10.0 < w.end and w.start < 13.0]
    assert build_cues([seg], 100.0, 120.0, CFG, LC) == []


def test_low_confidence_cues_are_flagged_and_marked_only_on_request(tmp_path):
    seg = seg_of([("hola", 1.0, 1.4), ("palabra", 1.5, 1.9), ("dudosa", 2.0, 2.4), ("otra", 2.5, 2.9)],
                 prob=[0.9, 0.1, 0.2, 0.9])
    lay = compute_layout(1920, 1080, CFG)
    res = write_subtitles(tmp_path / "a.ass", tmp_path / "a.srt", [seg], 0.0, 10.0, CFG, lay)
    assert res.low_confidence == ["hola palabra dudosa otra"] or len(res.low_confidence) == 1
    plain = (tmp_path / "a.ass").read_text(encoding="utf-8")
    assert "&H0080FF&" not in plain and "palabra dudosa" in plain.replace("\\N", " ")
    marked_cfg = Settings(sub_mark_low_conf=True)
    write_subtitles(tmp_path / "b.ass", None, [seg], 0.0, 10.0, marked_cfg, compute_layout(1920, 1080, marked_cfg))
    marked = (tmp_path / "b.ass").read_text(encoding="utf-8")
    assert marked.count("&H0080FF&") == 2
    stripped = re.sub(r"\{[^}]*\}", "", marked).replace("\\N", " ")   # el texto es el mismo con o sin marcas
    assert "hola palabra dudosa otra" in stripped


def test_ass_and_srt_documents(tmp_path):
    seg = speak("esto es una frase bastante larga para que haga falta partirla", 1.0, wps=4.0)
    lay = compute_layout(1920, 1080, CFG)
    cues = build_cues([seg], 0.0, 20.0, CFG, LC)
    ass = ass_document(cues, lay, CFG)
    assert "PlayResX: 1080" in ass and "WrapStyle: 2" in ass and "Alignment" in ass
    style = [ln for ln in ass.splitlines() if ln.startswith("Style:")][0].split(",")
    assert style[2] == str(CFG.sub_font_size) and style[18] == str(lay.sub_alignment)
    assert style[21] == str(lay.sub_margin_v) and style[19] == str(lay.margin_l) and style[20] == str(lay.margin_r)
    assert "\\N" in ass
    srt = srt_document(cues)
    assert srt.startswith("1\n00:00:01,") and " --> " in srt and srt.count("\n\n") == len(cues)
    assert _ts(61.257) == "0:01:01.26" and _ts(-3) == "0:00:00.00"


def test_braces_in_text_cannot_inject_ass_overrides(tmp_path):
    seg = seg_of([("{\\an9}hack", 1.0, 1.5), ("{texto}", 1.6, 2.0)])
    write_ass(tmp_path / "e.ass", [seg], 0.0, 5.0)
    body = (tmp_path / "e.ass").read_text(encoding="utf-8").split("[Events]")[1].split("Default,,0,0,0,,")[1]
    assert "{" not in body and "}" not in body and "\\an9" not in body


def test_legacy_helpers_still_work():
    lines = subtitle_lines([sentence("esta es la primera frase del clip", 100, 4)], 99.0, 110.0)
    assert lines and lines[0][0] == pytest.approx(1.0, abs=0.01)
    assert write_ass.__name__ == "write_ass"


# --- métricas ----------------------------------------------------------------------------------
def test_cue_metrics_and_speech_rate():
    m = cue_metrics([(0.0, 1.0, "hola amigo"), (1.04, 1.5, "esto es muy rápido de leer"), (4.0, 6.0, "uno\\Ndos")],
                    duration=8.0, max_cps=17.0, min_duration=1.0)
    assert m["cues"] == 3 and m["coverage_pct"] == pytest.approx(100 * (1.0 + 0.46 + 2.0) / 8.0, abs=0.2)
    assert m["pct_over_max_cps"] == pytest.approx(33.3, abs=0.1) and m["cues_shorter_than_min"] == 1
    assert m["gaps_over_1s"] == 1 and m["max_lines"] == 2 and m["min_cue_duration_s"] == 0.46
    assert cue_metrics([], 10.0) == {"cues": 0, "coverage_pct": 0.0}
    rate = speech_rate_cps([speak("hola amigos como están todos hoy", 2.0, wps=4.0)], 0.0, 20.0)
    assert 20 <= rate <= 40            # habla rápida: no se puede subtitular más lento que se habla
    assert speech_rate_cps([], 0, 10) is None


# --- layout y márgenes seguros -----------------------------------------------------------------
@pytest.mark.parametrize("platform", ["universal", "tiktok", "reels", "shorts"])
@pytest.mark.parametrize("layout", ["blur-fit", "focus"])
@pytest.mark.parametrize("src", [(1920, 1080), (1280, 720), (1080, 1920), (1920, 800)])
def test_gameplay_and_subtitles_stay_inside_the_safe_area(platform, layout, src):
    cfg = Settings(layout=layout, safe_platform=platform, fg_x=0.2)
    lay = compute_layout(*src, cfg)
    m = SAFE_MARGINS[platform]
    fw, fh, fx, fy = lay.fg
    assert fy >= m["top"] and fy + fh + 40 <= lay.sub_margin_v + 1           # gameplay dentro y subtítulos debajo
    assert lay.sub_margin_v + lay.sub_block_h <= H - m["bottom"] + 1          # el bloque de subtítulos no pisa la UI
    assert 0 <= fx and fx + fw <= W and fw % 2 == 0 and fh % 2 == 0
    cw, ch, cx, cy = lay.crop
    assert 0 <= cx and cx + cw <= src[0] and cy == 0 and ch <= src[1]
    assert lay.sub_alignment == 8


def test_focus_makes_the_gameplay_bigger_and_fit_shows_the_whole_frame():
    fit = compute_layout(1920, 1080, Settings(layout="blur-fit"))
    focus = compute_layout(1920, 1080, Settings(layout="focus", fg_zoom=1.35))
    assert fit.crop[:2] == (1920, 1080) and fit.fg[1] == 608                     # cuadro completo, sin recortar
    assert focus.fg[1] > fit.fg[1] * 1.3 and focus.crop[0] < 1920                # más grande, recorta costados
    left = compute_layout(1920, 1080, Settings(layout="focus", fg_zoom=1.35, fg_x=0.0)).crop[2]
    right = compute_layout(1920, 1080, Settings(layout="focus", fg_zoom=1.35, fg_x=1.0)).crop[2]
    assert left == 0 and right == 1920 - focus.crop[0]                           # la facecam puede quedar a cualquier lado


def test_full_bleed_layouts_put_subtitles_above_the_bottom_margin():
    for lay in ("fill", "crop"):
        lo = compute_layout(1920, 1080, Settings(layout=lay, safe_platform="tiktok"))
        assert lo.fg is None and lo.sub_alignment == 2 and lo.sub_margin_v == SAFE_MARGINS["tiktok"]["bottom"]


def test_universal_margins_are_the_union_of_all_platforms():
    u = SAFE_MARGINS["universal"]
    for side in u:
        assert u[side] == max(SAFE_MARGINS[p][side] for p in ("tiktok", "reels", "shorts"))
    assert line_chars(68, 60, 120) == 25 and line_chars(90, 60, 120) < 25


def test_filter_graph_reflects_layout_and_background_options():
    f = build_filter(Settings(layout="focus", fg_zoom=1.35, fg_x=0.1, bg_blur=40, bg_dim=0.2), "s.ass")
    assert "boxblur=40:4" in f and "eq=brightness=-0.2" in f and "overlay=" in f and "ass=s.ass" in f
    assert "crop=1422:1080:" in f and f.endswith("[v]")
    plain = build_filter(Settings(layout="blur-fit", bg_blur=0, bg_dim=0.0), "s.ass")
    assert "boxblur" not in plain and "eq=" not in plain and "crop=1920:1080:0:0" in plain
    assert "overlay" not in build_filter(Settings(layout="fill"), "s.ass")


# --- ajuste de límites -------------------------------------------------------------------------
def scene():
    """Charla tranquila, una reacción fuerte a ~30 s y más charla después; frases de 2-4 s."""
    segs = [
        sentence("estaba mirando el mapa a ver por donde ir", 5, 3.0),
        sentence("bueno creo que por aquí está bien", 10, 3.0),
        sentence("y de repente aparece un coche enorme", 24, 3.0),
        sentence("pero no puede ser mira lo que ha pasado", 27.5, 3.0),     # momento: ~29-31 s
        sentence("madre mía qué locura", 31.5, 2.0),
        sentence("bueno vamos a seguir con lo que tocaba hacer hoy", 40, 4.0),
        sentence("y ahora toca ir hacia la isla del norte porque", 46, 4.0),
        sentence("hay que recoger unas cosas", 50.2, 2.0),
    ]
    return sorted(segs, key=lambda s: s.start)


def candidate_at(ctx, anchor, start=None, end=None):
    cand = Candidate(start if start is not None else anchor - 6, end if end is not None else anchor + 6, anchor, ["audio"])
    cand.rank = 1
    return cand


def test_reaction_end_is_found_from_the_volume_drop():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs)            # fuerte entre 28 y 33 s, el resto calmado
    c = candidate_at(ctx, 30.0)
    end = reaction_end(c, ctx)
    assert end is not None and 32.5 <= end <= 35.0


def test_refined_clip_ends_on_a_complete_sentence_soon_after_the_reaction():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs, Settings(min_clip=10.0))
    c = candidate_at(ctx, 30.0, 24.0, 46.0)         # ventana original larga: sobran ~12 s tras la reacción
    r = refine_window(c, ctx)
    assert r.end < c.end - 5 and r.end >= 33.5 + 0.0   # recorta los segundos innecesarios
    assert r.end <= 31.5 + 2.0 + 1.5                    # termina tras «madre mía qué locura» (frase completa) + margen
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]
    assert any(n.startswith("fin") for n in r.notes) and r.changed


def test_a_reaction_spanning_several_sentences_before_its_peak_is_included_whole():
    """Réplica del video sintético: «Oh my god. No way. He is right behind me. That was insane.» (30–36 s) con el ancla
    (pico de audio) en la ÚLTIMA frase. El clip no puede empezar después de que empezó la reacción."""
    segs = [sentence("so we are dropping into the city now", 5, 3.0),
            sentence("I think there is a squad on the north side", 19, 3.2),
            sentence("Oh my god.", 30.0, 0.6), sentence("No way.", 31.5, 0.6),
            sentence("He is right behind me.", 32.8, 1.3), sentence("That was insane.", 34.9, 0.9),
            sentence("Alright that fight went well now we need to heal", 42, 2.5)]
    ctx = make_ctx(90, [(30, 36)], segs)
    c = candidate_at(ctx, 35.5, 26.25, 43.9)
    r = refine_window(c, ctx)
    assert r.start <= 30.0 - 0.2 and r.end >= 35.8, (r.start, r.end)           # contiene la reacción completa
    assert r.reaction_start is not None and r.reaction_start <= 30.6
    assert any("reacción empieza" in n for n in r.notes) or r.start <= c.start
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]


def test_reaction_is_chained_across_the_volume_dips_between_words():
    """El volumen real cae entre palabra y palabra (aquí hasta ~0.8 s de calma): una caída sostenida de 1 s no debe
    partir la reacción. Es el caso del video sintético real, que la réplica de volumen continuo no cubría."""
    segs = [sentence("so we are dropping into the city now", 5, 3.0),
            sentence("Oh my god.", 30.0, 0.6), sentence("No way.", 31.5, 0.6),
            sentence("He is right behind me.", 32.8, 1.3), sentence("That was insane.", 34.9, 0.9),
            sentence("Alright that fight went well now we need to heal", 42, 2.5)]
    ctx = make_ctx(90, [(30.0, 30.7), (31.5, 32.1), (32.8, 34.1), (34.9, 35.8)], segs)   # fuerte solo mientras se habla
    c = candidate_at(ctx, 35.5, 26.25, 43.9)
    assert reaction_start(c, ctx) <= 30.1 and reaction_end(c, ctx) >= 35.7
    r = refine_window(c, ctx)
    assert r.start <= 30.0 - 0.2 and r.end >= 35.8 + 0.5
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]
    # una frase lejana (pausa > refine_burst_gap) no se incluye como parte de la reacción
    far = [*segs[:1], sentence("esto fue hace un rato", 24.0, 1.5), *segs[1:]]
    ctx2 = make_ctx(90, [(30.0, 30.7), (31.5, 32.1), (32.8, 34.1), (34.9, 35.8)], far)
    assert reaction_start(candidate_at(ctx2, 35.5, 26.25, 43.9), ctx2) > 29.0


def test_a_long_loud_monologue_after_the_moment_does_not_drag_the_clip_on():
    """Locutor que sigue hablando fuerte y sin pausas durante 20 s: la reacción no se extiende más de refine_max_post."""
    segs = [sentence("¡No puede ser!", 30.0, 1.2)] + [sentence(f"y sigo hablando muy fuerte sin parar {i}", 31.6 + i * 2.1, 1.8)
                                                       for i in range(10)]
    ctx = make_ctx(90, [(30.0, 53.0)], segs, Settings(min_clip=8.0, refine_max_post=6.0, refine_max_extend=20.0))
    c = candidate_at(ctx, 30.5, 25.0, 45.0)
    r = refine_window(c, ctx)
    assert r.end <= 30.5 + 6.0 + 2.5 + 1.0, r      # pico + máximo + una frase de cierre + margen
    assert r.start <= 30.0 and not check_boundaries(ctx.transcript, r.start, r.end)["errors"]
    wide = make_ctx(90, [(30.0, 53.0)], segs, Settings(min_clip=8.0, refine_max_post=25.0, refine_max_extend=20.0))
    assert refine_window(candidate_at(wide, 30.5, 25.0, 45.0), wide).end > r.end + 5   # configurable


def test_the_refined_window_always_contains_the_whole_reaction():
    rng = random.Random(21)
    for _ in range(40):
        segs, t = [], 1.0
        while t < 120:
            d = rng.uniform(0.5, 4.0)
            segs.append(sentence("palabra " * rng.randint(1, 6) + "fin.", t, d))
            t += d + rng.choice([0.1, 0.3, 0.8, 2.0])
        a = rng.uniform(15, 100)
        ctx = make_ctx(130, [(a, a + rng.uniform(2, 7))], segs)
        from streamclipper.selection.candidates import build_window
        anchor = a + rng.uniform(0.0, 3.0)
        c = candidate_at(ctx, anchor)
        c.start, c.end = build_window(anchor, ctx)
        r = refine_window(c, ctx)
        rs, re_ = r.reaction_start, r.reaction_end
        if rs is None or re_ is None or r.end - r.start >= ctx.settings.max_clip - 1:
            continue
        for s in segs:   # ninguna frase de la reacción queda fuera
            if s.start >= rs - 0.05 and s.end <= re_ + 0.05:
                assert s.start >= r.start - 0.05 and s.end <= r.end + 0.05, (s.start, s.end, r.start, r.end, rs, re_)


def test_always_keeps_a_minimum_after_the_main_moment():
    segs = scene()
    ctx = make_ctx(80, [(28, 29.5)], segs, Settings(min_clip=8.0, refine_min_post=4.0))
    r = refine_window(candidate_at(ctx, 28.5, 22.0, 40.0), ctx)
    assert r.end >= 28.5 + 4.0


def test_incomplete_final_sentence_is_completed_with_the_next_one():
    segs = [sentence("esto fue una locura total y encima la", 20, 3.0),
            sentence("moto explotó delante de todos nosotros", 23.1, 3.0),
            sentence("luego seguimos jugando tranquilos un rato", 32, 3.0)]
    ctx = make_ctx(60, [(19, 22)], segs, Settings(min_clip=5.0, refine_tail=0.5, refine_min_post=0.5))
    r = refine_window(candidate_at(ctx, 21.0, 15.0, 25.0), ctx)
    assert r.end >= 26.0                                # incluye «moto explotó delante de todos nosotros»
    assert any("completa la frase" in n for n in r.notes)
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]


def test_previous_context_is_added_only_when_the_moment_depends_on_it():
    dep = [sentence("estábamos esperando a que llegara el tren", 14, 3.0),
           sentence("pero se ha ido sin nosotros", 18, 2.5),           # empieza con conector: depende de la anterior
           sentence("luego hablamos de otra cosa distinta", 30, 3.0)]
    ctx = make_ctx(60, [(18, 20)], dep, Settings(refine_min_clip=None, min_clip=3.0, refine_min_post=1.0))
    r = refine_window(candidate_at(ctx, 19.0, 17.0, 26.0), ctx)
    assert r.start <= 14.0 and any("frase anterior" in n for n in r.notes)

    indep = [sentence("estábamos esperando a que llegara el tren", 14, 3.0),
             sentence("¡Mierda, mierda, mierda!", 18, 2.0),             # exclamación corta: NO depende
             sentence("luego hablamos de otra cosa distinta", 30, 3.0)]
    ctx = make_ctx(60, [(18, 20)], indep, Settings(refine_min_clip=None, min_clip=3.0, refine_min_post=1.0))
    r = refine_window(candidate_at(ctx, 19.0, 17.0, 26.0), ctx)
    assert r.start >= 16.5 and not any("frase anterior" in n for n in r.notes)


def test_cuts_snap_to_quiet_moments_and_never_inside_speech():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs, Settings(min_clip=10.0))
    r = refine_window(candidate_at(ctx, 30.0, 24.0, 46.0), ctx)
    for t in (r.start, r.end):
        assert not any(s.start + 0.05 < t < s.end - 0.05 for s in segs)


def test_duration_limits_are_configurable_and_respected():
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs, Settings(refine_min_clip=20.0, refine_max_clip=25.0))
    r = refine_window(candidate_at(ctx, 30.0, 24.0, 46.0), ctx)
    assert 20.0 - 1e-6 <= r.end - r.start <= 25.0 + 1e-6
    assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"]
    short = make_ctx(80, [(28, 33)], segs, Settings(min_clip=5.0, max_clip=12.0, refine_min_post=2.0))
    r2 = refine_window(candidate_at(short, 30.0, 24.0, 46.0), short)
    assert r2.end - r2.start <= 12.0 + 1e-6


def test_refinement_never_cuts_a_sentence_over_random_transcripts():
    rng = random.Random(11)
    for _ in range(40):
        segs, t = [], 1.0
        while t < 150:
            d = rng.uniform(1.0, 8.0)
            segs.append(sentence("palabra " * rng.randint(2, 9) + "fin.", t, d))
            t += d + rng.choice([0.05, 0.2, 0.6, 2.0, 5.0])
        loud = [(rng.uniform(10, 140), rng.uniform(1, 4)) for _ in range(2)]
        ctx = make_ctx(160, [(a, a + d) for a, d in loud], segs, Settings(refine_min_post=rng.choice([0.5, 2.5, 5.0])))
        for anchor in (rng.uniform(8, 150) for _ in range(6)):
            c = candidate_at(ctx, anchor)
            from streamclipper.selection.candidates import build_window
            c.start, c.end = build_window(anchor, ctx)
            r = refine_window(c, ctx)
            assert 0 <= r.start < r.end <= 160
            assert not check_boundaries(ctx.transcript, r.start, r.end)["errors"], (anchor, r)
            assert r.end - r.start <= ctx.settings.max_clip + 1e-6


def test_refinement_is_separate_from_selection_and_does_not_touch_the_ranking():
    segs = scene() + [sentence("otra parte distinta del video con más cosas", 100, 3.0),
                      sentence("¡madre mía no puede ser qué locura!", 104, 3.0)]
    segs.sort(key=lambda s: s.start)
    ctx = make_ctx(160, [(28, 33), (104, 108)], segs)
    cands, chosen = find_clips(ctx)
    before = [(c.rank, c.start, c.end, c.anchor, c.score, [x.value for x in c.components]) for c in cands]
    refined = refine_selected(chosen, ctx)
    after = [(c.rank, c.start, c.end, c.anchor, c.score, [x.value for x in c.components]) for c in cands]
    assert before == after                                   # ni ventanas ni puntuaciones ni orden se tocaron
    assert len(refined) == len(chosen)
    off = make_ctx(160, [(28, 33), (104, 108)], segs, Settings(refine_bounds=False))
    cands2, chosen2 = find_clips(off)
    assert [(c.start, c.end, c.score) for c in cands2] == [(c.start, c.end, c.score) for c in cands]
    assert [(r.start, r.end, r.changed) for r in refine_selected(chosen2, off)] == [(c.start, c.end, False) for c in chosen2]


def test_falls_back_to_the_original_window_if_the_adjustment_would_cut_a_sentence(monkeypatch):
    segs = scene()
    ctx = make_ctx(80, [(28, 33)], segs, Settings(min_clip=10.0))
    c = candidate_at(ctx, 30.0, 24.0, 46.0)
    import streamclipper.selection.refine as ref
    monkeypatch.setattr(ref, "check_boundaries", lambda *a, **k: {"errors": ["x"], "warnings": []})
    r = ref.refine_window(c, ctx)
    assert (r.start, r.end, r.changed) == (24.0, 46.0, False) and "ventana original" in r.notes[0]


# --- configuración -----------------------------------------------------------------------------
def test_recommended_model_is_small_for_spanish():
    assert recommended_model("es") == "small" and recommended_model(None) == "small"
    assert recommended_model("en") == "base"
    assert Settings().whisper_model == "base"       # el valor de la API no cambia: base sigue disponible y es más rápido


@pytest.mark.parametrize("bad", [
    {"fg_zoom": 3.0}, {"fg_x": 1.5}, {"bg_blur": 200}, {"bg_dim": 0.9}, {"safe_platform": "youtube"},
    {"sub_max_words": 0}, {"sub_max_lines": 5}, {"sub_font_size": 20}, {"sub_min_duration": 0.0},
    {"sub_max_cps": 3.0}, {"sub_line_chars": 3}, {"refine_lead": 9.0}, {"refine_reaction_drop_db": 0.0},
    {"refine_calm_hold": 0.0}, {"layout": "cinemascope"},
])
def test_new_settings_are_validated(bad):
    with pytest.raises(InvalidInputError):
        Settings(**bad).validate()


def test_new_defaults_are_valid_and_old_results_still_load():
    Settings().validate()
    from streamclipper.pipeline import settings_from_result
    old = {"settings": {"whisper_model": "small", "layout": "blur-fit", "data_dir": "x"}}   # sin los campos nuevos
    cfg = settings_from_result(old)
    assert cfg.sub_max_words == 7 and cfg.refine_bounds is True and cfg.safe_platform == "universal"


def test_cli_style_overrides_only_include_what_was_given():
    from streamclipper.cli import _build_parser, _style_overrides
    a = _build_parser().parse_args(["rerender", "job", "--fg-zoom", "1.5", "--sub-max-words", "5", "--no-refine"])
    assert _style_overrides(a) == {"fg_zoom": 1.5, "sub_max_words": 5, "refine_bounds": False}
    assert _style_overrides(_build_parser().parse_args(["rerender", "job"])) == {}
    p = _build_parser().parse_args(["process", "x.mp4"])
    assert p.model is None and p.layout == "blur-fit"

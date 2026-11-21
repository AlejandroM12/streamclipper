"""Smart Highlights: señales de audio, contexto de transcripción, imagen, ranking, evaluación e integración.

Las pruebas con audio/video sintético usan eventos de verdad conocida: demuestran que cada señal hace lo que dice, NO que Smart
Highlights sea mejor con material real (eso se mide con valoraciones humanas: ver docs/smart-highlights.md).
"""
import json
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from streamclipper import evaluation_smart as es
from streamclipper.analysis.audio import analyze_audio
from streamclipper.analysis.audio_features import analyze_features
from streamclipper.analysis.narrative import analyze_narrative
from streamclipper.analysis.visual import MAX_SAMPLED_S, analyze_visual
from streamclipper.app.manager import JobManager
from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.models import Candidate, Segment, Transcript, Word
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.heuristic import HeuristicScorer
from streamclipper.selection.selector import find_clips, select_candidates
from streamclipper.selection.smart import PENALTY, POSITIVE, SmartScorer
from streamclipper.webutil import ApiError

R = 16000
rng = np.random.default_rng(7)


# --- audio sintético con eventos conocidos ---------------------------------------------------------------
def _voice(dur, f0=140, amp=0.1):
    t = np.arange(int(dur * R)) / R
    s = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return amp * s * (0.6 + 0.4 * np.sin(2 * np.pi * (3.7 + 0.8 * np.sin(2 * np.pi * 0.3 * t)) * t)) / 3


def _laugh(dur, amp=0.3):
    t = np.arange(int(dur * R)) / R
    s = sum(np.sin(2 * np.pi * 260 * k * t) / k for k in range(1, 9))
    return amp * s * (np.sin(2 * np.pi * 5.0 * t) > 0).astype(float) / 3


def _noise(dur, amp=0.1):
    return amp * rng.standard_normal(int(dur * R))


def _explosions(dur, every=2.0, amp=0.9):
    x = np.zeros(int(dur * R))
    for t0 in np.arange(1, dur - 1, every):
        n, i = int(0.4 * R), int(t0 * R)
        x[i:i + n] += amp * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.1 * R))
    return x + _noise(dur, 0.02)


# plano: 0-60 habla | 60-70 REACCIÓN | 70-110 habla | 110-120 RISA | 120-170 habla | 170-200 EXPLOSIONES | 200-240 habla | 240-260 SILENCIO
PLAN = [("habla", 60, lambda d: _voice(d)), ("reaccion", 10, lambda d: _voice(d, 220, 0.6)), ("habla", 40, lambda d: _voice(d)),
        ("risa", 10, lambda d: _laugh(d) + _voice(d, 150, 0.03)), ("habla", 50, lambda d: _voice(d)), ("explosiones", 30, _explosions),
        ("habla", 40, lambda d: _voice(d)), ("silencio", 20, lambda d: _noise(d, 0.002))]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    xs, marks, t = [], {}, 0
    for name, d, f in PLAN:
        xs.append(f(d))
        marks[name] = (t, t + d)
        t += d
    path = tmp_path_factory.mktemp("audio") / "synth.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(R)
        w.writeframes((np.clip(np.concatenate(xs), -1, 1) * 32767).astype(np.int16).tobytes())
    curve = analyze_audio(path)
    return path, curve, analyze_features(path, curve), marks


def test_features_have_the_curve_grid(synth):
    _, curve, f, _ = synth
    for arr in (f.voice, f.flatness, f.centroid, f.laugh, f.context_delta, f.constancy, f.periodic, f.quiet):
        assert len(arr) == len(curve.db)
    assert f.laugh.min() >= 0 and f.laugh.max() <= 1 and f.periodic.max() <= 1 and f.constancy.max() <= 1


def test_repetitive_explosions_are_flat_and_periodic_but_speech_is_not(synth):
    _, _, f, m = synth
    a, b = m["explosiones"]
    assert f.window(f.flatness, a + 2, b - 2).mean() > 0.7           # ruido plano, no voz
    assert f.window(f.periodic, a + 5, b - 5).max() > 0.8            # picos a intervalos regulares
    a, b = m["habla"] if "habla" in m else (0, 60)
    assert f.window(f.flatness, 5, 55).mean() < 0.1 and f.window(f.periodic, 5, 40).max() < 0.4


def test_silence_is_detected_relative_to_the_usual_level(synth):
    _, _, f, m = synth
    a, b = m["silencio"]
    assert f.window(f.quiet, a + 1, b - 1).all() and not f.window(f.quiet, 5, 55).any()


def test_regular_vocal_bursts_are_flagged_only_with_enough_evidence(synth):
    _, _, f, m = synth
    a, b = m["risa"]
    assert f.window(f.laugh, a + 1, b - 1).max() >= 0.3              # ráfagas regulares 3–8 Hz
    assert f.window(f.laugh, 5, 55).max() == 0.0 and f.window(f.laugh, 125, 165).max() == 0.0   # habla irregular: nada


def test_sudden_reaction_stands_out_against_the_previous_30_seconds(synth):
    _, _, f, m = synth
    a, b = m["reaccion"]
    assert f.window(f.context_delta, a, b).max() > 10 and f.window(f.context_delta, 20, 50).max() < 9


# --- contexto de transcripción ---------------------------------------------------------------------------
def seg(text, start, dur, prob=0.95, **kw):
    toks = text.split()
    step = dur / len(toks)
    words = [Word(start + i * step, start + (i + 1) * step - 0.02, t, prob) for i, t in enumerate(toks)]
    return Segment(start, start + dur, text, words, **kw)


def ctx_for(segs, duration=120.0, curve=None):
    from tests.helpers import make_ctx
    return make_ctx(duration, [], segs)


def cand(start, end, anchor):
    return Candidate(start=start, end=end, anchor=anchor, triggers=["audio"])


def test_narrative_needs_a_setup_before_and_a_payoff_after():
    segs = [seg("A ver si esto funciona?", 30, 3), seg("No puede ser, increíble!", 40, 3), seg("Qué locura.", 44, 2)]
    ctx = ctx_for(segs)
    n = analyze_narrative(cand(28, 48, 40.5), ctx)
    assert n.available and n.setup == 1.0 and n.payoff > 0 and n.arc > 0
    assert any("a ver" in e for e in n.evidence["planteamiento"]) and n.evidence["desenlace"]
    # sin planteamiento previo, una reacción sola no forma arco
    only = analyze_narrative(cand(38, 48, 40.5), ctx_for([seg("No puede ser, increíble!", 40, 3)]))
    assert only.setup == 0.0 and only.arc == 0.0


def test_exclamations_are_not_penalized_as_incomplete_but_split_sentences_are():
    closed = ctx_for([seg("Vamos con todo.", 10, 2), seg("No, no, no.", 20, 2)])
    assert analyze_narrative(cand(9, 23, 20.5), closed).incomplete == 0.0
    cut = ctx_for([seg("y entonces pasó que", 10, 2), seg("el coche se fue", 12.1, 2), seg("y después de todo el", 20, 2), seg("lío terminó bien.", 22.1, 2)])
    n = analyze_narrative(cand(10, 22.0, 15), cut)
    assert n.incomplete >= 0.5 and n.evidence["cortes"]


def test_dubious_transcription_and_loops_are_flagged():
    loop = [seg("gracias por ver", 10 + 3 * i, 2.5, prob=0.3, compression_ratio=3.0, avg_logprob=-1.5) for i in range(4)]
    n = analyze_narrative(cand(8, 24, 15), ctx_for(loop))
    assert n.flagged >= 0.9
    clean = analyze_narrative(cand(8, 24, 15), ctx_for([seg("hola a todos hoy jugamos", 10, 3), seg("vamos a ver qué pasa.", 14, 3)]))
    assert clean.flagged == 0.0


def test_narrative_is_unavailable_without_any_transcribed_words():
    assert analyze_narrative(cand(0, 20, 10), ctx_for([])).available is False


# --- imagen ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def video_with_cut(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / "cut.mp4"
    inputs = []
    for color in ("black", "white", "black"):
        inputs += ["-f", "lavfi", "-t", "12", "-i", f"color={color}:s=320x180:r=10"]
    subprocess.run([str(Settings().ffmpeg), "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", "concat=n=3:v=1:a=0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_visual_detects_a_scene_cut_and_not_a_static_scene(video_with_cut):
    cfg = Settings()
    cut, static = cand(10, 20, 12), cand(2, 8, 5)
    v = analyze_visual(video_with_cut, cfg.ffmpeg, [cut, static])
    assert v.available and v.per_candidate[id(cut)].cuts >= 1 and v.per_candidate[id(cut)].spike > 0.5
    assert v.per_candidate[id(static)].cuts == 0 and v.per_candidate[id(static)].spike < 0.2


def test_visual_samples_a_limited_amount_and_degrades_without_the_video(video_with_cut, tmp_path):
    cfg = Settings()
    many = [cand(10, 20, 6 + i * 0.5) for i in range(40)]
    v = analyze_visual(video_with_cut, cfg.ffmpeg, many, max_seconds=30)
    assert v.sampled_seconds <= 30 and len(v.per_candidate) < 40
    assert MAX_SAMPLED_S <= 900
    missing = analyze_visual(tmp_path / "no-existe.mp4", cfg.ffmpeg, many)
    assert missing.available is False and "video" in missing.reason


# --- Smart Highlights: degradación, pesos, penalizaciones, duplicados -----------------------------------------
def test_weights_are_priors_that_add_up():
    assert sum(POSITIVE.values()) == pytest.approx(1.0)
    assert set(PENALTY) == {"musica_o_ruido", "audio_repetitivo", "aire_muerto", "transcripcion_dudosa", "frase_incompleta"}


def synth_ctx(synth, with_words=True):
    path, curve, feats, m = synth
    segs = []
    if with_words:
        segs += [seg("A ver que pasa si lo intento?", 54, 4), seg("No puede ser, increible, no!", 60.5, 3.5), seg("Qué locura, tío.", 64.5, 3)]
        segs += [seg("bueno seguimos jugando tranquilamente.", 20 + 6 * i, 3) for i in range(3)]
    tr = Transcript("es", segs)
    return AnalysisContext(tr, curve, 260.0, Settings(max_clips=3)), feats, path


def test_without_new_signals_smart_degrades_exactly_to_the_classic_ranking(synth):
    ctx, _, _ = synth_ctx(synth, with_words=False)
    smart = SmartScorer(features=None, use_visual=False)
    cs, chs = find_clips(ctx, smart)
    cc, chc = find_clips(ctx, HeuristicScorer())
    assert [(round(c.score, 6), c.start, c.end) for c in cs] == [(round(c.score, 6), c.start, c.end) for c in cc]
    assert [c.start for c in chs] == [c.start for c in chc]
    assert smart.info()["effective"].startswith("clásico") and smart.info()["missing"]


def test_smart_penalizes_repetitive_loud_noise_that_the_classic_ranks_high(synth):
    ctx, feats, _ = synth_ctx(synth)
    a, b = synth[3]["explosiones"]
    cl = HeuristicScorer().score(SmartScorer().generate(ctx), ctx)
    classic_expl = max((c for c in cl if a <= c.anchor <= b), key=lambda c: c.score)
    classic_react = max((c for c in cl if 60 <= c.anchor <= 70), key=lambda c: c.score)
    scorer = SmartScorer(feats, use_visual=False)
    cs, chosen = find_clips(ctx, scorer)
    expl = max((c for c in cs if a <= c.anchor <= b), key=lambda c: c.score)
    react = max((c for c in cs if 60 <= c.anchor <= 70), key=lambda c: c.score)
    names = {x.name: x for x in expl.components}
    assert names["audio_repetitivo"].value > 0.3 and names["musica_o_ruido"].value > 0.3     # las penalizaciones se activan
    assert names["audio_repetitivo"].contribution < 0 and names["musica_o_ruido"].contribution < 0
    assert react.score > expl.score                                                          # y la reacción real queda por encima
    assert (react.score - expl.score) > (classic_react.score - classic_expl.score)           # con más margen que en el clásico
    assert react.anchor in [c.anchor for c in chosen]


def test_every_component_explains_itself_and_nothing_is_called_confidence(synth):
    ctx, feats, _ = synth_ctx(synth)
    cs, _ = find_clips(ctx, SmartScorer(feats, use_visual=False))
    d = cs[0].to_dict()
    assert "confidence" not in json.dumps(d).lower() and "confianza" not in d
    assert all(c["note"] for c in d["components"] if c["name"] != "visual")
    pos = [c for c in cs[0].components if c.weight > 0]
    assert sum(c.weight for c in pos) == pytest.approx(1.0)                                  # pesos efectivos renormalizados
    assert cs[0].score == pytest.approx(max(0.0, sum(c.contribution for c in cs[0].components)))


def test_missing_signals_renormalize_weights_and_are_reported(synth):
    ctx, feats, _ = synth_ctx(synth, with_words=False)                                       # sin transcripción
    scorer = SmartScorer(feats, use_visual=False)
    cs, _ = find_clips(ctx, scorer)
    info = scorer.info()
    assert info["effective"] == "smart" and any("transcripción" in m for m in info["missing"]) and any("imagen" in m for m in info["missing"])
    assert "arco_narrativo" not in {c.name for c in cs[0].components}
    assert sum(c.weight for c in cs[0].components if c.weight > 0) == pytest.approx(1.0)


def test_nearby_candidates_are_not_both_selected():
    from tests.helpers import make_ctx
    ctx = make_ctx(200, [(50, 52), (60, 62), (120, 122)],
                   [seg("No puede ser increíble mierda!", 50, 2.5), seg("Dios mío qué locura, no!", 60, 2.5), seg("Joder vaya locura tío!", 120, 2.5)])
    scorer = SmartScorer(features=None, use_visual=False)
    cs, chosen = find_clips(ctx, scorer)
    cl_chosen = select_candidates(cs, ctx)
    anchors = sorted(c.anchor for c in chosen)
    assert all(b - a >= 20 for a, b in zip(anchors, anchors[1:]))
    assert len(cl_chosen) >= len(chosen)


# --- evaluación y valoraciones humanas ------------------------------------------------------------------------
def test_split_is_deterministic_and_covers_both_sets():
    fps = [f"{i:016x}" for i in range(16)]
    assert [es.split_for(f) for f in fps] == [es.split_for(f) for f in fps]
    assert {es.split_for(f) for f in fps} == {"dev", "test"}


def test_human_labels_match_by_overlap_and_unrated_clips_stay_unrated():
    labels = [{"start": 100, "end": 120, "rating": "good", "id": "a"}, {"start": 300, "end": 320, "rating": "bad", "id": "b"}]
    assert es.label_for((102, 119), labels) == "good" and es.label_for((299, 322), labels) == "bad"
    assert es.label_for((200, 220), labels) is None and es.label_for((115, 160), labels) is None   # IoU < 0.3: no se inventa


def test_tuning_is_disabled_and_explains_why(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    (tmp_path / "reviews").mkdir()
    (tmp_path / "reviews" / "x.ratings.json").write_text(json.dumps({"fingerprint": "x", "ratings": {
        "j:clip_01": {"rating": "good", "start": 1, "end": 9}, "j:clip_02": {"rating": "good", "start": 20, "end": 29}}}))
    t = es.tuning_readiness(cfg)
    assert t["enabled"] is False and "DESHABILITADO" in t["message"] and "falsos negativos" in t["message"]
    r = es.ratings_report(cfg)
    assert "NO CONCLUYENTE" in r["verdict"] and r["modes"]["smart"]["rated"] == 0 and r["modes"]["classic"]["rated"] == 2


def test_ratings_are_grouped_by_the_detector_that_generated_them(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    (tmp_path / "reviews").mkdir()
    ratings = {f"s{i}:c": {"rating": r, "start": 0, "end": 9, "algorithm": {"settings": {"detector": d}}}
               for i, (d, r) in enumerate([("smart", "good"), ("smart", "bad"), ("classic", "good"), ("classic", "edit")])}
    (tmp_path / "reviews" / "y.ratings.json").write_text(json.dumps({"fingerprint": "y", "ratings": ratings}))
    r = es.ratings_report(cfg)["modes"]
    assert (r["smart"]["good"], r["smart"]["bad"], r["classic"]["good"], r["classic"]["edit"]) == (1, 1, 1, 1)


# --- parámetros de la app y endpoint de evaluación -----------------------------------------------------------
@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"\x00" * 64)
    return p


def make_manager(tmp_path):
    return JobManager(Settings(data_dir=tmp_path / "data"), runners={"process": lambda j, c: threading.Event().wait(2), "edit": lambda j, c: None, "vod": lambda j, c: None})


def test_detector_is_validated_and_part_of_the_duplicate_key(tmp_path, video):
    m = make_manager(tmp_path)
    classic = m.submit_process(str(video), {"detector": "classic"})
    smart = m.submit_process(str(video), {"detector": "smart", "smart_visual": False})      # mismo video, otro detector: no es duplicado
    assert classic.params["detector"] == "classic" and smart.params["detector"] == "smart" and smart.params["smart_visual"] is False
    with pytest.raises(ApiError, match="ya está en la cola"):
        m.submit_process(str(video), {"detector": "smart", "smart_visual": False})
    with pytest.raises(ApiError, match="Detección desconocida"):
        m.submit_process(str(video), {"detector": "magia"})
    assert m._clean_params({})["detector"] == "classic"                                      # por defecto: la clásica (línea base)
    m.shutdown(timeout=0.1)


def test_settings_reject_an_unknown_detector():
    with pytest.raises(InvalidInputError, match="Detector desconocido"):
        Settings(detector="magia").validate()
    Settings(detector="smart").validate()


# --- integración real (FFmpeg + Whisper): Smart produce clips y reutiliza la transcripción ----------------------
SAMPLE = REPO_ROOT / "data" / "samples" / "synthetic.mp4"


@pytest.mark.integration
def test_smart_pipeline_produces_real_clips_and_reuses_the_transcript(tmp_path):
    from streamclipper.pipeline import run_pipeline
    if not SAMPLE.exists():
        pytest.skip("falta data/samples/synthetic.mp4 (ver tools/make_synthetic.py)")
    data = tmp_path / "data"
    first = run_pipeline(str(SAMPLE), Settings(data_dir=data, language="en"))
    assert first["status"] == "completed" and first["detector"]["effective"] == "clásico"
    t0 = time.monotonic()
    second = run_pipeline(str(SAMPLE), Settings(data_dir=data, language="en", detector="smart", reuse_transcript=True))
    assert second["status"] == "completed", second["error"]
    assert second["transcript_reused_from"] == first["job_id"]                              # no se retranscribió
    assert second["stages"]["transcribir"]["seconds"] < 2 and time.monotonic() - t0 < 60
    assert second["detector"]["requested"] == "smart" and second["detector"]["effective"] == "smart"
    assert second["clips"] and all(Path(second["job_dir"], c["file"]).is_file() if not Path(c["file"]).is_absolute() else Path(c["file"]).is_file()
                                   for c in second["clips"])
    smart_c = [c for c in second["candidates"]]
    assert any(x["weight"] < 0 for x in smart_c[0]["components"]) and any(x["name"] == "energia_contexto" for x in smart_c[0]["components"])
    assert first["candidates"][0]["components"][0]["name"] == "intensidad_audio"            # el clásico no cambió

import random
import sys

import pytest

from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.core.memory import ResourceMonitor, own_memory_mb
from streamclipper.export import pick_alternatives
from streamclipper.jobs import resolve_job
from streamclipper.media.ffmpeg import run_cancellable
from streamclipper.models import Candidate, ScoreComponent, Segment, Transcript, Word
from streamclipper.pipeline import new_job_dir
from streamclipper.report import build_summary, render_markdown
from streamclipper.selection.candidates import build_window
from tests.helpers import make_ctx, sentence


# --- límites de frase --------------------------------------------------------------------------
def _t(*segs):
    return Transcript("en", list(segs))


def test_boundaries_flags_cut_inside_sentence_and_word():
    t = _t(sentence("this sentence is going to be cut in half here", 10, 6))
    assert check_boundaries(t, 12.0, 30.0)["errors"]          # inicio dentro de la frase
    assert check_boundaries(t, 0.0, 13.0)["errors"]           # fin dentro de la frase
    assert not check_boundaries(t, 9.8, 16.4)["errors"]       # límites en el hueco


def test_boundaries_semantic_warnings():
    t = _t(sentence("and then he just", 10, 2), Segment(12.5, 14, "ran away", [Word(12.5, 14, "ran")]))
    b = check_boundaries(t, 9.9, 14.2)
    assert not b["errors"]
    assert len(b["warnings"]) == 2  # primera en minúscula y última sin puntuación


def test_windows_never_cut_sentences_even_with_tiny_gaps():
    """Frases pegadas (huecos de 0.05-0.3 s) y anclas aleatorias: el margen no debe invadir la frase vecina."""
    rng = random.Random(1)
    for _ in range(40):
        segs, t = [], 0.5
        while t < 150:
            d = rng.uniform(1.5, 9.0)
            segs.append(sentence("word " * rng.randint(3, 12) + "end.", t, d))
            t += d + rng.choice([0.05, 0.1, 0.3, 0.8, 2.0])
        dur = 160
        ctx = make_ctx(dur, [], segs)
        tr = Transcript("en", segs)
        for anchor in (rng.uniform(3, dur - 3) for _ in range(8)):
            s, e = build_window(anchor, ctx)
            assert 0 <= s < e <= dur
            assert not check_boundaries(tr, s, e)["errors"], (anchor, s, e)
            assert ctx.settings.min_clip - 1e-6 <= e - s <= ctx.settings.max_clip + 1e-6


def test_hard_trim_still_lands_between_sentences():
    segs = [sentence("a long monologue " * 3 + "ends.", 5 + i * 7, 6.5) for i in range(12)]
    ctx = make_ctx(120, [], segs, Settings(pre_context=30, post_context=30, max_clip=25))
    s, e = build_window(45, ctx)
    assert not check_boundaries(Transcript("en", segs), s, e)["errors"]
    assert e - s <= 25


# --- memoria -----------------------------------------------------------------------------------
def test_own_memory_is_reported():
    own = own_memory_mb()
    assert own and own[0] > 10 and own[1] >= own[0] - 1


def test_resource_monitor_sees_child_process():
    with ResourceMonitor(interval=0.1) as m:
        # sys._base_executable: el python.exe de un venv es un lanzador que crea otro proceso hijo
        run_cancellable([sys._base_executable, "-c", "import time; x=b'x'*60_000_000; time.sleep(1.2)"])
    s = m.summary()
    assert s["python_peak_mb"] > 10
    assert s["ffmpeg_peak_mb"] > 20  # el hijo reservó 60 MB
    assert s["total_peak_mb_approx"] >= s["python_peak_mb"]
    assert s["system_min_available_gb"] is not None


# --- reporte / jobs / alternativos -------------------------------------------------------------
def _fake_result():
    cand = {"rank": 1, "start": 10.0, "end": 30.0, "duration": 20.0, "anchor": 15.0, "score": 0.7,
            "triggers": ["audio"], "explanation": "intensidad_audio 0.90x0.30", "text": "hola", "components": []}
    clip = {"name": "clip_01", "rank": 1, "file": "x.mp4", "start": 10.0, "end": 30.0, "duration": 20.0,
            "score": 0.7, "validation": {"ok": True, "info": {}},
            "boundaries": {"errors": [], "warnings": ["ojo"]}}
    return {"job_id": "j", "status": "completed", "source": "v.mp4", "error": None, "warnings": ["w1"],
            "settings": {"whisper_model": "base", "whisper_compute_type": "int8"},
            "media": {"duration": 100.0}, "transcript": {"language": "en", "words": 5, "sentences": 2},
            "stages": {"transcribir": {"seconds": 20, "ok": True}, "analizar_audio": {"seconds": 1, "ok": True},
                       "seleccionar": {"seconds": 0.1, "ok": True}, "renderizar": {"seconds": 9, "ok": True}},
            "total_seconds": 31, "resources": {"python_peak_mb": 800}, "candidates": [cand], "selected": [1],
            "clips": [clip], "alt_clips": []}


def test_report_contains_required_fields():
    s = build_summary(_fake_result())
    assert s["video_duration_s"] == 100.0 and s["whisper_model"] == "base"
    assert s["timings_s"]["transcription"] == 20 and s["timings_s"]["analysis"] == 1.1
    assert s["timings_s"]["rendering"] == 9 and s["timings_s"]["transcription_realtime_factor"] == 0.2
    assert s["clips_generated"] == 1 and s["candidates"][0]["selected"] is True
    assert s["clips"][0]["start"] == 10.0 and s["warnings"] == ["w1"]
    md = render_markdown(s)
    assert "Modelo Whisper" in md and "ojo" in md and "no una probabilidad" in md


def test_job_dirs_are_separate_per_model(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    a, da = new_job_dir(cfg, "base")
    b, db = new_job_dir(cfg, "small", label="Test 1!")
    assert da != db and "base" in a and "small-test1" in b


def test_resolve_job_by_prefix_and_ambiguity(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    for n in ("20260101-base-aaaa", "20260101-small-bbbb"):
        d = cfg.jobs_dir / n
        d.mkdir(parents=True)
        (d / "result.json").write_text("{}")
    assert resolve_job("20260101-base", cfg).name == "20260101-base-aaaa"
    with pytest.raises(InvalidInputError):
        resolve_job("20260101", cfg)
    with pytest.raises(InvalidInputError):
        resolve_job("nope", cfg)


def _cand(rank, a, b, score=0.6, intensity=0.8, text="alpha bravo charlie delta echo foxtrot"):
    return Candidate(a, b, (a + b) / 2, ["audio"], [ScoreComponent("intensidad_audio", intensity, 0.3),
                     ScoreComponent("reaccion_verbal", 0.0, 0.25)], score, text, rank)


def test_pick_alternatives_skips_overlaps_and_weak_signals():
    cands = [_cand(1, 0, 20), _cand(2, 10, 30), _cand(3, 50, 70, text="golf hotel india juliet kilo lima"),
             _cand(4, 100, 120, intensity=0.1, text="mike november oscar papa quebec romeo"),
             _cand(5, 150, 170, text="sierra tango uniform victor whiskey xray")]
    out = pick_alternatives(cands, taken=[cands[0]], count=5, max_overlap=0.15, text_trigger=0.5)
    assert [c.rank for c in out] == [3, 5]  # 2 solapa con 1; 4 no tiene señal propia

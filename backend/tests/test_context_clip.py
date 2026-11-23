"""Clips con contexto narrativo: construcción de intervalos, bloques, solapes, regeneración y compatibilidad.

Las pruebas unitarias usan transcripciones y audio sintéticos de verdad conocida (el evento, el planteamiento y el desenlace están en
tiempos conocidos). Las de integración usan FFmpeg y Whisper reales.
"""
import hashlib
import json
import re
import threading
from pathlib import Path

import pytest

from streamclipper import auto
from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.app.manager import JobManager
from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.models import Candidate, Segment, Transcript, Word
from streamclipper.pipeline import settings_from_result
from streamclipper.selection.context_clip import PRESETS, build_window, plan_windows
from streamclipper.webutil import ApiError
from tests.helpers import make_ctx, sentence as S


def build(dur, loud, sents, anchor, cwin, preset="tiktok"):
    cfg = Settings(clip_preset=preset)
    ctx = make_ctx(dur, loud, sents, cfg)
    c = Candidate(start=cwin[0], end=cwin[1], anchor=anchor, triggers=["audio"])
    c.rank = 1
    return plan_windows([c], ctx)[0], ctx


# escena del ejemplo: grito en ~120, la jugada empieza en 105, la conversación sigue hasta ~130 y después hay una pausa
SCENE = [S("hola chat bienvenidos", 88, 2.5), S("mira voy a saltar desde acá", 105, 3.5), S("si lo logro es increíble", 109.3, 3.2),
         S("tres dos uno", 113.5, 2.5), S("¡No puede ser! ¡No puede ser!", 117, 3.5), S("¡Lo logré!", 121.2, 1.8), S("jajaja qué locura", 124, 3),
         S("chat viste eso", 127.8, 2.2), S("bueno sigamos con lo otro", 134, 2.5)]
CHAT = [S(f"frase continua número {i} de la charla", 60 + i * 3.3, 3.0) for i in range(30)]       # 60–160 s sin pausas
CHAT[18] = S("¡No puede ser! ¡increíble!", 60 + 18 * 3.3, 3.0)


def test_a_10_second_candidate_becomes_a_contextual_clip_of_about_40_seconds():
    r, ctx = build(260, [(119, 123)], CHAT, 120, (112, 130))
    ev = r.context["event"]
    assert ev[1] - ev[0] < 15                                                    # el evento detectado es corto (≈10 s)
    assert 30 <= r.end - r.start <= 50                                           # el clip construido ronda los 40 s
    assert r.start < ev[0] - 5 and r.end > ev[1] + 5                             # hay planteamiento antes y desenlace después
    assert r.context["confidence"] == "media" and r.context["start_kind"] == "limite"   # conversación continua: tope, no pausa


def test_start_coincides_with_the_natural_pause_where_the_situation_begins():
    r, _ = build(200, [(117, 121)], SCENE, 119, (110, 128))
    assert 104.0 <= r.start <= 105.0                                             # la jugada empieza en 105: no en 118 ni en 88
    assert r.context["start_kind"] == "pausa" and "pausa natural" in r.context["start_reason"]
    assert not any(s.start >= 88 and s.end <= 93 for s in [s for s in SCENE if s.start >= r.start])   # no arrastra la escena anterior


def test_end_keeps_the_outcome_and_waits_for_the_conversation_to_finish():
    r, _ = build(200, [(117, 121)], SCENE, 119, (110, 128))
    assert r.end >= 130.0                                                        # «chat viste eso» (el cierre) queda incluido
    assert r.end < 134                                                           # y no invade la escena siguiente
    assert r.context["end_kind"] == "pausa" and r.context["confidence"] == "alta"
    assert not r.context["estimated"]


def test_sentences_are_never_cut_in_any_scenario():
    for sents, loud, anchor, cwin in ((SCENE, [(117, 121)], 119, (110, 128)), (CHAT, [(119, 123)], 120, (112, 130))):
        for preset in PRESETS:
            r, ctx = build(260, loud, sents, anchor, cwin, preset)
            assert check_boundaries(ctx.transcript, r.start, r.end)["errors"] == []


def test_conversation_with_silences_stops_at_the_natural_pause_and_adds_no_dead_air():
    g = [S("primera idea sobre el juego", 70, 3.0), S("segunda idea de otra cosa", 76.5, 3.0), S("mira esto que raro", 83, 2.5), S("¡No puede ser!", 87, 2.5),
         S("qué locura", 90.2, 2.0), S("bueno ya está", 98, 2.0)]
    r, _ = build(200, [(87, 90)], g, 88, (80, 96))
    assert r.start >= 80.5                                                       # no incluye las ideas anteriores (hay pausas ≥ 2,5 s)
    assert r.end <= 97                                                           # ni el silencio ni lo que viene después de la pausa
    assert r.context["confidence"] == "alta"
    assert r.end - r.start < 30                                                  # situación corta y completa: no se rellena hasta 30 s
    assert any("situación completa" in n for n in r.notes)


def test_short_complete_situations_are_allowed_and_documented():
    r, _ = build(200, [(117, 121)], SCENE, 119, (110, 128), "extendido")
    assert r.end - r.start < PRESETS["extendido"]["min"]                         # 26 s < 45 s del preset, y está bien
    assert any("no se agrega relleno" in n for n in r.notes)


def test_candidates_near_the_start_and_end_of_the_material_are_clamped():
    near_start = [S("hola hola chat", 1.0, 2.0), S("¡Mirad esto no puede ser!", 5.0, 3.0), S("qué fuerte", 8.5, 2.0)]
    r, _ = build(100, [(5, 8)], near_start, 6, (0, 16))
    assert r.start >= 0 and r.context["start_kind"] == "material"
    near_end = [S("a ver si sale", 92, 2.5), S("¡No puede ser increíble!", 95, 3.0)]
    r, _ = build(100, [(95, 98)], near_end, 96, (88, 100))
    assert r.end <= 100 and r.context["end_kind"] == "material"


def test_incomplete_or_missing_transcription_degrades_to_a_flagged_estimate():
    r, _ = build(200, [(100, 104)], [], 101, (94, 112))                          # sin una sola palabra
    assert r.context["estimated"] and r.context["confidence"] == "baja"           # sin una sola palabra: todo es estimación y se dice
    assert r.end > r.start and r.start >= 0
    empty = [Segment(100, 104, "", []), Segment(104.5, 108, "   ", [])]            # frases vacías (transcripción incompleta)
    r2, _ = build(200, [(100, 104)], empty, 101, (94, 112))
    assert r2.end > r2.start and r2.context["confidence"] in ("baja", "media")


def test_backup_expansion_is_conservative_and_never_pads_dead_air():
    # silencio relativo antes y después del evento (curva muy por debajo del nivel habitual): no hay nada que agregar
    cfg = Settings(clip_preset="tiktok")
    ctx = make_ctx(200, [(100, 104)], [], cfg)
    import numpy as np
    ctx.curve.excess[:] = -20.0
    ctx.curve.excess[ctx.curve.index(100):ctx.curve.index(104)] = 12.0
    c = Candidate(start=94, end=112, anchor=101, triggers=["audio"])
    c.rank = 1
    r = plan_windows([c], ctx)[0]
    assert r.end - r.start < 20 and not np.isnan(r.start)                       # no se rellena con silencio hasta 30 s


def test_presets_orient_the_limits_without_forcing_a_duration():
    lens = {p: build(260, [(119, 123)], CHAT, 120, (112, 130), p)[0] for p in PRESETS}
    d = {p: r.end - r.start for p, r in lens.items()}
    assert d["cortos"] <= 30.5 and d["cortos"] < d["tiktok"] <= d["extendido"] <= 90.5
    assert 30 <= d["tiktok"] <= 60
    assert all(r.bounds == (PRESETS[p]["min"], PRESETS[p]["max"]) for p, r in lens.items())


def test_exceptional_length_needs_a_continuous_conversation_with_natural_boundaries():
    # conversación densa de ~75 s con pausas largas en ambos extremos: se permite pasar de 60 s (hasta 90)
    dense = [S(f"esto es una frase bastante larga número {i} sobre la jugada de hoy", 100 + i * 3.1, 3.0) for i in range(24)]   # 100–174 s
    dense[12] = S("¡No puede ser! ¡increíble!", 100 + 12 * 3.1, 3.0)
    r, _ = build(300, [(137, 141)], dense, 138, (130, 148), "tiktok")
    assert r.end - r.start <= 60.5                                               # sin pausas naturales en los extremos: no se justifica
    bounded = [S("fin de otra escena", 90, 2.0)] + dense + [S("otra escena distinta", 185, 2.0)]
    r2, _ = build(300, [(137, 141)], bounded, 138, (130, 148), "tiktok")
    assert r2.end - r2.start <= 90.5
    if r2.end - r2.start > 60:
        assert r2.context["exceptional"] and any("excepcional" in n for n in r2.notes)


def test_every_decision_is_recorded_separately_and_original_timestamps_are_untouched():
    r, ctx = build(200, [(117, 121)], SCENE, 119, (110, 128))
    x = r.context
    assert x["original"] == [110, 128]                                           # el intervalo del candidato no se altera
    assert x["event"][0] <= 119 <= x["event"][1] and x["contextual"][0] <= x["event"][0] and x["contextual"][1] >= x["event"][1]
    assert x["final"][0] <= x["contextual"][0] + 0.1 and x["start_reason"] and x["end_reason"] and x["confidence"] in ("alta", "media", "baja")
    assert abs(x["final"][0] - r.start) < 0.02 and abs(x["final"][1] - r.end) < 0.02


def test_plan_is_deterministic_and_legacy_preset_keeps_the_previous_behavior():
    a, _ = build(260, [(119, 123)], CHAT, 120, (112, 130))
    b, _ = build(260, [(119, 123)], CHAT, 120, (112, 130))
    assert (a.start, a.end, a.context) == (b.start, b.end, b.context)
    assert Settings().clip_preset == "legacy"
    with pytest.raises(InvalidInputError, match="Preset de duración desconocido"):
        Settings(clip_preset="largo").validate()


# --- bloques de 20 min: el contexto cruza los límites ---------------------------------------------------------------------------
def block(index, start, end, total, segs_abs):
    """Un bloque con su transcripción en tiempo del bloque (desde dl_start) a partir de frases en tiempo absoluto."""
    dl_start, dl_end = max(0, start - 30), end + 30
    segs = [Segment(s.start - dl_start, s.end - dl_start, s.text, [Word(w.start - dl_start, w.end - dl_start, w.text, w.prob) for w in s.words])
            for s in segs_abs if dl_start <= s.start and s.end <= dl_end]
    return {"start": start, "end": end, "dl_start": dl_start, "dl_end": dl_end, "last": index == total - 1, "transcript": Transcript("es", segs)}


def test_context_is_gathered_across_a_block_boundary_without_duplicates():
    # conversación continua que cruza el límite de 1200 s: el planteamiento (1176–1198) cae en el bloque 0 y la reacción en el bloque 1
    ab = [S("otra cosa que no tiene que ver", 1150, 3.0)] + [S(f"voy a intentar esto paso {i} del plan", 1176 + i * 3.2, 3.0) for i in range(7)] + [
          S("¡No puede ser increíble!", 1198.5, 4.0), S("¡Lo logré!", 1203.0, 2.0), S("qué locura chat", 1205.5, 3.0), S("bueno sigamos", 1214.5, 2.5)]
    blocks = [block(0, 0, 1200, 2, ab), block(1, 1200, 2400, 2, ab)]
    merged = auto.merge_transcripts(blocks, 1140, 1240)
    texts = [s.text for s in merged.segments]
    assert len(texts) == len(set(texts)) == len(ab)                              # cada frase una sola vez (zona de contexto duplicada)
    assert [s.start for s in merged.segments] == sorted(s.start for s in merged.segments)
    # y el constructor usa ese contexto continuo: empieza en el planteamiento del bloque anterior y termina en el desenlace del siguiente
    cfg = Settings(clip_preset="tiktok")
    cand = Candidate(start=1190, end=1210, anchor=1200, triggers=["audio"])
    w = build_window(cand, merged.segments, None, 2400, cfg, "tiktok", event=(1198, 1206))
    assert w.start <= 1180 and w.end >= 1208                                     # el planteamiento (bloque 0) y el desenlace (bloque 1)
    assert w.context["start_kind"] in ("pausa", "limite") and check_boundaries(merged, w.start, w.end)["errors"] == []


def test_blocks_keep_absolute_times_and_ownership_by_core():
    ab = [S("frase en el borde del bloque", 1199.0, 2.0)]                        # punto medio 1200: pertenece al bloque 1
    blocks = [block(0, 0, 1200, 2, ab), block(1, 1200, 2400, 2, ab)]
    merged = auto.merge_transcripts(blocks, 1190, 1210)
    assert len(merged.segments) == 1 and merged.segments[0].start == pytest.approx(1199.0)


def test_clips_that_overlap_after_expansion_are_resolved_before_rendering():
    def item(i, anchor, score):
        return {"id": f"0:{i}", "start": anchor - 8, "end": anchor + 8, "anchor": anchor, "score": score, "block": 0}
    plan = [item(1, 100, 0.9), item(2, 130, 0.8), item(3, 400, 0.7)]             # anclas a 30 s: pasan la deduplicación global
    reserve = [item(4, 700, 0.6)]
    expanded = {"0:1": (80, 140), "0:2": (110, 170), "0:3": (390, 430), "0:4": (690, 730)}    # al expandirse, 1 y 2 se pisan
    res = auto.resolve_overlaps(plan, reserve, lambda c: expanded[c["id"]])
    ids = [c["id"] for c in res["accepted"]]
    assert ids == ["0:1", "0:3", "0:4"] and [d["id"] for d in res["dropped"]] == ["0:2"] and res["backfilled"] == 1
    assert all("provisional" in c for c in res["accepted"])
    near = auto.resolve_overlaps([item(1, 100, 0.9), item(2, 150, 0.8)], [], lambda c: {"0:1": (90, 130), "0:2": (132, 170)}[c["id"]])
    assert [c["id"] for c in near["accepted"]] == ["0:1"]                        # a menos de 3 s entre sí: el mismo momento


# --- aplicación: parámetros, compatibilidad con sesiones antiguas y modos ----------------------------------------------------------
def test_app_defaults_to_the_tiktok_preset_and_validates_it(tmp_path):
    m = JobManager(Settings(data_dir=tmp_path / "d"), runners={k: (lambda j, c: threading.Event().wait(1)) for k in ("process", "edit", "vod", "context")},
                   inspector=lambda u: {"duration": 7200, "title": "t", "acceptable": True, "auto": {"available": True}})
    assert m._clean_params({})["clip_preset"] == "tiktok"
    assert m._clean_params({"clip_preset": "cortos"})["clip_preset"] == "cortos"
    with pytest.raises(ApiError, match="Duración de clips desconocida"):
        m._clean_params({"clip_preset": "enorme"})
    v = tmp_path / "v.mp4"
    v.write_bytes(b"0" * 64)
    a = m.submit_process(str(v), {"clip_preset": "cortos"})
    b = m.submit_process(str(v), {"clip_preset": "extendido"})                   # otro preset: otro resultado, no es duplicado
    assert a.params["clip_preset"] == "cortos" and b.params["clip_preset"] == "extendido"
    kick = "https://kick.com/xqc/videos/01a1fae2-a5c0-754b-888b-c05cc64e9741"
    assert m.submit_auto(kick, {"target": 10}, {"clip_preset": "extendido"}).params["clip_preset"] == "extendido"
    assert m.submit_context("s", "clip_01", "tiktok").kind == "context"
    with pytest.raises(ApiError):
        m.submit_context("s", "clip_01", "gigante")
    m.shutdown(timeout=0.1)


def test_sessions_from_older_versions_keep_the_previous_behavior(tmp_path):
    from streamclipper.app.sessions import session_detail
    cfg = Settings(data_dir=tmp_path / "d")
    d = cfg.jobs_dir / "20250101-000000-small-viejo"
    (d / "out").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "job_id": d.name, "source": str(d / "v.mp4"), "status": "completed", "warnings": [], "selected": [1], "source_fingerprint": "x",
        "settings": {"whisper_model": "small", "language": "es", "data_dir": str(cfg.data_dir)}, "media": {"duration": 120.0},
        "candidates": [{"rank": 1, "start": 8, "end": 24, "anchor": 15, "score": 0.5, "triggers": [], "text": "x", "components": [
            {"name": n, "value": 0.5, "weight": 0.3, "contribution": 0.15, "note": "", "window": None, "evidence": []} for n in ("intensidad_audio", "dinamica_audio", "reaccion_verbal", "frases", "contexto", "duracion", "calidad_transcripcion")]}],
        "clips": [{"name": "clip_01", "rank": 1, "start": 10.0, "end": 20.0, "duration": 10.0, "score": 0.5, "file": str(d / "out" / "clip_01.mp4")}]}))
    (d / "transcript.json").write_text(json.dumps({"language": "es", "segments": [], "evidence_version": 2}))
    assert settings_from_result(json.loads((d / "result.json").read_text()), data_dir=cfg.data_dir).clip_preset == "legacy"
    det = session_detail(cfg, d.name)
    c = det["clips"][0]
    assert c["context"] is None and c["proposal"] is None and c["versions"][0]["duration"] == 10.0     # el clip viejo se muestra como siempre


# --- integración real: FFmpeg + Whisper sobre un video con momentos conocidos ---------------------------------------------------------
SAMPLE = REPO_ROOT / "data" / "samples" / "synthetic.mp4"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def srt_times(p: Path) -> list[tuple[float, float]]:
    out = []
    for m in re.finditer(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", p.read_text(encoding="utf-8")):
        a = [int(x) for x in m.groups()]
        out.append((a[0] * 3600 + a[1] * 60 + a[2] + a[3] / 1000, a[4] * 3600 + a[5] * 60 + a[6] + a[7] / 1000))
    return out


@pytest.mark.integration
def test_real_pipeline_builds_contextual_clips_and_regeneration_keeps_the_original(tmp_path):
    from streamclipper.edit import EditParams, render_edit
    from streamclipper.expand import prepare_context
    from streamclipper.pipeline import run_pipeline
    if not SAMPLE.exists():
        pytest.skip("falta data/samples/synthetic.mp4")
    data = tmp_path / "data"
    legacy = run_pipeline(str(SAMPLE), Settings(data_dir=data, language="en"))
    ctxd = run_pipeline(str(SAMPLE), Settings(data_dir=data, language="en", clip_preset="tiktok", reuse_transcript=True))
    assert legacy["status"] == ctxd["status"] == "completed" and len(ctxd["clips"]) == len(legacy["clips"]) == 2
    for old, new in zip(sorted(legacy["clips"], key=lambda c: c["start"]), sorted(ctxd["clips"], key=lambda c: c["start"])):
        assert new["duration"] >= old["duration"] - 0.5                        # el contexto nunca acorta
        assert new["context"]["original"] == new["selected_window"] and new["context"]["final"][0] <= new["start"] + 0.05
        assert new["boundaries"]["errors"] == []
        cues = srt_times(Path(ctxd["job_dir"]) / "out" / (new["name"] + ".srt"))
        assert cues and cues[0][0] >= 0 and cues[-1][1] <= new["duration"] + 0.2    # subtítulos dentro del clip, sin desfase
        assert any(new["start"] < ev < new["end"] for ev in (30.0, 73.0))          # el momento real sigue dentro del clip
    # regeneración desde el clip ORIGINAL (de la sesión previa): la original no cambia y la versión nueva se agrega
    job_dir = Path(legacy["job_dir"])
    clip = legacy["clips"][0]
    mp4 = Path(clip["file"])
    before = sha(mp4)
    prop = prepare_context(job_dir, clip["name"], "tiktok", Settings(data_dir=data))
    assert prop["source"] == "own" and prop["duration"] >= prop["original"]["duration"] - 0.5 and prop["context"]["start_reason"]
    meta = render_edit(job_dir, clip["name"], EditParams(start=prop["start"], end=prop["end"], note=prop["context"]), Settings(data_dir=data))
    assert sha(mp4) == before and mp4.is_file()                                # el original no se tocó
    ver = job_dir / "edits" / clip["name"] / "v001"
    assert (ver / "clip.mp4").is_file() and meta["context_note"]["confidence"] and meta["duration"] == pytest.approx(prop["end"] - prop["start"], abs=0.05)
    cues = srt_times(ver / "clip.srt")
    assert cues and cues[-1][1] <= meta["duration"] + 0.2

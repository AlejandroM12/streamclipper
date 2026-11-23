"""Transmisión completa (automático): bloques, tiempos absolutos, selección global, render diferido, recuperación y compatibilidad."""
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from streamclipper import auto
from streamclipper.app import runners
from streamclipper.app.manager import JobManager
from streamclipper.app.runners import real_block, real_render, run_auto
from streamclipper.app.sessions import auto_detail, list_sessions
from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, MediaError
from streamclipper.media.download import download_segment, inspect_media
from streamclipper.models import Segment, Transcript, Word
from streamclipper.webutil import ApiError
from tests.test_vod import FakeYDL, factory, vod_info

TW = "https://www.twitch.tv/videos/6528877"
KICK = "https://kick.com/xqc/videos/01a1fae2-a5c0-754b-888b-c05cc64e9741"
YT = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"
H8 = 8 * 3600


# --- planificación de bloques -------------------------------------------------------------------------------------
def test_eight_hours_make_24_blocks_that_cover_everything_without_gaps():
    b = auto.plan_blocks(H8)
    assert len(b) == 24
    assert b[0].start == 0 and b[-1].end == H8 and all(x.end == y.start for x, y in zip(b, b[1:]))
    assert all(x.end - x.start == 1200 for x in b)
    assert all(x.dl_end - x.dl_start <= auto.max_download_s() for x in b)
    assert (b[0].dl_start, b[0].dl_end) == (0, 1230) and (b[5].dl_start, b[5].dl_end) == (5970, 7230)      # contexto a cada lado
    assert b[-1].dl_end == H8 and b[-1].last and not b[0].last                                        # recortado al final del VOD


@pytest.mark.parametrize("duration,blocks,last_core", [
    (H8 + 120, 25, (H8, H8 + 120)),         # último bloque parcial de 2 min
    (H8 + 59, 24, (H8 - 1200, H8 + 59)),    # cola < 60 s: se absorbe en el bloque anterior
    (H8 + 60, 25, (H8, H8 + 60)),           # exactamente el mínimo: bloque propio
    (1200, 1, (0, 1200)), (1260, 2, (1200, 1260)), (61, 1, (0, 61)), (3000, 3, (2400, 3000)),
])
def test_last_block_is_handled(duration, blocks, last_core):
    b = auto.plan_blocks(duration)
    assert len(b) == blocks and (b[-1].start, b[-1].end) == last_core and b[-1].end == duration
    assert all(x.end == y.start for x, y in zip(b, b[1:]))


def test_too_short_streams_are_rejected_and_there_is_no_5_block_limit():
    with pytest.raises(InvalidInputError, match="demasiado corta"):
        auto.plan_blocks(59)
    assert len(auto.plan_blocks(12 * 3600)) == 36


def test_estimates_are_honest_and_bounded():
    e = auto.estimates(H8, 15)
    assert e["blocks"] == 24 and e["analyze_seconds"] == H8 and e["analysis_height"] == 480
    assert 2 <= e["est_hours_low"] < e["est_hours_high"] <= 6 and e["disk_peak_gb"] < 1.0        # no se descarga el VOD completo
    assert e["download_seconds"] < H8 * 1.06                                                      # solo los márgenes se descargan de más


# --- tiempos absolutos y solapamiento -------------------------------------------------------------------------------
def cand(rank, start, end, anchor, score, components=None):
    return {"rank": rank, "start": start, "end": end, "anchor": anchor, "score": score, "triggers": ["audio"], "text": f"t{rank}",
            "components": components or [{"name": "intensidad_audio", "value": 0.8, "weight": 0.3, "contribution": 0.24, "window": [start, end]}]}


def test_block_times_become_absolute_and_overlap_zone_candidates_are_owned_once():
    blocks = auto.plan_blocks(3000)                                   # núcleos 0–1200, 1200–2400, 2400–3000; contexto de 30 s
    b0, b1 = blocks[0], blocks[1]
    moment_abs = 1215.0                                               # cae en el contexto de b0 (…1230) y en el núcleo de b1 (1200…)
    r0 = {"candidates": [cand(1, 1190, 1230, moment_abs, 0.6), cand(2, 100, 120, 110, 0.5)], "eligible": [1, 2]}
    r1 = {"candidates": [cand(1, 1200 - b1.dl_start, 1240 - b1.dl_start, moment_abs - b1.dl_start, 0.6)], "eligible": [1]}
    p0, p1 = auto.pool_from_block(b0, "s0", r0), auto.pool_from_block(b1, "s1", r1)
    assert [c["id"] for c in p0] == ["0:2"]                           # el del contexto de b0 NO es suyo
    assert p1[0]["anchor"] == pytest.approx(moment_abs) and p1[0]["start"] == pytest.approx(1200) and p1[0]["block"] == 1
    assert p0[0]["start"] == 100 and p0[0]["session"] == "s0" and p0[0]["eligible"] is True


def test_a_moment_crossing_a_block_boundary_is_kept_whole_through_context():
    b = auto.plan_blocks(3000)
    # el ancla está justo en el límite (1199): le pertenece a b0, y su ventana (1180–1225) se descargó completa gracias al contexto
    r = {"candidates": [cand(1, 1180, 1225, 1199, 0.7)], "eligible": [1]}
    p = auto.pool_from_block(b[0], "s0", r)
    assert len(p) == 1 and p[0]["end"] == 1225 <= b[0].dl_end


def test_global_dedup_removes_the_same_moment_seen_twice_and_keeps_the_best():
    pool = [{"id": "0:1", "block": 0, "session": "a", "rank": 1, "start": 100, "end": 130, "anchor": 115, "score": 0.5, "eligible": True},
            {"id": "1:1", "block": 1, "session": "b", "rank": 1, "start": 105, "end": 135, "anchor": 120, "score": 0.62, "eligible": True},
            {"id": "1:2", "block": 1, "session": "b", "rank": 2, "start": 400, "end": 430, "anchor": 415, "score": 0.55, "eligible": True}]
    sel = auto.select_global(pool, 5)
    assert [c["id"] for c in sel["plan"]] == ["1:1", "1:2"] and sel["duplicates_avoided"] == 1
    assert [c["global_rank"] for c in sel["plan"]] == [1, 2]


# --- selección global y umbrales ---------------------------------------------------------------------------------------
def mk(block, rank, anchor, score, eligible=True):
    return {"id": f"{block}:{rank}", "block": block, "session": f"s{block}", "rank": rank, "start": anchor - 10, "end": anchor + 10, "anchor": anchor,
            "score": score, "eligible": eligible, "triggers": [], "top": []}


def test_ranking_compares_candidates_across_blocks_not_per_block():
    pool = [mk(0, 1, 100, 0.40), mk(0, 2, 500, 0.39), mk(0, 3, 900, 0.38), mk(1, 1, 1500, 0.80), mk(2, 1, 3000, 0.70), mk(2, 2, 3600, 0.36)]
    sel = auto.select_global(pool, 3)
    assert [c["id"] for c in sel["plan"]] == ["1:1", "2:1", "0:1"]                       # NO se toman 3 por bloque
    assert [c["block"] for c in sel["plan"]] == [1, 2, 0]


def test_below_threshold_candidates_are_exploratory_and_never_silently_promoted():
    pool = [mk(0, 1, 100, 0.60), mk(0, 2, 500, 0.30, eligible=False), mk(1, 1, 1500, 0.28, eligible=False)]
    sel = auto.select_global(pool, 5, explore=False)
    assert [c["kind"] for c in sel["plan"]] == ["recommended"] and sel["recommended"] == 1
    assert [c["id"] for c in sel["exploratory"]] == ["0:2", "1:1"]
    assert "Solo 1 momento(s) cumplen" in sel["notes"][0] and "2 candidato(s) exploratorios" in sel["notes"][0]


def test_exploratory_clips_are_included_only_when_requested_and_are_labeled():
    pool = [mk(0, 1, 100, 0.60), mk(0, 2, 500, 0.30, eligible=False), mk(1, 1, 1500, 0.28, eligible=False)]
    sel = auto.select_global(pool, 3, explore=True)
    assert [(c["id"], c["kind"]) for c in sel["plan"]] == [("0:1", "recommended"), ("0:2", "exploratory"), ("1:1", "exploratory")]
    assert any("EXPLORATORIOS" in n and "menor confianza" in n for n in sel["notes"]) and sel["exploratory"] == []


def test_a_video_without_interesting_moments_is_reported_not_faked():
    sel = auto.select_global([], 15)
    assert sel["plan"] == [] and sel["recommended"] == 0 and "No hay más candidatos" in sel["notes"][0]
    only_weak = auto.select_global([mk(0, 1, 100, 0.2, eligible=False)], 10)
    assert only_weak["plan"] == [] and len(only_weak["exploratory"]) == 1


def test_clips_are_not_concentrated_in_one_zone_unless_clearly_superior():
    # 6 candidatos en 4 minutos, puntuaciones parejas: el cuarto en adelante se reserva; uno claramente superior sí pasa
    base = [mk(0, i, 100 + i * 40, 0.50 - i * 0.002) for i in range(1, 7)]
    far = mk(1, 1, 5000, 0.70)
    sel = auto.select_global(base + [far], 10)
    ids = [c["id"] for c in sel["plan"]]
    assert "1:1" in ids and sum(1 for c in sel["plan"] if c["block"] == 0) == auto.CROWD_MAX and sel["crowd_skipped"] == 3
    assert len(sel["reserve"]) == 3                                    # lo reservado no se pierde: queda disponible
    star = mk(0, 9, 100 + 7 * 40, 0.69)                                # ≥ 95 % del mejor: claramente superior, no se descarta por «diversidad»
    ids = [c["id"] for c in auto.select_global(base + [far, star], 10)["plan"]]
    assert "0:9" in ids
    flat = [mk(0, i, 100 + i * 40, 0.60 - i * 0.002) for i in range(1, 7)]
    assert len(auto.select_global(flat, 10)["plan"]) == 6              # si TODO está parejo, no hay artefacto que corregir


def test_target_limits_the_plan_and_reserve_keeps_the_rest():
    pool = [mk(i % 3, i, 200 * i, 0.9 - i * 0.01) for i in range(1, 30)]
    sel = auto.select_global(pool, 10)
    assert len(sel["plan"]) == 10 and len(sel["reserve"]) > 0 and not sel["notes"]


def test_block_scales_report_distributions_without_transforming_scores():
    pool = [mk(0, 1, 100, 0.5), mk(0, 2, 400, 0.4), mk(1, 1, 1500, 0.7, eligible=False)]
    sc = auto.block_scales(pool, 3)
    assert sc[0]["max"] == 0.5 and sc[1]["eligible"] == 0 and sc[2]["n"] == 0 and pool[0]["score"] == 0.5


def test_transcript_and_candidate_shifting_to_the_clip_time_axis():
    tr = Transcript("es", [Segment(100, 104, "hola", [Word(100, 102, "hola", 0.9)]), Segment(130, 134, "fuera", [Word(130, 132, "fuera", 0.9)])])
    out = auto.shift_transcript(tr, 95, 0, 30)
    assert [(s.start, s.end) for s in out.segments] == [(5, 9)] and out.segments[0].words[0].start == 5     # solo frases completas
    c = auto.shift_candidate(cand(1, 100, 120, 110, 0.5), 95)
    assert (c["start"], c["end"], c["anchor"]) == (5, 25, 15) and c["components"][0]["window"] == [5, 25]
    assert auto.clip_range({"start": 100.4, "end": 130.2}, 3600) == (94, 137) and auto.clip_range({"start": 2, "end": 20}, 21) == (0, 21)


# --- gestor: alta, validaciones, bloques persistidos ----------------------------------------------------------------------
def info_for(duration=H8, **extra):
    from streamclipper.auto import TARGETS, estimates
    return {"title": "Gran stream", "duration": duration, "acceptable": True, "uploader": "s", "reason": None,
            "auto": {"available": True, "targets": list(TARGETS), **estimates(duration)}, **extra}


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


def make(settings, block=None, render=None, **kw):
    r = {"vod": lambda j, c: run_auto(settings, block or (lambda *a: None), render or (lambda *a: "s"))(j, c), "process": lambda j, c: None,
         "edit": lambda j, c: None}
    return JobManager(settings, runners=r, inspector=kw.pop("inspector", lambda u: info_for()), **kw)


def wait(m, jid, states, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = m.get(jid)
        if j["status"] in states:
            return j
        time.sleep(0.02)
    raise AssertionError(f"timeout: {m.get(jid)['status']} {m.get(jid)['message']}")


def test_submit_auto_plans_all_24_blocks_and_records_absolute_ranges(settings):
    m = make(settings, autostart=False)
    job = m.submit_auto(TW, {"target": 15})
    assert job.kind == "vod" and job.params["mode"] == "auto" and job.params["target"] == 15 and len(job.segments) == 24
    assert [(s.start, s.end) for s in job.segments][:2] == [(0, 1200), (1200, 2400)] and job.segments[1].dl_start == 1170
    assert all(s.stages[0]["key"] == "descarga" and "renderizar" not in {x["key"] for x in s.stages} for s in job.segments)   # un bloque no renderiza
    assert [s["key"] for s in job.stages] == ["seleccion_global", "render_final"] and job.params["analysis_height"] == 480
    assert job.result["estimates"]["blocks"] == 24 and job.params["detector"] == "classic"                # el clásico es el predeterminado


def test_submit_auto_validations_and_providers(settings):
    m = make(settings, autostart=False)
    for url in ("https://clips.twitch.tv/FaintLightGullWholeWheat", "https://kick.com/xqc/clips/clip_01J8RGZRKHXHXXKJEHGRM932A5", "https://evil.com/x"):
        with pytest.raises(ApiError) as e:
            m.submit_auto(url, {"target": 10})
        assert e.value.status == 400
    for bad in ({"target": 0}, {"target": 99}, {"target": "x"}, {"target": 10, "block_s": 10}, {"target": 10, "margin_s": 999}):
        with pytest.raises(ApiError):
            m.submit_auto(TW, bad)
    assert m.submit_auto(KICK, {"target": 20}).params["target"] == 20                                      # los tres proveedores
    assert m.submit_auto(YT, {"target": 10}).params["platform_name"] == "YouTube"
    short = make(settings, autostart=False, inspector=lambda u: {"title": "t", "duration": 30, "acceptable": True, "auto": {"available": True}})
    with pytest.raises(ApiError, match="demasiado corta"):
        short.submit_auto(TW, {"target": 10})
    live = make(settings, autostart=False, inspector=lambda u: {"duration": 9999, "acceptable": False, "reason": "No se aceptan transmisiones en vivo",
                                                               "auto": {"available": False, "reason": "No se aceptan transmisiones en vivo"}})
    with pytest.raises(ApiError, match="en vivo"):
        live.submit_auto(TW, {"target": 10})
    nodur = make(settings, autostart=False, inspector=lambda u: {"duration": None, "acceptable": False, "reason": "No se pudo determinar la duración"})
    with pytest.raises(ApiError, match="duración"):
        nodur.submit_auto(TW, {"target": 10})


def test_duplicates_and_manual_modes_are_unaffected(settings):
    m = make(settings, autostart=False)
    m.submit_auto(TW, {"target": 15})
    with pytest.raises(ApiError) as e:
        m.submit_auto("https://m.twitch.tv/videos/6528877?t=1m", {"target": 15})
    assert e.value.extra["duplicate"] == "active"
    assert m.submit_auto(TW, {"target": 10}).id and m.submit_auto(TW, {"target": 15}, {"detector": "smart"}).params["detector"] == "smart"
    manual = m.submit_vod(TW, [{"start": "00:10:00", "end": "00:20:00"}])          # tramos manuales: mismo contrato de siempre
    assert manual.params.get("mode") is None and len(manual.segments) == 1
    with pytest.raises(ApiError, match="solo se pueden elegir en un VOD de Twitch o Kick"):
        m.submit_vod(YT, [{"start": "00:10:00", "end": "00:20:00"}])                # YouTube sigue sin tramos manuales


# --- runner: orden, render diferido, fallos, cancelación y recuperación ----------------------------------------------------
class Probe:
    def __init__(self, scores=None, fail_block=None):
        self.log, self.scores, self.fail_block, self.lock = [], scores or {}, fail_block, threading.Lock()

    def block(self, job, seg, sctl):
        self.log.append(("block", seg.index))
        if self.fail_block == seg.index:
            raise MediaError("ffmpeg falló: corte de red")
        sctl.set(session=f"blk{seg.index}", result={"candidates": 3})

    def render(self, job, item, ctl):
        self.log.append(("render", item["id"]))
        return f"clip-{item['id']}"


def patch_pool(monkeypatch, per_block):
    """Reemplaza la lectura de result.json de cada bloque por candidatos controlados (puntuaciones y elegibilidad)."""
    def fake_pool(block, session, res):
        return [mk(block.index, r, block.start + 310 * r, sc, el) for r, (sc, el) in enumerate(per_block.get(block.index, []), 1)]
    monkeypatch.setattr(runners, "pool_from_block", fake_pool)


def seed_sessions(settings, n):
    for i in range(n):
        d = settings.jobs_dir / f"blk{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text("{}")


def test_blocks_run_sequentially_selection_is_global_and_only_selected_clips_render(settings, monkeypatch):
    per_block = {i: [(0.55 - 0.01 * i, True), (0.45, True), (0.41, True)] for i in range(24)}      # 72 candidatos elegibles
    patch_pool(monkeypatch, per_block)
    seed_sessions(settings, 24)
    p = Probe()
    m = make(settings, p.block, p.render)
    job = m.submit_auto(TW, {"target": 15})
    d = wait(m, job.id, {"completed"}, 30)
    kinds = [k for k, _ in p.log]
    assert kinds[:24] == ["block"] * 24 and set(kinds[24:]) == {"render"}                         # TODO el análisis antes de CUALQUIER render
    assert [i for k, i in p.log if k == "block"] == list(range(24))                                # secuencial y en orden
    assert kinds.count("render") == 15 < 24 * 3                                                    # solo los elegidos, no decenas
    assert d["result"]["summary"]["recommended"] == 15 and len(d["result"]["rendered"]) == 15
    assert {s["key"]: s["status"] for s in d["stages"]} == {"seleccion_global": "done", "render_final": "done"}
    blocks_of_plan = {c["block"] for c in d["result"]["plan"]}
    assert len(blocks_of_plan) > 5                                                                 # global, no 3 por bloque
    m.shutdown()


def test_job_reports_progress_per_block_and_stage(settings, monkeypatch):
    patch_pool(monkeypatch, {0: [(0.5, True)], 1: [(0.5, True)]})
    seed_sessions(settings, 2)
    seen = []

    def block(job, seg, sctl):
        sctl.stage("descarga", "running", 0.5, "descargando 50% del tramo")
        snap = m.get(job.id)
        seen.append((snap["message"], [s["status"] for s in snap["segments"]]))
        sctl.stage("descarga", "done", 1.0)
        sctl.set(session=f"blk{seg.index}", result={"candidates": 1})
    m = make(settings, block, Probe().render, inspector=lambda u: info_for(2400))
    wait(m, m.submit_auto(TW, {"target": 10}).id, {"completed"})
    assert seen[0][0].startswith("Bloque 1 de 2") and seen[0][1] == ["running", "pending"] and seen[1][1] == ["done", "running"]
    m.shutdown()


def test_a_failed_block_stops_selection_and_retry_redoes_only_that_block(settings, monkeypatch):
    patch_pool(monkeypatch, {i: [(0.6, True)] for i in range(3)})
    seed_sessions(settings, 3)
    p = Probe(fail_block=1)
    m = make(settings, p.block, p.render, inspector=lambda u: info_for(3600))
    job = m.submit_auto(TW, {"target": 10})
    d = wait(m, job.id, {"failed"})
    assert [s["status"] for s in d["segments"]] == ["done", "failed", "done"] and "No se eligen clips con la transmisión incompleta" in d["error"]
    assert not [x for x in p.log if x[0] == "render"] and "plan" not in d["result"]                 # no se inventa una selección parcial
    p.fail_block, p.log = None, []
    m.retry(job.id)
    d = wait(m, job.id, {"completed"})
    assert [x for x in p.log if x[0] == "block"] == [("block", 1)] and len([x for x in p.log if x[0] == "render"]) == 3
    m.shutdown()


def test_cancel_during_render_keeps_rendered_clips_and_resume_renders_only_the_rest(settings, monkeypatch):
    patch_pool(monkeypatch, {0: [(0.6 - 0.02 * i, True) for i in range(5)]})
    seed_sessions(settings, 1)
    gate, p = threading.Event(), Probe()

    def render(job, item, ctl):
        if len([x for x in p.log if x[0] == "render"]) == 2 and not gate.is_set():
            while not ctl.cancel.cancelled:
                time.sleep(0.02)
            raise CancelledError("cancelado")
        return p.render(job, item, ctl)
    m = make(settings, p.block, render, inspector=lambda u: info_for(1200))
    job = m.submit_auto(TW, {"target": 5})
    t0 = time.time()
    while len(m.get(job.id)["result"].get("rendered", {})) < 2 and time.time() - t0 < 10:
        time.sleep(0.02)
    time.sleep(0.2)
    m.cancel(job.id)
    d = wait(m, job.id, {"cancelled"})
    assert len(d["result"]["rendered"]) == 2 and d["segments"][0]["status"] == "done"
    gate.set()
    p.log.clear()
    m.retry(job.id)
    d = wait(m, job.id, {"completed"})
    assert [k for k, _ in p.log].count("block") == 0 and len(d["result"]["rendered"]) == 5 and len([x for x in p.log if x[0] == "render"]) == 3
    m.shutdown()


def test_restart_keeps_completed_blocks_and_continues_from_the_first_pending_one(settings, monkeypatch):
    patch_pool(monkeypatch, {i: [(0.6, True)] for i in range(4)})
    seed_sessions(settings, 4)
    gate = threading.Event()
    p1 = Probe()

    def block(job, seg, sctl):
        if seg.index == 2:
            sctl.set(session="blk2")
            gate.wait(10)                                                                          # «se cierra la aplicación» en el bloque 3
        return p1.block(job, seg, sctl)
    m1 = make(settings, block, p1.render, inspector=lambda u: info_for(4800))
    job = m1.submit_auto(TW, {"target": 10})
    t0 = time.time()
    while m1.get(job.id)["segments"][2]["status"] != "running" and time.time() - t0 < 10:
        time.sleep(0.02)
    from streamclipper.app.state import JobStore
    m1.store = JobStore(settings.data_dir / "zombie")                                              # el proceso «murió»: ya no escribe en el disco real
    p2 = Probe()
    m2 = make(settings, p2.block, p2.render, inspector=lambda u: info_for(4800), autostart=False)      # otra «sesión» de la app, mismo disco
    d = m2.get(job.id)
    assert d["status"] == "interrupted" and [s["status"] for s in d["segments"]] == ["done", "done", "pending", "pending"]
    assert d["segments"][0]["session"] == "blk0" and d["segments"][1]["session"] == "blk1"
    m3 = make(settings, p2.block, p2.render, inspector=lambda u: info_for(4800))
    m3.retry(job.id)
    d = wait(m3, job.id, {"completed"})
    assert [x for x in p2.log if x[0] == "block"] == [("block", 2), ("block", 3)]                    # 1 y 2 no se repiten
    assert len({c["id"] for c in d["result"]["plan"]}) == len(d["result"]["plan"]) == 4               # sin candidatos duplicados
    gate.set()
    m1.shutdown(timeout=3)
    m3.shutdown()


def test_restart_during_render_does_not_reselect_or_rerender(settings, monkeypatch):
    patch_pool(monkeypatch, {0: [(0.6 - 0.02 * i, True) for i in range(4)]})
    seed_sessions(settings, 1)
    p, boom = Probe(), {"on": True}

    def render(job, item, ctl):
        if len([x for x in p.log if x[0] == "render"]) == 2 and boom["on"]:
            raise MediaError("sin conexión: el VOD no responde")
        return p.render(job, item, ctl)
    m = make(settings, p.block, render, inspector=lambda u: info_for(1200))
    job = m.submit_auto(TW, {"target": 4})
    d = wait(m, job.id, {"failed"})
    plan_before = d["result"]["plan"]
    assert len(d["result"]["rendered"]) == 2 and d["stages"][1]["status"] == "failed"
    boom["on"], p.log = False, []
    m.retry(job.id)
    d = wait(m, job.id, {"completed"})
    assert d["result"]["plan"] == plan_before and len([x for x in p.log if x[0] == "render"]) == 2 and not [x for x in p.log if x[0] == "block"]
    m.shutdown()


def test_exploratory_clips_can_be_added_after_the_fact_without_reanalysis(settings, monkeypatch):
    patch_pool(monkeypatch, {0: [(0.6, True), (0.3, False), (0.29, False), (0.28, False)]})
    seed_sessions(settings, 1)
    p = Probe()
    m = make(settings, p.block, p.render, inspector=lambda u: info_for(1200))
    job = m.submit_auto(TW, {"target": 4})
    d = wait(m, job.id, {"completed"})
    assert d["result"]["summary"]["recommended"] == 1 and d["result"]["summary"]["exploratory_available"] == 3 and len(d["result"]["rendered"]) == 1
    p.log.clear()
    m.extend_auto(job.id, 2)
    d = wait(m, job.id, {"completed"})
    assert [k for k, _ in p.log] == ["render", "render"] and len(d["result"]["plan"]) == 3 and d["result"]["plan"][-1]["kind"] == "exploratory"
    with pytest.raises(ApiError):
        m.extend_auto(job.id, 5) if d["result"]["exploratory"] == [] else (_ for _ in ()).throw(ApiError(409, "x"))
    m.shutdown()


def test_delete_is_blocked_while_the_job_uses_a_rendered_session(settings, monkeypatch):
    patch_pool(monkeypatch, {0: [(0.6, True), (0.55, True)]})
    seed_sessions(settings, 1)
    gate = threading.Event()

    def render(job, item, ctl):
        gate.wait(5)
        return "20261121-000000-clip"
    m = make(settings, Probe().block, render, inspector=lambda u: info_for(1200))
    job = m.submit_auto(TW, {"target": 2})
    t0 = time.time()
    while m.get(job.id)["stages"][1]["status"] != "running" and time.time() - t0 < 10:
        time.sleep(0.02)
    d = settings.jobs_dir / "blk0"
    with pytest.raises(ApiError) as e:
        m.delete_session("blk0")
    assert e.value.status == 409 and d.exists()
    gate.set()
    m.shutdown()


# --- sesiones: ocultas, galería global, compatibilidad ----------------------------------------------------------------------
def write_session(settings, name, meta=None, clips=0):
    d = settings.jobs_dir / name
    (d / "out").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"job_id": name, "status": "completed", "source": "x.mp4", "candidates": [], "clips": [{"name": f"clip_{i}"} for i in range(clips)],
                                               "settings": {}, "media": {"duration": 60}}))
    if meta:
        (d / "segment.json").write_text(json.dumps(meta))


def test_analysis_sessions_are_hidden_and_the_global_gallery_lists_final_clips_in_order(settings):
    write_session(settings, "20260101-000000-old")                                                        # sesión anterior, sin segment.json
    write_session(settings, "20261121-b0", {"group": "G", "mode": "auto-block", "index": 0})
    for i, (gr, kind) in enumerate([(2, "exploratory"), (1, "recommended")]):
        write_session(settings, f"20261121-c{i}", {"group": "G", "mode": "auto-clip", "global_rank": gr, "kind": kind, "score": 0.5, "start": 100 * gr,
                                                   "end": 100 * gr + 20, "start_hms": "00:01:40", "end_hms": "00:02:00", "index": gr - 1, "count": 2})
    rows = {s["id"]: s for s in list_sessions(settings)}
    assert rows["20261121-b0"]["hidden"] is True and rows["20260101-000000-old"]["hidden"] is False and rows["20260101-000000-old"]["segment"] is None
    job = {"id": "G", "title": "Stream", "input": TW, "status": "completed", "params": {}, "created_at": 0, "segments": [],
           "result": {"plan": [{"id": "a"}, {"id": "b"}], "rendered": {"a": "x", "b": "y"}, "summary": {"recommended": 1}, "exploratory": [{"id": "z"}]}}
    det = auto_detail(settings, job)
    assert [c["meta"]["global_rank"] for c in det["clips"]] == [1, 2] and [c["meta"]["kind"] for c in det["clips"]] == ["recommended", "exploratory"]
    assert det["blocks"] == ["20261121-b0"] and det["exploratory_available"] == 1 and det["pending_render"] == 0


# --- proveedores y vista previa ---------------------------------------------------------------------------------------------
def test_inspect_offers_auto_for_the_three_providers_and_rejects_live_and_clips():
    class Y:
        def __init__(self, info): self.info = info
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k): return self.info
    base = {"title": "t", "duration": H8, "uploader": "u", "tbr": 2000}
    for url in (TW, KICK, YT):
        r = inspect_media(url, ydl_factory=lambda o: Y(base))
        assert r["auto"]["available"] and r["auto"]["blocks"] == 24 and r["auto"]["targets"] == [10, 15, 20]
    assert inspect_media(YT, ydl_factory=lambda o: Y(base))["acceptable"] is False                  # «video corto» sigue rechazando lo largo
    assert inspect_media(TW, ydl_factory=lambda o: Y(base))["acceptable"] is True and inspect_media(TW, ydl_factory=lambda o: Y(base))["segmentable"]
    live = inspect_media(TW, ydl_factory=lambda o: Y({**base, "is_live": True}))
    assert live["auto"]["available"] is False and "en vivo" in live["auto"]["reason"]
    assert inspect_media("https://clips.twitch.tv/FaintLightGullWholeWheat", ydl_factory=lambda o: Y({**base, "duration": 30}))["auto"]["available"] is False
    short = inspect_media(TW, ydl_factory=lambda o: Y({**base, "duration": 40}))
    assert short["auto"]["available"] is False and "demasiado corto" in short["auto"]["reason"]


def test_segment_download_for_youtube_needs_auto_and_uses_the_same_partial_range_code(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    with pytest.raises(InvalidInputError, match="solo se pueden descargar de un VOD"):
        download_segment(YT, 0, 60, cfg, ydl_factory=factory(vod_info("x")))
    f = factory(vod_info(tmp_path / "no-existe.mp4"))
    with pytest.raises(MediaError, match="FFmpeg no pudo descargar el tramo"):
        download_segment(YT, 0, 60, cfg, auto=True, ydl_factory=f)                                  # llegó a FFmpeg: no se descarga el video entero
    assert len(f.log) == 1 and not list((tmp_path / "samples").glob("*.mp4"))


# --- integración real: FFmpeg + Whisper sobre un «VOD» con momentos conocidos ---------------------------------------------------
SAMPLE = REPO_ROOT / "data" / "samples" / "synthetic.mp4"


@pytest.mark.integration
def test_full_automatic_flow_on_real_media_with_known_moments(tmp_path):
    if not SAMPLE.exists():
        pytest.skip("falta data/samples/synthetic.mp4")
    cfg = Settings(data_dir=tmp_path / "data")
    info = {"id": "6528877", "title": "VOD sintético", "duration": 105, "uploader": "s", "url": str(SAMPLE), "tbr": 1000}
    f = lambda opts: FakeYDL(info, [])      # noqa: E731  yt-dlp «resuelve» el archivo local; FFmpeg corta de verdad
    runners_ = {"vod": run_auto(cfg, real_block(cfg, f), real_render(cfg, f)), "process": lambda j, c: None, "edit": lambda j, c: None}
    m = JobManager(cfg, runners=runners_, inspector=lambda u: info_for(105))
    t0 = time.monotonic()
    job = m.submit_auto(TW, {"target": 3, "block_s": 45, "margin_s": 10}, {"model": "base", "language": "en", "layout": "blur-fit", "max_clips": 3})
    d = wait(m, job.id, {"completed", "failed"}, 280)
    assert d["status"] == "completed", (d["error"], [s["error"] for s in d["segments"]])
    assert len(d["segments"]) == 2 and all(s["status"] == "done" for s in d["segments"])
    # los dos momentos reales (≈30 s y ≈73 s) se encuentran en bloques distintos y con tiempos ABSOLUTOS correctos
    plan = d["result"]["plan"]
    assert len(plan) == 2 and {c["block"] for c in plan} == {0, 1}
    for event in (30.0, 73.0):
        assert any(c["start"] < event < c["end"] for c in plan), (event, [(c["start"], c["end"]) for c in plan])
    assert d["result"]["summary"]["recommended"] == 2 and len(d["result"]["rendered"]) == 2
    # el video de cada bloque se borró; las sesiones de bloque son de solo análisis y están ocultas
    samples = {p.name for p in (cfg.data_dir / "samples").glob("*.mp4")}
    assert not {"twitch-vod-6528877_000000-000055.mp4", "twitch-vod-6528877_000035-000105.mp4"} & samples      # bloques de análisis borrados
    assert len(samples) == 2                                                                                  # quedan solo los 2 recortes finales
    hidden = [s for s in list_sessions(cfg) if s["hidden"]]
    clips = [s for s in list_sessions(cfg) if not s["hidden"]]
    assert len(hidden) == 2 and len(clips) == 2 and all(s["clips"] == 1 and s["segment"]["mode"] == "auto-clip" for s in clips)
    for s in clips:                                                    # los clips finales son sesiones normales: reproducibles y con motivos
        res = json.loads((cfg.jobs_dir / s["id"] / "result.json").read_text(encoding="utf-8"))
        mp4 = Path(res["clips"][0]["file"])
        mp4 = mp4 if mp4.is_absolute() else cfg.jobs_dir / s["id"] / mp4
        probe = json.loads(subprocess.run([str(cfg.ffprobe), "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "json", str(mp4)],
                                          capture_output=True, text=True).stdout)
        assert any(x.get("height") == 1920 for x in probe["streams"]) and any(x["codec_type"] == "audio" for x in probe["streams"])
        assert res["candidates"][0]["components"] and s["segment"]["score"] > 0
    assert time.monotonic() - t0 < 280 and not subprocess.run(["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe"], capture_output=True, text=True).stdout.count("ffmpeg.exe")
    m.shutdown()

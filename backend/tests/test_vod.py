"""VOD de Twitch y Kick por tramos: validación de intervalos, descarga parcial, cola, cancelación, recuperación y API."""
import http.client
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from streamclipper.app import runners
from streamclipper.app.manager import JobManager
from streamclipper.app.runners import run_vod
from streamclipper.app.server import AppContext, make_server
from streamclipper.app.sessions import group_detail, list_sessions
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, MediaError, StreamClipperError
from streamclipper.media import download as dl
from streamclipper.media.download import (
    DownloadLimits, DownloadResult, download_segment, format_hms, parse_hms, validate_segments,
)
from streamclipper.webutil import ApiError

TW = "https://www.twitch.tv/videos/6528877"
KICK = "https://kick.com/xqc/videos/01a1fae2-a5c0-754b-888b-c05cc64e9741"
DURATION = 3 * 3600 + 25 * 60   # 3 h 25 min


# --- intervalos -------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,seconds", [("00:00:00", 0), ("01:25:00", 5100), ("1:02:03", 3723), ("100:00:00", 360000), (90, 90), (" 00:10:00 ", 600)])
def test_parse_hms_accepts_valid_times(text, seconds):
    assert parse_hms(text) == seconds


@pytest.mark.parametrize("bad", ["", None, "1:2:3", "00:60:00", "00:00:60", "abc", "00:10", "-00:10:00", "00:10:00.5", True, -5, 1.5, "00:10:00; rm"])
def test_parse_hms_rejects_invalid_times(bad):
    with pytest.raises(InvalidInputError):
        parse_hms(bad)


def seg(a, b):
    return {"start": a, "end": b}


def test_valid_segments_are_sorted_and_touching_is_not_overlap():
    assert validate_segments([seg("01:00:00", "01:10:00"), seg("00:10:00", "00:20:00"), seg("00:20:00", "00:30:00")], DURATION) == \
        [(600, 1200), (1200, 1800), (3600, 4200)]


@pytest.mark.parametrize("segments,fragment", [
    ([], "al menos un tramo"),
    (None, "al menos un tramo"),
    ([seg("00:00:00", "00:00:00")], "posterior al inicio"),          # vacío
    ([seg("00:10:00", "00:05:00")], "posterior al inicio"),          # invertido
    ([seg("00:00:00", "00:00:05")], "mínimo"),
    ([seg("00:00:00", "00:20:01")], "máximo por tramo es 20 min"),
    ([seg("00:00:00", "01:00:00")], "máximo por tramo"),
    ([seg("03:20:00", "03:30:00")], "el VOD dura 03:25:00"),         # fuera de la duración total
    ([seg("03:30:00", "03:40:00")], "el VOD dura"),
    ([seg("00:00:00", "00:10:00"), seg("00:09:59", "00:15:00")], "se superponen"),
    ([seg("00:00:00", "00:10:00"), seg("00:00:00", "00:10:00")], "se superponen"),
    ([seg("00:05:00", "00:15:00"), seg("00:00:00", "00:10:00")], "se superponen"),   # desordenados y superpuestos
    ([seg("00:00:00", "00:10:00")] * 6, "hasta 5 tramos"),
    ([seg("00:00", "00:10:00")], "Tramo 1: Hora inválida"),
    ([{"start": "00:00:00"}], "faltan el inicio y el final"),
    (["xx"], "Tramo 1"),
])
def test_invalid_segments_are_rejected(segments, fragment):
    with pytest.raises(InvalidInputError, match=fragment):
        validate_segments(segments, DURATION)


def test_exactly_five_segments_of_twenty_minutes_are_allowed():
    segs = [seg(format_hms(i * 1300), format_hms(i * 1300 + 1200)) for i in range(5)]
    assert len(validate_segments(segs, DURATION)) == 5


# --- descarga parcial real (FFmpeg real sobre un archivo local que hace de «VOD») ---------------------
@pytest.fixture(scope="module")
def fake_vod(tmp_path_factory) -> Path:
    """30 s con video (keyframe cada 1 s) y audio: hace de flujo HLS ya resuelto."""
    p = tmp_path_factory.mktemp("vod") / "vod.mp4"
    subprocess.run([str(Settings().ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=15:duration=30", "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                    "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


class FakeYDL:
    def __init__(self, info, log):
        self.info, self.log = info, log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        self.log.append(url)
        return self.info


def vod_info(url, duration=3600, **extra):
    return {"id": "6528877", "title": "Mi VOD", "duration": duration, "uploader": "streamer", "url": str(url), "tbr": 3000, **extra}


def factory(info, log=None):
    log = log if log is not None else []
    f = lambda opts: FakeYDL(info, log)  # noqa: E731
    f.log = log
    return f


def probe_duration(p):
    out = subprocess.run([str(Settings().ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout
    return float(out)


def test_partial_download_cuts_only_the_requested_range_and_verifies_it(tmp_path, fake_vod):
    cfg, msgs = Settings(data_dir=tmp_path), []
    res = download_segment(TW, 10, 20, cfg, ydl_factory=factory(vod_info(fake_vod)), progress=msgs.append)
    assert res.path == tmp_path / "samples" / "twitch-vod-6528877_000010-000020.mp4" and res.path.is_file()
    assert 9 <= probe_duration(res.path) <= 12            # ~10 s pedidos, NO los 30 s de la fuente
    assert res.duration == pytest.approx(probe_duration(res.path), abs=0.1) and (res.start, res.end) == (10, 20)
    side = json.loads(res.path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert side["kind"] == "vod-segment" and side["segment_start_s"] == 10 and side["requested_duration_s"] == 10
    assert side["title"] == "Mi VOD [00:00:10–00:00:20]" and "extra_seconds" in side and side["measured_duration_s"] > 0
    assert any("%" in m for m in msgs)                    # avance real de FFmpeg
    assert not list((tmp_path / "samples").glob(".downloading-*"))


def test_segment_download_uses_range_flags_and_never_a_full_download(tmp_path, fake_vod, monkeypatch):
    seen = []
    real = subprocess.Popen

    def spy(cmd, *a, **k):
        seen.append([str(c) for c in cmd])
        return real(cmd, *a, **k)
    monkeypatch.setattr(dl.subprocess, "Popen", spy)
    download_segment(TW, 5, 15, Settings(data_dir=tmp_path), ydl_factory=factory(vod_info(fake_vod)))
    cmd = seen[0]
    assert cmd[cmd.index("-ss") + 1] == "5" and cmd.count("-t") == 2 and cmd[cmd.index("-t") + 1] == "10"
    assert cmd.index("-ss") < cmd.index("-i") and "copy" in cmd         # corte por entrada, sin recodificar
    assert sum(Path(c[0]).stem == "ffmpeg" for c in seen) == 1          # un solo FFmpeg: no hay descarga previa del VOD


def test_ffmpeg_failure_is_a_clear_error_and_does_not_fall_back_to_full_download(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    f = factory(vod_info(tmp_path / "no-existe.mp4"))
    with pytest.raises(MediaError, match="FFmpeg no pudo descargar el tramo"):
        download_segment(TW, 10, 30, cfg, ydl_factory=f)
    assert len(f.log) == 1                                              # tampoco reintenta con otro método
    assert not list((tmp_path / "samples").glob("*.mp4")) and not list((tmp_path / "samples").glob(".downloading-*"))


def test_range_outside_the_source_is_rejected_by_ffprobe_verification(tmp_path, fake_vod):
    cfg = Settings(data_dir=tmp_path)   # la «fuente» dura 30 s pero se piden 30 s a partir del s 20
    with pytest.raises(MediaError, match="en lugar de los 30 s pedidos"):
        download_segment(TW, 20, 50, cfg, ydl_factory=factory(vod_info(fake_vod)))
    assert not list((tmp_path / "samples").glob("*.mp4"))


def test_segment_must_fit_in_the_vod_and_in_the_limits_before_any_ffmpeg(tmp_path, fake_vod, monkeypatch):
    monkeypatch.setattr(dl.subprocess, "Popen", lambda *a, **k: pytest.fail("no debe ejecutar FFmpeg"))
    cfg = Settings(data_dir=tmp_path)
    with pytest.raises(InvalidInputError, match="el VOD dura"):
        download_segment(TW, 3590, 3610, cfg, ydl_factory=factory(vod_info(fake_vod)))
    with pytest.raises(InvalidInputError, match="máximo por tramo"):
        download_segment(TW, 0, 1300, cfg, ydl_factory=factory(vod_info(fake_vod)))
    with pytest.raises(InvalidInputError, match="El tramo pesaría"):
        download_segment(TW, 0, 600, cfg, DownloadLimits(max_bytes=10_000_000), ydl_factory=factory(vod_info(fake_vod)))
    for bad in ("https://clips.twitch.tv/FaintLightGullWholeWheat", "https://www.youtube.com/watch?v=aqz-KE-bpKQ"):
        with pytest.raises(InvalidInputError, match="solo se pueden descargar de un VOD"):
            download_segment(bad, 0, 60, cfg, ydl_factory=factory(vod_info(fake_vod)))
    with pytest.raises(InvalidInputError, match="en vivo"):
        download_segment(TW, 0, 60, cfg, ydl_factory=factory(vod_info(fake_vod, is_live=True)))
    with pytest.raises(InvalidInputError, match="duración"):
        download_segment(TW, 0, 60, cfg, ydl_factory=factory(vod_info(fake_vod, duration=None)))


def test_existing_segment_is_reused_without_touching_the_network(tmp_path, fake_vod):
    cfg = Settings(data_dir=tmp_path)
    download_segment(TW, 10, 20, cfg, ydl_factory=factory(vod_info(fake_vod)))
    f = factory(vod_info(fake_vod))
    r = download_segment(TW, 10, 20, cfg, ydl_factory=f)
    assert r.reused and f.log == []


def test_no_cookies_or_credentials_in_segment_downloads(tmp_path, fake_vod):
    seen = {}

    def spy(opts):
        seen.update(opts)
        return FakeYDL(vod_info(fake_vod), [])
    download_segment(TW, 0, 10, Settings(data_dir=tmp_path), ydl_factory=spy)
    assert seen["usenetrc"] is False and seen["ignoreconfig"] is True and seen["noplaylist"] is True
    assert not {"cookiefile", "cookiesfrombrowser", "username", "password"} & set(seen)


def test_cancelling_kills_ffmpeg_promptly(tmp_path):
    procs, real = [], subprocess.Popen

    def spy(cmd, *a, **k):
        p = real([__import__("sys").executable, "-c", "import time; print('out_time_us=1000000', flush=True); time.sleep(60)"], *a, **k)
        procs.append(p)
        return p
    token = CancelToken()
    threading.Timer(0.8, token.cancel).start()
    t0 = time.time()
    orig = dl.subprocess.Popen
    dl.subprocess.Popen = spy
    try:
        with pytest.raises(CancelledError):
            dl._run_ffmpeg_cut(["ffmpeg"], 10, token, lambda m: None)
    finally:
        dl.subprocess.Popen = orig
    assert time.time() - t0 < 6
    assert procs[0].poll() is not None                      # el proceso no queda huérfano


# --- gestor: alta, cola, cancelación, reintento y recuperación ------------------------------------------
def inspector(duration=DURATION, **extra):
    return lambda url: {"title": "Gran VOD", "duration": duration, "acceptable": True, "segmentable": True, "uploader": "s", "reason": None, **extra}


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


def make(settings, do_segment, **kw):
    return JobManager(settings, runners={"vod": run_vod(settings, do_segment), "process": lambda j, c: None, "edit": lambda j, c: None},
                      inspector=kw.pop("inspector", inspector()), **kw)


def wait(m, jid, states, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = m.get(jid)
        if j["status"] in states:
            return j
        time.sleep(0.02)
    raise AssertionError(f"timeout: {m.get(jid)['status']}")


def ok_segment(calls):
    def do(job, seg, sctl):
        calls.append(seg.index)
        sctl.stage("descarga", "done", 1.0)
        sctl.set(session=f"sess-{seg.index}", result={"clips": 2})
    return do


SEGS = [seg("00:10:00", "00:20:00"), seg("01:00:00", "01:15:00"), seg("02:00:00", "02:05:00")]


def test_submit_vod_validates_and_records_url_and_times(settings):
    m = make(settings, ok_segment([]), autostart=False)
    job = m.submit_vod(KICK, SEGS)
    assert job.kind == "vod" and job.input == KICK and job.title == "Gran VOD"
    assert [(s.start, s.end) for s in job.segments] == [(600, 1200), (3600, 4500), (7200, 7500)]
    assert all(s.stages[0]["key"] == "descarga" and s.status == "pending" for s in job.segments)
    for bad, frag in (([seg("00:00:00", "00:30:00")], "máximo por tramo"), ([seg("03:25:00", "03:30:00")], "el VOD dura"), ([], "al menos")):
        with pytest.raises(ApiError, match=frag) as e:
            m.submit_vod(TW, bad)
        assert e.value.status == 400
    for url in ("https://clips.twitch.tv/FaintLightGullWholeWheat", "https://www.youtube.com/watch?v=aqz-KE-bpKQ", "https://evil.com/x"):
        with pytest.raises(ApiError) as e:
            m.submit_vod(url, SEGS[:1])
        assert e.value.status == 400
    m2 = make(settings, ok_segment([]), autostart=False, inspector=lambda u: {"duration": None, "acceptable": False, "reason": "El video no está disponible"})
    with pytest.raises(ApiError, match="no está disponible"):
        m2.submit_vod(TW, SEGS[:1])


def test_duplicate_vod_jobs_are_rejected(settings):
    m = make(settings, ok_segment([]), autostart=False)
    m.submit_vod(TW, SEGS)
    with pytest.raises(ApiError) as e:
        m.submit_vod("https://m.twitch.tv/videos/6528877?t=1m", list(reversed(SEGS)))   # mismas ranuras, otro orden y URL equivalente
    assert e.value.status == 409 and e.value.extra["duplicate"] == "active"
    assert m.submit_vod(TW, SEGS[:1]).id                                               # otros tramos del mismo VOD: no es duplicado


def test_segments_run_sequentially_in_one_heavy_job(settings):
    order, active, peak, lock = [], 0, 0, threading.Lock()

    def do(job, seg, sctl):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        order.append((job.id, seg.index))
        time.sleep(0.05)
        sctl.set(session=f"s{seg.index}")
        with lock:
            active -= 1
    m = make(settings, do)
    a = m.submit_vod(TW, SEGS)
    b = m.submit_vod(KICK, SEGS[:2])                    # otro VOD: espera en la cola
    wait(m, a.id, {"completed"})
    wait(m, b.id, {"completed"})
    assert peak == 1
    assert [i for j, i in order if j == a.id] == [0, 1, 2] and order[3][0] == b.id   # primero todo el VOD A, luego el B
    m.shutdown()


def test_progress_is_per_segment_and_real(settings):
    def do(job, seg, sctl):
        sctl.stage("descarga", "running", 0.4, "descargando 40% del tramo")
        snap = m.get(job.id)
        assert snap["segments"][seg.index]["stages"][0]["fraction"] == 0.4 and snap["segments"][seg.index]["status"] == "running"
        assert all(s["status"] == "pending" for s in snap["segments"][seg.index + 1:])      # los siguientes siguen en espera
        assert "Tramo %d de 3" % (seg.index + 1) in snap["message"]
        sctl.stage("descarga", "done", 1.0)
    m = make(settings, do)
    wait(m, m.submit_vod(TW, SEGS).id, {"completed"})
    m.shutdown()


def test_failed_segment_does_not_stop_the_rest_and_retry_only_redoes_failed(settings):
    calls, mode = [], {"fail": True}

    def do(job, seg, sctl):
        calls.append(seg.index)
        if seg.index == 1 and mode["fail"]:
            raise MediaError("ffmpeg falló: error de red")
        sctl.set(session=f"s{seg.index}")
    m = make(settings, do)
    j = m.submit_vod(TW, SEGS)
    d = wait(m, j.id, {"failed"})
    assert calls == [0, 1, 2]                                   # el tramo 2 se procesó pese al fallo del 1
    assert [s["status"] for s in d["segments"]] == ["done", "failed", "done"]
    assert "1 de 3 tramo(s) fallaron" in d["error"] and d["segments"][1]["hint"]
    mode["fail"] = False
    calls.clear()
    m.retry(j.id)
    d = wait(m, j.id, {"completed"})
    assert calls == [1]                                         # solo el fallido; los terminados no se repiten
    assert [s["status"] for s in d["segments"]] == ["done", "done", "done"] and d["segments"][0]["session"] == "s0"
    with pytest.raises(ApiError, match="Solo se pueden reintentar"):
        m.retry(j.id)
    m.shutdown()


def test_unexpected_error_in_a_segment_is_contained(settings):
    def do(job, seg, sctl):
        if seg.index == 0:
            raise RuntimeError("raro")
    m = make(settings, do)
    d = wait(m, m.submit_vod(TW, SEGS[:2]).id, {"failed"})
    assert d["segments"][0]["status"] == "failed" and "Error inesperado" in d["segments"][0]["error"] and d["segments"][1]["status"] == "done"
    m.shutdown()


def test_cancel_keeps_finished_segments_and_continue_skips_them(settings):
    gate, calls = threading.Event(), []

    def do(job, seg, sctl):
        calls.append(seg.index)
        if seg.index == 1 and not gate.is_set():
            while not sctl.cancel.cancelled:
                time.sleep(0.02)
            raise CancelledError("cancelado")
        sctl.set(session=f"s{seg.index}")
    m = make(settings, do)
    j = m.submit_vod(TW, SEGS)
    t0 = time.time()
    while m.get(j.id)["segments"][1]["status"] != "running" and time.time() - t0 < 5:
        time.sleep(0.02)
    m.cancel(j.id)
    d = wait(m, j.id, {"cancelled"})
    assert [s["status"] for s in d["segments"]] == ["done", "cancelled", "pending"] and "1 de 3 tramo(s) quedaron listos" in d["message"]
    gate.set()
    calls.clear()
    m.retry(j.id)
    d = wait(m, j.id, {"completed"})
    assert calls == [1, 2] and d["segments"][0]["session"] == "s0"
    m.shutdown()


def test_restart_recovers_and_continues_pending_segments_only(settings):
    gate = threading.Event()

    def blocker(job, seg, sctl):
        sctl.set(session=f"s{seg.index}")
        if seg.index == 1:
            gate.wait(10)                                      # «la aplicación se cierra» en el tramo 2
    m1 = make(settings, blocker)
    j = m1.submit_vod(TW, SEGS)
    t0 = time.time()
    while m1.get(j.id)["segments"][1]["status"] != "running" and time.time() - t0 < 5:
        time.sleep(0.02)
    # Estado en disco tal como quedaría tras un cierre brusco (sin shutdown limpio)
    m2 = make(settings, ok_segment([]), autostart=False)
    d = m2.get(j.id)
    assert d["status"] == "interrupted"
    assert [s["status"] for s in d["segments"]] == ["done", "pending", "pending"] and d["segments"][0]["session"] == "s0"
    calls = []
    m3 = make(settings, ok_segment(calls))
    m3.retry(j.id)
    d = wait(m3, j.id, {"completed"})
    assert calls == [1, 2] and d["segments"][0]["session"] == "s0"       # el tramo 1 no se repite
    gate.set()
    m1.shutdown(timeout=3)
    m3.shutdown()


def test_cannot_delete_a_session_of_a_running_vod(settings):
    gate = threading.Event()

    def do(job, seg, sctl):
        sctl.set(session="20261121-100000-seg")
        gate.wait(10)
    m = make(settings, do)
    j = m.submit_vod(TW, SEGS[:1])
    t0 = time.time()
    while not m.get(j.id)["segments"][0]["session"] and time.time() - t0 < 5:
        time.sleep(0.02)
    with pytest.raises(ApiError) as e:
        m.delete_session("20261121-100000-seg")
    assert e.value.status == 409
    gate.set()
    m.shutdown()


# --- relación VOD ↔ tramos ↔ clips (con el runner real, descarga y pipeline simulados) -----------------
def fake_pipeline(settings, monkeypatch, clips=2):
    def fake_download(url, start, end, cfg, limits, **kw):
        p = cfg.data_dir / "samples" / f"seg-{start}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        kw["progress"]("descargando 50% del tramo")
        return DownloadResult(p, "6528877", title="Gran VOD", duration=end - start, platform="twitch", start=start, end=end)

    def fake_run(source, cfg, cancel, progress, label=None, on_job_dir=None, **kw):
        jid = f"20261121-{Path(source).stem.split('-')[-1]}-small-app"
        d = cfg.jobs_dir / jid
        (d / "out").mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"job_id": jid, "status": "completed", "source": source, "clips": [], "candidates": [],
                                                   "settings": {"whisper_model": "small"}}))
        on_job_dir(jid, d)
        progress("transcribir", 1.0, "")
        return {"job_id": jid, "status": "completed", "clips": [{"name": f"clip_0{i}"} for i in range(clips)], "candidates": []}
    monkeypatch.setattr(runners, "download_segment", fake_download)
    monkeypatch.setattr(runners, "run_pipeline", fake_run)


def test_real_runner_links_sessions_to_their_vod_and_segment(settings, monkeypatch):
    fake_pipeline(settings, monkeypatch)
    m = JobManager(settings, inspector=inspector())       # runners por defecto: run_vod real
    j = m.submit_vod(KICK, SEGS[:2], {"model": "small", "language": "es", "layout": "blur-fit", "max_clips": 3})
    d = wait(m, j.id, {"completed"})
    assert [s["status"] for s in d["segments"]] == ["done", "done"] and all(s["session"] for s in d["segments"])
    assert d["segments"][0]["result"]["clips"] == 2 and d["segments"][0]["stages"][0]["detail"] == "600 s descargados"
    sessions = list_sessions(settings)
    assert len(sessions) == 2 and {s["segment"]["group"] for s in sessions} == {j.id}
    g = group_detail(settings, j.id)
    assert g["title"] == "Gran VOD" and g["url"] == KICK and [s["segment"]["start_hms"] for s in g["segments"]] == ["00:10:00", "01:00:00"]
    meta = g["segments"][0]["segment"]
    assert (meta["index"], meta["count"], meta["start"], meta["end"], meta["platform"], meta["vod_id"]) == (0, 2, 600, 1200, "kick", "01a1fae2-a5c0-754b-888b-c05cc64e9741")
    with pytest.raises(ApiError):
        group_detail(settings, "no-existe")
    m.shutdown()


# --- API HTTP ---------------------------------------------------------------------------------------
@pytest.fixture
def server(settings, monkeypatch):
    fake_pipeline(settings, monkeypatch)
    m = JobManager(settings, inspector=inspector())
    srv = make_server(AppContext(settings, m, picker=lambda: "", opener=lambda p: None), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_port, m
    srv.shutdown()
    m.shutdown()


def req(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"} if body is not None else {})
    r = c.getresponse()
    return r.status, json.loads(r.read())


def test_api_vod_flow_and_grouped_view(server):
    port, m = server
    st, j = req(port, "POST", "/api/jobs", {"source": TW, "segments": [seg("00:10:00", "00:20:00"), seg("00:30:00", "00:40:00")]})
    assert st == 201 and j["kind"] == "vod" and len(j["segments"]) == 2
    wait(m, j["id"], {"completed"})
    st, jobs = req(port, "GET", "/api/jobs")
    assert jobs["jobs"][0]["segments"][1]["status"] == "done"
    st, g = req(port, "GET", f"/api/groups/{j['id']}")
    assert st == 200 and len(g["segments"]) == 2 and g["segments"][1]["segment"]["end_hms"] == "00:40:00"
    st, sessions = req(port, "GET", "/api/sessions")
    assert all(s["segment"]["group"] == j["id"] for s in sessions["sessions"])
    assert req(port, "GET", "/api/groups/nope")[0] == 404
    assert req(port, "GET", "/api/groups/..%2f..")[0] in (400, 404)


def test_api_rejects_bad_segments_and_keeps_clip_and_normal_flows(server, settings):
    port, m = server
    st, r = req(port, "POST", "/api/jobs", {"source": TW, "segments": [seg("00:00:00", "00:30:00")]})
    assert st == 400 and "máximo por tramo" in r["error"]
    st, r = req(port, "POST", "/api/jobs", {"source": TW, "segments": [seg("00:00:00", "00:10:00"), seg("00:05:00", "00:15:00")]})
    assert st == 400 and "se superponen" in r["error"]
    st, r = req(port, "POST", "/api/jobs", {"source": "https://clips.twitch.tv/FaintLightGullWholeWheat", "segments": [seg("00:00:00", "00:00:30")]})
    assert st == 400 and "VOD" in r["error"]
    # sin «segments» el flujo anterior sigue igual (YouTube, clips, archivos): se valida con las reglas de siempre
    assert req(port, "POST", "/api/jobs", {"source": "https://evil.example/x"})[0] == 400
    assert req(port, "POST", "/api/jobs", {"source": ""})[0] == 400


def test_old_sessions_are_not_modified_or_grouped(settings):
    d = settings.jobs_dir / "20260101-000000-old"
    (d / "out").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"job_id": d.name, "status": "completed", "source": "x.mp4", "clips": [], "candidates": []}))
    before = sorted(p.name for p in d.rglob("*"))
    s = list_sessions(settings)
    assert s[0]["segment"] is None and sorted(p.name for p in d.rglob("*")) == before    # listar no escribe nada
    with pytest.raises(StreamClipperError):
        raise StreamClipperError("sanity")

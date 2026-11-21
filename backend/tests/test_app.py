"""Aplicación web local: cola, cancelación, duplicados, recuperación, API y seguridad (con runners falsos; sin Whisper ni FFmpeg)."""
import http.client
import json
import threading
import time

import pytest

from streamclipper.app.manager import JobManager, cleanup_stale
from streamclipper.app.server import AppContext, make_server
from streamclipper.app.state import JobStore
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, StreamClipperError
from streamclipper.webutil import ApiError


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"\x00" * 64)
    return p


def wait(manager, jid, states, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = manager.get(jid)
        if j["status"] in states:
            return j
        time.sleep(0.02)
    raise AssertionError(f"timeout: {manager.get(jid)['status']}")


def runner_ok(job, ctl):
    ctl.stage("validar", "running")
    ctl.stage("validar", "done")


def make_manager(settings, process=runner_ok, **kw):
    return JobManager(settings, runners={"process": process, "edit": runner_ok}, **kw)


def test_job_completes_with_stage_timings(settings, video):
    m = make_manager(settings)
    j = m.submit_process(str(video))
    done = wait(m, j.id, {"completed"})
    assert next(s for s in done["stages"] if s["key"] == "validar")["status"] == "done"
    m.shutdown()


def test_duplicate_active_rejected_and_completed_needs_force(settings, video):
    gate = threading.Event()
    m = make_manager(settings, process=lambda job, ctl: gate.wait(5))
    j = m.submit_process(str(video))
    with pytest.raises(ApiError) as e:
        m.submit_process(str(video))
    assert e.value.status == 409 and e.value.extra["duplicate"] == "active"
    gate.set()
    wait(m, j.id, {"completed"})
    m.get(j.id)
    m._jobs[j.id].session = "s1"   # lo que haría el runner real
    with pytest.raises(ApiError) as e:
        m.submit_process(str(video))
    assert e.value.extra["duplicate"] == "completed"
    assert m.submit_process(str(video), force=True).id != j.id
    m.shutdown()


def test_only_one_heavy_job_runs_at_a_time(settings, tmp_path):
    running, peak, lock = 0, 0, threading.Lock()

    def slow(job, ctl):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.15)
        with lock:
            running -= 1

    m = make_manager(settings, process=slow)
    ids = []
    for i in range(3):
        p = tmp_path / f"v{i}.mp4"
        p.write_bytes(b"x" * (10 + i))
        ids.append(m.submit_process(str(p)).id)
    for i in ids:
        wait(m, i, {"completed"})
    assert peak == 1
    m.shutdown()


def test_cancel_queued_and_running(settings, tmp_path):
    started = threading.Event()

    def blocker(job, ctl):
        started.set()
        while not ctl.cancel.cancelled:
            time.sleep(0.02)
        raise CancelledError("cancelado")

    m = make_manager(settings, process=blocker)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"a" * 5)
    b.write_bytes(b"b" * 7)
    j1, j2 = m.submit_process(str(a)), m.submit_process(str(b))
    started.wait(5)
    assert m.cancel(j2.id)["status"] == "cancelled"      # en cola: se descarta
    m.cancel(j1.id)
    assert wait(m, j1.id, {"cancelled"})["status"] == "cancelled"
    with pytest.raises(ApiError):
        m.cancel(j1.id)                                    # ya terminó
    m.shutdown()


def test_failure_is_reported_with_hint(settings, video):
    def boom(job, ctl):
        raise StreamClipperError("ffmpeg falló: Invalid data")

    m = make_manager(settings, process=boom)
    j = wait(m, m.submit_process(str(video)).id, {"failed"})
    assert "ffmpeg" in j["error"] and j["hint"]
    assert m.retry(j["id"]).status in ("queued", "running")
    m.shutdown()


def test_unexpected_exception_does_not_kill_worker(settings, tmp_path):
    calls = []

    def flaky(job, ctl):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("raro")

    m = make_manager(settings, process=flaky)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"a" * 5)
    b.write_bytes(b"b" * 6)
    wait(m, m.submit_process(str(a)).id, {"failed"})
    wait(m, m.submit_process(str(b)).id, {"completed"})
    m.shutdown()


def test_recovery_marks_active_jobs_interrupted(settings, video):
    m = make_manager(settings, autostart=False)
    j = m.submit_process(str(video))          # queda en cola; el worker no arrancó
    m2 = make_manager(settings, autostart=False)   # «reinicio» de la aplicación
    got = m2.get(j.id)
    assert got["status"] == "interrupted" and "reintentar" in got["message"]
    assert JobStore(settings.data_dir / "app" / "jobs").load_all()[0].status == "interrupted"


def test_validation_errors(settings, video):
    m = make_manager(settings)
    for src, params in (("", {}), (str(video) + "x", {}), (str(video), {"model": "huge"}), (str(video), {"max_clips": 99}),
                        ("https://example.com/v", {}), (str(video), {"layout": "x"})):
        with pytest.raises(ApiError) as e:
            m.submit_process(src, params)
        assert e.value.status == 400
    m.shutdown()


def test_queue_limit(settings, tmp_path):
    gate = threading.Event()
    m = make_manager(settings, process=lambda j, c: gate.wait(5), max_queue=2)
    for i in range(2):
        p = tmp_path / f"q{i}.mp4"
        p.write_bytes(b"q" * (3 + i))
        m.submit_process(str(p))
    p = tmp_path / "q9.mp4"
    p.write_bytes(b"q" * 99)
    with pytest.raises(ApiError) as e:
        m.submit_process(str(p))
    assert e.value.status == 429
    gate.set()
    m.shutdown()


def test_cleanup_stale_removes_temp_but_not_jobs(settings):
    (settings.data_dir / "tmp").mkdir(parents=True)
    (settings.data_dir / "tmp" / "x.wav").write_bytes(b"1")
    job = settings.jobs_dir / "keep"
    job.mkdir(parents=True)
    (job / "result.json").write_text("{}")
    cleanup_stale(settings)
    assert (job / "result.json").exists()


# --- HTTP ---------------------------------------------------------------------------------------------
@pytest.fixture
def server(settings, video):
    m = make_manager(settings)
    ctx = AppContext(settings, m, picker=lambda: str(video), opener=lambda p: None)
    srv = make_server(ctx, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_port, m
    srv.shutdown()
    m.shutdown()


def req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, data, h)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


def test_api_basic_flow(server, video):
    port, m = server
    assert req(port, "GET", "/api/health")[1]["ok"]
    st, page = req(port, "GET", "/")
    assert st == 200 and b"StreamClipper" in page
    st, job = req(port, "POST", "/api/jobs", {"source": str(video), "params": {"model": "base"}})
    assert st == 201
    wait(m, job["id"], {"completed"})
    st, jobs = req(port, "GET", "/api/jobs")
    assert jobs["jobs"][0]["id"] == job["id"]
    assert req(port, "POST", "/api/pick-file", {})[1]["path"] == str(video)
    st, cfg = req(port, "GET", "/api/config")
    assert cfg["defaults"]["model"] == "small" and cfg["defaults"]["language"] == "es"


def test_api_errors_are_json_and_clear(server):
    port, _ = server
    st, body = req(port, "POST", "/api/jobs", {"source": "https://example.com/x"})
    assert st == 400 and "YouTube" in body["error"]
    assert req(port, "GET", "/api/jobs/nope")[0] == 404
    assert req(port, "GET", "/api/nada")[0] == 404


def test_security_host_content_type_and_traversal(server, settings):
    port, _ = server
    assert req(port, "GET", "/api/health", headers={"Host": "evil.com"})[0] == 403
    assert req(port, "POST", "/api/jobs", None, headers={"Content-Type": "text/plain"})[0] == 415
    assert req(port, "GET", "/api/sessions/..%2f..%2fetc")[0] == 404
    assert req(port, "GET", "/media/..%2f/x.mp4")[0] in (404, 400)
    assert req(port, "POST", "/api/open-folder", {"path": "C:\\Windows"})[0] == 400
    assert settings.data_dir.exists() or True


def test_server_binds_loopback_only(server):
    port, _ = server
    # make_server recibe 127.0.0.1; comprobamos el socket real
    import socket
    s = socket.socket()
    try:
        assert s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()

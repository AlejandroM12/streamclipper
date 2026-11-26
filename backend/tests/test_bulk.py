"""Regeneración masiva (1.5, etapa 4): validación, secuencia, aislamiento de fallos, cancelación, reintento y persistencia."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from streamclipper.app import runners as rn
from streamclipper.app.manager import JobManager
from streamclipper.app.runners import run_bulk
from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.webutil import ApiError
from tests.test_editor import make_session

ITEMS = [{"session": "20261124-100000-small-a", "clip": "clip_01"}, {"session": "20261124-100000-small-b", "clip": "clip_01"},
         {"session": "20261124-100000-small-c", "clip": "clip_01"}]


def wait(m, jid, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        j = m.get(jid)
        j = SimpleNamespace(**j) if isinstance(j, dict) else j
        if j.status not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError("el trabajo no terminó")


def manager(tmp_path, **runners):
    cfg = Settings(data_dir=tmp_path / "d")
    for s in ("a", "b", "c"):
        make_session(cfg, f"20261124-100000-small-{s}", context=False)
    return cfg, JobManager(cfg, runners={"bulk": run_bulk(cfg), "process": lambda j, c: None, "edit": lambda j, c: None, "vod": lambda j, c: None, **runners})


def test_submit_validation(tmp_path):
    cfg, m = manager(tmp_path)
    for mode, items, opts in (("otro", ITEMS, {}), ("render", [], {}), ("render", ITEMS * 14, {}), ("render", [{"session": "../x", "clip": "clip_01"}], {}),
                              ("render", [{"session": "a", "clip": "c l"}], {}), ("render", [ITEMS[0], ITEMS[0]], {}),
                              ("render", ITEMS, {"preset": "gigante"}), ("render", ITEMS, {"composition": {"preset": "reacciones", "main": [2, 2, 1, 1]}})):
        with pytest.raises(ApiError):
            m.submit_bulk(mode, items, opts)
    m.shutdown(timeout=0.2)


def test_items_run_one_at_a_time_in_order_with_progress_and_persisted_state(tmp_path, monkeypatch):
    cfg, m = manager(tmp_path)
    live, peak, order = 0, 0, []

    def fake_render(d, clip, params, settings, cancel=None, progress=None):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        order.append((d.name, params.composition["preset"]))
        time.sleep(0.05)
        live -= 1
        return {"version": "edit-v001", "duration": params.end - params.start}
    monkeypatch.setattr(rn, "render_edit", fake_render)
    job = m.submit_bulk("render", ITEMS, {"extend": False, "composition": {"preset": "gameplay", "main": [0.1, 0.1, 0.5, 0.5]}})
    j = wait(m, job.id)
    assert j.status == "completed" and peak == 1                                   # nunca dos clips a la vez
    assert [o[0] for o in order] == [i["session"] for i in ITEMS] and all(p == "gameplay" for _, p in order)
    assert [i["status"] for i in j.result["items"]] == ["done"] * 3 and all(i["version"] == "edit-v001" for i in j.result["items"])
    saved = json.loads((cfg.data_dir / "app" / "jobs" / f"{job.id}.json").read_text(encoding="utf-8"))    # el estado queda en disco
    assert [i["status"] for i in saved["result"]["items"]] == ["done"] * 3 and saved["status"] == "completed"
    m.shutdown(timeout=0.5)


def test_a_failed_clip_does_not_stop_the_others_and_retry_only_repeats_the_failures(tmp_path, monkeypatch):
    cfg, m = manager(tmp_path)
    calls, broken = [], {"20261124-100000-small-b"}

    def fake_render(d, clip, params, settings, cancel=None, progress=None):
        calls.append(d.name)
        if d.name in broken:
            raise InvalidInputError("La fuente ya no está disponible.")
        return {"version": "edit-v001", "duration": 10.0}
    monkeypatch.setattr(rn, "render_edit", fake_render)
    job = m.submit_bulk("render", ITEMS, {"extend": False})
    j = wait(m, job.id)
    assert j.status == "failed" and [i["status"] for i in j.result["items"]] == ["done", "failed", "done"]
    assert "1 de 3" in j.error and "ya no está disponible" in j.result["items"][1]["error"] and calls == [i["session"] for i in ITEMS]
    broken.clear()
    m.retry(job.id)
    j2 = wait(m, job.id)
    assert j2.status == "completed" and [i["status"] for i in j2.result["items"]] == ["done"] * 3
    assert calls[3:] == ["20261124-100000-small-b"]                                # solo se repitió el clip fallido
    m.shutdown(timeout=0.5)


def test_cancel_stops_after_the_current_clip_and_keeps_finished_ones(tmp_path, monkeypatch):
    cfg, m = manager(tmp_path)
    started, release = threading.Event(), threading.Event()

    def slow_render(d, clip, params, settings, cancel=None, progress=None):
        started.set()
        release.wait(5)
        return {"version": "edit-v001", "duration": 10.0}
    monkeypatch.setattr(rn, "render_edit", slow_render)
    job = m.submit_bulk("render", ITEMS, {"extend": False})
    assert started.wait(5)
    m.cancel(job.id)
    release.set()
    j = wait(m, job.id)
    st = [i["status"] for i in j.result["items"]]
    assert j.status == "cancelled" and st[0] in ("done", "cancelled") and st[1:] == ["pending", "pending"]
    m.retry(job.id)                                                                 # un trabajo cancelado se puede continuar
    assert wait(m, job.id).status == "completed"
    m.shutdown(timeout=0.5)


def test_propose_without_extending_returns_the_original_interval_and_render_needs_a_proposal_when_extending(tmp_path):
    cfg, m = manager(tmp_path)
    j = wait(m, m.submit_bulk("propose", ITEMS[:1], {"extend": False}).id)
    p = j.result["items"][0]["proposal"]
    assert j.status == "completed" and p["plain"] and p["duration"] == 5.0 and p["start"] == 4.0
    j = wait(m, m.submit_bulk("render", ITEMS[:1], {"extend": True}).id)             # sin propuesta previa: error claro, nada se renderiza
    assert j.status == "failed" and "Ver propuestas" in j.result["items"][0]["error"]
    m.shutdown(timeout=0.5)


SAMPLE = REPO_ROOT / "data" / "samples" / "synthetic.mp4"


@pytest.mark.integration
def test_real_bulk_regeneration_of_two_clips_with_a_composition_keeps_originals(tmp_path):
    from streamclipper.pipeline import run_pipeline
    if not SAMPLE.exists():
        pytest.skip("falta data/samples/synthetic.mp4")
    cfg = Settings(data_dir=tmp_path / "data", language="en")
    res = run_pipeline(str(SAMPLE), cfg)
    sess = Path(res["job_dir"]).name
    assert len(res["clips"]) == 2
    before = {c["name"]: Path(c["file"]).read_bytes() for c in res["clips"]}
    m = JobManager(cfg, runners={"bulk": run_bulk(cfg), "process": lambda j, c: None, "edit": lambda j, c: None, "vod": lambda j, c: None})
    items = [{"session": sess, "clip": c["name"]} for c in res["clips"]]
    comp = {"preset": "reacciones", "cams": [[0, 0, 0.3, 0.3]], "main": [0.1, 0.3, 0.8, 0.7]}
    j = wait(m, m.submit_bulk("render", items, {"extend": False, "composition": comp}).id, timeout=300)
    assert j.status == "completed", j.error
    for it in j.result["items"]:
        v = Path(res["job_dir"]) / "edits" / it["clip"] / "v001"
        meta = json.loads((v / "meta.json").read_text(encoding="utf-8"))
        assert (v / "clip.mp4").is_file() and meta["layout"] == "stack" and meta["validation"]["ok"] and it["version"] == "edit-v001"
    assert {c["name"]: Path(c["file"]).read_bytes() for c in res["clips"]} == before      # los originales no cambiaron
    m.shutdown(timeout=0.5)

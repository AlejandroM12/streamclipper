"""Editor manual de clips (1.5, etapa 2): eje de tiempo único, conversión a tiempo absoluto, forma de onda, fotogramas y previsualización."""
import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from streamclipper.app.manager import JobManager
from streamclipper.app.server import AppContext, make_server
from streamclipper.core.config import Settings
from streamclipper.editor import editor_data, waveform

FF = Settings().ffmpeg


def make_video(path: Path, seconds: int = 12, w: int = 640, h: int = 360, audio: bool = True) -> Path:
    cmd = [str(FF), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=15:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True, capture_output=True)
    return path


def seg(a, b, text):
    return {"start": a, "end": b, "text": text, "words": []}


def make_session(cfg: Settings, name: str, *, context: bool, source_exists: bool = True) -> Path:
    """Sesión del modo automático: la fuente de la sesión es un recorte (t=0 ↔ VOD 1000 s) y, con `context`, hay un tramo recuperado
    (t=0 ↔ VOD 960 s): el eje del editor es el del tramo, 40 s por delante del eje de la fuente."""
    d = cfg.jobs_dir / name
    (d / "out").mkdir(parents=True)
    src = d / "source.mp4"
    if source_exists:
        make_video(src, 12)
    comps = [{"name": n, "value": 0.5, "weight": 0.3, "contribution": 0.15, "note": "", "window": None, "evidence": []}
             for n in ("intensidad_audio", "dinamica_audio", "reaccion_verbal", "frases", "contexto", "duracion", "calidad_transcripcion")]
    (d / "result.json").write_text(json.dumps({
        "job_id": name, "source": str(src), "status": "completed", "warnings": [], "selected": [1], "source_fingerprint": "x",
        "settings": {"whisper_model": "small", "language": "es", "data_dir": str(cfg.data_dir)}, "media": {"duration": 12.0},
        "candidates": [{"rank": 1, "start": 3, "end": 10, "anchor": 6, "score": 0.5, "triggers": [], "text": "x", "components": comps}],
        "clips": [{"name": "clip_01", "rank": 1, "start": 4.0, "end": 9.0, "duration": 5.0, "score": 0.5, "file": str(d / "out" / "clip_01.mp4"),
                   "selected_window": [3, 10], "context": {"event": [5.0, 8.0], "original": [3, 10], "contextual": [3, 10], "final": [4, 9]}}]}))
    (d / "transcript.json").write_text(json.dumps({"language": "es", "segments": [seg(2, 4, "hola"), seg(4.5, 8, "mira esto")], "evidence_version": 2}))
    (d / "segment.json").write_text(json.dumps({"mode": "auto-clip", "cut": [1000, 1012], "url": "https://kick.com/x/videos/abc", "group": "g",
                                                "start": 1003, "end": 1010, "index": 0, "count": 1}))
    if context:
        c = d / "context"
        c.mkdir()
        make_video(c / "source.mp4", 60)
        (c / "meta.json").write_text(json.dumps({"source": "source.mp4", "offset": 960, "range": [960, 1020], "url": "https://kick.com/x/videos/abc"}))
        (c / "transcript.json").write_text(json.dumps({"language": "es", "segments": [seg(40, 42, "hola"), seg(44.5, 48, "mira esto")], "evidence_version": 2}))
    return d


def test_context_axis_maps_the_original_and_exposes_absolute_vod_time(tmp_path):
    cfg = Settings(data_dir=tmp_path / "d")
    d = make_session(cfg, "20261124-100000-small-auto-clip-a", context=True)
    e = editor_data(d, "clip_01", cfg)
    assert e["available"] and e["axis"] == "context" and e["media"].endswith("/context/source.mp4") and abs(e["duration"] - 60) < 0.5
    assert e["offset_abs"] == 960 and e["can_fetch_more"] and not e["needs_vod_context"]
    assert e["original"] == [44.0, 49.0]                                     # 4–9 s del clip en la fuente + 40 s de diferencia de ejes
    assert e["event"] == [45.0, 48.0] and e["candidate"] == [43.0, 50.0] and e["anchor"] == 46.0
    assert [s[:2] for s in e["sentences"]] == [[40, 42], [44.5, 48]]           # transcripción del tramo, en su propio eje
    assert abs(len(e["waveform"]) * e["wave_hop"] - 60) < 1 and max(e["waveform"]) > 0.3
    assert e["offset_abs"] + e["original"][0] == 1004.0                        # = VOD 1000 s (inicio de la fuente) + 4 s: mismo instante en ambos ejes


def test_context_axis_translates_old_source_versions_to_the_editor_axis(tmp_path):
    cfg = Settings(data_dir=tmp_path / "d")
    d = make_session(cfg, "20261124-100000-small-auto-clip-b", context=True)
    for ver, start, end, ctx in (("v001", 4.0, 9.0, False), ("v002", 41.0, 52.0, True)):      # v001 en el eje de la fuente; v002 en el del tramo
        vd = d / "edits" / "clip_01" / ver
        vd.mkdir(parents=True)
        (vd / "meta.json").write_text(json.dumps({"version": "edit-" + ver, "start": start, "end": end, "context": ctx,
                                                  "context_note": {"confidence": "alta"} if ctx else None, "composition": {"preset": "reacciones"} if ctx else None}))
    e = editor_data(d, "clip_01", cfg, with_wave=False)
    assert [(v["id"], v["start"], v["end"]) for v in e["versions"]] == [("edit-v001", 44.0, 49.0), ("edit-v002", 41.0, 52.0)]
    assert e["versions"][1]["context"] and e["versions"][1]["composition"] == "reacciones"


def test_source_axis_when_only_the_trimmed_clip_exists_and_it_asks_for_vod_context(tmp_path):
    cfg = Settings(data_dir=tmp_path / "d")
    d = make_session(cfg, "20261124-100000-small-auto-clip-c", context=False)
    e = editor_data(d, "clip_01", cfg, with_wave=False)
    assert e["axis"] == "source" and e["original"] == [4.0, 9.0] and e["offset_abs"] == 1000 and e["needs_vod_context"] and not e["can_fetch_more"]


def test_missing_source_is_reported_not_crashed(tmp_path):
    cfg = Settings(data_dir=tmp_path / "d")
    d = make_session(cfg, "20261124-100000-small-auto-clip-d", context=False, source_exists=False)
    e = editor_data(d, "clip_01", cfg)
    assert e["available"] is False and "ya no está disponible" in e["reason"]


def test_waveform_is_cached_and_handles_media_without_audio(tmp_path):
    v = make_video(tmp_path / "a.mp4", 6)
    c = tmp_path / "cache" / "wave.json"
    w = waveform(v, c, FF)
    assert len(w) == 12 and c.is_file() and waveform(v, c, FF) == w
    assert waveform(make_video(tmp_path / "n.mp4", 4, audio=False), tmp_path / "cache" / "n.json", FF) == []


@pytest.fixture
def api(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data")
    m = JobManager(cfg, runners={"process": lambda j, c: None, "edit": lambda j, c: None, "vod": lambda j, c: None, "context": lambda j, c: None})
    srv = make_server(AppContext(cfg, m, picker=lambda: "", opener=lambda p: None), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}{path}", data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    yield cfg, call
    srv.shutdown()
    m.shutdown()


def test_http_editor_frame_and_composition_preview(api):
    cfg, call = api
    make_session(cfg, "20261124-100000-small-auto-clip-e", context=True)
    base = "/api/sessions/20261124-100000-small-auto-clip-e"
    st, e = call(base + "/editor?clip=clip_01")
    assert st == 200 and e["axis"] == "context" and e["original"] == [44.0, 49.0]
    st, f = call(base + "/frame?t=45")
    assert st == 200 and f["image"].startswith("data:image/jpeg;base64,") and len(f["image"]) > 1000
    comp = {"preset": "reacciones", "cams": [[0, 0, 0.3, 0.3]], "main": [0.1, 0.3, 0.8, 0.7]}
    st, p = call(base + "/compose-preview", {"composition": comp, "t": 45})
    assert st == 200 and p["image"].startswith("data:image/jpeg") and p["framing"]["regions"] == 2 and p["fallback"] is None
    st, p2 = call(base + "/compose-preview", {"composition": {"preset": "reacciones"}, "t": 45})
    assert st == 200 and "se usa Original" in p2["fallback"] and p2["framing"]["layout"] == "blur-fit"       # sin regiones: no inventa nada
    st, err = call(base + "/compose-preview", {"composition": {"preset": "reacciones", "main": [0.9, 0.9, 0.5, 0.5]}, "t": 1})
    assert st == 400                                                                                         # región fuera del cuadro
    assert call("/api/sessions/20261124-100000-small-auto-clip-e/editor?clip=clip_99")[0] == 400


def test_http_edit_accepts_a_composition_and_rejects_a_bad_one(api):
    cfg, call = api
    make_session(cfg, "20261124-100000-small-auto-clip-f", context=True)
    base = "/api/sessions/20261124-100000-small-auto-clip-f"
    body = {"clip": "clip_01", "start": 41.0, "end": 52.0, "context": True, "composition": {"preset": "gameplay", "main": [0.2, 0.2, 0.6, 0.6]}}
    st, job = call(base + "/edit", body)
    assert st == 201 and job["params"]["composition"]["preset"] == "gameplay" and job["params"]["context"] is True
    st, _ = call(base + "/edit", {**body, "composition": {"preset": "reacciones", "cams": [[0, 0, 0.01, 0.01]]}})
    assert st == 400

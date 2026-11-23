"""Regresión: «No se pudo cargar: Cannot read properties of undefined (reading 'max_minutes')».

Causa: la página (index.html, leída del disco en cada pedido) pedía `CONFIG.url_limits.max_minutes`, pero un servidor iniciado ANTES
de renombrar `youtube_limits` → `url_limits` seguía devolviendo el formato viejo; además, en Windows se podían abrir dos instancias
en el mismo puerto y la nueva nunca recibía tráfico. Estas pruebas fijan el contrato página ↔ API y reproducen el fallo en un
navegador real (se omiten si no hay Chrome).
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from streamclipper.app import server as srv_mod
from streamclipper.app.manager import JobManager
from streamclipper.app.server import API_CONTRACT, AppContext, make_server
from streamclipper.core.config import Settings
from streamclipper.media.download import inspect_media

HTML = (Path(srv_mod.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
CHROME = next((p for p in (os.environ.get("CHROME_PATH"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", shutil.which("chrome"), shutil.which("google-chrome"))
               if p and Path(p).is_file()), None)
needs_chrome = pytest.mark.skipif(CHROME is None, reason="requiere Chrome para probar la interfaz real")


def _fn(name: str) -> str:
    """Texto de una función de la página (hasta la siguiente función de primer nivel)."""
    m = re.search(rf"^(?:async )?function {name}\b.*?(?=^(?:async )?function |^/\* ----------)", HTML, re.S | re.M)
    assert m, name
    return m.group(0)


@pytest.fixture
def running(tmp_path):
    cfg = Settings(data_dir=tmp_path / "data")
    m = JobManager(cfg, runners={"process": lambda j, c: None, "edit": lambda j, c: None, "vod": lambda j, c: None})
    ctx = AppContext(cfg, m, picker=lambda: "", opener=lambda p: None)
    srv = make_server(ctx, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_port, ctx, cfg
    srv.shutdown()
    m.shutdown()


def get(port, path):
    import urllib.request
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as r:
        return r.read()


# --- contrato estático: lo que la página lee tiene que existir en lo que la API devuelve ------------------------
def test_every_config_key_the_page_reads_exists_in_api_config(running):
    port, _, _ = running
    cfg = json.loads(get(port, "/api/config"))
    top = set(re.findall(r"CONFIG\.([a-z_]+)", HTML))
    assert {"url_limits", "defaults", "models", "languages", "detectors", "platforms", "data_dir"} <= top      # la página sí usa estas
    missing = sorted(k for k in top if k not in cfg)
    assert not missing, f"la página lee CONFIG.{missing} pero /api/config no los devuelve: {sorted(cfg)}"
    for nested in set(re.findall(r"CONFIG\.url_limits\.([a-z_]+)", HTML)):               # el caso exacto del error
        assert nested in cfg["url_limits"], nested
    defaults = set(re.findall(r"\bD\.([a-z_]+)", _fn("viewNew")))
    assert defaults and defaults <= set(cfg["defaults"]), defaults - set(cfg["defaults"])
    assert "youtube_limits" not in cfg and "youtube_limits" not in HTML                  # el nombre viejo no debe quedar en ningún lado


def test_every_inspect_key_the_page_reads_exists_in_inspect_result():
    class Y:
        def __init__(self, o): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k):
            return {"title": "t", "duration": 5000, "uploader": "u", "tbr": 1000}
    for url in ("https://www.twitch.tv/videos/6528877", "https://clips.twitch.tv/FaintLightGullWholeWheat", "https://www.youtube.com/watch?v=aqz-KE-bpKQ"):
        r = inspect_media(url, ydl_factory=Y)
        check = re.search(r"async function checkUrl\(\).*?(?=\n  const segReset)", HTML, re.S).group(0)
        read = set(re.findall(r"\br\.([a-z_]+)", check)) | set(re.findall(r"\bsegInfo\.([a-z_]+)", _fn("viewNew")))
        assert {"platform_label", "title", "acceptable", "segmentable", "duration"} <= read          # la extracción encontró lo esperado
        assert read <= set(r), read - set(r)


def test_page_and_api_come_from_the_same_process_and_share_the_contract(running):
    port, ctx, _ = running
    page = get(port, "/").decode("utf-8")
    assert "__API_CONTRACT__" not in page and f'PAGE_CONTRACT = "{API_CONTRACT}"' in page
    assert json.loads(get(port, "/api/config"))["contract"] == API_CONTRACT == json.loads(get(port, "/api/health"))["contract"]
    # la página se carga una vez al arrancar: cambiar el archivo en disco NO cambia lo que sirve este proceso
    assert get(port, "/") == ctx.page


def test_a_second_instance_cannot_share_the_port():
    a = AppContext.__new__(AppContext)
    s1 = srv_mod.AppServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        with pytest.raises(OSError):
            srv_mod.AppServer(("127.0.0.1", s1.server_port), BaseHTTPRequestHandler)
        # y el reintento con otro puerto (lo que hace serve_app) funciona
        s3 = srv_mod.AppServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        assert s3.server_port != s1.server_port
        s3.server_close()
    finally:
        s1.server_close()
    del a


# --- navegador real --------------------------------------------------------------------------------------------
def _dom(url: str, budget: int = 7000) -> str:
    prof = tempfile.mkdtemp(prefix="sc-chrome-")
    try:
        out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-first-run",
                              f"--virtual-time-budget={budget}", "--dump-dom", url], capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    m = re.search(r'<main id="view">(.*?)</main>', out, re.S)
    assert m, out[:300]
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1)))


@needs_chrome
def test_browser_reproduces_the_stale_server_and_shows_a_clear_message(running):
    """Servidor viejo: devuelve `youtube_limits` y no `contract`, pero la página ya es la nueva. Antes: TypeError al leer max_minutes."""
    port, ctx, _ = running
    old_cfg = json.dumps({"version": "0.1.0", "defaults": {"model": "small"}, "models": [], "languages": ["es"], "layouts": ["blur-fit"],
                          "presets": [], "platforms": ["universal"], "youtube_limits": {"max_minutes": 20, "max_mb": 1500, "max_height": 1080},
                          "data_dir": "x"}).encode()

    class Old(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            body, ctype = (old_cfg, "application/json") if self.path == "/api/config" else (
                (ctx.page, "text/html; charset=utf-8") if self.path == "/" else (b"{}", "application/json"))
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    old = ThreadingHTTPServer(("127.0.0.1", 0), Old)
    threading.Thread(target=old.serve_forever, daemon=True).start()
    try:
        text = _dom(f"http://127.0.0.1:{old.server_port}/#/")
    finally:
        old.shutdown()
    assert "Cannot read properties" not in text and "max_minutes" not in text
    assert "versiones distintas" in text and "iniciar_streamclipper.bat" in text


def _old_format_session(cfg: Settings, name: str) -> None:
    """Sesión de una versión anterior: sin detector, sin transcripción reutilizada, sin segment.json, sin bloque `detector`."""
    d = cfg.jobs_dir / name
    (d / "out").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "job_id": name, "source": str(d / "v.mp4"), "status": "completed", "clips": [], "candidates": [], "selected": [], "warnings": [],
        "settings": {"whisper_model": "small", "language": "es", "layout": "blur-fit", "max_clips": 3, "data_dir": str(cfg.data_dir)},
        "media": {"duration": 120.0}, "source_fingerprint": "abc", "transcript": {"language": "es"}}))
    (d / "transcript.json").write_text(json.dumps({"language": "es", "segments": [], "evidence_version": 2}))


@needs_chrome
def test_browser_loads_new_jobs_sessions_and_status_screens_with_old_sessions_present(running):
    port, _, cfg = running
    _old_format_session(cfg, "20250101-000000-small-viejo")
    sess = json.loads(get(port, "/api/sessions"))["sessions"][0]
    assert sess["detector"] == "classic" and sess["segment"] is None and sess["reused_transcript"] is None   # compatibilidad hacia atrás
    base = f"http://127.0.0.1:{port}/"
    nuevo = _dom(base + "#/")
    assert "Nuevo video" in nuevo and "Detección de momentos" in nuevo and "hasta 20 min, 1500 MB y 1080p" in nuevo   # el texto que fallaba
    assert "Trabajos" in nuevo and "Todavía no procesaste" in nuevo
    sesiones = _dom(base + "#/sesiones")
    assert "Sesiones anteriores" in sesiones and "v.mp4" in sesiones and "Eliminar sesión" in sesiones
    assert "Detección:" in _dom(base + "#/sesion/20250101-000000-small-viejo")                                           # galería de sesión vieja
    estado = _dom(base + "#/estado")
    assert "Evaluación de detectores" in estado and "Análisis visual" in estado
    for text in (nuevo, sesiones, estado):
        assert "No se pudo cargar" not in text


@needs_chrome
def test_browser_vod_segment_editor_still_appears_for_a_vod(running, monkeypatch):
    """La vista previa de un VOD (segmentable) sigue alimentando el editor de tramos con el contrato actual."""
    port, ctx, _ = running
    monkeypatch.setattr(srv_mod, "inspect_media", lambda url, limits: {
        "platform": "kick", "platform_label": "Kick", "kind": "vod", "kind_label": "grabación (VOD)", "title": "VOD de prueba", "duration": 7200,
        "uploader": "u", "acceptable": True, "segmentable": True, "reason": None, "warnings": [], "estimated_mb": None,
        "max_segments": 5, "max_segment_s": 1200, "min_segment_s": 10})
    assert json.loads(subprocess.run(["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json", "-d", '{"url":"https://kick.com/a/videos/'
                                      '01a1fae2-a5c0-754b-888b-c05cc64e9741"}', f"http://127.0.0.1:{port}/api/inspect"], capture_output=True).stdout)["segmentable"]


# --- transmisión completa (automático): interfaz con clics reales ---------------------------------------------------
DRIVER = Path(__file__).parent / "cdp_driver.mjs"
needs_ui = pytest.mark.skipif(CHROME is None or shutil.which("node") is None, reason="requiere Chrome y Node para operar la interfaz")


def drive(url: str, steps: list[dict], read: str) -> dict:
    spec = Path(tempfile.mkdtemp(prefix="sc-spec-")) / "spec.json"
    spec.write_text(json.dumps({"url": url, "steps": steps, "read": read}), encoding="utf-8")
    try:
        out = subprocess.run(["node", str(DRIVER), CHROME, str(spec)], capture_output=True, timeout=150).stdout.decode("utf-8", "replace")
    finally:
        shutil.rmtree(spec.parent, ignore_errors=True)
    res = json.loads(out.strip().splitlines()[-1])
    assert res["ok"], res
    return res["value"]


def auto_info(duration=8 * 3600):
    from streamclipper.auto import TARGETS, estimates
    return {"platform": "kick", "platform_label": "Kick", "kind": "vod", "kind_label": "grabación (VOD)", "title": "Stream de 8 horas", "duration": duration,
            "uploader": "u", "acceptable": True, "segmentable": True, "reason": None, "warnings": [], "estimated_mb": None, "max_segments": 5,
            "max_segment_s": 1200, "min_segment_s": 10, "auto": {"available": True, "targets": list(TARGETS), **estimates(duration)}}


@needs_ui
def test_ui_automatic_mode_plans_24_blocks_and_starts_a_real_job(running, monkeypatch):
    port, ctx, _ = running
    info = auto_info()
    monkeypatch.setattr(srv_mod, "inspect_media", lambda url, limits: info)
    ctx.manager.inspector = lambda url: info
    gate = threading.Event()
    ctx.manager.runners["vod"] = lambda job, c: gate.wait(20)                     # el trabajo queda «en curso» sin tocar la red
    url = "https://kick.com/xqc/videos/01a1fae2-a5c0-754b-888b-c05cc64e9741"
    state = drive(f"http://127.0.0.1:{port}/#/", [
        {"js": f"(() => {{ const i = document.querySelector('input[type=text]'); i.value = '{url}'; i.dispatchEvent(new Event('input')); }})()",
         "wait": "document.body.innerText.includes('Tiempo estimado')"},
        {"js": "(() => { const s = [...document.querySelectorAll('select')].find((x) => [...x.options].some((o) => o.value === '20')); s.value = '20'; s.dispatchEvent(new Event('change')); })()"},
        {"js": "[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Procesar').click()",
         "wait": "document.querySelector('#jobs') && document.querySelector('#jobs').innerText.includes('Bloques completados')"},
    ], "({ text: document.querySelector('#view').innerText, mode: [...document.querySelectorAll('select')][0].value })")
    gate.set()
    text = state["text"]
    assert "24 bloques de 20 min" in text and "01:00:00" not in text.split("Se analizará")[0]                       # modo elegido solo para un stream largo
    assert "Bloques completados: 0 de 24 (0 %)" in text and "objetivo 20 clips" in text and "Detección Clásica" in text   # clásico por defecto
    assert "Cantidad de clips" not in text.split("Trabajos")[0]                                                    # el objetivo reemplaza a «cantidad de clips»
    jobs = json.loads(get(port, "/api/jobs"))["jobs"]
    assert jobs[0]["params"]["mode"] == "auto" and len(jobs[0]["segments"]) == 24 and jobs[0]["params"]["detector"] == "classic"


@needs_ui
def test_ui_manual_modes_still_work_for_a_vod_and_a_short_video(running, monkeypatch):
    port, ctx, _ = running
    monkeypatch.setattr(srv_mod, "inspect_media", lambda url, limits: auto_info(2 * 3600))
    url = "https://kick.com/xqc/videos/01a1fae2-a5c0-754b-888b-c05cc64e9741"
    text = drive(f"http://127.0.0.1:{port}/#/", [
        {"js": f"(() => {{ const i = document.querySelector('input[type=text]'); i.value = '{url}'; i.dispatchEvent(new Event('input')); }})()",
         "wait": "document.body.innerText.includes('Tiempo estimado')"},
        {"js": "(() => { const s = document.querySelector('select'); s.value = 'segments'; s.dispatchEvent(new Event('change')); })()",
         "wait": "document.body.innerText.includes('Tramos a procesar')"},
    ], "document.querySelector('#view').innerText")
    assert "Tramos a procesar" in text and "Agregar tramo" in text and "Tiempo estimado" not in text           # tramos manuales intactos
    short = dict(auto_info(300), segmentable=False, auto={"available": False})
    monkeypatch.setattr(srv_mod, "inspect_media", lambda url, limits: short)
    text = drive(f"http://127.0.0.1:{port}/#/", [
        {"js": "(() => { const i = document.querySelector('input[type=text]'); i.value = 'https://www.youtube.com/watch?v=aqz-KE-bpKQ'; i.dispatchEvent(new Event('input')); })()",
         "wait": "document.body.innerText.includes('Cumple los límites')"}], "document.querySelector('#view').innerText")
    assert "Modo de procesamiento" not in text and "Cantidad de clips" in text                                     # video corto: formulario de siempre


@needs_ui
def test_ui_auto_gallery_and_sessions_hide_analysis_sessions_and_group_final_clips(running):
    port, _, cfg = running
    _old_format_session(cfg, "20250101-000000-small-viejo")
    for name, meta in (("20261121-b0", {"group": "G1", "mode": "auto-block", "index": 0}),
                       ("20261121-c0", {"group": "G1", "mode": "auto-clip", "global_rank": 1, "kind": "recommended", "score": 0.5, "start": 100, "end": 120,
                                        "vod_title": "Stream largo", "platform": "kick", "detector": "classic", "block": 0, "index": 0, "count": 1,
                                        "start_hms": "00:01:40", "end_hms": "00:02:00", "top": []})):
        d = cfg.jobs_dir / name
        (d / "out").mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"job_id": name, "status": "completed", "source": "x.mp4", "candidates": [], "clips": [], "settings": {}, "media": {"duration": 60}}))
        (d / "segment.json").write_text(json.dumps(meta))
    text = drive(f"http://127.0.0.1:{port}/#/sesiones", [{"wait": "document.body.innerText.includes('Transmisión completa')"}], "document.querySelector('#view').innerText")
    assert "Stream largo" in text and "1 clip(s) final(es): 1 recomendados, 0 exploratorios · 1 bloque(s) analizados" in text
    assert text.count("Eliminar transmisión") == 1 and "v.mp4" in text                                          # la sesión antigua sigue visible, el análisis oculto
    assert "20261121-b0" not in text


# --- clips con contexto: selector de duración y «Ampliar con contexto» -------------------------------------------------------------
def _ctx_session(cfg: Settings, name: str) -> None:
    """Sesión con un clip de 10 s ya construido y una propuesta de contexto preparada (como la dejaría el trabajo `context`)."""
    d = cfg.jobs_dir / name
    (d / "out").mkdir(parents=True)
    (d / "context").mkdir()
    info = {"preset": "tiktok", "original": [8.0, 24.0], "event": [10.0, 20.0], "contextual": [4.0, 41.0], "final": [3.6, 41.5], "start_kind": "pausa",
            "end_kind": "pausa", "start_reason": "pausa natural de 6.0 s antes: ahí empieza la situación",
            "end_reason": "pausa natural de 4.0 s después: la situación terminó", "confidence": "alta", "estimated": False, "exceptional": False,
            "notes": ["contexto: 6 s antes y 21 s después del momento (10 s)"], "method": "transcripción + audio"}
    comps = [{"name": n, "value": 0.5, "weight": 0.3, "contribution": 0.15, "note": "", "window": None, "evidence": []}
             for n in ("intensidad_audio", "dinamica_audio", "reaccion_verbal", "frases", "contexto", "duracion", "calidad_transcripcion")]
    (d / "result.json").write_text(json.dumps({
        "job_id": name, "source": str(d / "v.mp4"), "status": "completed", "warnings": [], "selected": [1], "source_fingerprint": "x",
        "settings": {"whisper_model": "small", "language": "es", "data_dir": str(cfg.data_dir)}, "media": {"duration": 120.0},
        "candidates": [{"rank": 1, "start": 8, "end": 24, "anchor": 15, "score": 0.5, "triggers": [], "text": "x", "components": comps}],
        "clips": [{"name": "clip_01", "rank": 1, "start": 10.0, "end": 20.0, "duration": 10.0, "score": 0.5, "file": str(d / "out" / "clip_01.mp4"),
                   "selected_window": [8, 24], "context": None}]}))
    (d / "transcript.json").write_text(json.dumps({"language": "es", "segments": [], "evidence_version": 2}))
    (d / "context" / "proposal-clip_01.json").write_text(json.dumps({
        "clip": "clip_01", "preset": "tiktok", "source": "own", "start": 3.6, "end": 41.5, "duration": 37.9, "original": {"start": 10.0, "end": 20.0, "duration": 10.0},
        "context": info, "abs": None}))


@needs_ui
def test_ui_offers_the_three_duration_presets_with_tiktok_by_default_and_sends_the_choice(running, monkeypatch):
    port, ctx, _ = running
    short = dict(auto_info(300), segmentable=False, auto={"available": False})
    monkeypatch.setattr(srv_mod, "inspect_media", lambda url, limits: short)
    ctx.manager.runners["process"] = lambda j, c: threading.Event().wait(2)
    sels = drive(f"http://127.0.0.1:{port}/#/", [
        {"js": "(() => { const i = document.querySelector('input[type=text]'); i.value = 'https://www.youtube.com/watch?v=aqz-KE-bpKQ'; i.dispatchEvent(new Event('input')); })()",
         "wait": "document.body.innerText.includes('Cumple los límites')"},
        {"js": "(() => { const s = [...document.querySelectorAll('select')].find((x) => [...x.options].some((o) => o.value === 'cortos')); s.value = 'cortos'; s.dispatchEvent(new Event('change')); })()"},
        {"js": "[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Procesar').click()",
         "wait": "document.querySelector('#jobs') && document.querySelector('#jobs').innerText.includes('Clips cortos')"},
    ], "[...document.querySelectorAll('select')].map((s) => [...s.options].map((o) => o.value).join('|') + ':' + s.value)")
    assert any(x.startswith("cortos|tiktok|extendido") for x in sels)
    assert json.loads(get(port, "/api/jobs"))["jobs"][0]["params"]["clip_preset"] == "cortos"
    cfg = json.loads(get(port, "/api/config"))
    assert cfg["defaults"]["clip_preset"] == "tiktok" and [p["id"] for p in cfg["clip_presets"]] == ["cortos", "tiktok", "extendido"]


@needs_ui
def test_ui_expand_with_context_previews_the_proposal_and_regenerates_keeping_the_original(running):
    port, ctx, cfg = running
    _ctx_session(cfg, "20261121-100000-small-ctx")
    ctx.manager.runners["context"] = lambda j, c: None                             # la propuesta ya está preparada en disco
    ctx.manager.runners["edit"] = lambda j, c: None
    out = drive(f"http://127.0.0.1:{port}/#/sesion/20261121-100000-small-ctx", [
        {"wait": "document.body.innerText.includes('Ampliar con contexto')"},
        {"js": "[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Ampliar con contexto').click()",
         "wait": "document.body.innerText.includes('Buscar contexto')"},
        {"js": "[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Buscar contexto').click()",
         "wait": "document.body.innerText.includes('Confirmar y regenerar')"},
    ], "document.querySelector('.dlg').innerText")
    assert "Original: 10 s" in out and "propuesto: 37.9 s" in out and "pausa natural" in out and "confianza alta" in out
    assert "se conservan" in out and "Ver la propuesta" in out                      # previsualizar + el original no se toca
    drive(f"http://127.0.0.1:{port}/#/sesion/20261121-100000-small-ctx", [
        {"wait": "document.body.innerText.includes('Ampliar con contexto')"},
        {"js": "[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Ampliar con contexto').click()"},
        {"js": "[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Buscar contexto').click()",
         "wait": "document.body.innerText.includes('Confirmar y regenerar')"},
        {"js": "[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Confirmar y regenerar').click()"},
        {"wait": "true"},
    ], "true")
    jobs = json.loads(get(port, "/api/jobs"))["jobs"]
    edit = next(j for j in jobs if j["kind"] == "edit")
    assert edit["params"]["from_context"] is True and edit["params"]["note"]["confidence"] == "alta"             # el motivo lo pone el servidor
    assert edit["params"]["start"] == 3.6 and edit["params"]["end"] == 41.5


@needs_ui
def test_ui_gallery_shows_final_duration_and_clip_without_context_for_old_clips(running):
    port, _, cfg = running
    _ctx_session(cfg, "20261121-100000-small-ctx")
    text = drive(f"http://127.0.0.1:{port}/#/sesion/20261121-100000-small-ctx", [{"wait": "document.body.innerText.includes('Duración final')"}],
                 "document.querySelector('#view').innerText")
    assert "Duración final 10 s" in text and "recorte ajustado a la reacción (sin contexto automático)" in text    # un clip de 10 s se explica, no se esconde

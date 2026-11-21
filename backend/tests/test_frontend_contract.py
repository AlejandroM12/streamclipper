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
    assert "Nuevo video" in nuevo and "Detección de momentos" in nuevo and "Hasta 20 min, 1500 MB y 1080p" in nuevo   # el texto que fallaba
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

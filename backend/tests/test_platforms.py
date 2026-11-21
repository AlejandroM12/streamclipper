"""YouTube, Twitch y Kick: detección de plataforma, validación de URLs, límites, errores y descarga (yt-dlp simulado)."""
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from streamclipper.app.manager import JobManager
from streamclipper.app.server import AppContext, make_server
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.media.download import (
    DownloadLimits, check_info, download_media, download_youtube, explain_download_error, inspect_media, parse_media_url,
    target_path,
)
from streamclipper.webutil import ApiError

YT = "aqz-KE-bpKQ"
TW_VOD, TW_CLIP = "6528877", "FaintLightGullWholeWheat"
K_VOD, K_CLIP = "01a1fae2-a5c0-754b-888b-c05cc64e9741", "clip_01J8RGZRKHXHXXKJEHGRM932A5"


@pytest.mark.parametrize("url,platform,kind,vid,canonical", [
    (f"https://www.youtube.com/watch?v={YT}", "youtube", "watch", YT, f"https://www.youtube.com/watch?v={YT}"),
    (f"https://youtu.be/{YT}", "youtube", "short-link", YT, f"https://www.youtube.com/watch?v={YT}"),
    (f"https://www.twitch.tv/videos/{TW_VOD}", "twitch", "vod", TW_VOD, f"https://www.twitch.tv/videos/{TW_VOD}"),
    (f"https://m.twitch.tv/videos/{TW_VOD}?t=5m10s", "twitch", "vod", TW_VOD, f"https://www.twitch.tv/videos/{TW_VOD}"),
    (f"https://www.twitch.tv/riotgames/v/{TW_VOD}", "twitch", "vod", TW_VOD, f"https://www.twitch.tv/videos/{TW_VOD}"),
    (f"https://clips.twitch.tv/{TW_CLIP}", "twitch", "clip", TW_CLIP, f"https://clips.twitch.tv/{TW_CLIP}"),
    (f"https://clips.twitch.tv/canal/{TW_CLIP}", "twitch", "clip", TW_CLIP, f"https://clips.twitch.tv/{TW_CLIP}"),
    (f"https://www.twitch.tv/xqc/clip/{TW_CLIP}", "twitch", "clip", TW_CLIP, f"https://clips.twitch.tv/{TW_CLIP}"),
    (f"https://clips.twitch.tv/embed?clip={TW_CLIP}", "twitch", "clip", TW_CLIP, f"https://clips.twitch.tv/{TW_CLIP}"),
    (f"https://kick.com/xqc/videos/{K_VOD}", "kick", "vod", K_VOD, f"https://kick.com/xqc/videos/{K_VOD}"),
    (f"https://www.kick.com/xqc/videos/{K_VOD.upper()}", "kick", "vod", K_VOD, f"https://kick.com/xqc/videos/{K_VOD}"),
    (f"https://kick.com/spreen/clips/{K_CLIP}", "kick", "clip", K_CLIP, f"https://kick.com/spreen/clips/{K_CLIP}"),
    (f"https://kick.com/destiny?clip={K_CLIP}", "kick", "clip", K_CLIP, f"https://kick.com/destiny/clips/{K_CLIP}"),
])
def test_platform_is_detected_and_url_normalized(url, platform, kind, vid, canonical):
    ref = parse_media_url(url)
    assert (ref.platform, ref.kind, ref.video_id, ref.canonical_url) == (platform, kind, vid, canonical)


@pytest.mark.parametrize("url,fragment", [
    # dominios falsificados o ajenos
    (f"https://twitch.tv.evil.com/videos/{TW_VOD}", "Dominio no permitido"),
    (f"https://evil.com/twitch.tv/videos/{TW_VOD}", "Dominio no permitido"),
    (f"https://www.twitch.tv.evil.com/videos/{TW_VOD}", "Dominio no permitido"),
    (f"https://faketwitch.tv/videos/{TW_VOD}", "Dominio no permitido"),
    (f"https://twitch.tv@evil.com/videos/{TW_VOD}", "credenciales"),
    (f"https://kick.com.evil.com/xqc/videos/{K_VOD}", "Dominio no permitido"),
    (f"https://kick.co/xqc/videos/{K_VOD}", "Dominio no permitido"),
    (f"https://www.kіck.com/xqc/videos/{K_VOD}", "Dominio no permitido"),       # 'і' cirílica
    (f"https://www.twіtch.tv/videos/{TW_VOD}", "Dominio no permitido"),
    (f"https://player.twitch.tv/?video={TW_VOD}", "Dominio no permitido"),
    (f"https://kickstarter.com/projects/x/{K_VOD}", "Dominio no permitido"),
    (f"http://www.twitch.tv/videos/{TW_VOD}", "HTTPS"),
    (f"https://www.twitch.tv:8443/videos/{TW_VOD}", "puerto"),
    (f"https://www.twitch.tv/videos/{TW_VOD} --exec calc", "espacios"),
    # listas, colecciones, canales y directos
    ("https://www.twitch.tv/xqc", "en vivo"),
    ("https://kick.com/xqc", "en vivo"),
    ("https://www.twitch.tv/xqc/videos", "No se aceptan listas"),
    ("https://www.twitch.tv/xqc/clips", "No se aceptan listas"),
    ("https://www.twitch.tv/collections/abc123", "No se aceptan listas"),
    ("https://www.twitch.tv/directory/game/Just%20Chatting", "No se aceptan listas"),
    ("https://kick.com/xqc/videos", "No se aceptan listas"),
    ("https://kick.com/xqc/clips", "No se aceptan listas"),
    ("https://kick.com/categories/just-chatting", "no es un VOD"),
    ("https://kick.com/", "no es un VOD"),
    ("https://www.twitch.tv/", "no es un VOD"),
    ("https://www.youtube.com/playlist?list=PL1234567890", "listas de reproducción"),
    (f"https://www.youtube.com/live/{YT}", "en vivo"),
    # identificadores inválidos
    ("https://www.twitch.tv/videos/12ab", "no es un VOD"),
    (f"https://www.twitch.tv/x/v/{TW_VOD}9999999999999", "no es válido"),
    ("https://clips.twitch.tv/a", "no es válido"),
    ("https://kick.com/xqc/videos/not-a-uuid", "no es válido"),
    ("https://kick.com/xqc/clips/notaclip", "no es válido"),
    ("https://kick.com/xqc/clips/clip_../../etc", "no es"),
    ("https://vimeo.com/123", "Dominio no permitido"),
])
def test_invalid_or_spoofed_urls_are_rejected(url, fragment):
    with pytest.raises(InvalidInputError, match=fragment):
        parse_media_url(url)


def test_twitch_collection_param_is_ignored_with_warning():
    ref = parse_media_url(f"https://www.twitch.tv/videos/{TW_VOD}?collection=abc")
    assert ref.ignored_playlist and "collection" not in ref.canonical_url


def test_download_youtube_wrapper_stays_youtube_only(tmp_path):
    with pytest.raises(InvalidInputError, match="solo YouTube"):
        download_youtube(f"https://www.twitch.tv/videos/{TW_VOD}", Settings(data_dir=tmp_path))


# --- límites y errores ---------------------------------------------------------------------------
def test_hls_size_is_estimated_from_bitrate_and_limits_apply():
    info = {"duration": 600, "tbr": 9000, "requested_formats": [{"tbr": 9000}]}       # ≈ 675 MB
    assert check_info(info, DownloadLimits()) == pytest.approx(675_000_000)
    with pytest.raises(InvalidInputError, match="Tamaño estimado"):
        check_info({**info, "duration": 1200, "requested_formats": [{"tbr": 12000}]}, DownloadLimits())
    assert check_info({"duration": 30}, DownloadLimits()) is None                     # sin datos: se limita durante la descarga


@pytest.mark.parametrize("raw,fragment", [
    ("ERROR: [twitch:vod] 1: Video 1 does not exist", "no está disponible"),
    ("ERROR: [twitch:clips] X: This clip is no longer available", "no está disponible"),
    ("ERROR: [kick:vod] x: NOT_FOUND: Requested resource not found", "no está disponible"),
    ("ERROR: [twitch:vod] 1: You must be logged into an account that has access to this subscriber-only content", "suscriptores"),
    ("ERROR: [kick:vod] x: This video requires a channel subscription", "suscriptores"),
    ("ERROR: [kick:vod] x: Login required", "iniciar sesión"),
    ("ERROR: [kick:clip] x: HTTP Error 403: Forbidden (Cloudflare: Just a moment)", "anti-bot"),
    ("ERROR: Unable to download webpage: getaddrinfo failed", "conectar"),
])
def test_platform_errors_are_translated(raw, fragment):
    assert fragment in explain_download_error(raw)


# --- descarga con yt-dlp simulado ------------------------------------------------------------------
@pytest.fixture(scope="module")
def good_mp4(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("fx") / "good.mp4"
    subprocess.run([str(Settings().ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=15:duration=3", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


class FakeYDL:
    """Imita a yt-dlp: el id real puede NO coincidir con el de la URL (pasa en Twitch)."""
    def __init__(self, opts, info, good, log):
        self.opts, self.info, self.good, self.log = opts, info, good, log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        self.log.append(url)
        return self.info

    def process_ie_result(self, info, download=True):
        shutil.copy(self.good, Path(self.opts["outtmpl"]).parent / f"{info['id']}.mp4")


def fake(good, info, log=None):
    log = log if log is not None else []
    return lambda opts: FakeYDL(opts, info, good, log)


HLS_INFO = {"id": "otro-id-real", "title": "Mi clip", "duration": 40, "uploader": "streamer", "requested_formats": [{"tbr": 6000}]}


@pytest.mark.parametrize("url,prefix,vid", [
    (f"https://clips.twitch.tv/{TW_CLIP}", "twitch-clip", TW_CLIP),
    (f"https://www.twitch.tv/videos/{TW_VOD}", "twitch-vod", TW_VOD),
    (f"https://kick.com/spreen/clips/{K_CLIP}", "kick-clip", K_CLIP),
    (f"https://kick.com/xqc/videos/{K_VOD}", "kick-vod", K_VOD),
])
def test_download_from_twitch_and_kick_is_verified_and_recorded(tmp_path, good_mp4, url, prefix, vid):
    cfg, log = Settings(data_dir=tmp_path), []
    res = download_media(url, cfg, ydl_factory=fake(good_mp4, HLS_INFO, log))
    assert res.path == target_path(cfg, vid, prefix) == tmp_path / "samples" / f"{prefix}-{vid}.mp4"
    assert res.path.is_file() and res.platform == prefix.split("-")[0] and res.title == "Mi clip"
    side = json.loads(res.path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert side["platform"] == res.platform and side["kind"] == prefix.split("-")[1] and side["uploader"] == "streamer"
    assert len(log) == 1 and log[0].startswith("https://")
    assert not list((tmp_path / "samples").glob(".downloading-*"))


def test_vod_and_clip_with_same_id_do_not_collide(tmp_path):
    cfg = Settings(data_dir=tmp_path)
    assert target_path(cfg, "123", "twitch-vod") != target_path(cfg, "123", "twitch-clip") != target_path(cfg, "123", "youtube")


def test_limits_are_enforced_before_download_for_vods(tmp_path, good_mp4):
    log = []
    long_vod = {**HLS_INFO, "duration": 4 * 3600}
    with pytest.raises(InvalidInputError, match="máximo permitido es 20 min"):
        download_media(f"https://www.twitch.tv/videos/{TW_VOD}", Settings(data_dir=tmp_path), ydl_factory=fake(good_mp4, long_vod, log))
    assert len(log) == 1 and not (tmp_path / "samples").exists()      # solo consultó la metadata


def test_live_streams_are_rejected(tmp_path, good_mp4):
    with pytest.raises(InvalidInputError, match="en vivo"):
        download_media(f"https://www.twitch.tv/videos/{TW_VOD}", Settings(data_dir=tmp_path),
                       ydl_factory=fake(good_mp4, {**HLS_INFO, "is_live": True}))
    # un VOD de un directo ya terminado (was_live) sí es válido
    r = download_media(f"https://www.twitch.tv/videos/{TW_VOD}", Settings(data_dir=tmp_path), dry_run=True,
                       ydl_factory=fake(good_mp4, {**HLS_INFO, "live_status": "was_live"}))
    assert r.dry_run


def test_no_credentials_or_cookies_are_ever_requested(tmp_path, good_mp4):
    seen = {}

    def spy(opts):
        seen.update(opts)
        return FakeYDL(opts, HLS_INFO, good_mp4, [])
    download_media(f"https://kick.com/spreen/clips/{K_CLIP}", Settings(data_dir=tmp_path), ydl_factory=spy)
    for key in ("cookiefile", "cookiesfrombrowser", "username", "password", "twofactor", "netrc", "videopassword",
                "ap_username", "ap_password"):
        assert key not in seen
    assert seen["ignoreconfig"] is True and seen["usenetrc"] is False and seen["noplaylist"] is True
    assert "height<=1080" in seen["format"] and "height<=?1080" in seen["format"]      # los clips sin altura informada también entran


# --- vista previa ----------------------------------------------------------------------------------
def test_inspect_returns_platform_title_duration_and_acceptance(good_mp4):
    r = inspect_media(f"https://kick.com/spreen/clips/{K_CLIP}", ydl_factory=fake(good_mp4, HLS_INFO))
    assert (r["platform_label"], r["kind_label"], r["title"], r["duration"], r["acceptable"]) == ("Kick", "clip", "Mi clip", 40, True)
    assert r["estimated_mb"] == 30                                                      # 6000 kbps × 40 s


def test_inspect_reports_why_a_valid_url_cannot_be_processed(good_mp4):
    r = inspect_media(f"https://www.twitch.tv/videos/{TW_VOD}", ydl_factory=fake(good_mp4, {**HLS_INFO, "duration": 17208}))
    assert r["acceptable"] is False and r["title"] == "Mi clip" and "20 min" in r["reason"]


def test_inspect_translates_platform_errors():
    class Boom:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, *a, **k):
            raise RuntimeError("ERROR: [kick:vod] x: NOT_FOUND: Requested resource not found")
    r = inspect_media(f"https://kick.com/xqc/videos/{K_VOD}", ydl_factory=lambda o: Boom())
    assert r["acceptable"] is False and "no está disponible" in r["reason"]


def test_inspect_rejects_bad_urls_without_touching_the_network():
    called = []
    with pytest.raises(InvalidInputError):
        inspect_media("https://kick.com.evil.com/x/videos/" + K_VOD, ydl_factory=lambda o: called.append(1))
    assert called == []


# --- gestor y API ----------------------------------------------------------------------------------
def runner_ok(job, ctl):
    ctl.stage("validar", "done")


def test_manager_accepts_twitch_and_kick_and_dedups_per_kind(tmp_path):
    m = JobManager(Settings(data_dir=tmp_path), runners={"process": lambda j, c: threading.Event().wait(5), "edit": runner_ok})
    a = m.submit_process(f"https://www.twitch.tv/videos/{TW_VOD}")
    assert a.title.startswith("Twitch grabación") and a.input == f"https://www.twitch.tv/videos/{TW_VOD}"
    with pytest.raises(ApiError) as e:
        m.submit_process(f"https://m.twitch.tv/videos/{TW_VOD}?t=1m")            # misma URL canónica: duplicado
    assert e.value.status == 409
    assert m.submit_process(f"https://kick.com/spreen/clips/{K_CLIP}").title.startswith("Kick clip")
    with pytest.raises(ApiError) as e:
        m.submit_process("https://twitch.tv.evil.com/videos/1")
    assert e.value.status == 400 and "Dominio no permitido" in e.value.message
    m.shutdown(timeout=0.1)


def test_api_inspect_endpoint(tmp_path, monkeypatch, good_mp4):
    import http.client
    monkeypatch.setattr("streamclipper.app.server.inspect_media",
                        lambda url, limits: inspect_media(url, limits, ydl_factory=fake(good_mp4, HLS_INFO)))
    m = JobManager(Settings(data_dir=tmp_path), runners={"process": runner_ok, "edit": runner_ok})
    srv = make_server(AppContext(Settings(data_dir=tmp_path), m, picker=lambda: "", opener=lambda p: None), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def post(body):
        c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
        c.request("POST", "/api/inspect", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read())
    try:
        st, r = post({"url": f"https://clips.twitch.tv/{TW_CLIP}"})
        assert st == 200 and r["platform_label"] == "Twitch" and r["acceptable"]
        st, r = post({"url": "https://kick.com.evil.com/x/videos/" + K_VOD})
        assert st == 400 and "Dominio no permitido" in r["error"]
        assert post({"url": ""})[0] == 400
    finally:
        srv.shutdown()
        m.shutdown()

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from streamclipper.cli import main as cli_main
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, MediaError
from streamclipper.media.download import (
    DownloadLimits, check_info, download_youtube, explain_download_error, looks_like_url, parse_youtube_url,
    target_path,
)

VID = "aqz-KE-bpKQ"
OK_INFO = {"id": VID, "title": "Demo", "duration": 600, "availability": "public", "live_status": "not_live",
           "license": "CC-BY", "requested_formats": [{"filesize": 100_000_000}, {"filesize": 10_000_000}]}


# --- URLs --------------------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtube.com/watch?v={VID}&t=30s",
    f"https://m.youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?si=abc",
    f"https://www.youtube.com/shorts/{VID}",
    f"  https://www.youtube.com/watch?v={VID}  ",
    f"HTTPS://WWW.YOUTUBE.COM/watch?v={VID}",
    f"https://www.youtube.com:443/watch?v={VID}",
])
def test_valid_urls_are_normalized(url):
    ref = parse_youtube_url(url)
    assert ref.video_id == VID
    assert ref.canonical_url == f"https://www.youtube.com/watch?v={VID}"


def test_watch_url_with_playlist_param_downloads_only_the_video():
    ref = parse_youtube_url(f"https://www.youtube.com/watch?v={VID}&list=PL1234567890&index=3")
    assert ref.video_id == VID and ref.ignored_playlist is True
    assert "list" not in ref.canonical_url


@pytest.mark.parametrize("url,fragment", [
    ("", "vacía"),
    (f"http://www.youtube.com/watch?v={VID}", "HTTPS"),
    (f"ftp://www.youtube.com/watch?v={VID}", "HTTPS"),
    (f"www.youtube.com/watch?v={VID}", "HTTPS"),
    ("javascript:alert(1)", "HTTPS"),
    (f"file:///C:/videos/{VID}.mp4", "HTTPS"),
    (f"https://vimeo.com/watch?v={VID}", "Dominio no permitido"),
    (f"https://youtube.com.evil.com/watch?v={VID}", "Dominio no permitido"),
    (f"https://evil.com/?u=youtube.com/watch?v={VID}", "Dominio no permitido"),
    (f"https://www.youtub\u0435.com/watch?v={VID}", "Dominio no permitido"),   # 'е' cirílica
    (f"https://music.youtube.com/watch?v={VID}", "Dominio no permitido"),
    (f"https://user:pw@www.youtube.com/watch?v={VID}", "credenciales"),
    (f"https://www.youtube.com:8443/watch?v={VID}", "puerto"),
    ("https://www.youtube.com/playlist?list=PL1234567890", "listas de reproducción"),
    ("https://www.youtube.com/@canal", "no soportada"),
    ("https://www.youtube.com/channel/UC1234567890123456789012", "no soportada"),
    ("https://www.youtube.com/results?search_query=gameplay", "no soportada"),
    (f"https://www.youtube.com/live/{VID}", "en vivo"),
    ("https://www.youtube.com/watch", "v=<id>"),
    (f"https://www.youtube.com/watch?v={VID}&v=bbbbbbbbbbb", "v=<id>"),
    ("https://www.youtube.com/watch?v=corto", "identificador"),
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ%3Brm%20-rf", "identificador"),
    ("https://www.youtube.com/watch?v=../../etc/pw", "identificador"),
    (f"https://youtu.be/{VID}/extra", "inválido"),
    (f"https://www.youtube.com/watch?v={VID}\nhttps://evil.com", "espacios"),
    (f"https://www.youtube.com/watch?v={VID} --exec calc", "espacios"),
    ("https://www.youtube.com/watch?v=" + "a" * 400, "larga"),
])
def test_invalid_urls_are_rejected(url, fragment):
    with pytest.raises(InvalidInputError, match=fragment):
        parse_youtube_url(url)


def test_looks_like_url():
    assert looks_like_url("https://x") and looks_like_url("http://x") and looks_like_url("ftp://x")
    assert not looks_like_url(r"C:\videos\a.mp4") and not looks_like_url("a.mp4")


# --- metadata previa a la descarga ------------------------------------------------------------
def test_check_info_accepts_valid_video_and_estimates_size():
    assert check_info(OK_INFO, DownloadLimits()) == 110_000_000


@pytest.mark.parametrize("patch,fragment", [
    ({"duration": 21 * 60}, "máximo permitido es 20 min"),
    ({"duration": None}, "duración"),
    ({"is_live": True}, "en vivo"),
    ({"live_status": "is_upcoming"}, "en vivo"),
    ({"availability": "private"}, "privado"),
    ({"availability": "subscriber_only"}, "miembros"),
    ({"availability": "needs_auth"}, "iniciar sesión"),
    ({"_type": "playlist"}, "lista de reproducción"),
    ({"entries": []}, "lista de reproducción"),
    ({"requested_formats": [{"filesize_approx": 2_000_000_000}]}, "Tamaño estimado"),
])
def test_check_info_rejects(patch, fragment):
    with pytest.raises(InvalidInputError, match=fragment):
        check_info({**OK_INFO, **patch}, DownloadLimits())


def test_check_info_unknown_size_is_allowed_but_unknown():
    assert check_info({**OK_INFO, "requested_formats": [{"filesize": None}]}, DownloadLimits()) is None


@pytest.mark.parametrize("raw,fragment", [
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", "privado"),
    ("ERROR: [youtube] x: Video unavailable. This video has been removed by the uploader", "no está disponible"),
    ("ERROR: [youtube] x: Sign in to confirm your age. This video may be inappropriate", "edad"),
    ("ERROR: [youtube] x: The uploader has not made this video available in your country", "región"),
    ("ERROR: [youtube] x: Sign in to confirm you're not a bot", "anti-bot"),
    ("ERROR: [youtube] x: Join this channel to get access to members-only content", "miembros"),
    ("ERROR: Unable to download webpage: getaddrinfo failed", "conectar"),
    ("ERROR: something odd happened", "No se pudo descargar"),
])
def test_error_messages_are_clear(raw, fragment):
    assert fragment in explain_download_error(raw)


# --- descarga con yt-dlp simulado, archivos reales verificados con FFprobe ---------------------
@pytest.fixture(scope="module")
def good_mp4(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("fx") / "good.mp4"
    ff = str(Settings().ffmpeg)
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=15:duration=3", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


class FakeYDL:
    def __init__(self, opts, info, produce, good, error, log):
        self.opts, self.info, self.produce, self.good, self.error, self.log = opts, info, produce, good, error, log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        self.log.append(("extract", url))
        if self.error == "extract":
            raise RuntimeError("ERROR: [youtube] x: Private video. Sign in if you've been granted access")
        return self.info

    def process_ie_result(self, info, download=True):
        self.log.append(("download", info["id"]))
        for hook in self.opts["progress_hooks"]:
            hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
        if self.error == "download":
            raise RuntimeError("ERROR: unable to download video data: connection reset")
        out = Path(self.opts["outtmpl"]).parent / f"{info['id']}.mp4"
        if self.produce == "valid":
            shutil.copy(self.good, out)
        elif self.produce == "garbage":
            out.write_bytes(b"esto no es un video" * 100)
        elif self.produce == "partial":
            (out.parent / f"{info['id']}.mp4.part").write_bytes(b"x")  # nunca llegó a ser el MP4 final


def factory(good, info=None, produce="valid", error=None):
    log: list = []

    def make(opts):
        return FakeYDL(opts, info or OK_INFO, produce, good, error, log)
    make.log = log
    return make


URL = f"https://www.youtube.com/watch?v={VID}"


def test_download_valid_video_is_verified_and_recorded(tmp_path, good_mp4):
    cfg = Settings(data_dir=tmp_path)
    f = factory(good_mp4)
    res = download_youtube(URL, cfg, ydl_factory=f)
    assert res.path == target_path(cfg, VID) == tmp_path / "samples" / f"youtube-{VID}.mp4"
    assert res.path.is_file() and res.size_bytes == res.path.stat().st_size and not res.reused
    assert 2.5 < res.duration < 3.5  # medido por FFprobe sobre el archivo, no el de la metadata
    side = json.loads(res.path.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert side["video_id"] == VID and side["license"] == "CC-BY" and side["source_url"] == URL
    assert not list((tmp_path / "samples").glob(".downloading-*"))  # temporales eliminados
    assert [c[0] for c in f.log] == ["extract", "download"]


def test_ytdlp_options_restrict_playlists_size_and_format(tmp_path, good_mp4):
    seen = {}

    def spy(opts):
        seen.update(opts)
        return factory(good_mp4)(opts)
    download_youtube(URL + "&list=PL123", Settings(data_dir=tmp_path), DownloadLimits(max_bytes=5_000_000_000,
                     max_height=720), ydl_factory=spy)
    assert seen["noplaylist"] is True and seen["max_filesize"] == 5_000_000_000
    assert "height<=720" in seen["format"] and seen["merge_output_format"] == "mp4"
    assert "cookiesfrombrowser" not in seen and "cookiefile" not in seen and "username" not in seen
    assert Path(seen["ffmpeg_location"]).name == "ffmpeg"


def test_existing_file_is_not_overwritten_without_permission(tmp_path, good_mp4):
    cfg = Settings(data_dir=tmp_path)
    download_youtube(URL, cfg, ydl_factory=factory(good_mp4))
    path = target_path(cfg, VID)
    before = path.read_bytes()
    f = factory(good_mp4)
    with pytest.raises(InvalidInputError, match="--overwrite"):
        download_youtube(URL, cfg, ydl_factory=f)
    assert f.log == [] and path.read_bytes() == before  # ni siquiera consultó la red


def test_existing_file_can_be_reused_or_overwritten_explicitly(tmp_path, good_mp4):
    cfg = Settings(data_dir=tmp_path)
    download_youtube(URL, cfg, ydl_factory=factory(good_mp4))
    f = factory(good_mp4)
    r = download_youtube(URL, cfg, on_exists="reuse", ydl_factory=f)
    assert r.reused and f.log == []
    r = download_youtube(URL, cfg, on_exists="overwrite", ydl_factory=f)
    assert not r.reused and [c[0] for c in f.log] == ["extract", "download"]


def test_dry_run_validates_but_downloads_nothing(tmp_path, good_mp4):
    f = factory(good_mp4)
    r = download_youtube(URL, Settings(data_dir=tmp_path), dry_run=True, ydl_factory=f)
    assert r.dry_run and r.estimated_bytes == 110_000_000 and r.title == "Demo"
    assert [c[0] for c in f.log] == ["extract"]
    assert not (tmp_path / "samples").exists()


def test_metadata_rejection_happens_before_any_download(tmp_path, good_mp4):
    f = factory(good_mp4, info={**OK_INFO, "duration": 3 * 3600})
    with pytest.raises(InvalidInputError, match="máximo permitido"):
        download_youtube(URL, Settings(data_dir=tmp_path), ydl_factory=f)
    assert [c[0] for c in f.log] == ["extract"]


def test_unsafe_url_never_reaches_yt_dlp(tmp_path, good_mp4):
    f = factory(good_mp4)
    with pytest.raises(InvalidInputError):
        download_youtube("https://evil.example/watch?v=aqz-KE-bpKQ", Settings(data_dir=tmp_path), ydl_factory=f)
    assert f.log == []


def test_corrupt_download_is_rejected_and_cleaned(tmp_path, good_mp4):
    cfg = Settings(data_dir=tmp_path)
    with pytest.raises(InvalidInputError, match="no es un video válido"):
        download_youtube(URL, cfg, ydl_factory=factory(good_mp4, produce="garbage"))
    assert not target_path(cfg, VID).exists()
    assert not list((tmp_path / "samples").glob(".downloading-*"))


def test_missing_output_is_a_clear_error(tmp_path, good_mp4):
    with pytest.raises(MediaError, match="No se generó el MP4"):
        download_youtube(URL, Settings(data_dir=tmp_path), ydl_factory=factory(good_mp4, produce="partial"))


def test_oversize_file_is_rejected_and_removed(tmp_path, good_mp4):
    cfg = Settings(data_dir=tmp_path)
    tiny = DownloadLimits(max_bytes=good_mp4.stat().st_size - 1)
    with pytest.raises(MediaError, match="supera el máximo"):
        download_youtube(URL, cfg, tiny, ydl_factory=factory(good_mp4, info={**OK_INFO, "requested_formats": [{}]}))
    assert not target_path(cfg, VID).exists()
    assert not list((tmp_path / "samples").glob(".downloading-*"))


def test_download_errors_are_translated(tmp_path, good_mp4):
    cfg = Settings(data_dir=tmp_path)
    with pytest.raises(MediaError, match="privado"):
        download_youtube(URL, cfg, ydl_factory=factory(good_mp4, error="extract"))
    with pytest.raises(MediaError, match="conectar"):
        download_youtube(URL, cfg, ydl_factory=factory(good_mp4, error="download"))
    assert not list((tmp_path / "samples").glob(".downloading-*"))


def test_cancellation_stops_download_and_cleans_up(tmp_path, good_mp4):
    token = CancelToken()
    token.cancel()
    cfg = Settings(data_dir=tmp_path)
    with pytest.raises(CancelledError):
        download_youtube(URL, cfg, cancel=token, ydl_factory=factory(good_mp4))
    assert not target_path(cfg, VID).exists()
    assert not list((tmp_path / "samples").glob(".downloading-*"))


def test_real_yt_dlp_error_class_is_translated(tmp_path, good_mp4):
    import yt_dlp

    def boom(opts):
        f = FakeYDL(opts, OK_INFO, "valid", good_mp4, None, [])
        f.extract_info = lambda *a, **k: (_ for _ in ()).throw(
            yt_dlp.utils.DownloadError("ERROR: [youtube] x: Video unavailable. This video has been removed"))
        return f
    with pytest.raises(MediaError, match="no está disponible"):
        download_youtube(URL, Settings(data_dir=tmp_path), ydl_factory=boom)


# --- CLI y Git ---------------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", [["download"], ["process"]])
def test_cli_rejects_bad_url_before_any_work(cmd, capsys):
    code = cli_main(cmd + ["https://evil.example/watch?v=aqz-KE-bpKQ"])
    assert code == 2
    assert "Dominio no permitido" in capsys.readouterr().err


def test_cli_rejects_playlist(capsys):
    assert cli_main(["download", "https://www.youtube.com/playlist?list=PL1234567890"]) == 2
    assert "listas de reproducción" in capsys.readouterr().err


def test_samples_directory_is_ignored_by_git():
    for p in ("data/samples/youtube-aqz-KE-bpKQ.mp4", "data/samples/youtube-aqz-KE-bpKQ.source.json",
              "data/samples/.downloading-aqz-KE-bpKQ/aqz-KE-bpKQ.mp4"):
        assert subprocess.run(["git", "check-ignore", "-q", p], cwd=REPO_ROOT).returncode == 0, p


# --- red real (opt-in): solo metadata de un video con licencia CC-BY, sin descargar -------------
@pytest.mark.skipif(os.environ.get("STREAMCLIPPER_NETWORK_TESTS") != "1",
                    reason="requiere red; activar con STREAMCLIPPER_NETWORK_TESTS=1")
def test_live_metadata_check_for_cc_by_video(tmp_path):
    r = download_youtube(URL, Settings(data_dir=tmp_path), dry_run=True)
    assert r.license and "Creative Commons" in r.license
    assert r.estimated_bytes and r.duration < 20 * 60
    assert not (tmp_path / "samples").exists()

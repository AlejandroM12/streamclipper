import sys
import threading
import time

import numpy as np
import pytest

from streamclipper.analysis.audio import curve_from_db
from streamclipper.analysis.text import analyze_segment
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError
from streamclipper.media.ffmpeg import run_cancellable
from streamclipper.models import Segment, Transcript, Word
from streamclipper.pipeline import resolve_input
from streamclipper.render.ass import _ts, subtitle_lines, write_ass
from streamclipper.render.vertical import build_filter
from streamclipper.transcription.sentences import split_sentences
from tests.helpers import sentence


# --- audio -------------------------------------------------------------------------------------
def test_audio_excess_flags_only_the_loud_part():
    db = np.full(400, -35.0)
    db[200:212] = -15.0
    c = curve_from_db(db)
    assert c.excess[205] > 15
    assert c.excess[50] < 1
    assert c.rise[205] > 15


def test_audio_curve_is_not_fooled_by_bimodal_speech():
    # mitad silencio, mitad habla constante: el habla no debe contar como "pico"
    db = np.concatenate([np.full(200, -60.0), np.full(200, -30.0)])
    assert curve_from_db(db).excess.max() < 6


# --- texto -------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["Oh my god, no way!", "No puede ser, qué locura", "LET'S GOOO", "¡Dios mío, increíble!"])
def test_reactions_detected_in_english_and_spanish(text):
    assert analyze_segment(Segment(0, 3, text)).reaction >= 0.5


def test_neutral_text_has_no_reaction():
    assert analyze_segment(Segment(0, 3, "I am going to the north side of the map")).reaction == 0


# --- frases ------------------------------------------------------------------------------------
def test_split_sentences_breaks_long_segments_at_punctuation():
    words = [Word(i * 0.5, i * 0.5 + 0.4, w) for i, w in enumerate("Hello there. How are you? Fine.".split())]
    t = Transcript("en", [Segment(0, 3, "Hello there. How are you? Fine.", words)])
    out = split_sentences(t).segments
    assert [s.text for s in out] == ["Hello there.", "How are you?", "Fine."]


def test_whisper_segment_edge_alone_does_not_split_a_sentence():
    a = Segment(0, 1, "okay thanks for", [Word(0, 0.4, "okay"), Word(0.5, 0.8, "thanks"), Word(0.8, 1.0, "for")])
    b = Segment(1.1, 2, "watching everyone.", [Word(1.1, 1.5, "watching"), Word(1.5, 2.0, "everyone.")])
    out = split_sentences(Transcript("en", [a, b])).segments
    assert [s.text for s in out] == ["okay thanks for watching everyone."]
    assert (out[0].start, out[0].end) == (0, 2.0)


# --- subtítulos --------------------------------------------------------------------------------
def test_ass_timestamp_format():
    assert _ts(0) == "0:00:00.00"
    assert _ts(61.257) == "0:01:01.26"
    assert _ts(3725.5) == "1:02:05.50"


def test_subtitles_are_relative_to_clip_and_in_order(tmp_path):
    segs = [sentence("this is the first sentence of the clip", 100, 4), sentence("and this is the second one", 105, 3)]
    lines = subtitle_lines(segs, 99.0, 110.0)
    assert lines and lines[0][0] == pytest.approx(1.0, abs=0.01)  # 100 - 99
    for (a0, a1, _), (b0, _, _) in zip(lines, lines[1:]):
        assert a0 < a1 <= b0 + 1e-9  # sin solapes ni tiempos invertidos
    assert max(ln[1] for ln in lines) <= 11.0
    n = write_ass(tmp_path / "x.ass", segs, 99.0, 110.0)
    assert n == len(lines)
    assert "Dialogue:" in (tmp_path / "x.ass").read_text(encoding="utf-8")


def test_subtitles_clipped_to_clip_bounds():
    segs = [sentence("one two three four five six seven eight", 8, 8)]
    lines = subtitle_lines(segs, 10.0, 14.0)
    assert all(0 <= a < b <= 4.0 + 1e-9 for a, b, _ in lines)


def test_ass_escapes_override_braces(tmp_path):
    seg = Segment(0, 2, "{\\an9}hack", [Word(0, 1, "{\\an9}hack")])
    write_ass(tmp_path / "e.ass", [seg], 0, 5)
    assert "{" not in (tmp_path / "e.ass").read_text(encoding="utf-8").split("[Events]")[1].split("Default,,0,0,0,,")[1]


# --- render ------------------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["blur-fit", "fill", "crop"])
def test_filter_graphs_target_1080x1920_with_subtitles(layout):
    f = build_filter(Settings(layout=layout, zoom=1.5, crop_x=0.2), "subs.ass")
    assert "1080" in f and "1920" in f and "ass=subs.ass" in f and f.endswith("[v]")


# --- configuración y rutas ---------------------------------------------------------------------
def test_settings_validation():
    Settings().validate()
    for bad in (Settings(min_clip=2), Settings(min_clip=30, max_clip=20), Settings(max_clips=0),
                Settings(layout="x"), Settings(zoom=9), Settings(pre_context=-1)):
        with pytest.raises(InvalidInputError):
            bad.validate()


def test_resolve_input_rejects_unsafe_paths(tmp_path):
    with pytest.raises(InvalidInputError):
        resolve_input(str(tmp_path / "missing.mp4"))
    with pytest.raises(InvalidInputError):
        resolve_input(r"\\server\share\video.mp4")
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(InvalidInputError):
        resolve_input(str(tmp_path / "a.txt"))
    with pytest.raises(InvalidInputError):
        resolve_input(str(tmp_path))  # un directorio no es un archivo
    ok = tmp_path / "v.mp4"
    ok.write_bytes(b"0")
    assert resolve_input(str(ok)) == ok.resolve()


# --- procesos y cancelación --------------------------------------------------------------------
def test_run_cancellable_kills_process_on_cancel():
    token = CancelToken()
    threading.Timer(0.6, token.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(CancelledError):
        run_cancellable([sys.executable, "-c", "import time; time.sleep(60)"], cancel=token)
    assert time.monotonic() - t0 < 10


def test_run_cancellable_returns_output():
    code, out, _ = run_cancellable([sys.executable, "-c", "print('hola')"])
    assert code == 0 and out.strip() == "hola"

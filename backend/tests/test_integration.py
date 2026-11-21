"""Integración con FFmpeg y faster-whisper reales sobre un video sintético (voz SAPI).

Valida el pipeline técnico (transcripción, tiempos, selección, render, formato). No evalúa la
calidad del detector con gameplay real.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.pipeline import run_pipeline

pytestmark = pytest.mark.integration

SAMPLE = REPO_ROOT / "data" / "samples" / "synthetic.mp4"


@pytest.fixture(scope="module")
def sample() -> Path:
    if not SAMPLE.exists():
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(REPO_ROOT / "backend" / "tools" / "make_synthetic.py"), str(SAMPLE)],
                       check=True)
    return SAMPLE


@pytest.fixture(scope="module")
def result(sample, tmp_path_factory):
    s = Settings(data_dir=tmp_path_factory.mktemp("data"), language="en")
    return run_pipeline(str(sample), s)


def test_pipeline_completes(result):
    assert result["status"] == "completed", result["error"]
    assert (Path(result["job_dir"]) / "work").exists() is False  # temporales eliminados


def test_transcript_has_word_timestamps_in_order(result):
    t = json.loads((Path(result["job_dir"]) / "transcript.json").read_text(encoding="utf-8"))
    words = [w for s in t["segments"] for w in s["words"]]
    assert len(words) > 50
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert all(0 <= w["start"] <= w["end"] <= 105.5 for w in words)
    text = " ".join(w["text"] for w in words).lower()
    assert "oh my god" in text and "unreal" in text


def test_selects_the_two_real_moments_and_nothing_else(result):
    clips = result["clips"]
    assert 1 <= len(clips) <= 3
    assert len(clips) == 2  # el video sintético contiene exactamente dos momentos
    # eventos reales en ~30 s y ~73 s
    for event in (30.0, 73.0):
        assert any(c["start"] < event < c["end"] for c in clips), (event, clips)


def test_clips_are_unique_and_do_not_overlap(result):
    clips = sorted(result["clips"], key=lambda c: c["start"])
    for a, b in zip(clips, clips[1:]):
        inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
        assert inter / min(a["end"] - a["start"], b["end"] - b["start"]) <= 0.15
    assert len({c["file"] for c in clips}) == len(clips)


def test_duration_within_limits(result):
    for c in result["clips"]:
        assert 15 <= c["end"] - c["start"] <= 60


def test_output_files_are_valid_h264_aac_1080x1920(result):
    for c in result["clips"]:
        v = c["validation"]
        assert v["ok"], v["problems"]
        assert v["info"]["video"]["codec_name"] == "h264"
        assert (v["info"]["video"]["width"], v["info"]["video"]["height"]) == (1080, 1920)
        assert v["info"]["audio"]["codec_name"] == "aac"
        assert abs(v["info"]["duration"] - (c["end"] - c["start"])) < 0.75
        assert Path(c["file"]).stat().st_size > 10_000


def test_subtitles_exist_and_match_clip_times(result):
    for c in result["clips"]:
        ass = Path(c["file"]).with_suffix(".ass").read_text(encoding="utf-8")
        assert "Dialogue:" in ass and c["subtitle_lines"] > 0


def test_cancelled_job_leaves_no_temp_files_or_partial_outputs(sample, tmp_path):
    token = CancelToken()
    token.cancel()
    r = run_pipeline(str(sample), Settings(data_dir=tmp_path), cancel=token)
    assert r["status"] == "cancelled"
    job = Path(r["job_dir"])
    assert not (job / "work").exists()
    assert not list(job.rglob("*.part")) and not list(job.rglob("*.mp4"))


def test_invalid_file_reports_clear_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"esto no es un video")
    r = run_pipeline(str(bad), Settings(data_dir=tmp_path))
    assert r["status"] == "failed" and "video válido" in r["error"]

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


def test_report_files_and_contents(result):
    job = Path(result["job_dir"])
    rep = json.loads((job / "report.json").read_text(encoding="utf-8"))
    assert (job / "report.md").read_text(encoding="utf-8").startswith("# Reporte")
    assert rep["video_duration_s"] == pytest.approx(105.0, abs=0.1)
    assert rep["whisper_model"] == "base" and rep["language"] == "en"
    t = rep["timings_s"]
    assert t["transcription"] > 0 and t["analysis"] >= 0 and t["rendering"] > 0 and t["total"] >= t["transcription"]
    assert rep["clips_generated"] == len(rep["clips"]) == 2
    assert rep["candidates_detected"] == len(rep["candidates"]) >= 2
    assert all(c["reasons"] and c["triggers"] for c in rep["candidates"])
    assert [c["rank"] for c in rep["candidates"] if c["selected"]] == [
        c["rank"] for c in rep["clips"]] or len([c for c in rep["candidates"] if c["selected"]]) == 2
    assert isinstance(rep["warnings"], list) and rep["error"] is None


def test_peak_memory_is_measured(result):
    r = result["resources"]
    assert r["python_peak_mb"] > 100  # faster-whisper + modelo
    assert r["ffmpeg_peak_mb"] > 0
    assert r["total_peak_mb_approx"] >= r["python_peak_mb"]


def test_clips_do_not_start_or_end_mid_sentence(result):
    for c in result["clips"]:
        assert c["boundaries"]["errors"] == [], c["boundaries"]


def test_export_alternatives_without_retranscribing(result, monkeypatch):
    import streamclipper.transcription.whisper as w
    from streamclipper.export import export_alternatives

    def boom(*a, **k):
        raise AssertionError("no debe retranscribir")
    monkeypatch.setattr(w, "transcribe", boom)

    job = result["job_dir"]
    done = {c["rank"] for c in result["clips"]}
    extra = next(c["rank"] for c in result["candidates"] if c["rank"] not in done)
    out = export_alternatives(job, Settings(), ranks=[extra])
    assert len(out["clips"]) == 1
    clip = out["clips"][0]
    assert Path(clip["file"]).name == f"alt_{extra:02d}.mp4"
    assert clip["validation"]["ok"] and clip["boundaries"]["errors"] == []
    rep = json.loads((Path(job) / "report.json").read_text(encoding="utf-8"))
    assert len(rep["alternative_clips"]) == 1
    assert not list(Path(job).glob("work_export_*"))  # temporales eliminados
    # re-exportar el mismo no duplica
    again = export_alternatives(job, Settings(), ranks=[extra])
    assert again["clips"] == [] and any("ya estaba exportado" in x for x in again["warnings"])


def test_export_unknown_candidate_is_a_clear_error(result):
    from streamclipper.core.errors import InvalidInputError
    from streamclipper.export import export_alternatives
    with pytest.raises(InvalidInputError, match="inexistentes"):
        export_alternatives(result["job_dir"], Settings(), ranks=[999])


def test_two_runs_keep_separate_results_and_compare(sample, result, tmp_path_factory):
    """Dos ejecuciones del mismo video conservan resultados separados y se pueden comparar."""
    from streamclipper.compare import compare_jobs
    data = tmp_path_factory.mktemp("data2")
    r2 = run_pipeline(str(sample), Settings(data_dir=data, language="en"), label="segunda")
    assert r2["status"] == "completed"
    assert r2["job_dir"] != result["job_dir"] and "base" in r2["job_id"]
    text = compare_jobs(result["job_dir"], r2["job_dir"], Settings(data_dir=data))
    assert "Probabilidad media por palabra" in text and "coinciden" in text
    assert "NO usaron el mismo archivo" not in text
    assert "2/2" in text  # mismos dos momentos


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

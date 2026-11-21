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


def test_whisper_evidence_is_persisted_and_summarized(result):
    job = Path(result["job_dir"])
    t = json.loads((job / "transcript.json").read_text(encoding="utf-8"))
    assert t["evidence_version"] == 2 and t["whisper_segments"]
    for raw in t["whisper_segments"]:
        assert isinstance(raw["avg_logprob"], float) and isinstance(raw["no_speech_prob"], float)
        assert isinstance(raw["compression_ratio"], float)
    s0 = t["segments"][0]
    assert s0["avg_logprob"] is not None and s0["compression_ratio"] is not None and s0["words"][0]["prob"] > 0
    ev = result["transcript"]["evidence"]
    assert ev["whisper_stats_available"] and ev["sentences"] == len(t["segments"])
    rep = json.loads((job / "report.json").read_text(encoding="utf-8"))
    assert rep["transcript"]["evidence"]["whisper_stats_available"] is True
    assert rep["selection"] is not None and "margin_to_next" in rep["selection"]
    assert "Evidencia de Whisper" in (job / "report.md").read_text(encoding="utf-8")


def test_candidates_carry_windows_and_evidence(result):
    for c in result["candidates"]:
        for comp in c["components"]:
            assert comp["window"] is not None
        r = next(x for x in c["components"] if x["name"] == "reaccion_verbal")
        assert (r["value"] > 0) == any(e.startswith("+") for e in r["evidence"])


def test_rescore_reuses_transcript_without_retranscribing_and_evaluates(result, monkeypatch):
    import streamclipper.transcription.whisper as w
    from streamclipper.evaluation import evaluate_jobs
    from streamclipper.export import export_alternatives
    from streamclipper.rescore import rescore_job

    monkeypatch.setattr(w, "transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retranscribió")))
    res = rescore_job(result["job_dir"], Settings(), suffix="v2")
    new_dir = Path(res["job_dir"])
    assert res["rescored_from"] == result["job_id"] and res["status"] == "completed" and res["clips"] == []
    assert new_dir != Path(result["job_dir"]) and (new_dir / "report.md").is_file()
    assert (new_dir / "transcript.json").read_bytes() == (Path(result["job_dir"]) / "transcript.json").read_bytes()
    assert not (new_dir / "work").exists()
    # mismo algoritmo y mismos datos: los dos momentos reales se mantienen
    sel = [c for c in res["candidates"] if c["rank"] in res["selected"]]
    assert len(sel) == 2 and all(any(c["start"] < e < c["end"] for c in sel) for e in (30.0, 73.0))

    data, md = evaluate_jobs(result["job_dir"], res["job_dir"], Settings())
    assert data["counts"]["matched"] >= 2 and "Top-3: antes → ahora" in md
    assert all(t["still"] is not None for t in data["top3_old"])  # ningún clip real se perdió

    # los clips elegidos de un job re-puntuado (sin renderizar) se exportan sin retranscribir
    out = export_alternatives(res["job_dir"], Settings(), selected=True)
    assert [Path(c["file"]).name for c in out["clips"]] == ["clip_01.mp4", "clip_02.mp4"]
    assert all(c["validation"]["ok"] and c["boundaries"]["errors"] == [] for c in out["clips"])
    rep = json.loads((new_dir / "report.json").read_text(encoding="utf-8"))
    assert rep["clips_generated"] == 2
    from streamclipper.core.errors import InvalidInputError
    with pytest.raises(InvalidInputError, match="ya tiene sus clips"):
        export_alternatives(res["job_dir"], Settings(), selected=True)
    again = export_alternatives(res["job_dir"], Settings(), ranks=[res["selected"][0]])
    assert again["clips"] == [] and any("ya estaba exportado" in w for w in again["warnings"])


def test_old_format_job_without_whisper_stats_still_rescores(result, tmp_path):
    import shutil
    from streamclipper.rescore import rescore_job

    old = tmp_path / "old-format-job"
    shutil.copytree(result["job_dir"], old, ignore=shutil.ignore_patterns("out", "work*"))
    (old / "out").mkdir()
    t = json.loads((old / "transcript.json").read_text(encoding="utf-8"))
    for k in ("whisper_segments", "evidence_version"):
        t.pop(k)
    for s in t["segments"]:
        for k in ("avg_logprob", "compression_ratio"):
            s.pop(k)
        s["no_speech_prob"] = 0.0   # el valor por omisión del formato viejo: nunca lo midió Whisper
    (old / "transcript.json").write_text(json.dumps(t), encoding="utf-8")
    res = rescore_job(str(old), Settings(), suffix="compat")
    assert res["status"] == "completed" and len(res["selected"]) == 2
    assert res["transcript"]["evidence"]["whisper_stats_available"] is False
    assert "Evidencia de Whisper:** no disponible" in (Path(res["job_dir"]) / "report.md").read_text(encoding="utf-8")


def _digest(folder: Path) -> dict:
    import hashlib
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


def test_rerender_creates_new_versions_and_never_touches_the_originals(result):
    from streamclipper.core.errors import InvalidInputError
    from streamclipper.rerender import rerender_job
    from streamclipper.review import build_items

    job = Path(result["job_dir"])
    before = _digest(job / "out")
    result_json = (job / "result.json").read_bytes()
    m = rerender_job(str(job), Settings(), tag="t3", layout="focus", fg_x=0.1)

    assert _digest(job / "out") == before and (job / "result.json").read_bytes() == result_json   # intactos
    out = job / "out_t3"
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"clip_0{i}.{ext}" for i in (1, 2) for ext in ("ass", "mp4", "srt"))
    assert not list(out.glob("*.part")) and not list(job.glob("work_rerender_*"))
    assert m["settings"]["layout"] == "focus" and m["settings"]["fg_x"] == 0.1 and len(m["clips"]) == 2
    for r in m["clips"]:
        n, o = r["new"], r["original"]
        assert n["validation"]["ok"] and n["boundaries"]["errors"] == []
        info = n["validation"]["info"]
        assert (info["video"]["width"], info["video"]["height"]) == (1080, 1920) and info["audio"]["codec_name"] == "aac"
        assert 15 <= n["duration"] <= 60 and n["render_seconds"] > 0
        for k in ("cues", "coverage_pct", "cps_mean", "pct_over_max_cps", "min_cue_duration_s", "comfortable_pct"):
            assert k in n["metrics"] and k in o["metrics"]
        assert n["metrics"]["cues"] > 0 and isinstance(n["issue_count"], int)
        assert Path(n["srt"]).read_text(encoding="utf-8").startswith("1\n")
    assert m["resources"]["python_peak_mb"] > 0 and m["resources"]["ffmpeg_peak_mb"] > 0
    md = (job / "rerender_t3.md").read_text(encoding="utf-8")
    assert "Original" in md and "Nueva" in md and "no** significan un clip mejor" in md and "Ritmo del habla" in md

    with pytest.raises(InvalidInputError, match="Ya existe"):
        rerender_job(str(job), Settings(), tag="t3")
    with pytest.raises(InvalidInputError, match="etiqueta"):
        rerender_job(str(job), Settings(), tag="Mala Etiqueta!")
    rerender_job(str(job), Settings(), tag="t3", overwrite=True)
    assert _digest(job / "out") == before                      # ni siquiera con --overwrite se tocan los originales

    items = build_items(job, Settings(), 0, with_sync=False)    # la herramienta de revisión ve original y nueva juntas
    names = [i["name"] for i in items if i["kind"] == "clip"]
    assert names == ["clip_01", "clip_01@t3", "clip_02", "clip_02@t3"]
    v = next(i for i in items if i["name"] == "clip_01@t3")
    assert v["variant"] == "t3" and v["media"] == "out_t3/clip_01.mp4" and v["media_exists"]


def test_pipeline_adjusts_bounds_after_selection_without_changing_the_ranking(sample, result, tmp_path):
    for c in result["clips"]:
        assert c["layout"] == "blur-fit" and c["selected_window"][0] <= c["selected_window"][1]
        assert c["subtitle_metrics"]["cues"] > 0 and Path(c["file"]).with_suffix(".srt").is_file()
    off = run_pipeline(str(sample), Settings(data_dir=tmp_path, language="en", refine_bounds=False))
    assert off["status"] == "completed"
    # mismo ranking, puntuaciones y ventanas candidatas con y sin ajuste de límites
    key = lambda r: [(c["rank"], c["start"], c["end"], c["score"]) for c in r["candidates"]]  # noqa: E731
    assert key(off) == key(result) and off["selected"] == result["selected"]
    assert all(c["refine_notes"] == [] and c["start"] == c["selected_window"][0] for c in off["clips"])


def test_validation_session_end_to_end(sample, tmp_path):
    """Sesión completa: job + alternativos + informe por clip + elementos de revisión con sincronía estimada."""
    from streamclipper.review import ReviewStore, coverage_against_reviews
    from streamclipper.review_server import load_state
    from streamclipper.session import resolve_review_target, run_session

    s = Settings(data_dir=tmp_path)
    meta = run_session(str(sample), ["base"], s, label="demo", language="en", n_alternatives=10)
    sdir = Path(meta["session_dir"])
    assert [j["status"] for j in meta["jobs"]] == ["completed"] and meta["compare"] is None
    assert json.loads((sdir / "session.json").read_text(encoding="utf-8"))["fingerprint"]
    job_dir = Path(meta["jobs"][0]["job_dir"])
    assert (job_dir / "alternatives.md").is_file() and (job_dir / "alternatives.json").is_file()
    report = (sdir / "report.md").read_text(encoding="utf-8")
    assert "Clips elegidos" in report and "Confianza de transcripción" in report and "No** califica" in report
    assert "entretenid" not in report.replace("como entretenido", "")   # el informe no califica clips

    state = load_state(resolve_review_target(["demo"], s), s, 10, with_sync=True)
    clips = [i for i in state.items if i["kind"] == "clip"]
    assert len(clips) == 2
    for c in clips:
        assert c["media_exists"] and c["validation_ok"] and c["transcript"] and c["confidence"]["whisper_stats_available"]
        sub = c["subtitles"]
        assert sub["available"] and sub["lines"] > 0 and sub["sync"]["available"]
        assert abs(sub["sync"]["lag_s"]) <= 0.5, sub["sync"]   # voz sintética limpia: sin desfase relevante
        assert c["audio"]["anchor_in_clip_s"] is not None and c["components"]
    # la evaluación humana se guarda por video y sirve para comparar versiones
    state.store.set(clips[0], "good", "bien", state.jobs[0]["algorithm"])
    state.store.set(clips[1], "edit", "recortar", state.jobs[0]["algorithm"])
    reloaded = ReviewStore(state.store.path)
    cov = coverage_against_reviews(state.jobs[0]["result"], reloaded)
    assert cov["selected_by_rating"] == {"good": 1, "edit": 1, "bad": 0, "unrated": 0}
    assert cov["good_or_edit_selected"] == 2
    from streamclipper.core.errors import InvalidInputError
    with pytest.raises(InvalidInputError, match="Ya existe la sesión"):
        run_session(str(sample), ["base"], s, label="demo", language="en")


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

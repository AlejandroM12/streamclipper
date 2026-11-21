import http.client
import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.models import Segment, Transcript, Word
from streamclipper.review import (
    ReviewStore, alternatives_for, clip_confidence, coverage_against_reviews, read_ass, subtitle_report,
    write_alternatives,
)
from streamclipper.review_server import _PAGE, load_state, make_server


# --- ayudantes ---------------------------------------------------------------------------------
def words(text, start, step=0.4, prob=0.9):
    toks = text.split()
    ps = prob if isinstance(prob, list) else [prob] * len(toks)
    return [Word(start + i * step, start + i * step + step * 0.9, t, ps[i]) for i, t in enumerate(toks)]


def sent(text, start, prob=0.9, **stats):
    ws = words(text, start, prob=prob)
    return Segment(ws[0].start, ws[-1].end, text, ws, **stats)


def write_ass(path: Path, lines):
    def ts(t):
        cs = round(t * 100)
        return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"
    path.write_text("[Events]\n" + "".join(f"Dialogue: 0,{ts(a)},{ts(b)},Default,,0,0,0,,{t}\n" for a, b, t in lines),
                    encoding="utf-8")


def components(reaction=0.5):
    names = ["intensidad_audio", "dinamica_audio", "reaccion_verbal", "frases", "contexto", "duracion",
             "calidad_transcripcion"]
    w = [.3, .1, .25, .1, .1, .05, .1]
    return [{"name": n, "value": reaction if n == "reaccion_verbal" else 0.5, "weight": x, "contribution": 0.1,
             "note": f"nota {n}", "evidence": ["+ «x» (strong)"] if n == "reaccion_verbal" and reaction else [],
             "window": [0, 1]} for n, x in zip(names, w)]


def cand(rank, s, e, score=0.6, anchor=None):
    return {"rank": rank, "start": s, "end": e, "duration": e - s, "anchor": anchor or (s + e) / 2, "score": score,
            "triggers": ["audio"], "explanation": "", "components": components(),
            "text": " ".join(f"palabra{rank}x{i}" for i in range(8))}  # textos distintos: no se descartan por repetición


def make_job(root: Path, fp="fp123", model="base", with_clip=True) -> Path:
    job = root / "jobs" / f"job-{model}"
    (job / "out").mkdir(parents=True)
    src = root / "video.mp4"
    src.write_bytes(bytes(range(256)) * 40)
    cands = [cand(1, 10, 30, .7), cand(2, 100, 120, .65), cand(3, 200, 220, .6), cand(4, 300, 320, .55)]
    tr = Transcript("es", [sent("hola esto es una prueba <b>x</b>", 11), sent("otra frase distinta aqui", 105)],
                    evidence_version=1)
    (job / "transcript.json").write_text(json.dumps(tr.to_dict()), encoding="utf-8")
    clips = []
    if with_clip:
        (job / "out" / "clip_01.mp4").write_bytes(b"0123456789" * 100)
        write_ass(job / "out" / "clip_01.ass", [(1.0, 3.0, "hola esto es una prueba")])
        clips = [{"name": "clip_01", "rank": 1, "file": str(job / "out" / "clip_01.mp4"), "start": 10, "end": 30,
                  "duration": 20, "score": .7, "validation": {"ok": True, "info": {}},
                  "boundaries": {"errors": [], "warnings": []}}]
    result = {"job_id": job.name, "status": "completed", "source": str(src), "source_fingerprint": fp,
              "settings": {"whisper_model": model, "min_score": 0.35, "max_overlap": 0.15, "text_trigger": 0.5,
                           "data_dir": str(root)},
              "candidates": cands, "selected": [1], "clips": clips, "alt_clips": [], "media": {"duration": 400.0}}
    (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return job


# --- confianza y subtítulos --------------------------------------------------------------------
def test_clip_confidence_with_and_without_whisper_stats():
    t = Transcript("es", [sent("uno dos tres cuatro", 1, prob=[0.9, 0.2, 0.9, 0.3], avg_logprob=-1.4,
                              compression_ratio=2.6, no_speech_prob=0.2)])
    c = clip_confidence(t, 0, 10)
    assert c["words"] == 4 and c["mean_word_prob"] == 0.575 and c["words_below_0_4_pct"] == 50.0
    assert c["low_confidence_words"] == ["dos", "cuatro"] and c["whisper_stats_available"]
    assert c["flags"] == {"baja_confianza_whisper": 1, "repetitiva": 1, "posible_no_voz": 0}
    old = Transcript("es", [sent("uno dos", 1)], evidence_version=1)
    assert clip_confidence(old, 0, 10)["flags"] is None and clip_confidence(old, 50, 60)["words"] == 0


def test_subtitle_report_flags_facts(tmp_path):
    clip = {"file": str(tmp_path / "c.mp4"), "start": 100.0, "end": 120.0, "boundaries": {
        "errors": ["corte dentro de una frase"], "warnings": ["empieza en minúscula"]}}
    segs = [sent("esto sigue y sigue sin parar nunca jamas", 104.0, prob=[.9, .1, .1, .1, .9, .9, .9, .9]),
            sent("gracias por ver el video", 110.0)]
    write_ass(tmp_path / "c.ass", [(4.0, 6.0, "esto sigue y sigue sin parar nunca jamas"),
                                   (10.0, 10.5, "texto larguisimo para leer en muy poco tiempo"),
                                   (10.5, 12.0, "gracias por ver el video")])
    r = subtitle_report(clip, Transcript("es", segs), None, with_sync=False)
    text = " | ".join(r["issues"])
    assert r["available"] and r["lines"] == 3
    assert "primeros 4.0 s" in text and "últimos 8.0 s" in text         # bordes sin subtítulos
    assert "baja confianza" in text and "caracteres/segundo" in text
    assert "alucinación" in text and "límite de frase: corte" in text and "(aviso)" in text
    assert "sync" not in r  # sin estimación cuando se omite


def test_subtitle_report_without_ass_or_lines(tmp_path):
    clip = {"file": str(tmp_path / "x.mp4"), "start": 0, "end": 10, "boundaries": {}}
    assert subtitle_report(clip, Transcript("es", []), None, False)["available"] is False
    write_ass(tmp_path / "x.ass", [])
    assert "no tiene subtítulos" in subtitle_report(clip, Transcript("es", []), None, False)["issues"][0]


def test_read_ass_parses_times(tmp_path):
    write_ass(tmp_path / "a.ass", [(1.25, 3.5, "hola")])
    assert read_ass(tmp_path / "a.ass") == [(1.25, 3.5, "hola")]


# --- alternativos ------------------------------------------------------------------------------
def test_alternatives_exclude_selected_overlaps_and_are_limited(tmp_path):
    job = make_job(tmp_path)
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    result["candidates"].insert(1, cand(5, 12, 32, .69))     # solapa con el elegido
    alts = alternatives_for(result, 2, Settings())
    assert [a.rank for a in alts] == [2, 3] and all(a.start > 50 for a in alts)
    assert len(alternatives_for(result, 10, Settings())) == 3


def test_write_alternatives_files(tmp_path):
    job = make_job(tmp_path)
    items = write_alternatives(job, Settings(), 10)
    assert [i["rank"] for i in items] == [2, 3, 4]
    assert (job / "alternatives.json").is_file()
    md = (job / "alternatives.md").read_text(encoding="utf-8")
    assert "Candidatos alternativos" in md and "| 2 |" in md


# --- almacén de evaluaciones -------------------------------------------------------------------
def item(i, s, e, job="j1", model="base"):
    return {"id": f"{job}:{i}", "job_id": job, "model": model, "kind": "clip", "name": i, "rank": 1, "start": s,
            "end": e, "score": .5}


def test_store_set_clear_validate_and_persist(tmp_path):
    st = ReviewStore(tmp_path / "r.ratings.json", "v.mp4", "fp")
    st.set(item("clip_01", 10, 30), "good", "buen momento", {"git_commit": "abc"})
    st.set(item("clip_02", 50, 70), "bad")
    st.set(item("clip_03", 90, 110), "edit", "recortar el inicio")
    assert st.summary() == {"good": 1, "edit": 1, "bad": 1}
    again = ReviewStore(tmp_path / "r.ratings.json")           # se relee desde disco
    r = again.ratings["j1:clip_01"]
    assert r["rating"] == "good" and r["note"] == "buen momento" and r["algorithm"]["git_commit"] == "abc"
    assert r["start"] == 10 and r["end"] == 30 and r["rated_at"]
    st.set(item("clip_02", 50, 70), None)                      # limpiar
    assert "j1:clip_02" not in st.ratings
    st.set(item("clip_03", 90, 110), None, "solo una nota")    # nota sin evaluación
    assert st.ratings["j1:clip_03"]["rating"] is None and st.summary()["edit"] == 0
    with pytest.raises(InvalidInputError):
        st.set(item("x", 0, 1), "genial")
    with pytest.raises(InvalidInputError):
        st.set(item("x", 0, 1), "good", "n" * 2001)
    assert not list(tmp_path.glob("*.tmp"))                    # escritura atómica sin restos


def test_moments_merge_same_moment_across_models_and_flag_conflicts(tmp_path):
    st = ReviewStore(tmp_path / "r.json", "v", "fp")
    st.set(item("a", 100, 120, "base-job", "base"), "good")
    st.set(item("b", 101, 121, "small-job", "small"), "bad")
    st.set(item("c", 500, 520, "base-job", "base"), "edit")
    ms = st.moments()
    assert len(ms) == 2 and ms[0]["conflict"] is True and ms[1]["conflict"] is False
    assert ms[0]["rating"] in ("good", "bad") and len(ms[0]["ratings"]) == 2


def test_coverage_against_reviews_for_comparing_versions(tmp_path):
    st = ReviewStore(tmp_path / "r.json", "v", "fp")
    st.set(item("a", 10, 30), "good")
    st.set(item("b", 100, 120), "good")     # un momento bueno que el job NO eligió
    st.set(item("c", 400, 420), "bad")
    res = {"candidates": [cand(1, 10, 30), cand(2, 100, 120), cand(3, 400, 420), cand(4, 700, 720)],
           "selected": [1, 3, 4]}
    cov = coverage_against_reviews(res, st)
    assert cov["selected_by_rating"] == {"good": 1, "edit": 0, "bad": 1, "unrated": 1}
    assert cov["good_or_edit_selected"] == 1 and len(cov["good_or_edit_moments"]) == 2
    missed = next(w for w in cov["good_or_edit_moments"] if not w["selected"])
    assert missed["candidate_rank"] == 2 and missed["start"] == 100


# --- servidor local ----------------------------------------------------------------------------
@pytest.fixture()
def server(tmp_path):
    job = make_job(tmp_path)
    cfg = Settings(data_dir=tmp_path)
    state = load_state([job], cfg, n_alternatives=3, with_sync=False)
    srv = make_server(state, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, state, tmp_path
    srv.shutdown()
    srv.server_close()


def call(srv, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    h = {"Host": host or f"127.0.0.1:{srv.server_port}", **(headers or {})}
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, dict(r.getheaders()), data


def test_page_and_state(server):
    srv, state, _ = server
    s, h, body = call(srv, "GET", "/")
    assert s == 200 and "text/html" in h["Content-Type"] and "Revisión de clips".encode() in body
    s, _, body = call(srv, "GET", "/api/state")
    data = json.loads(body)
    assert s == 200 and data["labels"] == {"good": "Bueno para publicar", "edit": "Necesita edición", "bad": "No sirve"}
    kinds = [i["kind"] for i in data["items"]]
    assert kinds.count("clip") == 1 and kinds.count("candidate") == 3
    clip = next(i for i in data["items"] if i["kind"] == "clip")
    assert clip["media"] == "out/clip_01.mp4" and clip["subtitles"]["lines"] == 1 and clip["confidence"]["words"] > 0
    assert "<b>x</b>" in clip["transcript"]  # el texto de terceros viaja como dato, nunca como HTML


def test_ui_never_injects_data_as_html():
    assert "innerHTML" not in _PAGE and "insertAdjacentHTML" not in _PAGE and "document.write" not in _PAGE
    assert "textContent" in _PAGE or "createTextNode" in _PAGE


@pytest.mark.skipif(shutil.which("node") is None, reason="node no disponible")
def test_ui_javascript_is_syntactically_valid(tmp_path):
    js = re.search(r"<script>(.*?)</script>", _PAGE, re.S).group(1)
    f = tmp_path / "ui.js"
    f.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_rating_is_saved_to_disk_and_validated(server):
    srv, state, root = server
    hdr = {"Content-Type": "application/json"}
    s, _, body = call(srv, "POST", "/api/rating", json.dumps({"id": "job-base:clip_01", "rating": "good", "note": "ok"}), hdr)
    assert s == 200 and json.loads(body)["summary"]["good"] == 1
    saved = json.loads((root / "reviews" / "fp123.ratings.json").read_text(encoding="utf-8"))
    r = saved["ratings"]["job-base:clip_01"]
    assert r["rating"] == "good" and r["note"] == "ok" and r["start"] == 10 and r["algorithm"]["settings"]
    # una evaluación inexistente, mal formada o demasiado grande se rechaza
    assert call(srv, "POST", "/api/rating", json.dumps({"id": "job-base:clip_01", "rating": "genial"}), hdr)[0] == 400
    assert call(srv, "POST", "/api/rating", json.dumps({"id": "nope", "rating": "good"}), hdr)[0] == 404
    assert call(srv, "POST", "/api/rating", "no json", hdr)[0] == 400
    assert call(srv, "POST", "/api/rating", json.dumps({"id": "x"}), {"Content-Type": "text/plain"})[0] == 415
    assert call(srv, "POST", "/api/rating", json.dumps({"id": "job-base:clip_01", "note": "x" * 30000}), hdr)[0] == 413
    assert call(srv, "POST", "/api/otra", "{}", hdr)[0] == 404
    # sigue guardado lo anterior
    assert json.loads((root / "reviews" / "fp123.ratings.json").read_text(encoding="utf-8"))["ratings"]["job-base:clip_01"]["rating"] == "good"
    call(srv, "POST", "/api/rating", json.dumps({"id": "job-base:clip_01", "rating": None, "note": ""}), hdr)
    assert "job-base:clip_01" not in json.loads((root / "reviews" / "fp123.ratings.json").read_text(encoding="utf-8"))["ratings"]


def test_host_header_is_checked_against_dns_rebinding(server):
    srv, *_ = server
    assert call(srv, "GET", "/api/state", host="evil.example.com")[0] == 403
    assert call(srv, "GET", "/api/state", host=f"localhost:{srv.server_port}")[0] == 200
    assert call(srv, "POST", "/api/rating", "{}", {"Content-Type": "application/json"}, host="evil.com")[0] == 403


def test_media_serving_supports_ranges(server):
    srv, *_ = server
    s, h, body = call(srv, "GET", "/media/0/out/clip_01.mp4")
    assert s == 200 and len(body) == 1000 and h["Accept-Ranges"] == "bytes" and h["Content-Type"] == "video/mp4"
    s, h, body = call(srv, "GET", "/media/0/out/clip_01.mp4", headers={"Range": "bytes=10-19"})
    assert s == 206 and body == b"0123456789" and h["Content-Range"] == "bytes 10-19/1000"
    s, h, body = call(srv, "GET", "/media/0/out/clip_01.mp4", headers={"Range": "bytes=990-"})
    assert s == 206 and len(body) == 10
    s, h, body = call(srv, "GET", "/media/0/out/clip_01.mp4", headers={"Range": "bytes=-5"})
    assert s == 206 and len(body) == 5 and h["Content-Range"] == "bytes 995-999/1000"
    assert call(srv, "GET", "/media/0/out/clip_01.mp4", headers={"Range": "bytes=5000-6000"})[0] == 416
    assert call(srv, "GET", "/media/0/out/clip_01.mp4", headers={"Range": "bytes=abc"})[0] == 416
    s, h, body = call(srv, "GET", "/source/0", headers={"Range": "bytes=0-3"})
    assert s == 206 and body == bytes([0, 1, 2, 3])


def test_only_files_inside_the_job_folder_are_served(server):
    srv, state, root = server
    (root / "secret.mp4").write_bytes(b"secreto")
    (state.jobs[0]["dir"] / "result.json").exists()
    for p in ("/media/0/../../secret.mp4", "/media/0/%2e%2e/%2e%2e/secret.mp4", "/media/0/out/../../../secret.mp4",
              "/media/0/result.json", "/media/0/transcript.json", "/media/5/out/clip_01.mp4", "/media/0/out/nope.mp4",
              "/media/0/C:/Windows/win.ini"):
        assert call(srv, "GET", p)[0] == 404, p


def test_load_state_requires_same_video(tmp_path):
    a = make_job(tmp_path / "a", fp="AAA")
    b = make_job(tmp_path / "b", fp="BBB", model="small")
    with pytest.raises(InvalidInputError, match="mismo video"):
        load_state([a, b], Settings(data_dir=tmp_path), with_sync=False)
    c = make_job(tmp_path / "c", fp="AAA", model="small")
    st = load_state([a, c], Settings(data_dir=tmp_path), with_sync=False)
    assert {i["model"] for i in st.items} == {"base", "small"} and st.store.path.name == "AAA.ratings.json"

import random
from types import SimpleNamespace

import numpy as np
import pytest

from streamclipper.analysis.audio import HOP, curve_from_db
from streamclipper.compare import (
    Job, best_match, closeness, compare_candidates, compare_clips, compare_data, match_one_to_one,
    render_markdown, spearman, top_locations,
)
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.evaluation import evaluate_data, explain_change, render_markdown as eval_markdown
from streamclipper.models import Candidate, Segment, Transcript, Word
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.heuristic import HeuristicScorer
from streamclipper.selection.selector import TIE_MARGIN, find_clips, selection_report
from streamclipper.transcription.sentences import split_sentences
from streamclipper.transcription.whisper import transcribe
from tests.helpers import make_ctx, sentence
from tests.test_lexicon import seg


# --- evidencia de Whisper ----------------------------------------------------------------------
class _W:
    def __init__(self, s, e, w, p):
        self.start, self.end, self.word, self.probability = s, e, w, p


def _fake_model(segments):
    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return iter(segments), SimpleNamespace(language="es")
    return FakeModel


def test_transcribe_persists_whisper_stats_and_does_not_invent_missing_ones(monkeypatch, tmp_path):
    s1 = SimpleNamespace(start=0.0, end=2.0, text=" hola amigos.", avg_logprob=-0.31, no_speech_prob=0.02,
                         compression_ratio=1.4, temperature=0.0, words=[_W(0, 1, " hola", 0.9), _W(1, 2, " amigos.", 0.8)])
    s2 = SimpleNamespace(start=3.0, end=4.0, text="x", avg_logprob=-1.5, no_speech_prob=0.7,  # sin compression_ratio
                         words=[_W(3, 4, "x", 0.3)])
    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", _fake_model([s1, s2]))
    t = transcribe(tmp_path / "a.wav", Settings(), CancelToken(), 5.0)
    assert t.has_whisper_stats and len(t.whisper_segments) == 2
    assert (t.segments[0].avg_logprob, t.segments[0].no_speech_prob, t.segments[0].compression_ratio) == (-0.31, 0.02, 1.4)
    assert t.segments[1].compression_ratio is None and t.whisper_segments[1]["compression_ratio"] is None
    assert t.whisper_segments[1]["temperature"] is None  # el atributo no existía: no se inventa
    assert t.segments[0].words[1].prob == 0.8


def test_sentence_split_keeps_worst_case_stats_of_contributing_segments():
    a = Segment(0, 2, "uno dos", [Word(0, 1, "uno", .9), Word(1, 2, "dos", .9)], avg_logprob=-0.2,
                no_speech_prob=0.1, compression_ratio=1.0)
    b = Segment(2.1, 4, "tres.", [Word(2.1, 3, "tres", .9), Word(3, 4, "cuatro.", .9)], avg_logprob=-1.3,
                no_speech_prob=0.5, compression_ratio=2.9)
    out = split_sentences(Transcript("es", [a, b], whisper_segments=[{"index": 0}]))
    assert len(out.segments) == 1  # sin puntuación entre los segmentos: una sola frase
    s = out.segments[0]
    assert (s.avg_logprob, s.no_speech_prob, s.compression_ratio) == (-1.3, 0.5, 2.9)
    assert out.whisper_segments == [{"index": 0}] and out.has_whisper_stats


def test_sentence_split_of_old_transcript_keeps_stats_unknown():
    old = Transcript.from_dict({"language": "es", "segments": [{"start": 0, "end": 1, "text": "hola.", "words": [
        {"start": 0, "end": 1, "text": "hola.", "prob": 0.9}]}]})
    s = split_sentences(old).segments[0]
    assert s.avg_logprob is None and s.compression_ratio is None and not split_sentences(old).has_whisper_stats


# --- explicaciones coherentes con el valor -----------------------------------------------------
def _scored(ctx):
    cands, _ = find_clips(ctx)
    return cands


def test_everyday_vamos_is_explained_as_discarded_and_value_is_zero():
    segs = [seg("Vamos a revisar las cañas hoy", 0.9, start=20.0), sentence("so I am looking around the map", 4, 5)]
    ctx = make_ctx(60, [(20, 23)], sorted(segs, key=lambda s: s.start))
    cands = _scored(ctx)
    c = min(cands, key=lambda c: abs(c.anchor - 21))
    r = next(x for x in c.components if x.name == "reaccion_verbal")
    assert r.value == 0 and r.window is not None
    assert any(e.startswith("- «vamos»") and "uso cotidiano" in e for e in r.evidence)
    assert "sin reacción aceptada" in r.note and "descartadas: 1" in r.note and "vamos»" not in r.note.split(";")[0]


def test_reaction_outside_the_reaction_window_is_not_used_nor_claimed():
    # reacción fuerte a 9 s del ancla (dentro del clip, fuera de ±4 s): no debe sumar ni aparecer como motivo
    segs = sorted([sentence("oh my god no way", 5, 3), sentence("the circle is closing in around us now", 30, 5)],
                  key=lambda s: s.start)
    ctx = make_ctx(80, [(30, 33)], segs, Settings(pre_context=30, post_context=10))
    c = next(c for c in _scored(ctx) if c.start < 5 < c.end and abs(c.anchor - 31) < 3)
    r = next(x for x in c.components if x.name == "reaccion_verbal")
    assert r.value == 0 and not any(e.startswith("+") for e in r.evidence)
    assert "oh my god" not in r.note and f"{c.anchor - 4:.1f}" in r.note


def test_explanations_are_consistent_with_values_on_random_transcripts():
    rng = random.Random(7)
    words = ["vamos", "a", "ver", "oh", "my", "god", "no", "way", "mierda", "pescar", "la", "caña", "wow", "bro"]
    seen = {"pos": 0, "zero": 0, "discarded": 0, "frases": 0}
    for trial in range(25):
        segs, t = [], 1.0
        while t < 100:
            n = rng.randint(2, 8)
            toks = [rng.choice(words) for _ in range(n)]
            probs = [rng.choice([0.15, 0.5, 0.9]) for _ in range(n)]
            segs.append(seg(" ".join(toks), probs, start=t, step=0.35))
            t += n * 0.35 + rng.choice([0.3, 1.0, 3.0])
        loud = [(rng.uniform(5, 90), 2.0) for _ in range(3)]
        ctx = make_ctx(110, [(a, a + d) for a, d in loud], segs)
        for c in _scored(ctx):
            comps = {x.name: x for x in c.components}
            r = comps["reaccion_verbal"]
            plus = [e for e in r.evidence if e.startswith("+")]
            assert (r.value > 0) == bool(plus), (c.to_dict()["components"][2])
            for e in plus:  # todo motivo citado como usado figura en la nota
                term = e.split("«")[1].split("»")[0]
                assert term in r.note
            if r.value == 0:
                assert "sin reacción aceptada" in r.note
            assert r.window == pytest.approx((c.anchor - 4, c.anchor + 4))
            assert comps["frases"].window == pytest.approx((c.start, c.end))
            f = comps["frases"]
            assert (f.value > 0) == any(e.startswith("+") for e in f.evidence)
            assert all(x.window is not None for x in c.components)
            assert abs(c.score - sum(x.contribution for x in c.components)) < 1e-9
            seen["pos" if r.value > 0 else "zero"] += 1
            seen["discarded"] += any(e.startswith("-") for e in r.evidence)
            seen["frases"] += f.value > 0
    # el test no es vacío: cubre candidatos con y sin reacción, con descartes y con frases
    assert all(v > 0 for v in seen.values()), seen


def test_candidate_dict_roundtrip_keeps_window_and_evidence():
    ctx = make_ctx(60, [(20, 23)], [seg("oh my god no way", 0.9, start=20.0)])
    c = _scored(ctx)[0]
    back = Candidate.from_dict(c.to_dict())
    r = next(x for x in back.components if x.name == "reaccion_verbal")
    assert r.window and r.evidence


def test_old_candidate_dict_without_window_or_evidence_loads():
    d = {"rank": 1, "start": 0, "end": 20, "anchor": 10, "triggers": ["audio"], "score": 0.5, "text": "",
         "components": [{"name": "frases", "value": 0.1, "weight": 0.1, "note": "x", "contribution": 0.01}]}
    c = Candidate.from_dict(d)
    assert c.components[0].window is None and c.components[0].evidence == []


# --- fragilidad de la selección ----------------------------------------------------------------
def test_selection_report_flags_near_ties_and_stays_quiet_otherwise():
    ctx = make_ctx(200, [(20, 23), (100, 103)],
                   sorted([sentence("oh my god no way", 20, 3), sentence("oh my god no way", 100, 3),
                           sentence("so I am looking around", 60, 4)], key=lambda s: s.start))
    cands, chosen = find_clips(ctx)
    for i, c in enumerate(cands, 1):
        c.rank = i
    rep = selection_report(cands, chosen, ctx)
    assert set(rep) >= {"margin_to_next", "top_spread", "fragile", "message", "tie_margin"}
    # fuerzo un casi empate con el siguiente candidato elegible
    extra = Candidate(150, 170, 160, ["audio"], list(chosen[0].components), chosen[-1].score - 0.001, "otro texto aqui", 9)
    rep2 = selection_report(cands + [extra], chosen, ctx)
    assert rep2["fragile"] and rep2["margin_to_next"] == pytest.approx(0.001, abs=1e-3)
    assert "Selección frágil" in rep2["message"] and str(TIE_MARGIN) in rep2["message"]
    assert selection_report([], [], ctx)["fragile"] is False


# --- comparador ---------------------------------------------------------------------------------
def cd(rank, s, e, score, anchor=None):
    return {"rank": rank, "start": s, "end": e, "score": score, "anchor": anchor if anchor is not None else (s + e) / 2,
            "triggers": ["audio"], "explanation": "", "text": "", "components": []}


def test_best_match_never_returns_an_arbitrary_candidate_when_all_iou_are_zero():
    m, v = best_match(cd(1, 0, 10, .5), [cd(2, 50, 60, .4), cd(3, 70, 80, .3)])
    assert m is None and v == 0
    m, v = best_match(cd(1, 0, 10, .5), [cd(2, 8, 18, .4)], 0.5)
    assert m is None and 0 < v < 0.5  # solape real pero bajo el umbral pedido


def test_one_to_one_matching_uses_each_candidate_once():
    a = [cd(1, 0, 10, .5), cd(2, 1, 11, .4)]
    b = [cd(1, 0, 10, .5)]
    pairs = match_one_to_one(a, b)
    assert len(pairs) == 1 and pairs[0][0]["rank"] == 1


def test_spearman_cases():
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5]) == pytest.approx(0.943, abs=0.01)
    assert spearman([1, 2, 3], [1, 2, 3]) is None               # pocos pares
    assert spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) is None   # sin varianza
    assert spearman([1, 2, 2, 4, 5], [1, 2, 2, 4, 5]) == pytest.approx(1.0)  # empates


def test_candidate_sets_report_overlap_and_correlation():
    a = [cd(i, i * 100, i * 100 + 20, 1 - i / 100) for i in range(1, 9)]
    b = [cd(i, i * 100 + 1, i * 100 + 21, 1 - i / 100) for i in range(1, 7)] + [cd(7, 5000, 5020, .1)]
    out = compare_candidates(a, b)
    assert out["a_with_match"]["iou_0.5"] == 6 and out["b_with_match"]["iou_0.5"] == 6
    assert out["matched_pairs"] == 6 and out["spearman"] == pytest.approx(1.0)
    few = compare_candidates(a[:3], b[:3])
    assert few["spearman"] is None and "n/a" in few["spearman_note"]


def test_top3_location_found_partial_and_missing():
    a = [cd(1, 0, 20, .9), cd(2, 100, 120, .8), cd(3, 200, 220, .7), cd(4, 300, 320, .6)]
    b = [cd(1, 1, 21, .9), cd(5, 105, 125, .5), cd(9, 900, 920, .3)]   # 2º: IoU ≈ 0.6; 3º: ausente
    rows = top_locations(a, b)
    assert rows[0]["found"] and rows[0]["other_rank"] == 1 and not rows[0]["partial"]
    assert rows[1]["found"] and rows[1]["other_rank"] == 5
    assert rows[2]["found"] is False and rows[2]["best_iou"] == 0
    partial = top_locations([cd(1, 0, 20, .9)], [cd(7, 9, 29, .5)])
    assert partial[0]["found"] and partial[0]["partial"] is True and partial[0]["other_rank"] == 7


def test_top3_uses_distinct_moments_not_overlapping_neighbours():
    a = [cd(1, 0, 20, .9), cd(2, 2, 22, .89), cd(3, 100, 120, .8), cd(4, 200, 220, .7), cd(5, 300, 320, .6)]
    assert [r["rank"] for r in top_locations(a, a)] == [1, 3, 4]


def test_closeness_warning():
    close = [cd(1, 0, 20, .60), cd(2, 100, 120, .59), cd(3, 200, 220, .58), cd(4, 300, 320, .575)]
    far = [cd(1, 0, 20, .9), cd(2, 100, 120, .7), cd(3, 200, 220, .5), cd(4, 300, 320, .2)]
    assert closeness(close)["close"] and "sensible" in closeness(close)["message"]
    assert not closeness(far)["close"] and closeness(far)["message"] is None


def test_clips_metric_is_separate_and_never_shows_arbitrary_matches():
    out = compare_clips([cd(1, 0, 20, .9), cd(2, 100, 120, .8)], [cd(1, 1, 21, .9)])
    assert out["a"][0]["match"]["iou"] > 0.8 and out["a"][1]["match"] is None
    assert out["a_matched"] == 1 and out["b_matched"] == 1


def _result(cands, selected, clips=None, fp="x", model="base"):
    return {"job_id": model, "status": "completed", "source": "v.mp4", "error": None, "warnings": [],
            "settings": {"whisper_model": model, "whisper_compute_type": "int8"}, "media": {"duration": 1000.0},
            "stages": {}, "candidates": cands, "selected": selected, "clips": clips or [], "alt_clips": [],
            "source_fingerprint": fp, "transcript": {"language": "es", "words": 1}}


def test_full_comparison_markdown_has_all_sections_and_warnings(tmp_path):
    ca = [cd(1, 0, 20, .60), cd(2, 100, 120, .59), cd(3, 200, 220, .58), cd(4, 300, 320, .575)]
    cb = [cd(1, 1, 21, .50), cd(2, 400, 420, .49), cd(3, 500, 520, .48), cd(4, 100, 120, .475)]
    ja = Job(tmp_path / "A", _result(ca, [1, 2, 3], model="base"), {"segments": []})
    jb = Job(tmp_path / "B", _result(cb, [1, 2, 3], model="small"), {"segments": []})
    d = compare_data(ja, jb)
    md = render_markdown(d)
    for s in ("## 1. Conjuntos de candidatos", "## 2. Dónde aparece el top-3", "### Estabilidad del ranking",
              "## 3. Clips finales (métrica separada)", "sensible a cambios mínimos"):
        assert s in md
    assert d["clips"]["a_matched"] == 1  # solo el momento 0–20 coincide entre los clips finales
    assert "sin coincidencia" in md


def test_jobs_without_clips_use_selected_candidates():
    c = [cd(1, 0, 20, .6), cd(2, 100, 120, .5)]
    ja = Job(None, _result(c, [1, 2]), {"segments": []})
    ja.dir = type("D", (), {"name": "A"})()
    jb = Job(type("D", (), {"name": "B"})(), _result(c, [1]), {"segments": []})
    d = compare_data(ja, jb)
    assert d["clips_kind"]["a"].startswith("seleccionados") and len(d["clips"]["a"]) == 2


def test_different_sources_are_flagged():
    c = [cd(1, 0, 20, .6)]
    mk = lambda n, fp: Job(type("D", (), {"name": n})(), _result(c, [1], fp=fp), {"segments": []})
    assert "NO usaron el mismo archivo" in render_markdown(compare_data(mk("A", "1"), mk("B", "2")))


# --- evaluación antes/después ------------------------------------------------------------------
def comp(name, v, w, note="", evidence=None):
    return {"name": name, "value": v, "weight": w, "contribution": v * w, "note": note, "evidence": evidence or []}


def full(rank, s, e, reaction, note, evidence=None, frases=0.0, anchor=None, quality=0.8):
    comps = [comp("intensidad_audio", .6, .3), comp("dinamica_audio", 1.0, .1),
             comp("reaccion_verbal", reaction, .25, note, evidence), comp("frases", frases, .1),
             comp("contexto", 1.0, .1), comp("duracion", .8, .05), comp("calidad_transcripcion", quality, .1)]
    sc = sum(c["contribution"] for c in comps)
    return {"rank": rank, "start": s, "end": e, "anchor": anchor or (s + e) / 2, "score": round(sc, 3),
            "triggers": ["audio"], "explanation": "", "text": "", "components": comps}


def test_explain_change_cites_the_discarded_evidence():
    old = full(1, 0, 20, .5, "vamos")
    new = full(1, 0, 20, 0.0, "ventana…: sin reacción aceptada; descartadas: 1",
               ["- «vamos» (uso cotidiano: «vamos a …»)"])
    why = explain_change(old, new)
    assert len(why) == 1 and "reaccion_verbal" in why[0] and "uso cotidiano" in why[0]


def test_evaluation_before_after_detects_changes_regressions_and_still_selected(tmp_path):
    old = [full(1, 0, 20, .5, "vamos", frases=1.0), full(2, 100, 120, 1.0, "mierda, vamos"),
           full(3, 200, 220, .5, "vamos"), full(4, 300, 320, .0, "sin reacción detectada")]
    new = [full(1, 100, 120, .5, "mierda", ["+ «mierda» (strong)"]), full(2, 198, 220, 0.0, "sin reacción aceptada"),
           full(3, 300, 320, 0.0, "sin reacción aceptada"), full(4, 0, 20, 0.0, "sin reacción aceptada",
                                                                  ["- «vamos» (baja confianza de Whisper)"])]
    ja = Job(type("D", (), {"name": "old"})(), _result(old, [1, 2, 3]), {"segments": []})
    jb = Job(type("D", (), {"name": "new"})(), _result(new, [1, 2, 3]), {"segments": []})
    d = evaluate_data(ja, jb)
    assert d["counts"]["matched"] == 4 and d["counts"]["removed"] == 0
    # el clip 200–220 sigue elegido aunque ahora sea la ventana 198–220 (candidato vecino con otro rango)
    still = {t["clip"]["rank"]: t["still"] for t in d["top3_old"]}
    assert still[3] is not None and still[3]["rank"] == 2
    assert still[1] is None  # 0–20 ya no está elegido
    flags = " ".join(d["flags"])
    assert "Cambio esperado" in flags               # el clip 1 tenía solo «vamos»
    assert "Posible PÉRDIDA" not in flags
    assert "Seleccionado sin reacción verbal aceptada" in flags
    md = eval_markdown(d)
    assert "No determina cuál selección es mejor" in md and "Top-3: antes → ahora" in md


def test_evaluation_flags_possible_loss_when_non_vamos_evidence_disappears():
    old = [full(1, 0, 20, 1.0, "mierda, vamos"), full(2, 100, 120, .5, "vamos"), full(3, 200, 220, .5, "vamos")]
    new = [full(1, 100, 120, .5, "x"), full(2, 200, 220, .5, "x"), full(3, 0, 20, .0, "sin reacción aceptada")]
    d = evaluate_data(Job(type("D", (), {"name": "o"})(), _result(old, [1]), {"segments": []}),
                      Job(type("D", (), {"name": "n"})(), _result(new, [1]), {"segments": []}))
    assert any("Posible PÉRDIDA" in f and "mierda" in f for f in d["flags"])

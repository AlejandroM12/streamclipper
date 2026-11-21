import json

import numpy as np
import pytest

from streamclipper.analysis.audio import HOP, curve_from_db
from streamclipper.analysis.confidence import segment_flags, summarize
from streamclipper.analysis.lexicon import Lexicon
from streamclipper.analysis.text import analyze_segment
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.models import Segment, Transcript, Word


def seg(text: str, probs=0.9, start=10.0, step=0.4, **stats) -> Segment:
    toks = text.split()
    ps = probs if isinstance(probs, list) else [probs] * len(toks)
    words = [Word(start + i * step, start + i * step + step * 0.9, t, ps[i]) for i, t in enumerate(toks)]
    return Segment(start, words[-1].end, text, words, **stats)


def curve(loud_at: tuple[float, float] | None = None, n=400):
    db = np.full(n, -35.0) + np.random.default_rng(1).normal(0, 0.4, n)
    if loud_at:
        db[int(loud_at[0] / HOP):int(loud_at[1] / HOP)] = -15.0
    return curve_from_db(db)


def accepted(sig):
    return [h for h in sig.details if h.accepted]


# --- «vamos»: contexto, confianza y señales acústicas -----------------------------------------
@pytest.mark.parametrize("text", ["Vamos a revisar las cañas", "Vamos a empezar", "vamos para la isla ahora",
                                  "Bueno vamos con calma", "Vamos a ver si pica"])
def test_vamos_in_everyday_phrases_is_not_a_reaction(text):
    sig = analyze_segment(seg(text), curve())
    assert sig.reaction == 0 and not accepted(sig)
    assert any("uso cotidiano" in h.reason for h in sig.details)


def test_everyday_vamos_stays_ignored_even_if_shouted():
    # volumen alto y exclamación, pero «vamos a …» es perífrasis: sigue sin contar sola
    sig = analyze_segment(seg("¡Vamos a revisar las cañas!", start=10.0), curve((9.5, 12)))
    assert sig.reaction == 0


@pytest.mark.parametrize("text,probs", [("¡Vamos, vamos!", 0.9), ("¡Vamos, vamos, vamos!", 0.85)])
def test_repeated_confident_vamos_with_exclamation_is_a_strong_reaction(text, probs):
    sig = analyze_segment(seg(text, probs), curve())
    assert sig.reaction == 0.5 and accepted(sig)[0].weight == 1.0
    assert "repetición confiable" in accepted(sig)[0].reason


def test_repeated_vamos_without_exclamation_is_moderate_not_ignored_nor_strong():
    sig = analyze_segment(seg("vamos vamos", 0.9), curve())  # Whisper omitió los signos
    assert accepted(sig)[0].weight == 0.5 and sig.reaction == 0.25


def test_elongated_vamos_needs_no_punctuation():
    sig = analyze_segment(seg("vamooos", 0.9), curve())
    assert accepted(sig)[0].weight == 0.5 and "alargada" in accepted(sig)[0].reason
    shouted = analyze_segment(seg("VAMOOOOS", 0.9, start=10.0), curve((9.5, 12)))
    assert accepted(shouted)[0].weight == 1.0 and "volumen" in accepted(shouted)[0].reason


def test_acoustic_emphasis_upgrades_a_bare_vamos_without_any_punctuation():
    quiet = analyze_segment(seg("vamos", 0.9), curve())
    loud = analyze_segment(seg("vamos", 0.9, start=10.0), curve((9.8, 11)))
    assert quiet.reaction == 0 and not accepted(quiet)           # sola y en tono neutro: nada
    assert accepted(loud) and "volumen" in accepted(loud)[0].reason


def test_exclamation_alone_is_not_enough_for_long_sentences():
    sig = analyze_segment(seg("bueno amigos vamos eso fue todo hoy!", 0.9), curve())
    assert sig.reaction == 0


def test_without_audio_curve_acoustic_evidence_is_simply_unavailable():
    sig = analyze_segment(seg("vamos", 0.9), None)
    assert sig.reaction == 0 and "sin análisis acústico" in sig.details[0].reason


def test_low_confidence_vamos_is_rejected_with_reason():
    sig = analyze_segment(seg("vamos", 0.2, start=10.0), curve((9.8, 11)))
    assert sig.reaction == 0 and "baja confianza" in sig.details[0].reason


# --- repeticiones: confiables vs artefactos ---------------------------------------------------
def test_low_confidence_repetition_loop_does_not_raise_interest():
    # copia del artefacto observado: «vamos» ×5 con probabilidades 0.02..0.71 (media 0.51)
    loop = seg("vamos, vamos, vamos, vamos, vamos", [0.02, 0.45, 0.64, 0.71, 0.71])
    sig = analyze_segment(loop, curve())
    assert sig.interest == 0 and sig.reaction == 0
    assert sig.repetitions_counted == 0 and "confianza media 0.51" in sig.repetitions_discarded[0]


def test_legitimate_confident_repetition_still_counts():
    sig = analyze_segment(seg("no no no no", 0.92), curve())  # grito repetido, bien transcrito
    assert sig.repetitions_counted == 3 and sig.interest == 1.0 and not sig.repetitions_discarded


def test_repetition_flagged_by_whisper_compression_ratio_is_discarded():
    s = seg("no no no no", 0.92, compression_ratio=3.1)
    sig = analyze_segment(s, curve())
    assert sig.interest == 0 and "repetitiva" in sig.flags and sig.repetitions_discarded


# --- léxico general con confianza -------------------------------------------------------------
def test_normal_reactions_need_confident_words():
    ok = analyze_segment(seg("oh my god no way", 0.9), curve())
    bad = analyze_segment(seg("oh my god no way", 0.15), curve())
    assert ok.reaction == 1.0 and bad.reaction == 0
    assert all("baja confianza" in h.reason for h in bad.details)


def test_whisper_no_voice_flag_discards_hits_only_when_both_conditions_hold():
    both = seg("oh my god", 0.9, no_speech_prob=0.9, avg_logprob=-1.6)
    only_nsp = seg("oh my god", 0.9, no_speech_prob=0.9, avg_logprob=-0.3)  # Whisper no la trataría como silencio
    assert analyze_segment(both, curve()).reaction == 0
    assert analyze_segment(only_nsp, curve()).reaction > 0
    assert segment_flags(Segment(0, 1, "x", no_speech_prob=0.95)) == []  # sin avg_logprob no se infiere


def test_old_segments_without_whisper_stats_are_handled():
    old = Segment(0, 2, "oh my god", [Word(0, 0.6, "oh", 0.9), Word(0.7, 1.2, "my", 0.9), Word(1.3, 2, "god", 0.9)])
    assert old.avg_logprob is None and analyze_segment(old, curve()).reaction > 0


# --- léxico configurable ----------------------------------------------------------------------
def test_lexicon_file_can_extend_remove_and_override(tmp_path):
    cfg = tmp_path / "lex.json"
    cfg.write_text(json.dumps({
        "extend_strong": ["que bestia"], "remove": ["joder"],
        "disable_ambiguous": ["vamos"],
        "ambiguous": {"dale": {"pattern": "dale", "mundane_next": ["que"]}},
    }), encoding="utf-8")
    lex = Lexicon.from_file(cfg)
    assert analyze_segment(seg("que bestia"), curve(), lexicon=lex).reaction == 0.5
    assert analyze_segment(seg("joder"), curve(), lexicon=lex).reaction == 0
    assert analyze_segment(seg("vamos vamos"), curve(), lexicon=lex).reaction == 0  # regla desactivada
    assert not any(r.name == "vamos" for r in lex.ambiguous) and any(r.name == "dale" for r in lex.ambiguous)


@pytest.mark.parametrize("content", ['{"clave_rara": 1}', "no es json", '{"extend_strong": ["("]}', "[1]",
                                     '{"ambiguous": {"x": {"nope": 1}}}'])
def test_invalid_lexicon_is_a_clear_error(tmp_path, content):
    p = tmp_path / "bad.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(InvalidInputError):
        Lexicon.from_file(p)


def test_settings_validate_lexicon_path(tmp_path):
    with pytest.raises(InvalidInputError, match="léxico"):
        Settings(lexicon_path=str(tmp_path / "nope.json")).validate()


# --- evidencia de Whisper y compatibilidad -----------------------------------------------------
def test_transcript_roundtrip_keeps_whisper_stats():
    s = seg("hola mundo", 0.9, avg_logprob=-0.4, no_speech_prob=0.05, compression_ratio=1.2)
    t = Transcript("es", [s], whisper_segments=[{"index": 0, "avg_logprob": -0.4}])
    back = Transcript.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back.has_whisper_stats and back.segments[0].avg_logprob == -0.4
    assert back.segments[0].compression_ratio == 1.2 and back.whisper_segments[0]["index"] == 0


def test_old_transcript_json_loads_without_inventing_values():
    old = {"language": "es", "segments": [{"start": 0, "end": 1, "text": "hola", "no_speech_prob": 0.0,
                                           "words": [{"start": 0, "end": 1, "text": "hola", "prob": 0.8}]}]}
    t = Transcript.from_dict(old)
    assert not t.has_whisper_stats and t.evidence_version == 1
    s = t.segments[0]
    assert s.no_speech_prob is None and s.avg_logprob is None and s.compression_ratio is None  # 0.0 viejo = no medido
    assert s.words[0].prob == 0.8
    assert summarize(t)["whisper_stats_available"] is False and "sentences_low_confidence" not in summarize(t)


def test_summarize_counts_flagged_sentences():
    segs = [seg("a b c", 0.9, avg_logprob=-0.3, compression_ratio=1.1, no_speech_prob=0.1),
            seg("x x x x", 0.5, avg_logprob=-1.4, compression_ratio=3.0, no_speech_prob=0.8)]
    out = summarize(Transcript("es", segs))
    assert out["sentences_low_confidence"] == 1 and out["sentences_repetitive"] == 1
    assert out["sentences_possible_no_voice"] == 1

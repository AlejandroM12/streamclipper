from streamclipper.models import Candidate
from streamclipper.selection.candidates import build_window, generate_candidates
from streamclipper.selection.selector import find_clips, overlap_ratio
from tests.helpers import make_ctx, sentence

CALM = [sentence("so I am looking around the map for a good spot to land", 5 + 12 * i, 5) for i in range(8)]


def _with_event(t):
    return sorted(CALM + [sentence("oh my god no way that was insane", t, 4)], key=lambda s: s.start)


def test_window_respects_duration_limits_and_bounds():
    ctx = make_ctx(120, [(50, 53)], _with_event(50))
    s, e = build_window(51, ctx)
    assert 0 <= s < e <= 120
    assert ctx.settings.min_clip <= e - s <= ctx.settings.max_clip


def test_window_does_not_cut_sentences():
    segs = _with_event(50)
    ctx = make_ctx(120, [(50, 53)], segs)
    s, e = build_window(51, ctx)
    for seg in segs:  # ni el inicio ni el final caen dentro de una frase
        assert not (seg.start + 0.01 < s < seg.end - 0.01), (s, seg)
        assert not (seg.start + 0.01 < e < seg.end - 0.01), (e, seg)


def test_window_near_video_start_and_end():
    ctx = make_ctx(40, [(1, 3)], [sentence("oh my god", 1, 2)])
    s, e = build_window(2, ctx)
    assert s >= 0 and e <= 40 and e - s >= 15
    s, e = build_window(39, make_ctx(40, [(38, 40)], [sentence("oh my god", 38, 2)]))
    assert s >= 0 and e <= 40 and e - s >= 15


def test_max_clip_is_enforced_with_long_context():
    from streamclipper.core.config import Settings
    ctx = make_ctx(200, [(100, 103)], _with_event(100), Settings(pre_context=40, post_context=40, max_clip=30))
    s, e = build_window(101, ctx)
    assert e - s <= 30.0 + 1e-6


def test_flat_audio_without_reactions_yields_no_clips():
    ctx = make_ctx(120, [], CALM)
    _, chosen = find_clips(ctx)
    assert chosen == []  # no se fabrican clips


def test_detects_event_combining_audio_and_text():
    ctx = make_ctx(120, [(50, 53)], _with_event(50))
    _, chosen = find_clips(ctx)
    assert len(chosen) == 1
    c = chosen[0]
    assert c.start < 50 < c.end
    assert "audio" in c.triggers or "verbal" in c.triggers
    assert c.score > 0.35 and c.explanation()


def test_verbal_reaction_without_loudness_is_still_a_candidate():
    ctx = make_ctx(120, [], _with_event(50))
    cands = generate_candidates(ctx)
    assert any("verbal" in c.triggers for c in cands)


def test_selected_clips_do_not_overlap_and_respect_max():
    events = [(20, 23), (60, 63), (100, 103), (140, 143), (180, 183)]
    segs = sorted(CALM + [sentence(f"oh my god no way number {i}", a, 3) for i, (a, _) in enumerate(events)],
                  key=lambda s: s.start)
    ctx = make_ctx(200, events, segs)
    _, chosen = find_clips(ctx)
    assert len(chosen) <= ctx.settings.max_clips == 3
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            assert overlap_ratio(a, b) <= ctx.settings.max_overlap


def test_adjacent_candidates_for_same_moment_are_deduplicated():
    ctx = make_ctx(120, [(50, 56)], _with_event(50))
    cands, chosen = find_clips(ctx)
    assert len(cands) >= 1
    assert len(chosen) == 1


def test_overlap_ratio():
    a = Candidate(0, 20, 10, [])
    assert overlap_ratio(a, Candidate(20, 40, 30, [])) == 0
    assert overlap_ratio(a, Candidate(10, 30, 20, [])) == 0.5

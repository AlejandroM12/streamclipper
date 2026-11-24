"""Composición vertical por regiones (1.5, etapa 1): geometría, validación, respaldos y render real."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.render.compose import COMPOSITIONS, effective_rows, parse_composition, settings_overrides
from streamclipper.render.layout import H, SAFE_MARGINS, W, compute_layout
from streamclipper.render.vertical import build_filter

CAMS = [[0.0, 0.0, 0.26, 0.28], [0.745, 0.0, 0.255, 0.28]]
MAIN = [0.26, 0.14, 0.48, 0.84]
TRIVIA = {"preset": "reacciones", "cams": CAMS, "main": MAIN}


def layout_for(comp, src=(1920, 1080), **kw):
    cfg = Settings(**kw)
    ov, why = settings_overrides(parse_composition(comp))
    cfg = replace(cfg, **ov)
    return compute_layout(*src, cfg), cfg, why


def test_presets_and_validation():
    assert set(COMPOSITIONS) == {"original", "gameplay", "reacciones", "podcast", "personalizado"}
    assert parse_composition(None) is None and parse_composition({}) is None
    for bad in ({"preset": "nuevo"}, {"preset": "reacciones", "main": [0.5, 0.5, 0.9, 0.2]}, {"preset": "reacciones", "main": [0, 0, 0.01, 0.5]},
                {"preset": "reacciones", "main": "x"}, {"preset": "personalizado", "rows": [[[0, 0, .2, .2]] * 4]},
                {"preset": "podcast", "cams": [[0, 0, .2, .2]] * 7}, "texto"):
        with pytest.raises(InvalidInputError):
            parse_composition(bad)


def test_regions_are_never_invented_without_enough_evidence():
    for comp, word in (({"preset": "reacciones", "main": MAIN}, "cámara"), ({"preset": "reacciones", "cams": CAMS}, "contenido"),
                       ({"preset": "podcast"}, "participante"), ({"preset": "personalizado"}, "sin regiones")):
        lay, _, why = layout_for(comp)
        assert lay.mode == "blur-fit" and not lay.panels and word in why     # vuelve a Original y dice por qué
    lay, cfg, why = layout_for({"preset": "gameplay"})
    assert lay.mode == "focus" and "sin región" in why                      # el recorte horizontal de siempre
    lay, _, _ = layout_for({"preset": "original"})
    assert lay.mode == "blur-fit" and lay.fg == (W, 608, 0, lay.fg[3])


def test_reacciones_puts_cameras_on_top_and_main_below_inside_the_safe_area():
    lay, cfg, _ = layout_for(TRIVIA)
    m = SAFE_MARGINS[cfg.safe_platform]
    cams, main = lay.panels[:2], lay.panels[2]
    assert lay.mode == "stack" and len(lay.panels) == 3
    assert cams[0]["pos"][1] == cams[1]["pos"][1] < main["pos"][1]            # misma fila arriba; el contenido debajo
    assert cams[0]["pos"][0] + cams[0]["size"][0] <= cams[1]["pos"][0]         # lado a lado sin pisarse
    for p in lay.panels:                                                      # todo dentro del lienzo y de la zona segura
        (x, y), (w, h) = p["pos"], p["size"]
        assert x >= 0 and x + w <= W and y >= m["top"] and y + h <= H - m["bottom"] - lay.sub_block_h
        assert w % 2 == 0 and h % 2 == 0
    assert lay.sub_margin_v >= main["pos"][1] + main["size"][1]                # subtítulos debajo del contenido, sin taparlo
    assert lay.sub_margin_v + lay.sub_block_h <= H - m["bottom"] + 1
    # el contenido principal se ve mucho más grande que con Original (panel central: ~48 % del ancho del cuadro)
    orig, _, _ = layout_for({"preset": "original"})
    assert main["size"][0] > 1.2 * (0.48 * orig.fg[0])


def test_region_aspect_is_preserved_so_nothing_is_stretched():
    lay, _, _ = layout_for(TRIVIA)
    for p in lay.panels:
        (cw, ch, _, _), (pw, ph) = p["crop"], p["size"]
        assert abs(cw / ch - pw / ph) / (cw / ch) < 0.02


def test_podcast_stacks_participants_and_keeps_optional_content_last():
    two = {"preset": "podcast", "cams": [[0, 0.1, 0.5, 0.8], [0.5, 0.1, 0.5, 0.8]]}
    lay, _, _ = layout_for(two)
    ys = [p["pos"][1] for p in lay.panels]
    assert len(lay.panels) == 2 and ys[0] < ys[1]
    four = {"preset": "podcast", "cams": [[0, 0, .5, .5], [.5, 0, .5, .5], [0, .5, .5, .5], [.5, .5, .5, .5]]}
    lay4, _, _ = layout_for(four)
    assert len(lay4.panels) == 4 and len({p["pos"][1] for p in lay4.panels}) == 2   # rejilla 2x2
    with_main = {**two, "main": [0.1, 0.1, 0.8, 0.3]}
    assert len(layout_for(with_main)[0].panels) == 3


def test_personalizado_rows_and_platform_margins():
    rows = {"preset": "personalizado", "rows": [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 0.5, 0.5]], [[0.0, 0.5, 1.0, 0.5]]]}
    lay, cfg, _ = layout_for(rows, safe_platform="shorts")
    assert len(lay.panels) == 3 and cfg.safe_platform == "shorts"
    assert min(p["pos"][1] for p in lay.panels) >= SAFE_MARGINS["shorts"]["top"]


def test_filter_graph_is_well_formed_and_has_optional_subtitles():
    lay, cfg, _ = layout_for(TRIVIA)
    g = build_filter(cfg, "clip.ass", lay)
    assert g.count("overlay=") == 3 and "split=4" in g and "ass=clip.ass" in g and g.rstrip().endswith("[v]") and "lanczos" in g
    assert "ass=" not in build_filter(cfg, None, lay)                          # previsualización sin subtítulos


def test_framing_reports_scale_and_regions():
    lay, _, _ = layout_for(TRIVIA)
    f = lay.framing
    assert f["layout"] == "stack" and f["regions"] == 3 and 40 <= f["scale_pct"] <= 100 and "sin recortar más de lo declarado" in f["note"]
    assert effective_rows(parse_composition(TRIVIA))[0][0] == parse_composition(TRIVIA)["cams"]


SAMPLE = REPO_ROOT / "data" / "samples" / "synthetic.mp4"


@pytest.mark.integration
def test_real_render_with_composition_keeps_resolution_audio_and_subtitles(tmp_path):
    from streamclipper.edit import EditParams, render_edit
    from streamclipper.media.ffmpeg import probe
    from streamclipper.pipeline import run_pipeline
    if not SAMPLE.exists():
        pytest.skip("falta data/samples/synthetic.mp4")
    data = tmp_path / "data"
    res = run_pipeline(str(SAMPLE), Settings(data_dir=data, language="en"))
    job = Path(res["job_dir"])
    clip = res["clips"][0]
    comp = {"preset": "reacciones", "cams": [[0.0, 0.0, 0.3, 0.3]], "main": [0.1, 0.3, 0.8, 0.7]}
    meta = render_edit(job, clip["name"], EditParams(start=clip["start"], end=clip["end"], composition=comp), Settings(data_dir=data))
    out = job / "edits" / clip["name"] / "v001" / "clip.mp4"
    info = probe(out, Settings(data_dir=data).ffprobe)
    assert (info.width, info.height) == (1080, 1920) and info.has_audio and abs(info.duration - (clip["end"] - clip["start"])) < 0.3
    assert meta["layout"] == "stack" and meta["composition"]["preset"] == "reacciones" and meta["framing"]["regions"] == 2
    assert meta["validation"]["ok"] and (job / "edits" / clip["name"] / "v001" / "clip.srt").is_file()
    # sin regiones suficientes: vuelve a Original con aviso, no falla ni inventa recortes
    meta2 = render_edit(job, clip["name"], EditParams(start=clip["start"], end=clip["end"], composition={"preset": "reacciones"}), Settings(data_dir=data))
    assert meta2["layout"] == "blur-fit" and any("Composición" in w for w in meta2["warnings"])
    # la versión original y la primera versión no cambian
    assert json.loads((job / "edits" / clip["name"] / "v001" / "meta.json").read_text(encoding="utf-8"))["composition"]["preset"] == "reacciones"
    assert Path(clip["file"]).is_file()

"""Genera una versión nueva de los clips de un job (límites, subtítulos y presentación actuales) sin tocar los originales.

No retranscribe ni vuelve a descargar: reutiliza transcript.json y el video original (solo re-extrae el audio para
detectar el final de las reacciones). Los clips nuevos van a `out_<etiqueta>/` con los mismos nombres; el manifiesto
`rerender_<etiqueta>.json` y el informe `rerender_<etiqueta>.md` comparan cada clip antes/después con las mismas
métricas. La comparación describe diferencias medibles; no decide cuál versión es mejor.
"""
import json
import re
import shutil
import time
from pathlib import Path

from streamclipper.context import load_context
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.core.log import log_event, setup_logging
from streamclipper.core.memory import ResourceMonitor
from streamclipper.jobs import load_result, resolve_job
from streamclipper.models import Candidate, Transcript
from streamclipper.pipeline import settings_from_result
from streamclipper.render.clips import render_set
from streamclipper.render.subtitles import cue_metrics, speech_rate_cps
from streamclipper.review import read_ass, subtitle_report
from streamclipper.selection.refine import refine_selected

_TAG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")


def _subs_summary(clip: dict, transcript: Transcript, cfg: Settings, mp4: Path) -> dict:
    """Métricas y avisos de subtítulos, medidos igual para el clip original y el nuevo."""
    ass = mp4.with_suffix(".ass")
    lines = read_ass(ass) if ass.is_file() else []
    dur = clip["end"] - clip["start"]
    rep = subtitle_report({**clip, "file": str(mp4)}, transcript, cfg.ffmpeg, with_sync=True)
    return {
        "metrics": {**cue_metrics(lines, dur, cfg.sub_max_cps, cfg.sub_min_duration),
                    "speech_rate_cps": speech_rate_cps(transcript.segments, clip["start"], clip["end"])},
        "issues": rep.get("issues", []), "issue_count": len(rep.get("issues", [])),
        "sync": rep.get("sync"), "boundary_errors": len(clip.get("boundaries", {}).get("errors", [])),
        "boundary_warnings": len(clip.get("boundaries", {}).get("warnings", [])),
    }


def rerender_job(job_ref: str, settings: Settings, tag: str = "stage3", overwrite: bool = False,
                 cancel: CancelToken | None = None, progress=None, **overrides) -> dict:
    cancel = cancel or CancelToken()
    progress = progress or (lambda *_: None)
    if not _TAG.match(tag):
        raise InvalidInputError("La etiqueta debe usar letras minúsculas, dígitos, '-' o '_' (máx. 31)")
    job_dir = resolve_job(job_ref, settings)
    result = load_result(job_dir)
    originals = result.get("clips") or []
    if not originals:
        raise InvalidInputError("El job no tiene clips renderizados que re-generar (usá export --selected primero)")
    cfg = settings_from_result(result, **overrides)
    cfg.validate()
    out_dir = job_dir / f"out_{tag}"
    if out_dir.exists():
        if not overwrite:
            raise InvalidInputError(f"Ya existe {out_dir.name}. Usá otra --tag o --overwrite (los originales no se tocan)")
        shutil.rmtree(out_dir)
    setup_logging(job_dir / "job.log")
    work = job_dir / f"work_rerender_{int(time.time())}"
    out_dir.mkdir()
    by_rank = {c["rank"]: Candidate.from_dict(c) for c in result["candidates"]}
    transcript = Transcript.from_dict(json.loads((job_dir / "transcript.json").read_text(encoding="utf-8")))
    warnings: list[str] = []
    t0 = time.monotonic()
    try:
        with ResourceMonitor() as mon:
            ctx, src = load_context(job_dir, result, cfg, work / "ctx", cancel)
            names = [c["name"] for c in originals]
            cands = [by_rank[c["rank"]] for c in originals]
            windows = dict(zip(names, refine_selected(cands, ctx)))
            clips = render_set(src, list(zip(names, cands)), transcript, cfg, work, out_dir, cancel, progress,
                               warnings, refined=windows)
            rows = []
            for orig, new in zip(originals, clips):
                o_mp4 = job_dir / "out" / f"{orig['name']}.mp4"
                n_mp4 = out_dir / f"{new['name']}.mp4"
                o, n = _subs_summary(orig, transcript, cfg, o_mp4), _subs_summary(new, transcript, cfg, n_mp4)
                rows.append({
                    "name": orig["name"], "rank": orig["rank"], "score": orig["score"],
                    "original": {"start": orig["start"], "end": orig["end"], "duration": orig["duration"],
                                 "file": str(o_mp4), **o},
                    "new": {"start": new["start"], "end": new["end"], "duration": new["duration"], "file": str(n_mp4),
                            "srt": str(n_mp4.with_suffix(".srt")), "refine_notes": new["refine_notes"],
                            "render_seconds": new["render_seconds"], "layout": new["layout"], "framing": new["framing"],
                            "fade_in": new["fade_in"], "fade_out": new["fade_out"],
                            "subtitle_problems": new["subtitle_problems"], "duration_bounds": new["duration_bounds"],
                            "low_confidence_subtitles": new["low_confidence_subtitles"],
                            "validation": new["validation"], "boundaries": new["boundaries"], **n},
                })
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)   # nunca deja una versión a medias
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)

    n_orig = len(originals)
    orig_render_s = result.get("stages", {}).get("renderizar", {}).get("seconds")
    manifest = {
        "tag": tag, "job_id": result["job_id"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overrides": overrides, "out_dir": str(out_dir), "clips": rows, "warnings": warnings,
        "settings": {k: getattr(cfg, k) for k in (
            "whisper_model", "layout", "fg_zoom", "fg_x", "keep_regions", "bg_blur", "bg_dim", "safe_platform", "sub_max_words",
            "sub_line_chars", "sub_max_lines", "sub_min_duration", "sub_max_cps", "sub_font_size", "refine_bounds",
            "refine_lead", "refine_tail", "refine_min_post", "refine_min_clip", "refine_max_clip", "refine_fades",
            "min_clip", "max_clip")},
        "timing": {"new_render_total_s": round(sum(r["new"]["render_seconds"] for r in rows), 1),
                   "new_total_with_audio_analysis_s": round(time.monotonic() - t0, 1),
                   "original_render_stage_s": orig_render_s,
                   "original_render_per_clip_approx_s": round(orig_render_s / n_orig, 1) if orig_render_s else None,
                   "note": "El tiempo original es de toda la etapa de render de los clips juntos (aprox. por clip)."},
        "resources": mon.summary(),
    }
    (job_dir / f"rerender_{tag}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                                                  encoding="utf-8")
    (job_dir / f"rerender_{tag}.md").write_text(render_markdown(manifest), encoding="utf-8")
    log_event("rerendered", job=result["job_id"], tag=tag, clips=n_orig)
    manifest["job_dir"] = str(job_dir)
    return manifest


def framing_text(fr: dict | None) -> str:
    if not fr:
        return "—"
    if "visible_width_pct" in fr:
        return f"{fr['layout']}: se ve el {fr['visible_width_pct']} % del ancho"
    return (f"{fr['layout']}, zoom {fr['zoom_used']}: recorta {fr['crop_left_pct']} % a la izquierda y "
            f"{fr['crop_right_pct']} % a la derecha; {fr['note']}")


def _f(v, suffix="") -> str:
    return "—" if v is None else f"{v}{suffix}"


def render_markdown(m: dict) -> str:
    L = [f"# Versión «{m['tag']}» de los clips — {m['job_id']}", "",
         "Clips originales intactos en `out/`; versión nueva en "
         f"`{Path(m['out_dir']).name}/` (MP4, `.ass` y `.srt` editable). Las métricas se miden igual en ambos.", "",
         "> Menos avisos o mejores métricas **no** significan un clip mejor: revisá ambas versiones "
         "(`streamclipper review <sesión>`).", ""]
    s = m["settings"]
    keep = s.get("keep_regions") or []
    L += [f"- **Presentación:** {s['layout']} (zoom máx. {s['fg_zoom']}, x {s['fg_x']}"
          + (f"; regiones a conservar declaradas a mano: {', '.join(keep)}" if keep else "; sin regiones declaradas")
          + f"), fondo desenfocado {s['bg_blur']} / "
          f"oscurecido {s['bg_dim']}, márgenes {s['safe_platform']}",
          f"- **Subtítulos:** ≤ {s['sub_max_words']} palabras, ≤ {s['sub_max_lines']} líneas, lectura mínima "
          f"{s['sub_min_duration']} s, ≤ {s['sub_max_cps']} caracteres/s, fuente {s['sub_font_size']}",
          f"- **Límites:** ajuste {'activado' if s['refine_bounds'] else 'desactivado'}; duración del clip final "
          f"{s.get('refine_min_clip') or s['min_clip']}–{min(s.get('refine_max_clip') or s['max_clip'], s['max_clip'])} s; "
          f"fades {'sí' if s.get('refine_fades', True) else 'no'}", ""]
    for r in m["clips"]:
        o, n = r["original"], r["new"]
        om, nm = o["metrics"], n["metrics"]
        L += [f"## {r['name']} (candidato #{r['rank']}, puntuación {r['score']} — sin cambios)", "",
              "| | Original | Nueva |", "|---|---|---|",
              f"| Inicio – fin | {o['start']} – {o['end']} s | {n['start']} – {n['end']} s |",
              f"| Duración | {o['duration']} s | {n['duration']} s ({n['duration'] - o['duration']:+.2f} s) |",
              f"| Subtítulos (cantidad) | {om['cues']} | {nm['cues']} |",
              f"| Cobertura de subtítulos | {om['coverage_pct']} % | {nm['coverage_pct']} % |",
              f"| Velocidad (cps media / p90 / máx) | {_f(om.get('cps_mean'))} / {_f(om.get('cps_p90'))} / {_f(om.get('cps_max'))} | "
              f"{_f(nm.get('cps_mean'))} / {_f(nm.get('cps_p90'))} / {_f(nm.get('cps_max'))} |",
              f"| Ritmo del habla (caracteres/s mientras se habla) | {_f(om.get('speech_rate_cps'))} | {_f(nm.get('speech_rate_cps'))} |",
              f"| Subtítulos sobre {s['sub_max_cps']:g} cps | {_f(om.get('pct_over_max_cps'), ' %')} | {_f(nm.get('pct_over_max_cps'), ' %')} |",
              f"| Subtítulos más cortos que la lectura mínima | {_f(om.get('cues_shorter_than_min'))} | {_f(nm.get('cues_shorter_than_min'))} |",
              f"| Duración mínima de un subtítulo | {_f(om.get('min_cue_duration_s'), ' s')} | {_f(nm.get('min_cue_duration_s'), ' s')} |",
              f"| Huecos > 1 s entre subtítulos | {_f(om.get('gaps_over_1s'))} | {_f(nm.get('gaps_over_1s'))} |",
              f"| Subtítulos «cómodos» (cps, duración, ancho) | {_f(om.get('comfortable_pct'), ' %')} | {_f(nm.get('comfortable_pct'), ' %')} |",
              f"| Avisos automáticos | {o['issue_count']} | {n['issue_count']} |",
              f"| Defectos demostrables de subtítulos (fuera de duración, superposición, texto omitido, huecos, márgenes) | — | {len(n.get('subtitle_problems', []))} |",
              f"| Encuadre | cuadro completo (blur-fit) | {framing_text(n.get('framing'))} |",
              f"| Fades de audio (entrada / salida) | — | {n.get('fade_in', 0)} s / {n.get('fade_out', 0)} s |",
              f"| Errores / avisos de límite de frase | {o['boundary_errors']} / {o['boundary_warnings']} | "
              f"{n['boundary_errors']} / {n['boundary_warnings']} |",
              f"| Tiempo de render | ≈ {_f(m['timing']['original_render_per_clip_approx_s'], ' s')} (aprox.) | {n['render_seconds']} s |", ""]
        if n["refine_notes"]:
            L += ["**Cambios de límites:** " + "; ".join(n["refine_notes"]), ""]
        if n["low_confidence_subtitles"]:
            L += ["**Subtítulos de baja confianza (texto sin corregir):** "
                  + "; ".join(f"«{t[:40]}»" for t in n["low_confidence_subtitles"][:5]), ""]
        if n["issues"]:
            L += ["**Avisos de la versión nueva:**"] + [f"- {x}" for x in n["issues"]] + [""]
    t, res = m["timing"], m["resources"]
    L += ["## Totales", "",
          f"- Render de la versión nueva: {t['new_render_total_s']} s (con análisis de audio: {t['new_total_with_audio_analysis_s']} s). "
          f"Etapa de render original: {_f(t['original_render_stage_s'], ' s')}.",
          f"- Memoria máxima durante esta re-generación (aprox.): Python {res['python_peak_mb']} MB, FFmpeg {res['ffmpeg_peak_mb']} MB, "
          f"combinado {res['total_peak_mb_approx']} MB. No es comparable con el pico del job original, que incluía la transcripción.",
          "- El desfase de subtítulos es una estimación contra un detector de voz; si la correlación es baja se marca como no confiable."]
    return "\n".join(L) + "\n"


# --- comparación original / etapa 3 / etapa 3.1 ------------------------------------------------------------------
def _load_manifest(job_dir: Path, tag: str) -> dict:
    p = job_dir / f"rerender_{tag}.json"
    if not p.is_file():
        raise InvalidInputError(f"No existe la versión '{tag}' en {job_dir.name} (falta {p.name})")
    return json.loads(p.read_text(encoding="utf-8"))


def _framing_of(manifest: dict, clip: dict) -> dict | None:
    """Encuadre de una versión; las anteriores a la etapa 3.1 no lo guardaban: se recalcula con sus parámetros."""
    if clip["new"].get("framing"):
        return clip["new"]["framing"]
    from streamclipper.render.layout import compute_layout
    st = manifest["settings"]
    cfg = Settings(layout=st["layout"], fg_zoom=st["fg_zoom"], fg_x=st["fg_x"], keep_regions=st.get("keep_regions", []))
    return compute_layout(1920, 1080, cfg).framing


def compare_variants(job_ref: str, tags: list[str], settings: Settings) -> tuple[dict, str]:
    """Original vs. cada versión generada con `rerender`, con las mismas métricas. No juzga cuál es mejor."""
    job_dir = resolve_job(job_ref, settings)
    result = load_result(job_dir)
    manifests = {t: _load_manifest(job_dir, t) for t in tags}
    names = [c["name"] for c in result["clips"]]
    data: dict = {"job_id": result["job_id"], "model": result["settings"]["whisper_model"], "tags": tags, "clips": {},
                  "totals": {}}
    n_orig = len(names)
    orig_render = result.get("stages", {}).get("renderizar", {}).get("seconds")
    L = [f"# Comparación de versiones — {result['job_id']} (Whisper {data['model']})", "",
         "Versiones: **original** (`out/`) y " + ", ".join(f"**{t}** (`out_{t}/`)" for t in tags) + ". Todas se miden con "
         "las mismas métricas. Los archivos originales y las versiones anteriores no se modifican.", "",
         "> Esta tabla describe diferencias medibles. **Menos avisos o más cobertura no significan un clip mejor ni publicable**: "
         "se decide al revisarlos (`streamclipper review <sesión>`).", ""]
    for name in names:
        rows = {t: next(c for c in m["clips"] if c["name"] == name) for t, m in manifests.items()}
        first = rows[tags[0]]
        o = first["original"]
        cols = [("Original", o, None, None)] + [(t, rows[t]["new"], manifests[t], rows[t]) for t in tags]
        data["clips"][name] = {"rank": first["rank"], "score": first["score"], "versions": {
            "original": {k: o[k] for k in ("start", "end", "duration", "issue_count")}
            | {"metrics": o["metrics"]}}}
        L += [f"## {name} (candidato #{first['rank']}, puntuación {first['score']} — sin cambios)", "",
              "| | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]

        def row(label, fn):
            L.append(f"| {label} | " + " | ".join(fn(c) for c in cols) + " |")

        row("Inicio – fin (s)", lambda c: f"{c[1]['start']} – {c[1]['end']}")
        row("Duración", lambda c: f"{c[1]['duration']} s" + ("" if c[0] == "Original" else f" ({c[1]['duration'] - o['duration']:+.2f})"))
        row("Subtítulos", lambda c: str(c[1]["metrics"]["cues"]))
        row("Cobertura de subtítulos", lambda c: f"{c[1]['metrics']['coverage_pct']} %")
        row("Subtítulos > cps cómodo / «cómodos»", lambda c: f"{c[1]['metrics'].get('pct_over_max_cps')} % / {c[1]['metrics'].get('comfortable_pct')} %")
        row("Subtítulos más cortos que la lectura mínima", lambda c: str(c[1]["metrics"].get("cues_shorter_than_min")))
        row("Ritmo del habla (cps)", lambda c: str(c[1]["metrics"].get("speech_rate_cps")))
        row("Avisos automáticos", lambda c: str(c[1]["issue_count"]))
        row("Defectos demostrables de subtítulos", lambda c: "—" if c[0] == "Original" else (
            str(len(c[1].get("subtitle_problems"))) if c[1].get("subtitle_problems") is not None else "no medido"))
        row("Encuadre", lambda c: "cuadro completo (blur-fit)" if c[0] == "Original" else framing_text(_framing_of(c[2], c[3])))
        row("Fades (entrada / salida)", lambda c: "—" if c[0] == "Original" or "fade_in" not in c[1] else f"{c[1]['fade_in']} / {c[1]['fade_out']} s")
        row("Render por clip", lambda c: f"≈ {round(orig_render / n_orig, 1)} s (aprox.)" if c[0] == "Original" and orig_render
            else ("—" if c[0] == "Original" else f"{c[1]['render_seconds']} s"))
        for t in tags:
            nv = rows[t]["new"]
            data["clips"][name]["versions"][t] = {k: nv.get(k) for k in (
                "start", "end", "duration", "issue_count", "render_seconds", "fade_in", "fade_out", "subtitle_problems")} | {
                "metrics": nv["metrics"], "framing": _framing_of(manifests[t], rows[t])}
            if nv.get("refine_notes"):
                L.append("")
                L.append(f"- **{t}** — límites: " + "; ".join(nv["refine_notes"]))
        L.append("")
    L += ["## Costo de renderizado y memoria", "", "| Versión | Render de los clips | Memoria máx. (aprox.) |", "|---|---|---|",
          f"| Original | {orig_render if orig_render is not None else '—'} s (etapa de render del job) | pico del job completo "
          f"{(result.get('resources') or {}).get('total_peak_mb_approx', '—')} MB (incluye la transcripción) |"]
    for t, m in manifests.items():
        L.append(f"| {t} | {m['timing']['new_render_total_s']} s (total con análisis de audio: {m['timing']['new_total_with_audio_analysis_s']} s) | "
                 f"{m['resources']['total_peak_mb_approx']} MB (Python {m['resources']['python_peak_mb']}, FFmpeg {m['resources']['ffmpeg_peak_mb']}) |")
        data["totals"][t] = {"render_s": m["timing"]["new_render_total_s"], "memory_mb": m["resources"]["total_peak_mb_approx"]}
    L += ["", "La memoria de las versiones nuevas es solo la de la re-generación; no es comparable con el pico del job original."]
    return data, "\n".join(L) + "\n"

"""Evaluación comparativa del detector clásico y Smart Highlights sobre material existente, sin renderizar ni retranscribir.

Qué mide y qué NO puede medir:
- Mide, para cada modo, métricas objetivas (duplicados, bordes de frase, habla en el clip, silencio, tiempo, memoria) y la
  coincidencia con las valoraciones humanas guardadas.
- Una valoración solo existe para clips que alguien llegó a ver. Los clips que el detector NO eligió casi nunca tienen
  valoración, así que los falsos negativos no se pueden medir con los datos actuales y cualquier métrica basada en
  valoraciones favorece al detector que generó los clips valorados (sesgo de selección). Se informa siempre cuánto del top-3
  de cada modo está valorado.
- NO se ajusta nada con las valoraciones: los pesos de Smart son priores fijados antes de ver resultados. El ajuste
  automático está deshabilitado hasta cumplir los requisitos de `tuning_readiness`.
- Separación ajuste/evaluación: cada video cae en «dev» o «test» según su huella (par/impar). Un futuro ajuste solo podría
  usar «dev»; cualquier afirmación de mejora debe medirse en «test».
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from streamclipper.analysis.audio import analyze_audio
from streamclipper.analysis.boundaries import check_boundaries
from streamclipper.core.config import Settings
from streamclipper.core.memory import ResourceMonitor
from streamclipper.jobs import load_result
from streamclipper.media.ffmpeg import extract_audio
from streamclipper.models import Transcript
from streamclipper.pipeline import fingerprint, settings_from_result
from streamclipper.review import ReviewStore
from streamclipper.selection.base import AnalysisContext
from streamclipper.selection.selector import find_clips, overlap_ratio
from streamclipper.selection.smart import MIN_ANCHOR_GAP, build_smart_scorer

IOU_LABEL = 0.30                 # una valoración se asigna a un clip si se solapan al menos este IoU
MIN_VIDEOS, MIN_PER_MODE, MIN_NEG = 5, 10, 3   # requisitos mínimos para habilitar el ajuste automático
LABEL_SCORE = {"good": 1.0, "edit": 0.5, "bad": 0.0}


def split_for(fp: str) -> str:
    """Asignación determinista y previa a ver resultados: huella par → dev, impar → test."""
    return "dev" if int(fp[-1], 16) % 2 == 0 else "test"


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def human_labels(settings: Settings, fp: str, src: str) -> list[dict]:
    store = ReviewStore.for_source(settings, fp, src)
    return [{"start": v["start"], "end": v["end"], "rating": v["rating"], "id": k} for k, v in store.ratings.items() if v.get("rating")]


def label_for(win: tuple[float, float], labels: list[dict]) -> str | None:
    """Mejor valoración humana que cubre esta ventana (por IoU), o None si el clip nunca fue valorado."""
    best = max(((_iou(win, (lb["start"], lb["end"])), lb) for lb in labels), key=lambda x: x[0], default=(0.0, None))
    return best[1]["rating"] if best[1] and best[0] >= IOU_LABEL else None


def _metrics(chosen, ctx: AnalysisContext, transcript: Transcript, labels: list[dict]) -> dict:
    cfg = ctx.settings
    clips = []
    for c in sorted(chosen, key=lambda c: c.start):
        b = check_boundaries(transcript, c.start, c.end)
        spoken = sum(min(s.end, c.end) - max(s.start, c.start) for s in transcript.segments if s.end > c.start and s.start < c.end)
        clips.append({"start": round(c.start, 1), "end": round(c.end, 1), "anchor": round(c.anchor, 1), "score": round(c.score, 3),
                      "rank": c.rank, "boundary_errors": len(b["errors"]), "boundary_warnings": len(b["warnings"]),
                      "speech_cover": round(max(0.0, min(1.0, spoken / c.duration)), 2),
                      "quiet": round(float(ctx.curve.window(ctx.curve.excess < -8.0, c.start, c.end).mean()), 2),
                      "label": label_for((c.start, c.end), labels),
                      "top_components": [f"{x.name} {x.value:.2f}" for x in sorted(c.components, key=lambda x: abs(x.contribution), reverse=True)[:3]]})
    dups = sum(1 for i, a in enumerate(chosen) for b2 in chosen[i + 1:]
               if overlap_ratio(a, b2) > cfg.max_overlap or abs(a.anchor - b2.anchor) < MIN_ANCHOR_GAP)
    return {"clips": clips, "n": len(clips), "duplicate_pairs": dups,
            "labeled": sum(1 for x in clips if x["label"]),
            "good": sum(1 for x in clips if x["label"] == "good"), "edit": sum(1 for x in clips if x["label"] == "edit"),
            "bad": sum(1 for x in clips if x["label"] == "bad"),
            "boundary_errors": sum(x["boundary_errors"] for x in clips), "boundary_warnings": sum(x["boundary_warnings"] for x in clips),
            "mean_speech_cover": round(float(np.mean([x["speech_cover"] for x in clips])), 2) if clips else None,
            "mean_quiet": round(float(np.mean([x["quiet"] for x in clips])), 2) if clips else None}


def evaluate_session(job_dir: Path, settings: Settings, visual: bool = True, work_root: Path | None = None) -> dict:
    """Corre ambos detectores en memoria sobre la transcripción guardada y el audio re-extraído del video."""
    result = load_result(job_dir)
    src = Path(result["source"])
    if not src.is_file() or fingerprint(src) != result.get("source_fingerprint"):
        raise ValueError("el video original no está disponible o cambió")
    cfg = settings_from_result(result, data_dir=settings.data_dir, detector="classic", smart_visual=visual)
    transcript = Transcript.from_dict(json.loads((job_dir / "transcript.json").read_text(encoding="utf-8")))
    work = (work_root or settings.data_dir / "tmp") / f"eval-{job_dir.name}"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio.wav"
    try:
        extract_audio(src, wav, cfg.ffmpeg)
        curve = analyze_audio(wav)
        ctx = AnalysisContext(transcript, curve, result["media"]["duration"], cfg)
        labels = human_labels(settings, result["source_fingerprint"], str(src))
        out: dict = {"job": job_dir.name, "title": src.name, "duration": result["media"]["duration"], "fingerprint": result["source_fingerprint"],
                     "split": split_for(result["source_fingerprint"]), "labels": len(labels), "modes": {}}
        stored = [c["rank"] for c in result.get("candidates", []) if c["rank"] in result.get("selected", [])]
        for mode in ("classic", "smart"):
            t0 = time.monotonic()
            with ResourceMonitor() as mon:
                scorer = build_smart_scorer(wav, curve, src, cfg, None) if mode == "smart" else None
                cands, chosen = find_clips(ctx, scorer)
                for r, c in enumerate(cands, 1):
                    c.rank = r
            res = mon.summary()
            m = _metrics(chosen, ctx, transcript, labels)
            m.update(seconds=round(time.monotonic() - t0, 2), peak_ram_mb=res.get("total_peak_mb_approx"), candidates=len(cands))
            m["detector"] = scorer.info() if scorer else {"requested": "classic", "effective": "clásico"}
            out["modes"][mode] = m
        a = [(c["start"], c["end"]) for c in out["modes"]["classic"]["clips"]]
        b = [(c["start"], c["end"]) for c in out["modes"]["smart"]["clips"]]
        out["overlap_top"] = sum(1 for x in a if any(_iou(x, y) >= IOU_LABEL for y in b))
        out["classic_reproduces_stored"] = len(stored) == out["modes"]["classic"]["n"]
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def corpus(settings: Settings) -> list[Path]:
    """Sesiones reales utilizables: completadas, no derivadas, con transcripción, video original y clips."""
    from streamclipper.app.sessions import list_sessions
    out = []
    for s in list_sessions(settings):
        d = settings.jobs_dir / s["id"]
        if s["status"] == "completed" and not s["legacy"] and (d / "transcript.json").is_file() and Path(s["source"]).is_file():
            out.append(d)
    return out


def tuning_readiness(settings: Settings) -> dict:
    """El ajuste automático solo se habilita con datos suficientes. Hoy se informa por qué NO."""
    per_video: dict[str, dict] = {}
    for f in (settings.data_dir / "reviews").glob("*.ratings.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        r = [v["rating"] for v in data.get("ratings", {}).values() if v.get("rating")]
        per_video[data.get("fingerprint", f.stem)] = {"rated": len(r), "good": r.count("good"), "bad": r.count("bad"), "edit": r.count("edit")}
    ok_videos = [k for k, v in per_video.items() if v["rated"] >= MIN_PER_MODE and v["bad"] >= MIN_NEG and v["good"] >= MIN_NEG]
    need = []
    if len(ok_videos) < MIN_VIDEOS:
        need.append(f"al menos {MIN_VIDEOS} videos con ≥ {MIN_PER_MODE} clips valorados, ≥ {MIN_NEG} «good» y ≥ {MIN_NEG} «bad» (hay {len(ok_videos)})")
    need.append("valoraciones también de clips NO elegidos por el detector (para medir falsos negativos)")
    return {"enabled": False, "videos": per_video, "eligible_videos": len(ok_videos), "requirements": need,
            "message": "Ajuste automático DESHABILITADO: " + "; ".join(need) + "."}


def ratings_report(settings: Settings) -> dict:
    """Valoraciones reales agrupadas por detector con el que se generó cada clip (sesiones procesadas en cada modo)."""
    by_mode: dict[str, dict] = {m: {"good": 0, "edit": 0, "bad": 0, "videos": set()} for m in ("classic", "smart")}
    for f in (settings.data_dir / "reviews").glob("*.ratings.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for v in data.get("ratings", {}).values():
            if not v.get("rating"):
                continue
            mode = (v.get("algorithm", {}).get("settings", {}) or {}).get("detector", "classic")
            if mode in by_mode:
                by_mode[mode][v["rating"]] += 1
                by_mode[mode]["videos"].add(data.get("fingerprint", f.stem))
    out = {}
    for m, d in by_mode.items():
        n = d["good"] + d["edit"] + d["bad"]
        out[m] = {"rated": n, "good": d["good"], "edit": d["edit"], "bad": d["bad"], "videos": len(d["videos"]),
                  "good_rate": round(d["good"] / n, 2) if n else None}
    enough = all(out[m]["rated"] >= MIN_PER_MODE and out[m]["videos"] >= 3 for m in out)
    if enough:
        verdict = ("Hay datos mínimos para comparar, pero solo con clips que ya se habían visto: no hay forma de medir los falsos negativos. "
                   f"Tasa de «good»: clásico {out['classic']['good_rate']} vs smart {out['smart']['good_rate']}.")
    else:
        verdict = (f"NO CONCLUYENTE: valoraciones de Smart = {out['smart']['rated']} ({out['smart']['videos']} video/s), clásico = "
                   f"{out['classic']['rated']} ({out['classic']['videos']} video/s). Se necesitan ≥ {MIN_PER_MODE} clips valorados y ≥ 3 videos "
                   "por modo. No se puede afirmar que Smart Highlights mejore ni que empeore.")
    return {"modes": out, "verdict": verdict, "tuning": tuning_readiness(settings)}


def render_markdown(rows: list[dict], ratings: dict) -> str:
    L = ["# Comparación: ranking clásico vs Smart Highlights", "",
         "Misma transcripción y mismo audio para ambos; sin renderizar. Las valoraciones humanas disponibles son pocas y de clips que el "
         "clásico ya había elegido (sesgo de selección): ver límites en `docs/smart-highlights.md`.", ""]
    L += ["| Sesión | Video | Split | Modo | Clips | Duplicados | Bordes (err/avisos) | Habla en clip | Valorados (good/edit/bad) | Tiempo | RAM pico |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for mode, m in r["modes"].items():
            L.append(f"| `{r['job'][-24:]}` | {r['title'][:22]} | {r['split']} | {mode} | {m['n']} | {m['duplicate_pairs']} | "
                     f"{m['boundary_errors']}/{m['boundary_warnings']} | {m['mean_speech_cover']} | {m['labeled']} ({m['good']}/{m['edit']}/{m['bad']}) | "
                     f"{m['seconds']} s | {m['peak_ram_mb']} MB |")
    L += ["", "## Clips elegidos y por qué", ""]
    for r in rows:
        L.append(f"### `{r['job']}` ({r['title']}, {r['duration']:.0f} s) — coincidencia de top: {r['overlap_top']}/{r['modes']['classic']['n']}")
        for mode, m in r["modes"].items():
            info = m["detector"]
            L.append(f"- **{mode}** ({info.get('effective')}"
                     + (f"; faltó: {', '.join(info['missing'])}" if info.get("missing") else "") + f"; duplicados evitados: {info.get('duplicates_avoided', 0)})")
            for c in m["clips"]:
                L.append(f"  - {c['start']}–{c['end']} s, puntuación {c['score']}, valoración humana: {c['label'] or 'sin valorar'}; "
                         f"señales: {', '.join(c['top_components'])}")
        L.append("")
    L += ["## Valoraciones humanas guardadas", "", f"{ratings['verdict']}", "", f"{ratings['tuning']['message']}", ""]
    return "\n".join(L)

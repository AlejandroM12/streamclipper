"""Compara dos ejecuciones del mismo video (p. ej. Whisper base vs small)."""
import difflib
import json
import re
from pathlib import Path

from streamclipper.core.config import Settings
from streamclipper.jobs import load_result, resolve_job
from streamclipper.report import build_summary


def _load(ref: str, settings: Settings) -> tuple[Path, dict, dict]:
    d = resolve_job(ref, settings)
    result = load_result(d)
    tpath = d / "transcript.json"
    transcript = json.loads(tpath.read_text(encoding="utf-8")) if tpath.is_file() else {"segments": []}
    return d, result, transcript


def _words(t: dict) -> list[dict]:
    return [w for s in t["segments"] for w in s.get("words", [])]


def _tokens(t: dict) -> list[str]:
    return [re.sub(r"[^\w']", "", w["text"].lower()) for w in _words(t) if re.sub(r"[^\w']", "", w["text"])]


def _iou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union else 0.0


def compare_jobs(ref_a: str, ref_b: str, settings: Settings) -> str:
    da, ra, ta = _load(ref_a, settings)
    db, rb, tb = _load(ref_b, settings)
    sa, sb = build_summary(ra), build_summary(rb)
    notes = []
    if ra.get("source_fingerprint") != rb.get("source_fingerprint"):
        notes.append("⚠️ Los jobs NO usaron el mismo archivo de video: la comparación no es válida.")

    def avg_prob(t: dict) -> str:
        ws = _words(t)
        return f"{sum(w['prob'] for w in ws) / len(ws):.3f}" if ws else "—"

    sim = difflib.SequenceMatcher(None, _tokens(ta), _tokens(tb), autojunk=False).ratio() \
        if _tokens(ta) and _tokens(tb) else None

    def row(label, fa, fb) -> str:
        return f"| {label} | {fa} | {fb} |"

    def g(s, *path):
        for p in path:
            s = (s or {}).get(p)
        return "—" if s is None else s

    L = [f"# Comparación: {da.name} vs {db.name}", "", *notes, "",
         f"| | A: {sa['whisper_model']} | B: {sb['whisper_model']} |", "|---|---|---|",
         row("Estado", sa["status"], sb["status"]),
         row("Duración del video (s)", sa["video_duration_s"], sb["video_duration_s"]),
         row("Transcripción (s)", g(sa, "timings_s", "transcription"), g(sb, "timings_s", "transcription")),
         row("Factor tiempo real", g(sa, "timings_s", "transcription_realtime_factor"),
             g(sb, "timings_s", "transcription_realtime_factor")),
         row("Total (s)", g(sa, "timings_s", "total"), g(sb, "timings_s", "total")),
         row("Pico RAM combinado (MB, aprox.)", g(sa, "resources", "total_peak_mb_approx"),
             g(sb, "resources", "total_peak_mb_approx")),
         row("Idioma", sa["language"], sb["language"]),
         row("Palabras", g(sa, "transcript", "words"), g(sb, "transcript", "words")),
         row("Probabilidad media por palabra", avg_prob(ta), avg_prob(tb)),
         row("Candidatos", sa["candidates_detected"], sb["candidates_detected"]),
         row("Clips generados", sa["clips_generated"], sb["clips_generated"]),
         row("Advertencias", len(sa["warnings"]), len(sb["warnings"]))]
    if sim is not None:
        L += ["", f"Similitud de texto transcrito (A vs B): **{sim:.1%}**"]

    L += ["", "## Clips", "", "| Clip A | Mejor coincidencia en B (IoU) |", "|---|---|"]
    for ca in sa["clips"]:
        best = max(sb["clips"], key=lambda cb: _iou(ca, cb), default=None)
        m = f"{best['start']}–{best['end']}s (IoU {_iou(ca, best):.2f})" if best else "sin clips en B"
        L.append(f"| {ca['start']}–{ca['end']}s (#{ca['rank']}, {ca['score']}) | {m} |")
    matched = sum(1 for ca in sa["clips"] if any(_iou(ca, cb) >= 0.5 for cb in sb["clips"]))
    L += ["", f"Clips de A que coinciden (IoU ≥ 0.5) con alguno de B: **{matched}/{len(sa['clips'])}**.",
          "Las puntuaciones solo son comparables dentro de un mismo job."]
    return "\n".join(L) + "\n"


def save_comparison(text: str, name_a: str, name_b: str, settings: Settings) -> Path:
    out = settings.data_dir / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name_a}__vs__{name_b}.md"
    p.write_text(text, encoding="utf-8")
    return p

"""Compara dos ejecuciones del mismo video (p. ej. Whisper base vs small, o antes/después de un cambio).

Tres niveles, por separado:
  1. Conjuntos completos de candidatos (qué momentos detectó cada uno, con IoU y correlación de ranking).
  2. Ubicación del top-3 de cada job dentro del ranking del otro.
  3. Clips finales (lo que realmente se exportó).
Coincidir en los clips finales NO equivale a detectar lo mismo: el top-3 puede salir de un grupo de
candidatos casi empatados. Por eso el nivel 1 y los avisos de puntuaciones próximas van primero.
"""
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from streamclipper.core.config import Settings
from streamclipper.jobs import load_result, resolve_job
from streamclipper.report import build_summary
from streamclipper.selection.selector import TIE_MARGIN

IOU_MATCH = 0.5      # umbral de «mismo momento»
IOU_PARTIAL = 0.3    # por debajo de esto no se muestra ninguna coincidencia
MIN_PAIRS_FOR_CORR = 5


@dataclass
class Job:
    dir: Path
    result: dict
    transcript: dict


def load_job(ref: str, settings: Settings) -> Job:
    d = resolve_job(ref, settings)
    tpath = d / "transcript.json"
    transcript = json.loads(tpath.read_text(encoding="utf-8")) if tpath.is_file() else {"segments": []}
    return Job(d, load_result(d), transcript)


# --- utilidades de intervalos ------------------------------------------------------------------
def iou(x: dict, y: dict) -> float:
    inter = max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    union = (x["end"] - x["start"]) + (y["end"] - y["start"]) - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(x: dict, y: dict) -> float:
    inter = max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    return inter / min(x["end"] - x["start"], y["end"] - y["start"])


def best_match(c: dict, others: list[dict], min_iou: float = 0.0) -> tuple[dict | None, float]:
    """Mejor coincidencia por IoU; None si ninguna alcanza min_iou (nunca devuelve una arbitraria con IoU 0)."""
    best, best_iou = None, 0.0
    for o in others:
        v = iou(c, o)
        if v > best_iou:
            best, best_iou = o, v
    return (best, best_iou) if best is not None and best_iou >= max(min_iou, 1e-9) else (None, best_iou)


def match_one_to_one(a: list[dict], b: list[dict], thr: float = IOU_MATCH) -> list[tuple[dict, dict, float]]:
    """Emparejamiento voraz por IoU descendente: cada candidato se usa a lo sumo una vez."""
    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((a[i], b[j], v))
    return out


def _avg_ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Correlación de rangos de Spearman (con empates); None si no hay datos suficientes o varianza nula."""
    if len(x) != len(y) or len(x) < MIN_PAIRS_FOR_CORR:
        return None
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def distinct_top(cands: list[dict], k: int, max_overlap: float = 0.15) -> list[dict]:
    """Los k mejores momentos distintos (sin solaparse más de max_overlap), por rank."""
    out: list[dict] = []
    for c in sorted(cands, key=lambda c: c["rank"]):
        if all(_overlap_ratio(c, o) <= max_overlap for o in out):
            out.append(c)
        if len(out) == k:
            break
    return out


def closeness(cands: list[dict], k: int = 3) -> dict:
    """Qué tan apretadas están las puntuaciones del top-k y del siguiente momento distinto."""
    top = distinct_top(cands, k + 1)
    out = {"top_spread": None, "gap_to_next": None, "close": False, "message": None}
    if len(top) < 2:
        return out
    chosen = top[:k]
    out["top_spread"] = round(chosen[0]["score"] - chosen[-1]["score"], 3)
    if len(top) > k:
        out["gap_to_next"] = round(chosen[-1]["score"] - top[k]["score"], 3)
    parts = []
    if out["top_spread"] < TIE_MARGIN and len(chosen) > 1:
        parts.append(f"los {len(chosen)} primeros momentos se diferencian por solo {out['top_spread']:.3f}")
    if out["gap_to_next"] is not None and out["gap_to_next"] < TIE_MARGIN:
        parts.append(f"el {k + 1}.º momento (#{top[k]['rank']}) está a solo {out['gap_to_next']:.3f} del {k}.º")
    if parts:
        out["close"] = True
        out["message"] = "; ".join(parts) + f" (margen < {TIE_MARGIN}): el top-{k} es sensible a cambios mínimos."
    return out


# --- comparaciones -----------------------------------------------------------------------------
def compare_candidates(ca: list[dict], cb: list[dict]) -> dict:
    n_a, n_b = len(ca), len(cb)

    def covered(src, dst, thr):
        return sum(1 for x in src if best_match(x, dst, thr)[0] is not None)

    pairs = match_one_to_one(ca, cb)
    corr = spearman([p[0]["rank"] for p in pairs], [p[1]["rank"] for p in pairs])
    def near_anchor(src, dst, tol):
        return sum(1 for x in src if any(abs(x["anchor"] - y["anchor"]) <= tol for y in dst))

    return {
        "counts": {"a": n_a, "b": n_b},
        "a_with_match": {"iou_0.5": covered(ca, cb, IOU_MATCH), "iou_0.3": covered(ca, cb, IOU_PARTIAL)},
        "b_with_match": {"iou_0.5": covered(cb, ca, IOU_MATCH), "iou_0.3": covered(cb, ca, IOU_PARTIAL)},
        "anchors_within_1s": near_anchor(ca, cb, 1.0), "anchors_within_3s": near_anchor(ca, cb, 3.0),
        "matched_pairs": len(pairs),
        "spearman": None if corr is None else round(corr, 3),
        "spearman_note": None if corr is not None else
        f"n/a: se requieren al menos {MIN_PAIRS_FOR_CORR} candidatos emparejados (IoU ≥ {IOU_MATCH}) con rangos distintos",
    }


def top_locations(src: list[dict], dst: list[dict], k: int = 3) -> list[dict]:
    """Para cada uno de los k mejores momentos de src: dónde aparece en el ranking de dst."""
    out = []
    for c in distinct_top(src, k):
        m, v = best_match(c, dst, IOU_PARTIAL)
        row = {"rank": c["rank"], "start": c["start"], "end": c["end"], "score": c["score"]}
        if m is None:
            row |= {"found": False, "best_iou": round(v, 2)}
        else:
            row |= {"found": True, "other_rank": m["rank"], "other_score": m["score"], "iou": round(v, 2),
                    "partial": v < IOU_MATCH, "other_start": m["start"], "other_end": m["end"]}
        out.append(row)
    return out


def compare_clips(a: list[dict], b: list[dict]) -> dict:
    """Clips finales exportados. Solo se informa una coincidencia si el IoU es > 0."""
    def side(src, dst):
        rows = []
        for c in src:
            m, v = best_match(c, dst)
            rows.append({"start": c["start"], "end": c["end"], "rank": c["rank"], "score": c["score"],
                         "match": None if m is None else {"start": m["start"], "end": m["end"], "iou": round(v, 2)}})
        return rows
    ra, rb = side(a, b), side(b, a)
    return {"a": ra, "b": rb,
            "a_matched": sum(1 for r in ra if r["match"] and r["match"]["iou"] >= IOU_MATCH),
            "b_matched": sum(1 for r in rb if r["match"] and r["match"]["iou"] >= IOU_MATCH)}


def _avg_prob(t: dict) -> float | None:
    ws = [w for s in t["segments"] for w in s.get("words", [])]
    return sum(w["prob"] for w in ws) / len(ws) if ws else None


def _tokens(t: dict) -> list[str]:
    toks = (re.sub(r"[^\w']", "", w["text"].lower()) for s in t["segments"] for w in s.get("words", []))
    return [x for x in toks if x]


def compare_data(ja: Job, jb: Job) -> dict:
    sa, sb = build_summary(ja.result), build_summary(jb.result)
    ca, cb = ja.result.get("candidates", []), jb.result.get("candidates", [])
    ta, tb = _tokens(ja.transcript), _tokens(jb.transcript)
    return {
        "a": {"name": ja.dir.name, "summary": sa}, "b": {"name": jb.dir.name, "summary": sb},
        "same_source": ja.result.get("source_fingerprint") == jb.result.get("source_fingerprint"),
        "transcript_similarity": difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio() if ta and tb else None,
        "avg_prob": {"a": _avg_prob(ja.transcript), "b": _avg_prob(jb.transcript)},
        "candidates": compare_candidates(ca, cb),
        "top3_a_in_b": top_locations(ca, cb), "top3_b_in_a": top_locations(cb, ca),
        "closeness": {"a": closeness(ca), "b": closeness(cb)},
        "clips": compare_clips(_final_clips(ja.result), _final_clips(jb.result)),
        "clips_kind": {"a": _clips_kind(ja.result), "b": _clips_kind(jb.result)},
    }


def _final_clips(result: dict) -> list[dict]:
    """Clips exportados; si el job no renderizó (p. ej. re-puntuado), los candidatos seleccionados."""
    if result.get("clips"):
        return result["clips"]
    sel = set(result.get("selected", []))
    return [c for c in result.get("candidates", []) if c["rank"] in sel]


def _clips_kind(result: dict) -> str:
    return "renderizados" if result.get("clips") else "seleccionados (sin renderizar)"


# --- Markdown ----------------------------------------------------------------------------------
def _loc_line(r: dict, other: str) -> str:
    head = f"#{r['rank']} ({r['start']}–{r['end']}s, {r['score']})"
    if not r["found"]:
        return f"| {head} | sin coincidencia en {other} (IoU máx {r['best_iou']}) | — |"
    tag = " (parcial)" if r["partial"] else ""
    return f"| {head} | #{r['other_rank']} en {other}{tag}: {r['other_start']}–{r['other_end']}s | IoU {r['iou']}, puntuación {r['other_score']} |"


def render_markdown(d: dict) -> str:
    sa, sb = d["a"]["summary"], d["b"]["summary"]

    def g(s, *path):
        for p in path:
            s = (s or {}).get(p)
        return "—" if s is None else s

    ap = d["avg_prob"]
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    L = [f"# Comparación: {d['a']['name']} vs {d['b']['name']}", ""]
    if not d["same_source"]:
        L += ["⚠️ Los jobs NO usaron el mismo archivo de video: la comparación no es válida.", ""]
    L += [f"| | A: {sa['whisper_model']} | B: {sb['whisper_model']} |", "|---|---|---|",
          f"| Estado | {sa['status']} | {sb['status']} |",
          f"| Duración del video (s) | {sa['video_duration_s']} | {sb['video_duration_s']} |",
          f"| Transcripción (s) | {g(sa, 'timings_s', 'transcription')} | {g(sb, 'timings_s', 'transcription')} |",
          f"| Factor tiempo real | {g(sa, 'timings_s', 'transcription_realtime_factor')} | {g(sb, 'timings_s', 'transcription_realtime_factor')} |",
          f"| Total (s) | {g(sa, 'timings_s', 'total')} | {g(sb, 'timings_s', 'total')} |",
          f"| Pico RAM combinado (MB, aprox.) | {g(sa, 'resources', 'total_peak_mb_approx')} | {g(sb, 'resources', 'total_peak_mb_approx')} |",
          f"| Idioma | {sa['language']} | {sb['language']} |",
          f"| Palabras | {g(sa, 'transcript', 'words')} | {g(sb, 'transcript', 'words')} |",
          f"| Probabilidad media por palabra | {fmt(ap['a'])} | {fmt(ap['b'])} |",
          f"| Candidatos | {sa['candidates_detected']} | {sb['candidates_detected']} |",
          f"| Clips generados | {sa['clips_generated']} | {sb['clips_generated']} |",
          f"| Advertencias | {len(sa['warnings'])} | {len(sb['warnings'])} |"]
    if d["transcript_similarity"] is not None:
        L += ["", f"Similitud de texto transcrito (A vs B): **{d['transcript_similarity']:.1%}**"]

    c = d["candidates"]
    L += ["", "## 1. Conjuntos de candidatos", "",
          f"- Candidatos de A con el mismo momento en B (IoU ≥ {IOU_MATCH}): **{c['a_with_match']['iou_0.5']}/{c['counts']['a']}** "
          f"(IoU ≥ {IOU_PARTIAL}: {c['a_with_match']['iou_0.3']})",
          f"- Candidatos de B con el mismo momento en A (IoU ≥ {IOU_MATCH}): **{c['b_with_match']['iou_0.5']}/{c['counts']['b']}** "
          f"(IoU ≥ {IOU_PARTIAL}: {c['b_with_match']['iou_0.3']})",
          f"- Anclas de A con ancla de B a ≤ 1 s: {c['anchors_within_1s']}/{c['counts']['a']}; a ≤ 3 s: {c['anchors_within_3s']}/{c['counts']['a']}",
          f"- Pares emparejados uno a uno: {c['matched_pairs']}; correlación de ranking (Spearman): "
          + (f"**{c['spearman']}**" if c["spearman"] is not None else f"{c['spearman_note']}")]

    L += ["", "## 2. Dónde aparece el top-3 de cada uno en el ranking del otro", "",
          "| Top-3 de A | En el ranking de B | Detalle |", "|---|---|---|"]
    L += [_loc_line(r, "B") for r in d["top3_a_in_b"]] or ["| — | — | — |"]
    L += ["", "| Top-3 de B | En el ranking de A | Detalle |", "|---|---|---|"]
    L += [_loc_line(r, "A") for r in d["top3_b_in_a"]] or ["| — | — | — |"]

    L += ["", "### Estabilidad del ranking", ""]
    for key, name in (("a", "A"), ("b", "B")):
        cl = d["closeness"][key]
        if cl["message"]:
            L.append(f"- ⚠️ **{name}:** {cl['message']}")
        else:
            L.append(f"- {name}: sin empates cercanos (dispersión del top-3 {cl['top_spread']}, "
                     f"distancia al siguiente {cl['gap_to_next']}).")

    k = d["clips"]
    L += ["", "## 3. Clips finales (métrica separada)", "",
          f"A: {d['clips_kind']['a']}; B: {d['clips_kind']['b']}.", "",
          "| Clip de A | Mejor coincidencia en B |", "|---|---|"]
    for r in k["a"]:
        m = f"{r['match']['start']}–{r['match']['end']}s (IoU {r['match']['iou']})" if r["match"] else "sin coincidencia"
        L.append(f"| {r['start']}–{r['end']}s (#{r['rank']}, {r['score']}) | {m} |")
    L += ["", "| Clip de B | Mejor coincidencia en A |", "|---|---|"]
    for r in k["b"]:
        m = f"{r['match']['start']}–{r['match']['end']}s (IoU {r['match']['iou']})" if r["match"] else "sin coincidencia"
        L.append(f"| {r['start']}–{r['end']}s (#{r['rank']}, {r['score']}) | {m} |")
    L += ["", f"Clips de A que coinciden (IoU ≥ {IOU_MATCH}) con alguno de B: **{k['a_matched']}/{len(k['a'])}**; "
          f"de B con alguno de A: **{k['b_matched']}/{len(k['b'])}**.",
          "", "> Coincidir en los clips finales no implica mejor calidad, ni discrepar implica que detecten momentos "
          "distintos: mirá la sección 1. Las puntuaciones solo son comparables dentro de un mismo job."]
    return "\n".join(L) + "\n"


def compare_jobs(ref_a: str, ref_b: str, settings: Settings) -> str:
    return render_markdown(compare_data(load_job(ref_a, settings), load_job(ref_b, settings)))


def save_comparison(text: str, name_a: str, name_b: str, settings: Settings, kind: str = "vs") -> Path:
    out = settings.data_dir / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name_a}__{kind}__{name_b}.md"
    p.write_text(text, encoding="utf-8")
    return p

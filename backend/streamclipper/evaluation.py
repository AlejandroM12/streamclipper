"""Evaluación antes/después de un cambio de algoritmo sobre los mismos artefactos (job original vs re-puntuado).

Informa cambios de ranking, puntuación y top-3 con su explicación, y marca posibles regresiones. NO afirma
que el resultado sea mejor: sin criterio humano (qué momentos valen la pena) solo se puede describir qué
cambió y por qué.
"""
import re

from streamclipper.compare import (
    IOU_MATCH, IOU_PARTIAL, Job, best_match, closeness, load_job, match_one_to_one, spearman,
)
from streamclipper.core.config import Settings

CHANGE_EPS = 0.005     # cambio mínimo de contribución para mencionarlo
TOP_N = 25             # filas de la tabla de candidatos


def _comp(c: dict, name: str) -> dict:
    return next(x for x in c["components"] if x["name"] == name)


def _old_terms(c: dict) -> list[str]:
    """Términos verbales que el job ANTERIOR listaba en su nota (formato viejo: 'vamos, mierda')."""
    note = _comp(c, "reaccion_verbal")["note"]
    if "sin reacción" in note:
        return []
    note = note.split(": ", 1)[-1] if "ventana" in note else note
    return [t.strip().strip("«»") for t in re.split(r",|;", note.split("; descartadas")[0]) if t.strip()]


def explain_change(old: dict, new: dict) -> list[str]:
    out = []
    for oc in old["components"]:
        nc = _comp(new, oc["name"])
        d = nc["contribution"] - oc["contribution"]
        if abs(d) < CHANGE_EPS:
            continue
        txt = f"{oc['name']} {oc['contribution']:.3f}→{nc['contribution']:.3f} ({d:+.3f})"
        why = ""
        if oc["name"] in ("reaccion_verbal", "frases"):
            rej = [e[2:] for e in nc.get("evidence", []) if e.startswith("-")]
            acc = [e[2:] for e in nc.get("evidence", []) if e.startswith("+")]
            if rej:
                why = "; ".join(rej[:2])
            elif acc and d > 0:
                why = acc[0]
            elif d < 0:
                why = "ya no hay evidencia aceptada en la ventana"
        elif oc["name"] == "calidad_transcripcion":
            why = "ventana del clip distinta"
        elif oc["name"] in ("intensidad_audio", "dinamica_audio", "contexto", "duracion"):
            why = f"ancla {old['anchor']}→{new['anchor']} / ventana {old['start']}–{old['end']}→{new['start']}–{new['end']}"
        out.append(txt + (f": {why}" if why else ""))
    return out


def evaluate_data(jo: Job, jn: Job) -> dict:
    co, cn = jo.result["candidates"], jn.result["candidates"]
    pairs = match_one_to_one(co, cn, IOU_MATCH)
    pair_of_old = {p[0]["rank"]: p for p in pairs}
    matched_new = {p[1]["rank"] for p in pairs}
    sel_o = [c for c in co if c["rank"] in jo.result["selected"]]
    sel_n = [c for c in cn if c["rank"] in jn.result["selected"]]

    rows = []
    for o, n, v in sorted(pairs, key=lambda p: p[0]["rank"]):
        if o["rank"] <= TOP_N or n["rank"] <= TOP_N:
            rows.append({"old_rank": o["rank"], "new_rank": n["rank"], "start": o["start"], "end": o["end"],
                         "old_score": o["score"], "new_score": n["score"],
                         "d_score": round(n["score"] - o["score"], 3), "d_rank": o["rank"] - n["rank"],
                         "iou": round(v, 2), "why": explain_change(o, n),
                         "old_selected": o["rank"] in jo.result["selected"],
                         "new_selected": n["rank"] in jn.result["selected"]})
    removed = [o for o in co if o["rank"] not in pair_of_old]
    added = [n for n in cn if n["rank"] not in matched_new]

    def loc(c, pool):
        """Equivalente de c en pool por IoU (≥ 0.3); None si no hay: nunca una coincidencia arbitraria."""
        m, v = best_match(c, pool, IOU_PARTIAL)
        return None if m is None else {"rank": m["rank"], "score": m["score"], "iou": round(v, 2),
                                       "start": m["start"], "end": m["end"]}

    # «sigue elegido» se decide contra los clips seleccionados nuevos (mismo momento, aunque cambie la ventana)
    top_old = [{"clip": c, "now": loc(c, cn), "still": loc(c, sel_n) if best_match(c, sel_n, IOU_MATCH)[0] else None}
               for c in sel_o]
    top_new = [{"clip": c, "before": loc(c, co)} for c in sel_n]

    flags = []
    for t in top_old:
        c, now = t["clip"], t["now"]
        if t["still"] is None:
            other = [x for x in _old_terms(c) if x != "vamos"]
            where_ = (f"ahora es el candidato #{now['rank']} (puntuación {now['score']}, IoU {now['iou']})"
                      if now else "sin candidato equivalente")
            if other:
                flags.append(f"Posible PÉRDIDA: el clip {c['start']}–{c['end']}s (antes #{c['rank']}) tenía evidencia verbal "
                             f"distinta de «vamos» ({', '.join(other)}) y ya no está en la selección: {where_}.")
            else:
                flags.append(f"Cambio esperado: el clip {c['start']}–{c['end']}s (antes #{c['rank']}) ya no está en la selección "
                             f"({where_}); su evidencia verbal era solo «vamos» o ninguna.")
    for t in top_new:
        c = t["clip"]
        if _comp(c, "reaccion_verbal")["value"] == 0:
            flags.append(f"Seleccionado sin reacción verbal aceptada: el clip {c['start']}–{c['end']}s (#{c['rank']}) depende "
                         "de audio, contexto y calidad de transcripción; revisar si el pico es una reacción real o solo "
                         "voz más alta.")
        if _comp(c, "calidad_transcripcion")["value"] < 0.4:
            flags.append(f"Transcripción poco confiable en el clip {c['start']}–{c['end']}s (#{c['rank']}): "
                         f"{_comp(c, 'calidad_transcripcion')['note']}.")
        if t["before"] is None:
            flags.append(f"El clip nuevo {c['start']}–{c['end']}s (#{c['rank']}) no tenía candidato equivalente antes: "
                         "es un momento surgido de un ancla distinta.")
    if len(sel_n) < len(sel_o):
        flags.append(f"Se seleccionan menos clips ({len(sel_n)} contra {len(sel_o)}).")
    cl_o, cl_n = closeness(co), closeness(cn)
    if cl_n["close"]:
        flags.append("El top-3 nuevo sigue siendo frágil: " + cl_n["message"])
    lost_strong = [o for o in removed if o["rank"] <= 10]
    if lost_strong:
        flags.append("Candidatos del top-10 anterior sin equivalente nuevo (ancla desaparecida): "
                     + ", ".join(f"#{o['rank']} ({o['start']}–{o['end']}s)" for o in lost_strong))

    corr = spearman([p[0]["rank"] for p in pairs], [p[1]["rank"] for p in pairs])
    deltas = [p[1]["score"] - p[0]["score"] for p in pairs]
    return {
        "old": jo.dir.name, "new": jn.dir.name,
        "counts": {"old": len(co), "new": len(cn), "matched": len(pairs), "removed": len(removed), "added": len(added)},
        "spearman": None if corr is None else round(corr, 3),
        "score_delta": {"mean": round(sum(deltas) / len(deltas), 4) if deltas else None,
                        "min": round(min(deltas), 3) if deltas else None, "max": round(max(deltas), 3) if deltas else None,
                        "changed_over_0_02": sum(1 for d in deltas if abs(d) > 0.02)},
        "top3_old": top_old, "top3_new": top_new, "rows": rows,
        "removed": [{"rank": o["rank"], "start": o["start"], "end": o["end"], "score": o["score"],
                     "triggers": o["triggers"], "old_verbal": _old_terms(o)} for o in removed],
        "added": [{"rank": n["rank"], "start": n["start"], "end": n["end"], "score": n["score"],
                   "triggers": n["triggers"]} for n in added],
        "closeness": {"old": cl_o, "new": cl_n}, "flags": flags,
    }


def render_markdown(d: dict) -> str:
    c = d["counts"]
    L = ["# Evaluación antes/después", "", f"- **Antes:** `{d['old']}`", f"- **Después:** `{d['new']}`", "",
         f"- Candidatos: {c['old']} → {c['new']}; emparejados (IoU ≥ {IOU_MATCH}): {c['matched']}; "
         f"sin equivalente nuevo: {c['removed']}; nuevos sin equivalente anterior: {c['added']}",
         f"- Correlación de ranking (Spearman, pares emparejados): {d['spearman']}",
         f"- Cambio de puntuación en pares: media {d['score_delta']['mean']}, mín {d['score_delta']['min']}, "
         f"máx {d['score_delta']['max']}; con |Δ| > 0.02: {d['score_delta']['changed_over_0_02']}", "",
         "## Top-3: antes → ahora", "", "| Clip antes | Ahora en el ranking nuevo | ¿Sigue elegido? |", "|---|---|---|"]
    for t in d["top3_old"]:
        o, n, s = t["clip"], t["now"], t["still"]
        eq = (f"#{n['rank']} {n['start']}–{n['end']}s (puntuación {n['score']}, IoU {n['iou']})"
              if n else "sin candidato equivalente")
        L.append(f"| #{o['rank']} {o['start']}–{o['end']}s ({o['score']}) | {eq} | "
                 + (f"sí, como #{s['rank']}" if s else "no") + " |")
    L += ["", "| Clip nuevo | Equivalente en el ranking anterior |", "|---|---|"]
    for t in d["top3_new"]:
        n, b = t["clip"], t["before"]
        L.append(f"| #{n['rank']} {n['start']}–{n['end']}s ({n['score']}) | "
                 + (f"#{b['rank']} {b['start']}–{b['end']}s (puntuación {b['score']}, IoU {b['iou']})"
                    if b else "sin equivalente") + " |")

    L += ["", f"## Cambios por candidato (top-{25} antes o después)", "",
          "| Antes # | Ahora # | Δ rank | Puntuación | Δ | Momento | Sel. | Explicación |", "|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        sel = ("A" if r["old_selected"] else "") + ("→D" if r["new_selected"] else "")
        L.append(f"| {r['old_rank']} | {r['new_rank']} | {r['d_rank']:+d} | {r['old_score']}→{r['new_score']} | "
                 f"{r['d_score']:+.3f} | {r['start']}–{r['end']}s | {sel} | "
                 + ("; ".join(r["why"]).replace("|", "/") if r["why"] else "sin cambios relevantes") + " |")
    L += ["", f"## Candidatos sin equivalente nuevo ({len(d['removed'])})", ""]
    L += [f"- #{r['rank']} {r['start']}–{r['end']}s (puntuación {r['score']}, señales {r['triggers']}, "
          f"verbal anterior: {', '.join(r['old_verbal']) or '—'})" for r in d["removed"][:25]] or ["_Ninguno._"]
    L += ["", f"## Candidatos nuevos sin equivalente anterior ({len(d['added'])})", ""]
    L += [f"- #{r['rank']} {r['start']}–{r['end']}s (puntuación {r['score']}, señales {r['triggers']})"
          for r in d["added"][:25]] or ["_Ninguno._"]
    L += ["", "## Posibles regresiones y observaciones", ""] + [f"- ⚠️ {f}" for f in d["flags"]]
    if not d["flags"]:
        L.append("_Ninguna detectada automáticamente._")
    L += ["", "> Este informe describe qué cambió y por qué. No determina cuál selección es mejor: eso requiere "
          "revisar los clips (o una referencia con los momentos buenos). Más coincidencia entre modelos tampoco "
          "implica mejor calidad."]
    return "\n".join(L) + "\n"


def evaluate_jobs(ref_old: str, ref_new: str, settings: Settings) -> tuple[dict, str]:
    d = evaluate_data(load_job(ref_old, settings), load_job(ref_new, settings))
    return d, render_markdown(d)

"""Material para la revisión humana: detalle factual de cada clip/candidato y almacén de evaluaciones.

Nada de aquí califica un clip como entretenido: se informan hechos (tiempos, transcripción, confianza, señales de
audio, avisos de subtítulos) y se guarda lo que decide la persona: bueno para publicar / necesita edición / no sirve.
Las evaluaciones se guardan por video (huella del archivo) y por momento (inicio–fin), para poder compararlas
contra versiones futuras del algoritmo aunque cambien los jobs.
"""
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from streamclipper import __version__
from streamclipper.analysis.confidence import segment_flags
from streamclipper.compare import IOU_MATCH, best_match
from streamclipper.core.config import REPO_ROOT, Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.export import pick_alternatives
from streamclipper.jobs import load_result
from streamclipper.media.ffmpeg import run_cancellable
from streamclipper.models import Candidate, Transcript

RATINGS = {"good": "Bueno para publicar", "edit": "Necesita edición", "bad": "No sirve"}

# Umbrales de los AVISOS informativos de subtítulos (no intervienen en la selección)
SYNC_LAG_WARN_S = 0.3
SYNC_CORR_WARN = 0.3
EDGE_GAP_WARN_S = 2.5
LOW_PROB = 0.4
READ_CPS_WARN = 20.0     # caracteres por segundo difíciles de leer
_HALLUCINATION = re.compile(r"subt[ií]tulos|amara\.org|gracias por (ver|mirar)|suscr[ií]b|dale like|thanks for watching",
                            re.I)


def reviews_dir(settings: Settings) -> Path:
    return settings.data_dir / "reviews"


def algorithm_info(result: dict) -> dict:
    """Identifica la versión del algoritmo y su configuración, para comparar evaluaciones entre versiones."""
    git = None
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    settings = {k: v for k, v in result.get("settings", {}).items() if k != "data_dir"}
    return {"package_version": __version__, "git_commit": git, "settings": settings}


# --- transcripción y confianza -----------------------------------------------------------------
def clip_transcript(transcript: Transcript, start: float, end: float) -> str:
    return " ".join(s.text for s in transcript.segments if s.end > start and s.start < end)


def clip_confidence(transcript: Transcript, start: float, end: float) -> dict:
    segs = [s for s in transcript.segments if s.end > start and s.start < end]
    words = [w for s in segs for w in s.words if w.end > start and w.start < end]
    out: dict = {"words": len(words), "whisper_stats_available": transcript.has_whisper_stats,
                 "mean_word_prob": None, "words_below_0_4_pct": None, "low_confidence_words": [], "flags": None,
                 "min_avg_logprob": None, "max_compression_ratio": None}
    if words:
        out["mean_word_prob"] = round(sum(w.prob for w in words) / len(words), 3)
        out["words_below_0_4_pct"] = round(100 * sum(w.prob < LOW_PROB for w in words) / len(words), 1)
        out["low_confidence_words"] = [w.text for w in words if w.prob < LOW_PROB][:10]
    if transcript.has_whisper_stats and segs:
        flags = [segment_flags(s) for s in segs]
        out["flags"] = {k: sum(k in f for f in flags) for k in ("baja_confianza_whisper", "repetitiva", "posible_no_voz")}
        lp = [s.avg_logprob for s in segs if s.avg_logprob is not None]
        cr = [s.compression_ratio for s in segs if s.compression_ratio is not None]
        out["min_avg_logprob"] = round(min(lp), 3) if lp else None
        out["max_compression_ratio"] = round(max(cr), 3) if cr else None
    return out


# --- subtítulos --------------------------------------------------------------------------------
_DIALOGUE = re.compile(r"Dialogue: 0,(\d+):(\d+):(\d+)\.(\d+),(\d+):(\d+):(\d+)\.(\d+),Default,,0,0,0,,(.*)")


def read_ass(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = _DIALOGUE.match(ln)
        if m:
            g = [int(x) for x in m.groups()[:8]]
            out.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 100, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 100,
                        m.group(9)))
    return out


def _indicator(intervals, n: int, step: float) -> np.ndarray:
    x = np.zeros(n)
    for a, b in intervals:
        x[int(a / step):int(np.ceil(b / step))] = 1
    return x


def estimate_sync(mp4: Path, lines: list[tuple[float, float, str]], duration: float, ffmpeg: Path,
                  step: float = 0.1, max_lag: float = 2.0) -> dict:
    """Estima el desfase subtítulos↔voz comparando los subtítulos con la actividad de voz (VAD) del audio del
    clip renderizado. lag > 0 = subtítulos tardíos respecto de la voz. Es una estimación, no una medición."""
    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            code, _, err = run_cancellable([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", mp4, "-vn",
                                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], timeout=120)
            if code != 0:
                return {"available": False, "reason": f"no se pudo extraer el audio: {err.strip()[:100]}"}
            audio = decode_audio(str(wav), sampling_rate=16000)
        vad = [(s["start"] / 16000, s["end"] / 16000) for s in get_speech_timestamps(audio, VadOptions())]
        if not vad or not lines:
            return {"available": False, "reason": "sin voz detectada" if not vad else "sin subtítulos"}
        n = int(duration / step) + 1
        sub, sp = _indicator([(a, b) for a, b, _ in lines], n, step), _indicator(vad, n, step)
        xs, ys = sub - sub.mean(), sp - sp.mean()
        den = (np.linalg.norm(xs) * np.linalg.norm(ys)) or 1.0
        k = int(max_lag / step)
        corr = {}
        for lag in range(-k, k + 1):
            corr[lag] = float((xs[lag:] * ys[:len(ys) - lag]).sum() / den) if lag >= 0 else \
                float((xs[:lag] * ys[-lag:]).sum() / den)
        best = max(corr, key=corr.get)
        sub_total = sub.sum() * step
        return {"available": True, "lag_s": round(best * step, 2), "corr_at_best": round(corr[best], 2),
                "corr_at_zero": round(corr[0], 2),
                "subtitle_in_speech_pct": round(min(100.0, 100 * (sub * sp).sum() * step / sub_total), 0) if sub_total else None,
                "speech_without_subtitle_pct": round(100 * (sp * (1 - sub)).sum() / max(1.0, sp.sum()), 0)}
    except Exception as e:  # el análisis de sincronía nunca debe impedir la revisión
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def subtitle_report(clip: dict, transcript: Transcript, ffmpeg: Path | None, with_sync: bool = True) -> dict:
    """Avisos factuales sobre los subtítulos de un clip renderizado."""
    mp4 = Path(clip["file"])
    ass = mp4.with_suffix(".ass")
    if not ass.is_file():
        return {"available": False, "issues": ["no se encontró el archivo de subtítulos (.ass)"]}
    lines = read_ass(ass)
    dur = clip["end"] - clip["start"]
    issues: list[str] = []
    out: dict = {"available": True, "lines": len(lines), "issues": issues}
    if not lines:
        issues.append("el clip no tiene subtítulos (no se transcribió habla en este tramo)")
        return out | {"coverage_pct": 0.0}
    out["coverage_pct"] = round(100 * sum(min(b, dur) - a for a, b, _ in lines) / dur, 0)
    lead, tail = lines[0][0], dur - lines[-1][1]
    if lead >= EDGE_GAP_WARN_S:
        issues.append(f"los primeros {lead:.1f} s no tienen subtítulos")
    if tail >= EDGE_GAP_WARN_S:
        issues.append(f"los últimos {tail:.1f} s no tienen subtítulos")

    words = [w for s in transcript.segments for w in s.words if w.end > clip["start"] and w.start < clip["end"]]
    by_line = []
    for a, b, text in lines:
        ws = [w for w in words if clip["start"] + a - 0.3 <= w.start <= clip["start"] + b + 0.3]
        by_line.append((a, b, text, ws))
    weak = [t for a, b, t, ws in by_line if ws and sum(w.prob < LOW_PROB for w in ws) / len(ws) >= 0.4]
    if weak:
        issues.append(f"{len(weak)} línea(s) con muchas palabras de baja confianza (posible texto incorrecto): "
                      + "; ".join(f"«{t[:35]}»" for t in weak[:3]))
    fast = [t for a, b, t in lines if b > a and len(t) / (b - a) > READ_CPS_WARN]
    if fast:
        issues.append(f"{len(fast)} línea(s) con más de {READ_CPS_WARN:.0f} caracteres/segundo (difíciles de leer)")
    longw = [w for w in words if w.end - w.start > 1.5]
    if longw:
        issues.append(f"{len(longw)} palabra(s) con duración > 1.5 s (marcas de tiempo dudosas): "
                      + ", ".join(f"«{w.text}»" for w in longw[:4]))
    text_all = " ".join(t for _, _, t in lines)
    if _HALLUCINATION.search(text_all):
        issues.append("el texto contiene una frase típica de alucinación de Whisper (créditos/suscripción)")
    for m in clip.get("boundaries", {}).get("errors", []):
        issues.append("límite de frase: " + m)
    for m in clip.get("boundaries", {}).get("warnings", []):
        issues.append("límite de frase (aviso): " + m)

    if with_sync and ffmpeg is not None:
        sync = estimate_sync(mp4, lines, dur, ffmpeg)
        out["sync"] = sync
        if sync.get("available"):
            if abs(sync["lag_s"]) >= SYNC_LAG_WARN_S:
                issues.append(f"desfase estimado de {sync['lag_s']:+.1f} s respecto de la voz "
                              f"({'tardíos' if sync['lag_s'] > 0 else 'adelantados'})")
            if sync["corr_at_zero"] < SYNC_CORR_WARN:
                issues.append(f"baja correlación subtítulo↔voz ({sync['corr_at_zero']}): pueden faltar o sobrar líneas")
            if sync.get("speech_without_subtitle_pct") and sync["speech_without_subtitle_pct"] >= 20:
                issues.append(f"{sync['speech_without_subtitle_pct']:.0f} % de la voz detectada no tiene subtítulo")
    return out


# --- elementos de revisión ---------------------------------------------------------------------
def _audio_signals(cand: dict | None, clip_start: float) -> dict:
    if not cand:
        return {}
    comp = {c["name"]: c for c in cand["components"]}
    verbal = comp.get("reaccion_verbal", {})
    return {
        "anchor_in_clip_s": round(cand["anchor"] - clip_start, 1), "triggers": cand["triggers"],
        "intensity": {"value": comp["intensidad_audio"]["value"], "note": comp["intensidad_audio"]["note"]},
        "dynamics": {"value": comp["dinamica_audio"]["value"], "note": comp["dinamica_audio"]["note"]},
        "verbal": {"value": verbal.get("value"), "note": verbal.get("note"), "evidence": verbal.get("evidence", [])},
        "no_accepted_verbal_reaction": (verbal.get("value") == 0) if verbal else None,
    }


def _item(job_dir: Path, result: dict, transcript: Transcript, entry: dict, kind: str, settings: Settings,
          with_sync: bool, cand_by_rank: dict) -> dict:
    cand = cand_by_rank.get(entry["rank"])
    item = {
        "id": f"{result['job_id']}:{entry['name']}", "job_id": result["job_id"],
        "model": result["settings"]["whisper_model"], "kind": kind, "name": entry["name"], "rank": entry["rank"],
        "start": entry["start"], "end": entry["end"], "duration": round(entry["end"] - entry["start"], 2),
        "score": entry["score"], "triggers": entry.get("triggers") or (cand or {}).get("triggers", []),
        "components": [{"name": c["name"], "value": c["value"], "weight": c["weight"], "contribution": c["contribution"],
                        "note": c["note"], "evidence": c.get("evidence", [])} for c in (cand or {}).get("components", [])],
        "transcript": clip_transcript(transcript, entry["start"], entry["end"]),
        "confidence": clip_confidence(transcript, entry["start"], entry["end"]),
        "audio": _audio_signals(cand, entry["start"]), "media": None, "subtitles": None,
        "below_min_score": entry["score"] < result["settings"].get("min_score", 0.35),
    }
    if kind != "candidate":
        # la ruta sale del nombre dentro del job (no de la ruta absoluta guardada): sirve si la carpeta se movió
        mp4 = job_dir / "out" / f"{entry['name']}.mp4"
        item["media"] = f"out/{entry['name']}.mp4"
        item["media_exists"] = mp4.is_file()
        item["validation_ok"] = entry.get("validation", {}).get("ok")
        item["subtitles"] = subtitle_report({**entry, "file": str(mp4)}, transcript,
                                            settings.ffmpeg if with_sync else None, with_sync)
    return item


def build_items(job_dir: Path, settings: Settings, n_alternatives: int = 10, with_sync: bool = True) -> list[dict]:
    """Clips renderizados, alternativos renderizados y hasta n candidatos alternativos sin renderizar."""
    result = load_result(job_dir)
    transcript = Transcript.from_dict(json.loads((job_dir / "transcript.json").read_text(encoding="utf-8")))
    by_rank = {c["rank"]: c for c in result.get("candidates", [])}
    items = [_item(job_dir, result, transcript, c, "clip", settings, with_sync, by_rank) for c in result.get("clips", [])]
    items += [_item(job_dir, result, transcript, c, "alt", settings, with_sync, by_rank) for c in result.get("alt_clips", [])]
    for alt in alternatives_for(result, n_alternatives, settings):
        entry = {"name": f"cand_{alt.rank:02d}", "rank": alt.rank, "start": round(alt.start, 2), "end": round(alt.end, 2),
                 "score": round(alt.score, 3), "triggers": alt.triggers}
        items.append(_item(job_dir, result, transcript, entry, "candidate", settings, with_sync, by_rank))
    return items


def alternatives_for(result: dict, n: int, settings: Settings) -> list[Candidate]:
    """Hasta n candidatos alternativos: sin solape con lo elegido/exportado ni entre sí, y con señal propia."""
    cands = [Candidate.from_dict(d) for d in result.get("candidates", [])]
    rendered = {c["rank"] for c in result.get("clips", [])} | {a["rank"] for a in result.get("alt_clips", [])}
    taken = [c for c in cands if c.rank in set(result.get("selected", [])) | rendered]
    cfg = {**result.get("settings", {})}
    return pick_alternatives([c for c in cands if c not in taken], taken, n, cfg.get("max_overlap", 0.15),
                             cfg.get("text_trigger", 0.5))


def write_alternatives(job_dir: Path, settings: Settings, n: int = 10) -> list[dict]:
    """Escribe alternatives.json y alternatives.md en el job (lista para revisión, sin renderizar)."""
    items = [i for i in build_items(job_dir, settings, n, with_sync=False) if i["kind"] == "candidate"]
    (job_dir / "alternatives.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    L = [f"# Candidatos alternativos — {job_dir.name}", "",
         "Lista para revisión (sin renderizar). Para generar el MP4: `export <job> --ids <#>`.", "",
         "| # | Inicio–fin | Dur. | Puntuación | Señales | Confianza (prob. media / <0.4) | Texto |", "|---|---|---|---|---|---|---|"]
    for i in items:
        c = i["confidence"]
        L.append(f"| {i['rank']} | {i['start']}–{i['end']} s | {i['duration']} s | {i['score']}"
                 + (" (bajo el umbral)" if i["below_min_score"] else "") + f" | {', '.join(i['triggers'])} | "
                 f"{c['mean_word_prob']} / {c['words_below_0_4_pct']} % | {i['transcript'][:110].replace('|', '/')} |")
    (job_dir / "alternatives.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return items


# --- informe de sesión (legible) ---------------------------------------------------------------
def render_item_markdown(i: dict) -> list[str]:
    c, a = i["confidence"], i["audio"]
    kind = {"clip": "clip", "alt": "alternativo renderizado", "candidate": "candidato sin renderizar"}[i["kind"]]
    L = [f"#### `{i['name']}` — {kind} (candidato #{i['rank']}, Whisper {i['model']})", "",
         f"- **Tiempo:** {i['start']}–{i['end']} s del video ({i['duration']} s)",
         f"- **Puntuación:** {i['score']}" + (" (bajo el umbral mínimo)" if i["below_min_score"] else "")
         + f" · señales: {', '.join(i['triggers']) or '—'}",
         f"- **Confianza de transcripción:** {c['words']} palabras, prob. media {c['mean_word_prob']}, "
         f"{c['words_below_0_4_pct']} % con prob < 0.4"
         + (f"; Whisper: {c['flags']}" if c.get("flags") else "; sin estadísticas de Whisper (job anterior)")]
    if c["low_confidence_words"]:
        L.append(f"  - palabras de baja confianza: {', '.join(c['low_confidence_words'])}")
    if a:
        L.append(f"- **Audio:** ancla a {a['anchor_in_clip_s']} s del inicio; {a['intensity']['note']}; {a['dynamics']['note']}")
        L.append(f"- **Reacción verbal:** {a['verbal']['note']}"
                 + ("  _(sin reacción verbal aceptada: elegido por audio/contexto)_" if a["no_accepted_verbal_reaction"] else ""))
    L.append("- **Motivos (componentes):** " + ("; ".join(f"{x['name']} {x['value']}×{x['weight']}" for x in i["components"]) or "—"))
    s = i["subtitles"]
    if s is not None:
        if not s["available"]:
            L.append(f"- **Subtítulos:** {s['issues'][0]}")
        else:
            sync = s.get("sync") or {}
            L.append(f"- **Subtítulos:** {s['lines']} líneas, cobertura {s['coverage_pct']} %"
                     + (f", desfase estimado {sync['lag_s']:+.1f} s" if sync.get("available") else "")
                     + (" — avisos:" if s["issues"] else " — sin avisos"))
            L += [f"  - ⚠️ {x}" for x in s["issues"]]
    L += ["- **Transcripción:** " + (i["transcript"] or "_(sin habla transcrita)_"), ""]
    return L


def render_session_report(label: str, jobs: list[dict], items_by_job: dict[str, list[dict]], compare_path: str | None) -> str:
    L = [f"# Sesión de validación: {label}", "",
         "> Este informe describe hechos (tiempos, transcripción, confianza, señales, avisos). **No** califica ningún clip "
         "como entretenido ni publicable: eso lo decidís al revisarlos (`streamclipper review`).", ""]
    for j in jobs:
        L += [f"## Whisper {j['model']} — `{j['job_id']}` ({j['status']})", ""]
        if j["status"] != "completed":
            L += [f"Falló: {j.get('error')}", ""]
            continue
        its = items_by_job.get(j["job_id"], [])
        for title, kinds in (("Clips elegidos", ("clip",)), ("Alternativos renderizados", ("alt",)),
                             ("Candidatos alternativos para revisión", ("candidate",))):
            sel = [i for i in its if i["kind"] in kinds]
            if sel:
                L += [f"### {title}", ""]
                for i in sel:
                    L += render_item_markdown(i)
    if compare_path:
        L += ["## Comparación base vs small", "", f"Ver `{compare_path}`."]
    return "\n".join(L) + "\n"


# --- evaluaciones humanas ----------------------------------------------------------------------
class ReviewStore:
    """Evaluaciones de un video (por huella de archivo). Escritura atómica; un JSON legible por persona."""

    def __init__(self, path: Path, source: str = "", fingerprint: str = "") -> None:
        self.path = path
        self.data = {"fingerprint": fingerprint, "source": source, "ratings": {}}
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def for_source(cls, settings: Settings, fingerprint: str, source: str) -> "ReviewStore":
        return cls(reviews_dir(settings) / f"{fingerprint}.ratings.json", source, fingerprint)

    @property
    def ratings(self) -> dict:
        return self.data["ratings"]

    def set(self, item: dict, rating: str | None, note: str = "", algorithm: dict | None = None) -> None:
        if rating is not None and rating not in RATINGS:
            raise InvalidInputError(f"Evaluación inválida: {rating!r} (válidas: {', '.join(RATINGS)})")
        if len(note) > 2000:
            raise InvalidInputError("La nota es demasiado larga (máx. 2000 caracteres)")
        if rating is None and not note:
            self.ratings.pop(item["id"], None)
        else:
            self.ratings[item["id"]] = {
                "rating": rating, "note": note, "rated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "job_id": item["job_id"], "model": item["model"], "kind": item["kind"], "name": item["name"],
                "rank": item["rank"], "start": item["start"], "end": item["end"], "score": item["score"],
                "algorithm": algorithm or {},
            }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def summary(self) -> dict:
        counts = {k: 0 for k in RATINGS}
        for r in self.ratings.values():
            if r["rating"] in counts:
                counts[r["rating"]] += 1
        return counts

    def moments(self) -> list[dict]:
        """Une evaluaciones del mismo momento (IoU ≥ 0.5) venidas de modelos o versiones distintas.

        El rótulo es el de la evaluación más reciente; si discrepan se marca `conflict`."""
        out: list[dict] = []
        for r in sorted((r for r in self.ratings.values() if r["rating"]), key=lambda r: r["start"]):
            for m in out:
                if best_match(r, [m], IOU_MATCH)[0] is not None:
                    m["ratings"].append(r)
                    break
            else:
                out.append({"start": r["start"], "end": r["end"], "ratings": [r]})
        for m in out:
            latest = max(m["ratings"], key=lambda r: r["rated_at"])
            m["rating"] = latest["rating"]
            m["conflict"] = len({r["rating"] for r in m["ratings"]}) > 1
        return out


def coverage_against_reviews(result: dict, store: ReviewStore) -> dict:
    """Cómo se ubica un job frente a las evaluaciones humanas ya guardadas (sirve para comparar versiones).

    - Precisión: de los clips elegidos, cuántos caen sobre momentos evaluados como bueno/edición/no sirve.
    - Cobertura: de los momentos evaluados como bueno o edición, cuántos están entre los elegidos y en qué
      posición del ranking aparece el más cercano.
    """
    moments = store.moments()
    cands = result.get("candidates", [])
    sel = [c for c in cands if c["rank"] in set(result.get("selected", []))]
    chosen = {"good": 0, "edit": 0, "bad": 0, "unrated": 0}
    for c in sel:
        hit = next((m for m in moments if best_match(c, [m], IOU_MATCH)[0] is not None), None)
        chosen[hit["rating"] if hit else "unrated"] += 1
    wanted = []
    for m in moments:
        if m["rating"] in ("good", "edit"):
            in_sel = best_match(m, sel, IOU_MATCH)[0] is not None
            cand, _ = best_match(m, cands, IOU_MATCH)
            wanted.append({"start": m["start"], "end": m["end"], "rating": m["rating"], "selected": in_sel,
                           "candidate_rank": cand["rank"] if cand else None})
    return {"selected_by_rating": chosen, "good_or_edit_moments": wanted,
            "good_or_edit_selected": sum(w["selected"] for w in wanted),
            "rated_moments": len(moments), "conflicts": sum(m["conflict"] for m in moments)}

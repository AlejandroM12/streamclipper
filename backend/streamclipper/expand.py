"""«Ampliar con contexto»: propone un intervalo contextual nuevo para un clip YA generado, sin reprocesar el video completo.

- Clips de videos cortos y de tramos: la fuente de la sesión ya contiene todo el contexto; solo se extrae el audio de la ventana
  alrededor del clip y se aplica la construcción narrativa (selection/context_clip.py).
- Clips del modo automático: su fuente es un recorte de unos segundos. El contexto se reúne desde las transcripciones YA hechas de
  los bloques del trabajo (incluidos los vecinos si el clip cae cerca de un límite) y se vuelve a descargar solo el rango necesario
  del VOD (clip ± 40 s). Nunca se analiza ni se descarga el VOD completo.
- Los archivos originales no se tocan: todo lo nuevo va a `<sesión>/context/` y el clip regenerado es una versión nueva (edit-vNNN).
"""
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

from streamclipper.analysis.audio import analyze_audio
from streamclipper.analysis.transitions import detect_transitions
from streamclipper.auto import FETCH_S, merge_transcripts
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError, MediaError, StreamClipperError
from streamclipper.jobs import load_result
from streamclipper.media.download import DownloadLimits, download_segment
from streamclipper.media.ffmpeg import probe, run_cancellable
from streamclipper.models import Candidate, Segment, Transcript, Word
from streamclipper.pipeline import fingerprint, settings_from_result
from streamclipper.selection.context_clip import PRESETS, build_window

ProgressFn = Callable[[str, float, str], None]


def proposal_path(job_dir: Path, clip: str) -> Path:
    return job_dir / "context" / f"proposal-{clip}.json"


def load_proposal(job_dir: Path, clip: str) -> dict | None:
    p = proposal_path(job_dir, clip)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def _audio_curve(src: Path, a: float, b: float, cfg: Settings, work: Path, cancel: CancelToken):
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "ctx.wav"
    code, _, err = run_cancellable([cfg.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", src,
                                    "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], cancel)
    if code != 0:
        raise MediaError(f"No se pudo extraer el audio para analizar el contexto: {err.strip()[-200:]}")
    try:
        return analyze_audio(wav)
    finally:
        wav.unlink(missing_ok=True)


def _shifted(tr: Transcript, shift: float, lo: float, hi: float) -> list[Segment]:
    out = []
    for s in tr.segments:
        a, b = s.start - shift, s.end - shift
        if a >= lo - 0.01 and b <= hi + 0.01:
            out.append(Segment(a, b, s.text, [Word(w.start - shift, w.end - shift, w.text, w.prob) for w in s.words],
                               s.no_speech_prob, s.avg_logprob, s.compression_ratio))
    return out


def _group_blocks(settings: Settings, group: str) -> list[dict]:
    """Bloques de análisis de un trabajo automático (sesiones ocultas) con sus transcripciones guardadas."""
    out = []
    for d in settings.jobs_dir.iterdir() if settings.jobs_dir.is_dir() else []:
        try:
            m = json.loads((d / "segment.json").read_text(encoding="utf-8"))
            if m.get("group") != group or m.get("mode") != "auto-block":
                continue
            tr = Transcript.from_dict(json.loads((d / "transcript.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            continue
        out.append({"start": m["start"], "end": m["end"], "dl_start": m["dl_start"], "dl_end": m["dl_end"], "index": m["index"],
                    "last": m["index"] == m["count"] - 1, "transcript": tr})
    return sorted(out, key=lambda x: x["index"])


def _event_of(entry: dict) -> tuple[float, float]:
    """Evento del clip ya generado: su construcción previa si la tiene; si no (clips de 10 s de versiones anteriores), el clip mismo."""
    ctx = entry.get("context")
    if ctx and ctx.get("event"):
        return float(ctx["event"][0]), float(ctx["event"][1])
    return float(entry["start"]), float(entry["end"])


def prepare_context(job_dir: Path, clip: str, preset: str, settings: Settings, cancel: CancelToken | None = None,
                    progress: ProgressFn | None = None, ydl_factory=None) -> dict:
    cancel = cancel or CancelToken()
    progress = progress or (lambda *_: None)
    if preset not in PRESETS:
        raise InvalidInputError(f"Duración de clips desconocida: {preset}")
    result = load_result(job_dir)
    entry = next((c for c in result.get("clips", []) if c.get("name") == clip), None)
    if entry is None:
        raise InvalidInputError(f"El clip {clip} no existe en esta sesión.")
    cfg = settings_from_result(result, data_dir=settings.data_dir, clip_preset=preset)
    meta_p = job_dir / "segment.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    cand_raw = next((c for c in result.get("candidates", []) if c.get("rank") == entry.get("rank")), None)
    e0, e1 = _event_of(entry)                                    # en el eje de tiempo de la fuente de la sesión
    work = job_dir / "context" / ".work"
    progress("preparar", 0.0, "")
    try:
        if meta.get("mode") == "auto-clip":
            prop = _from_vod(job_dir, entry, cand_raw, (e0, e1), meta, preset, cfg, settings, cancel, work, ydl_factory)
        else:
            prop = _from_own_source(job_dir, result, entry, cand_raw, (e0, e1), preset, cfg, cancel, work)
        progress("preparar", 1.0, "")
        progress("proponer", 0.0, "")
        d = job_dir / "context"
        d.mkdir(exist_ok=True)
        shift = prop.pop("_orig_shift", 0.0)                      # el clip original expresado en el eje del editor
        prop.update(clip=clip, preset=preset, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    original={"start": round(float(entry["start"]) + shift, 2), "end": round(float(entry["end"]) + shift, 2)})
        prop["original"]["duration"] = round(prop["original"]["end"] - prop["original"]["start"], 2)
        prop["duration"] = round(prop["end"] - prop["start"], 2)
        proposal_path(job_dir, clip).write_text(json.dumps(prop, indent=2, ensure_ascii=False), encoding="utf-8")
        progress("proponer", 1.0, "")
        return prop
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _build(cand_raw: dict | None, ev_local: tuple[float, float], segs, curve, dur: float, shift: float, preset: str, cfg, transitions=None):
    """Construcción en el eje LOCAL de la ventana analizada. `shift` pasa los tiempos del candidato guardado a ese eje."""
    mid = (ev_local[0] + ev_local[1]) / 2
    cand = Candidate(start=cand_raw["start"] - shift if cand_raw else ev_local[0], end=cand_raw["end"] - shift if cand_raw else ev_local[1],
                     anchor=cand_raw["anchor"] - shift if cand_raw else mid, triggers=cand_raw.get("triggers", []) if cand_raw else [])
    return build_window(cand, segs, curve, dur, cfg, preset, event=ev_local, transitions=transitions)


def _axis(ctx: dict | None, off: float) -> dict | None:
    if not ctx:
        return None
    return {**ctx, **{k: [round(v + off, 2) for v in ctx[k]] for k in ("original", "event", "contextual", "final")}}


def _from_own_source(job_dir, result, entry, cand_raw, ev, preset, cfg, cancel, work) -> dict:
    src = Path(result.get("source", ""))
    if not src.is_file():
        raise InvalidInputError("El video de origen de esta sesión ya no está disponible, así que no se puede buscar más contexto. "
                                "Los clips y las versiones que ya existen no se modifican.")
    if result.get("source_fingerprint") and fingerprint(src) != result["source_fingerprint"]:
        raise InvalidInputError("El video de origen cambió desde que se procesó esta sesión; no se puede ampliar con seguridad.")
    dur = float(result["media"]["duration"])
    a, b = max(0.0, ev[0] - FETCH_S), min(dur, ev[1] + FETCH_S)
    tr = Transcript.from_dict(json.loads((job_dir / "transcript.json").read_text(encoding="utf-8")))
    curve = _audio_curve(src, a, b, cfg, work, cancel)
    trn = detect_transitions(src, cfg.ffmpeg, a, b, cancel)           # cambios visuales: límites entre situaciones distintas
    w = _build(cand_raw, (ev[0] - a, ev[1] - a), _shifted(tr, a, 0, b - a), curve, b - a, a, preset, cfg, [t - a for t in trn.times])
    # el eje del editor es el de la fuente de la sesión: local + a
    return {"source": "own", "start": round(w.start + a, 2), "end": round(w.end + a, 2), "context": _axis(w.context, a), "abs": None,
            "_orig_shift": 0.0, "transitions": trn.to_dict()}


def _from_vod(job_dir, entry, cand_raw, ev, meta, preset, cfg, settings, cancel, work, ydl_factory) -> dict:
    cut_a = float(meta["cut"][0])
    blocks = _group_blocks(settings, meta["group"])
    if not blocks:
        raise InvalidInputError("No se encontraron las transcripciones de los bloques de este trabajo: no se puede reunir el contexto.")
    vod_dur = max(b["dl_end"] for b in blocks)
    e0a, e1a = cut_a + ev[0], cut_a + ev[1]                      # evento en tiempo absoluto del VOD
    a, b = max(0, int(e0a - FETCH_S)), min(int(vod_dur), int(e1a + FETCH_S + 0.999))
    try:
        res = download_segment(meta["url"], a, b, cfg, DownloadLimits(), on_exists="reuse", cancel=cancel, ydl_factory=ydl_factory, auto=True)
    except StreamClipperError as e:
        raise InvalidInputError(f"No se pudo volver a descargar el contexto del VOD ({e}). Si la transmisión ya no está disponible en la "
                                "plataforma, no se puede ampliar; el clip original sigue intacto.") from None
    ctx_dir = job_dir / "context"
    ctx_dir.mkdir(exist_ok=True)
    dst = ctx_dir / "source.mp4"
    if dst.exists():
        dst.unlink()
    try:
        os.link(res.path, dst)                                    # mismo disco: sin duplicar el archivo
    except OSError:
        shutil.copy2(res.path, dst)
    merged = merge_transcripts(blocks, a, b)
    local = Transcript(merged.language, _shifted(merged, a, 0, b - a), [], 2)
    (ctx_dir / "transcript.json").write_text(json.dumps(local.to_dict(), ensure_ascii=False), encoding="utf-8")
    dur = probe(dst, cfg.ffprobe).duration
    (ctx_dir / "meta.json").write_text(json.dumps({"source": "source.mp4", "offset": a, "range": [a, b], "url": meta["url"]}), encoding="utf-8")
    curve = _audio_curve(dst, 0.0, dur, cfg, work, cancel)
    # eje del editor = eje del recorte de contexto (local). El candidato guardado está en el eje del recorte original: local = cut_a + t - a
    trn = detect_transitions(dst, cfg.ffmpeg, 0.0, dur, cancel)       # cambios visuales: límites entre situaciones distintas
    w = _build(cand_raw, (e0a - a, e1a - a), local.segments, curve, dur, a - cut_a, preset, cfg, trn.times)
    return {"source": "context", "start": round(w.start, 2), "end": round(w.end, 2), "context": w.context,
            "abs": {"start": round(w.start + a, 2), "end": round(w.end + a, 2), "context": _axis(w.context, a)}, "_orig_shift": cut_a - a,
            "transitions": trn.to_dict()}

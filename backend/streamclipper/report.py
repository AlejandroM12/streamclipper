"""Reporte de análisis por ejecución: report.json (estructurado) y report.md (legible)."""
import json
from pathlib import Path


def build_summary(result: dict) -> dict:
    stages = result.get("stages", {})

    def secs(*names: str) -> float | None:
        vals = [stages[n]["seconds"] for n in names if n in stages]
        return round(sum(vals), 2) if vals else None

    duration = (result.get("media") or {}).get("duration")
    t_trans = secs("transcribir")
    selected = set(result.get("selected", []))
    clips = result.get("clips", [])
    alts = result.get("alt_clips", [])
    return {
        "job_id": result["job_id"],
        "status": result["status"],
        "source": result["source"],
        "video_duration_s": round(duration, 2) if duration else None,
        "whisper_model": result["settings"]["whisper_model"],
        "whisper_compute_type": result["settings"]["whisper_compute_type"],
        "language": (result.get("transcript") or {}).get("language"),
        "transcript": result.get("transcript"),
        "timings_s": {
            "validation_and_audio_extraction": secs("validar", "extraer_audio"),
            "transcription": t_trans,
            "analysis": secs("analizar_audio", "seleccionar"),
            "rendering": secs("renderizar", "verificar"),
            "total": result.get("total_seconds"),
            "transcription_realtime_factor": round(t_trans / duration, 3) if t_trans and duration else None,
        },
        "resources": result.get("resources"),
        "candidates_detected": len(result.get("candidates", [])),
        "candidates": [
            {"rank": c["rank"], "start": c["start"], "end": c["end"], "score": c["score"],
             "triggers": c["triggers"], "reasons": c["explanation"], "selected": c["rank"] in selected,
             "text": c["text"][:200]}
            for c in result.get("candidates", [])
        ],
        "clips_generated": len(clips),
        "clips": [_clip_summary(c) for c in clips],
        "alternative_clips": [_clip_summary(c) for c in alts],
        "error": result.get("error"),
        "warnings": result.get("warnings", []),
        "settings": result["settings"],
    }


def _clip_summary(c: dict) -> dict:
    return {
        "name": c["name"], "rank": c["rank"], "file": c["file"], "start": c["start"], "end": c["end"],
        "duration": c["duration"], "score": c["score"],
        "validation_ok": c["validation"]["ok"], "format": c["validation"]["info"],
        "sentence_boundary_errors": c["boundaries"]["errors"],
        "sentence_boundary_warnings": c["boundaries"]["warnings"],
    }


def _fmt(v, suffix="") -> str:
    return "—" if v is None else f"{v}{suffix}"


def render_markdown(s: dict) -> str:
    t, r = s["timings_s"], s["resources"] or {}
    L = [f"# Reporte {s['job_id']}", "",
         f"- **Estado:** {s['status']}" + (f" — {s['error']}" if s["error"] else ""),
         f"- **Video:** `{s['source']}`",
         f"- **Duración del video:** {_fmt(s['video_duration_s'], ' s')}",
         f"- **Modelo Whisper:** {s['whisper_model']} ({s['whisper_compute_type']}), idioma: {_fmt(s['language'])}",
         f"- **Transcripción:** {_fmt(t['transcription'], ' s')} "
         f"(factor {_fmt(t['transcription_realtime_factor'])}× la duración del video)",
         f"- **Análisis (audio + selección):** {_fmt(t['analysis'], ' s')}",
         f"- **Renderizado y verificación:** {_fmt(t['rendering'], ' s')}",
         f"- **Total:** {_fmt(t['total'], ' s')}",
         f"- **RAM (aprox.):** pico Python {_fmt(r.get('python_peak_mb'), ' MB')}, "
         f"pico FFmpeg {_fmt(r.get('ffmpeg_peak_mb'), ' MB')}, "
         f"pico combinado {_fmt(r.get('total_peak_mb_approx'), ' MB')}, "
         f"mínimo libre del sistema {_fmt(r.get('system_min_available_gb'), ' GB')}",
         f"- **Candidatos detectados:** {s['candidates_detected']} — **clips generados:** {s['clips_generated']}",
         "", "## Clips generados", ""]
    for c in s["clips"] + s["alternative_clips"]:
        ok = "OK" if c["validation_ok"] else "FALLÓ"
        L.append(f"- `{c['name']}` (candidato #{c['rank']}): {c['start']}s → {c['end']}s "
                 f"({c['duration']} s), puntuación {c['score']}, validación {ok}")
        for m in c["sentence_boundary_errors"]:
            L.append(f"  - ❌ {m}")
        for m in c["sentence_boundary_warnings"]:
            L.append(f"  - ⚠️ {m}")
    if not s["clips"]:
        L.append("_Ninguno._")
    L += ["", "## Candidatos (por puntuación)", "",
          "| # | Inicio | Fin | Puntuación | Elegido | Señales | Motivos |", "|---|---|---|---|---|---|---|"]
    for c in s["candidates"][:25]:
        L.append(f"| {c['rank']} | {c['start']} | {c['end']} | {c['score']} | {'✅' if c['selected'] else ''} | "
                 f"{', '.join(c['triggers'])} | {c['reasons'].replace('|', '/')} |")
    L += ["", "> La puntuación es un ranking heurístico dentro de este video, no una probabilidad de viralización.",
          "", "## Advertencias y errores", ""]
    L += [f"- ⚠️ {w}" for w in s["warnings"]] or ["_Sin advertencias._"]
    if s["error"]:
        L.append(f"- ❌ {s['error']}")
    return "\n".join(L) + "\n"


def write_reports(job_dir: Path, result: dict) -> dict:
    summary = build_summary(result)
    (job_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (job_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary

"""Historial de sesiones y galería de clips (lee lo que ya produjo el pipeline; no duplica datos)."""
import json
import re
import time
from pathlib import Path

from streamclipper.core.config import Settings
from streamclipper.edit import list_edits
from streamclipper.jobs import load_result
from streamclipper.models import Transcript
from streamclipper.review import RATINGS, ReviewStore, build_items, clip_confidence, clip_transcript
from streamclipper.webutil import ApiError

_STAMP = re.compile(r"^(\d{8})-(\d{6})")


def _created(name: str, d: Path) -> float:
    m = _STAMP.match(name)
    if m:
        try:
            return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
        except ValueError:
            pass
    return d.stat().st_mtime


def _source_title(result: dict) -> str:
    src = Path(result.get("source", ""))
    side = src.with_suffix(".source.json")
    try:
        if side.is_file():
            t = json.loads(side.read_text(encoding="utf-8")).get("title")
            if t:
                return str(t)
    except (OSError, json.JSONDecodeError):
        pass
    return src.name or "video"


def summarize(d: Path, result: dict, active: set[str]) -> dict:
    status = result.get("status", "?")
    if status == "running":   # un job que quedó «running» en disco y no está activo se interrumpió
        status = "running" if d.name in active else "interrupted"
    variants = sorted(p.name.removeprefix("out_") for p in d.glob("out_*") if p.is_dir())
    legacy = any("name" not in c for c in result.get("clips", []))   # jobs de la etapa 2: clips sin nombre
    edits = 0 if legacy else sum(len(list_edits(d, c["name"])) for c in result.get("clips", []))
    return {
        "id": d.name, "title": _source_title(result), "source": result.get("source", ""), "created": _created(d.name, d),
        "status": status, "model": result.get("settings", {}).get("whisper_model"),
        "language": (result.get("transcript") or {}).get("language"),
        "video_seconds": (result.get("media") or {}).get("duration"), "clips": len(result.get("clips", [])),
        "candidates": len(result.get("candidates", [])), "variants": variants, "edits": edits,
        "warnings": len(result.get("warnings", [])), "error": result.get("error"),
        "total_seconds": result.get("total_seconds"), "rescored_from": result.get("rescored_from"),
        "legacy": legacy,
    }


def list_sessions(settings: Settings, active: set[str] | None = None, include_derived: bool = False) -> list[dict]:
    active = active or set()
    out = []
    if not settings.jobs_dir.is_dir():
        return out
    for d in settings.jobs_dir.iterdir():
        if not d.is_dir() or not (d / "result.json").is_file():
            continue
        try:
            result = load_result(d)
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("rescored_from") and not include_derived:
            continue   # análisis derivados (rescore) sin clips propios
        out.append(summarize(d, result, active))
    return sorted(out, key=lambda s: s["created"], reverse=True)


def _version_entry(job_id: str, clip: str, vid: str, label: str, rel: str, start: float, end: float, extra: dict) -> dict:
    return {"id": vid, "label": label, "rating_id": f"{job_id}:{clip}" + ("" if vid == "original" else f"@{vid}"),
            "media": rel, "start": round(start, 2), "end": round(end, 2), "duration": round(end - start, 2), **extra}


def session_detail(settings: Settings, name: str, active: set[str] | None = None) -> dict:
    d = settings.jobs_dir / name
    result = load_result(d)
    if any("name" not in c for c in result.get("clips", [])):
        raise ApiError(422, "Esta sesión es de una versión muy antigua (sus clips no tienen nombre) y no se puede abrir en la galería.")
    transcript = Transcript.from_dict(json.loads((d / "transcript.json").read_text(encoding="utf-8")))
    items = build_items(d, settings, 0, with_sync=False)
    by_clip: dict[str, list[dict]] = {}
    for it in items:
        if it["kind"] == "clip":
            by_clip.setdefault(it["name"].split("@")[0], []).append(it)
    job_id = result["job_id"]
    clips = []
    for c in result.get("clips", []):
        cname = c["name"]
        versions = []
        for it in by_clip.get(cname, []):
            vid = it["variant"]
            sub = it.get("subtitles") or {}
            versions.append(_version_entry(
                job_id, cname, vid, "Original" if vid == "original" else f"Versión {vid}", it["media"], it["start"], it["end"], {
                    "kind": "original" if vid == "original" else "variant", "media_exists": it.get("media_exists", True),
                    "issues": sub.get("issues", []), "cues": (sub.get("lines") or 0), "coverage_pct": sub.get("coverage_pct"),
                    "has_srt": (d / Path(it["media"]).parent / f"{Path(it['media']).stem}.srt").is_file()}))
        for m in list_edits(d, cname):
            versions.append(_version_entry(
                job_id, cname, m["version"], f"Editada {m['version'].removeprefix('edit-')}",
                f"edits/{cname}/{m['version'].removeprefix('edit-')}/clip.mp4", m["start"], m["end"], {
                    "kind": "edit", "media_exists": True, "issues": m.get("warnings", []), "cues": m.get("cue_count"),
                    "coverage_pct": (m.get("metrics") or {}).get("coverage_pct"), "has_srt": True,
                    "subtitles": m.get("subtitles"), "preset": m.get("preset"), "created_at": m.get("created_at"),
                    "framing": (m.get("framing") or {}).get("note")}))
        base = by_clip.get(cname, [None])[0] or {}
        cand = next((x for x in result.get("candidates", []) if x["rank"] == c["rank"]), {})
        clips.append({
            "name": cname, "rank": c["rank"], "score": c["score"], "triggers": c.get("triggers", []),
            "start": c["start"], "end": c["end"], "duration": c["duration"],
            "transcript": clip_transcript(transcript, c["start"], c["end"]),
            "confidence": clip_confidence(transcript, c["start"], c["end"]),
            "reasons": [{"name": x["name"], "value": x["value"], "note": x["note"]} for x in cand.get("components", [])],
            "refine_notes": c.get("refine_notes", []), "boundaries": c.get("boundaries", {}),
            "audio": base.get("audio", {}), "versions": versions,
        })
    store = ReviewStore.for_source(settings, result.get("source_fingerprint", ""), result.get("source", ""))
    ids = {v["rating_id"] for c in clips for v in c["versions"]}
    return {
        "session": summarize(d, result, active or set()), "labels": RATINGS, "clips": clips,
        "ratings": {k: v for k, v in store.ratings.items() if k in ids},
        "settings": {k: result.get("settings", {}).get(k) for k in ("layout", "safe_platform", "max_clips", "whisper_model", "language")},
        "warnings": result.get("warnings", []), "stages": result.get("stages", {}), "resources": result.get("resources"),
        "selection": result.get("selection"), "media_duration": (result.get("media") or {}).get("duration"),
        "has_source": Path(result.get("source", "")).is_file(),
    }


def rating_item(settings: Settings, name: str, clip: str, version: str) -> tuple[dict, ReviewStore, dict]:
    """(elemento, almacén, algoritmo) para guardar una evaluación de una versión de un clip."""
    from streamclipper.review import algorithm_info
    d = settings.jobs_dir / name
    detail = session_detail(settings, name)
    for c in detail["clips"]:
        if c["name"] != clip:
            continue
        for v in c["versions"]:
            if v["id"] == version:
                item = {"id": v["rating_id"], "job_id": detail["session"]["id"], "model": detail["session"]["model"] or "",
                        "kind": "clip", "name": f"{clip}@{version}" if version != "original" else clip, "rank": c["rank"],
                        "start": v["start"], "end": v["end"], "score": c["score"]}
                result = load_result(d)
                store = ReviewStore.for_source(settings, result.get("source_fingerprint", ""), result.get("source", ""))
                return item, store, algorithm_info(result)
    raise KeyError("clip o versión inexistente")

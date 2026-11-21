"""Exporta clips de candidatos alternativos de un job ya procesado, sin repetir la transcripción."""
import json
import shutil
import time
from pathlib import Path

from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError
from streamclipper.core.log import log_event, setup_logging
from streamclipper.jobs import load_result, resolve_job
from streamclipper.models import Candidate, Transcript
from streamclipper.pipeline import fingerprint, settings_from_result
from streamclipper.render.clips import render_set
from streamclipper.report import write_reports
from streamclipper.selection.selector import overlap_ratio, text_similarity


def _has_trigger(c: Candidate, text_trigger: float) -> bool:
    by = {x.name: x for x in c.components}
    return by["intensidad_audio"].value >= 0.35 or by["reaccion_verbal"].value >= text_trigger


def pick_alternatives(candidates: list[Candidate], taken: list[Candidate], count: int,
                      max_overlap: float, text_trigger: float) -> list[Candidate]:
    """Siguientes mejores candidatos que no se solapan ni repiten lo ya exportado y tienen señal propia."""
    picked: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: c.rank):
        if len(picked) >= count:
            break
        if c in taken or not _has_trigger(c, text_trigger):
            continue
        others = taken + picked
        if any(overlap_ratio(c, o) > max_overlap or text_similarity(c, o) > 0.6 for o in others):
            continue
        picked.append(c)
    return picked


def export_alternatives(
    job_ref: str,
    settings: Settings,
    count: int = 3,
    ranks: list[int] | None = None,
    cancel: CancelToken | None = None,
    progress=None,
    selected: bool = False,
    **setting_overrides,
) -> dict:
    """Renderiza candidatos alternativos (por cantidad o por número de ranking) reutilizando transcript.json.

    Con `selected=True` renderiza los clips ELEGIDOS de un job que no los renderizó (p. ej. re-puntuado con
    `rescore`), con los nombres clip_NN."""
    cancel = cancel or CancelToken()
    progress = progress or (lambda *_: None)
    job_dir = resolve_job(job_ref, settings)
    result = load_result(job_dir)
    if not result.get("candidates"):
        raise InvalidInputError("El job no tiene candidatos para exportar")
    src = Path(result["source"])
    if not src.is_file():
        raise InvalidInputError(f"El video original ya no existe: {src}")
    if fingerprint(src) != result.get("source_fingerprint"):
        raise InvalidInputError("El video original cambió desde que se procesó este job; reprocesalo")

    cfg = settings_from_result(result, **setting_overrides)
    cfg.validate()
    transcript = Transcript.from_dict(json.loads((job_dir / "transcript.json").read_text(encoding="utf-8")))
    cands = [Candidate.from_dict(d) for d in result["candidates"]]
    by_rank = {c.rank: c for c in cands}
    # «exportado» = renderizado. Los seleccionados de un job re-puntuado todavía no lo están.
    done = {c["rank"] for c in result.get("clips", [])} | {a["rank"] for a in result.get("alt_clips", [])}
    taken = [by_rank[r] for r in (set(result["selected"]) | done) if r in by_rank]  # para evitar solapes
    names = {}

    warnings: list[str] = []
    if selected:
        if result.get("clips"):
            raise InvalidInputError("Este job ya tiene sus clips seleccionados renderizados")
        if not result["selected"]:
            raise InvalidInputError("El job no seleccionó ningún clip")
        ranks = list(result["selected"])
        names = {r: f"clip_{i:02d}" for i, r in enumerate(ranks, 1)}
    if ranks:
        missing = [r for r in ranks if r not in by_rank]
        if missing:
            raise InvalidInputError(f"Candidatos inexistentes: {missing}. Hay {len(cands)} (ver report.md)")
        chosen = []
        for r in ranks:
            if r in done:
                warnings.append(f"El candidato #{r} ya estaba exportado; se omite")
            else:
                chosen.append(by_rank[r])
                clash = [o.rank for o in taken if overlap_ratio(by_rank[r], o) > cfg.max_overlap]
                if clash:
                    warnings.append(f"El candidato #{r} se solapa con el/los clip(s) ya exportado(s) #{clash}")
                if by_rank[r].score < cfg.min_score:
                    warnings.append(f"El candidato #{r} tiene puntuación {by_rank[r].score:.2f} "
                                    f"< umbral {cfg.min_score}: exportado por pedido explícito")
    else:
        chosen = pick_alternatives(cands, taken, count, cfg.max_overlap, cfg.text_trigger)
        if len(chosen) < count:
            warnings.append(f"Solo hay {len(chosen)} alternativo(s) sin solape/repetición con lo ya exportado")

    setup_logging(job_dir / "job.log")
    work = job_dir / f"work_export_{int(time.time())}"
    out_dir = job_dir / "out"
    out_dir.mkdir(exist_ok=True)
    work.mkdir()
    new_clips: list[dict] = []
    t0 = time.monotonic()
    out_name = lambda c: names.get(c.rank, f"alt_{c.rank:02d}")
    try:
        if chosen:
            new_clips = render_set(src, [(out_name(c), c) for c in chosen], transcript, cfg, work,
                                   out_dir, cancel, progress, warnings)
    except BaseException:
        for c in chosen:  # sin salidas a medias ante error o cancelación
            for ext in (".mp4", ".mp4.part", ".ass"):
                (out_dir / f"{out_name(c)}{ext}").unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)

    result.setdefault("clips" if selected else "alt_clips", []).extend(new_clips)
    result["warnings"].extend(warnings)
    result.setdefault("export_runs", []).append({
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "ranks": [c.rank for c in chosen],
        "seconds": round(time.monotonic() - t0, 2), "overrides": setting_overrides,
    })
    (job_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    write_reports(job_dir, result)
    log_event("alternatives_exported", job=result["job_id"], ranks=[c.rank for c in chosen])
    return {"job_dir": str(job_dir), "clips": new_clips, "warnings": warnings}

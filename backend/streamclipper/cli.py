import argparse
import gc
import json
import sys

from streamclipper.compare import compare_jobs, save_comparison
from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import StreamClipperError
from streamclipper.evaluation import evaluate_jobs
from streamclipper.export import export_alternatives
from streamclipper.jobs import load_result
from streamclipper.rescore import rescore_job
from streamclipper.review import ReviewStore, coverage_against_reviews, write_alternatives
from streamclipper.review_server import load_state, serve
from streamclipper.session import resolve_review_target, run_session
from streamclipper.jobs import resolve_job
from streamclipper.media.download import DownloadLimits, download_youtube, looks_like_url
from streamclipper.media.ffmpeg import probe
from streamclipper.pipeline import STAGES, resolve_input, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="streamclipper", description="Clips verticales desde streams de videojuegos")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("process", help="Analiza un MP4 y genera hasta N clips verticales")
    pr.add_argument("input", help="Ruta a un MP4 local, o URL HTTPS de YouTube (se descarga a data/samples/)")
    _add_download_args(pr)
    pr.add_argument("--model", default="base",
                    help="Modelo faster-whisper; acepta varios separados por coma (base,small) y los corre "
                         "uno tras otro, cada uno en su propio job")
    pr.add_argument("--label", default=None, help="Etiqueta opcional para el nombre del job")
    pr.add_argument("--language", default=None, help="Código de idioma (es, en); por defecto autodetecta")
    pr.add_argument("--max-clips", type=int, default=3)
    pr.add_argument("--min-clip", type=float, default=15.0, help="Duración mínima en s")
    pr.add_argument("--max-clip", type=float, default=60.0, help="Duración máxima en s")
    pr.add_argument("--pre", type=float, default=6.0, help="Contexto previo al momento (s)")
    pr.add_argument("--post", type=float, default=6.0, help="Contexto posterior al momento (s)")
    pr.add_argument("--min-score", type=float, default=0.35, help="Puntuación mínima para generar un clip")
    _add_lexicon_args(pr)
    pr.add_argument("--peak-db", type=float, default=6.0,
                    help="dB sobre el nivel habitual para considerar un pico de audio (calibración)")
    pr.add_argument("--text-trigger", type=float, default=0.5,
                    help="Reacción verbal mínima (0..1) para considerar una frase (calibración)")
    pr.add_argument("--layout", choices=["blur-fit", "fill", "crop"], default="blur-fit")
    pr.add_argument("--crop-x", type=float, default=0.5, help="Posición horizontal del recorte (0..1)")
    pr.add_argument("--crop-y", type=float, default=0.5, help="Posición vertical del recorte (0..1)")
    pr.add_argument("--zoom", type=float, default=1.0)
    pr.add_argument("--threads", type=int, default=4)
    pr.add_argument("--keep-temp", action="store_true")
    pr.add_argument("--json", action="store_true", help="Imprime el reporte completo en JSON")

    ex = sub.add_parser("export", help="Exporta clips de candidatos alternativos de un job, sin retranscribir")
    ex.add_argument("job", help="Id (o prefijo) del job, o ruta a su carpeta")
    g = ex.add_mutually_exclusive_group()
    g.add_argument("--alternatives", type=int, default=None, metavar="N",
                   help="Exporta los N mejores candidatos no elegidos que no se solapen (por defecto 3)")
    g.add_argument("--ids", type=int, nargs="+", metavar="RANK", help="Exporta candidatos concretos (columna # de report.md)")
    g.add_argument("--selected", action="store_true",
                   help="Renderiza los clips elegidos de un job que no los renderizó (p. ej. tras `rescore`)")
    ex.add_argument("--layout", choices=["blur-fit", "fill", "crop"], default=None)
    ex.add_argument("--crop-x", type=float, default=None)
    ex.add_argument("--crop-y", type=float, default=None)
    ex.add_argument("--zoom", type=float, default=None)

    se = sub.add_parser("session", help="Validación: procesa un video (URL o MP4) con varios modelos y prepara la revisión")
    se.add_argument("input", help="URL HTTPS de YouTube o ruta a un MP4")
    se.add_argument("--models", default="base,small", help="Modelos Whisper separados por coma (defecto base,small)")
    se.add_argument("--label", default=None, help="Nombre de la sesión (defecto: nombre del archivo)")
    se.add_argument("--language", default=None, help="Idioma hablado (es, en); por defecto autodetecta")
    se.add_argument("--alternatives", type=int, default=10, help="Candidatos alternativos a listar por modelo")
    se.add_argument("--threads", type=int, default=4)
    _add_download_args(se)

    rv = sub.add_parser("review", help="Abre la herramienta local para evaluar clips (bueno / edición / no sirve)")
    rv.add_argument("target", nargs="+", help="Nombre de sesión, o ids/fragmentos de uno o más jobs del mismo video")
    rv.add_argument("--port", type=int, default=8765)
    rv.add_argument("--no-open", action="store_true", help="No abrir el navegador automáticamente")
    rv.add_argument("--alternatives", type=int, default=10, help="Candidatos sin renderizar a incluir por job")
    rv.add_argument("--no-sync", action="store_true", help="Omitir la estimación de sincronía de subtítulos (más rápido)")

    al = sub.add_parser("alternatives", help="Escribe la lista de candidatos alternativos de un job (alternatives.md/json)")
    al.add_argument("job")
    al.add_argument("--n", type=int, default=10)

    rr = sub.add_parser("reviews", help="Resume las evaluaciones humanas y cómo se ubica un job frente a ellas")
    rr.add_argument("jobs", nargs="+")

    rs = sub.add_parser("rescore", help="Re-puntúa un job con el algoritmo actual, sin retranscribir ni descargar")
    rs.add_argument("job", help="Id (o fragmento) del job, o ruta a su carpeta")
    rs.add_argument("--suffix", default="rescored", help="Sufijo del job nuevo")
    rs.add_argument("--evaluate", action="store_true", help="Imprime y guarda la evaluación antes/después")
    _add_lexicon_args(rs)

    ev = sub.add_parser("evaluate", help="Evalúa antes/después: job original vs job re-puntuado")
    ev.add_argument("old")
    ev.add_argument("new")

    cp = sub.add_parser("compare", help="Compara dos jobs del mismo video (p. ej. base vs small)")
    cp.add_argument("job_a")
    cp.add_argument("job_b")

    pb = sub.add_parser("probe", help="Valida un video con FFprobe")
    pb.add_argument("input")

    dl = sub.add_parser("download", help="Descarga un video público de YouTube a data/samples/ (uso autorizado)")
    _add_download_args(dl, with_url=True)
    dl.add_argument("--overwrite", action="store_true", help="Reemplaza el archivo si ya existe")
    dl.add_argument("--dry-run", action="store_true",
                    help="Solo valida la URL y consulta duración/tamaño/licencia; no descarga")
    return p


def _add_lexicon_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lexicon", default=None, help="JSON que amplía o ajusta el léxico de reacciones (docs/lexicon.md)")
    p.add_argument("--min-word-prob", type=float, default=None,
                   help="Confianza mínima de Whisper para que una palabra de reacción cuente (defecto 0.4)")
    p.add_argument("--repeat-min-prob", type=float, default=None,
                   help="Confianza media mínima para que una repetición cuente (defecto 0.6)")


def _lexicon_overrides(args) -> dict:
    return {k: v for k, v in (("lexicon_path", args.lexicon), ("min_word_prob", args.min_word_prob),
                              ("repeat_min_prob", args.repeat_min_prob)) if v is not None}


def _add_download_args(p: argparse.ArgumentParser, with_url: bool = False) -> None:
    if with_url:
        p.add_argument("url", help="URL HTTPS de YouTube o YouTube Shorts (un único video)")
    p.add_argument("--max-minutes", type=float, default=20.0, help="Duración máxima permitida (min)")
    p.add_argument("--max-size-mb", type=int, default=1500, help="Tamaño máximo permitido (MB)")
    p.add_argument("--max-height", type=int, default=1080, help="Resolución máxima a descargar (alto en px)")


def _limits(args) -> DownloadLimits:
    return DownloadLimits(max_duration_s=args.max_minutes * 60, max_bytes=args.max_size_mb * 1024 * 1024,
                          max_height=args.max_height)


def _print_result(result: dict) -> None:
    r = result.get("resources") or {}
    print(f"\nTarea {result['job_id']}: {result['status']}")
    if result["error"]:
        print(f"Error: {result['error']}")
    for c in result["clips"]:
        ok = "OK" if c["validation"]["ok"] else "FALLÓ"
        print(f"  {c['file']}  [{c['start']:.1f}s-{c['end']:.1f}s] puntuación {c['score']:.2f} validación {ok}")
        print(f"    {c['explanation']}")
    for w in result["warnings"]:
        print(f"  Aviso: {w}")
    print(f"Tiempo total {result.get('total_seconds')} s | RAM pico aprox. {r.get('total_peak_mb_approx')} MB "
          f"(Python {r.get('python_peak_mb')} MB, FFmpeg {r.get('ffmpeg_peak_mb')} MB)")
    print(f"Reporte: {result['job_dir']}\\report.md")


def _print_download(res) -> None:
    if res.dry_run:
        print(f"Validación OK (no se descargó nada): '{res.title}' — {res.duration / 60:.1f} min, "
              f"tamaño estimado {_mb(res.estimated_bytes)}, licencia: {res.license or 'no informada'}")
        print(f"Se guardaría en {res.path}")
    elif res.reused:
        print(f"Ya estaba descargado, se reutiliza: {res.path}")
    else:
        print(f"Descargado y verificado con FFprobe: {res.path}\n  '{res.title}' — {res.duration / 60:.1f} min, "
              f"{_mb(res.size_bytes)}, licencia: {res.license or 'no informada'}")
    for w in res.warnings:
        print(f"  Aviso: {w}")


def _mb(n: int | None) -> str:
    return "desconocido" if n is None else f"{n / 1e6:.0f} MB"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # consolas cp1252 no pueden imprimir «⚠️», «—», etc.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "probe":
            print(json.dumps(vars(probe(resolve_input(args.input), Settings().ffprobe)), indent=2))
            return 0

        if args.cmd == "download":
            cancel = CancelToken()
            try:
                res = download_youtube(
                    args.url, Settings(), _limits(args), dry_run=args.dry_run, cancel=cancel,
                    on_exists="overwrite" if args.overwrite else "error",
                    progress=lambda m: print(f"  {m}", file=sys.stderr))
            except KeyboardInterrupt:
                cancel.cancel()
                print("Cancelado.", file=sys.stderr)
                return 130
            _print_download(res)
            return 0

        if args.cmd == "session":
            cancel = CancelToken()
            try:
                meta = run_session(args.input, [m.strip() for m in args.models.split(",") if m.strip()], Settings(),
                                   label=args.label, limits=_limits(args), n_alternatives=args.alternatives,
                                   cancel=cancel, progress=lambda m: print(m, file=sys.stderr),
                                   language=args.language)
            except KeyboardInterrupt:
                cancel.cancel()
                print("Cancelado.", file=sys.stderr)
                return 130
            print(f"\nSesión '{meta['label']}' lista en {meta['session_dir']}")
            for j in meta["jobs"]:
                print(f"  Whisper {j['model']}: {j['status']}" + (f" — {j['error']}" if j["error"] else "")
                      + f"  ({j['job_dir']})")
            print(f"Informe por clip: {meta['session_dir']}\\report.md"
                  + (f"\nComparación: {meta['compare']}" if meta["compare"] else ""))
            print(f"\nPara revisar y evaluar los clips:\n  streamclipper review {meta['label']}")
            return 0 if all(j["status"] == "completed" for j in meta["jobs"]) else 1

        if args.cmd == "review":
            s = Settings()
            dirs = resolve_review_target(args.target, s)
            if not dirs:
                print("Error: la sesión no tiene jobs completados", file=sys.stderr)
                return 2
            print("Preparando elementos de revisión (estimando sincronía de subtítulos)…", file=sys.stderr)
            serve(load_state(dirs, s, args.alternatives, with_sync=not args.no_sync), args.port, not args.no_open)
            return 0

        if args.cmd == "alternatives":
            s = Settings()
            d = resolve_job(args.job, s)
            items = write_alternatives(d, s, args.n)
            print(f"{len(items)} candidatos alternativos escritos en {d}\\alternatives.md y alternatives.json")
            return 0

        if args.cmd == "reviews":
            s = Settings()
            for ref in args.jobs:
                d = resolve_job(ref, s)
                res = load_result(d)
                store = ReviewStore.for_source(s, res.get("source_fingerprint", ""), res["source"])
                cov = coverage_against_reviews(res, store)
                counts = store.summary()
                print(f"\n{d.name}  (evaluaciones del video: {counts['good']} bueno, {counts['edit']} edición, "
                      f"{counts['bad']} no sirve; {cov['rated_moments']} momentos, {cov['conflicts']} con discrepancias)")
                sel = cov["selected_by_rating"]
                print(f"  Clips elegidos: {sel['good']} bueno, {sel['edit']} edición, {sel['bad']} no sirve, "
                      f"{sel['unrated']} sin evaluar")
                print(f"  Momentos buenos/edición cubiertos por los elegidos: {cov['good_or_edit_selected']}"
                      f"/{len(cov['good_or_edit_moments'])}")
                for w in cov["good_or_edit_moments"]:
                    print(f"    {w['start']}–{w['end']}s ({w['rating']}): "
                          + ("elegido" if w["selected"] else f"candidato #{w['candidate_rank']}" if w["candidate_rank"]
                             else "ningún candidato"))
            return 0

        if args.cmd == "rescore":
            s = Settings()
            old_name = resolve_job(args.job, s).name
            res = rescore_job(old_name, s, suffix=args.suffix, **_lexicon_overrides(args))
            print(f"Job re-puntuado: {res['job_id']}  (candidatos {len(res['candidates'])}, "
                  f"seleccionados {len(res['selected'])})\nReporte: {res['job_dir']}\\report.md")
            for w in res["warnings"][1:]:
                print(f"  Aviso: {w}")
            if args.evaluate:
                data, text = evaluate_jobs(old_name, res["job_id"], s)
                print("\n" + text)
                print(f"Guardado en {save_comparison(text, old_name, res['job_id'], s, kind='eval')}")
            return 0

        if args.cmd == "evaluate":
            s = Settings()
            data, text = evaluate_jobs(args.old, args.new, s)
            print(text)
            print(f"Guardado en {save_comparison(text, resolve_job(args.old, s).name, resolve_job(args.new, s).name, s, kind='eval')}")
            return 0

        if args.cmd == "compare":
            s = Settings()
            text = compare_jobs(args.job_a, args.job_b, s)
            path = save_comparison(text, resolve_job(args.job_a, s).name, resolve_job(args.job_b, s).name, s)
            print(text)
            print(f"Guardado en {path}")
            return 0

        cancel = CancelToken()

        def progress(stage: str, frac: float, msg: str) -> None:
            if msg or frac in (0.0, 1.0):
                n = STAGES.index(stage) + 1
                print(f"[{n}/{len(STAGES)}] {stage:<15} {'listo' if frac == 1.0 and not msg else msg}", file=sys.stderr)

        if args.cmd == "export":
            overrides = {k: v for k, v in (("layout", args.layout), ("crop_x", args.crop_x),
                                           ("crop_y", args.crop_y), ("zoom", args.zoom)) if v is not None}
            try:
                out = export_alternatives(args.job, Settings(), count=args.alternatives or 3, ranks=args.ids,
                                          cancel=cancel, progress=progress, selected=args.selected, **overrides)
            except KeyboardInterrupt:
                cancel.cancel()
                print("Cancelado.", file=sys.stderr)
                return 130
            for c in out["clips"]:
                print(f"  {c['file']}  [{c['start']:.1f}s-{c['end']:.1f}s] candidato #{c['rank']} "
                      f"puntuación {c['score']:.2f}")
            for w in out["warnings"]:
                print(f"  Aviso: {w}")
            if not out["clips"]:
                print("No había alternativos para exportar.")
            return 0

        if looks_like_url(args.input):  # URL: se descarga (o se reutiliza lo ya descargado) y se procesa el MP4
            try:
                res = download_youtube(args.input, Settings(), _limits(args), on_exists="reuse", cancel=cancel,
                                       progress=lambda m: print(f"  {m}", file=sys.stderr))
            except KeyboardInterrupt:
                cancel.cancel()
                print("Cancelado.", file=sys.stderr)
                return 130
            _print_download(res)
            args.input = str(res.path)

        models = [m.strip() for m in args.model.split(",") if m.strip()]
        job_ids, exit_code = [], 0
        for model in models:
            s = Settings(
                whisper_model=model, language=args.language, max_clips=args.max_clips,
                min_clip=args.min_clip, max_clip=args.max_clip, pre_context=args.pre, post_context=args.post,
                min_score=args.min_score, audio_peak_db=args.peak_db, text_trigger=args.text_trigger,
                **_lexicon_overrides(args),
                layout=args.layout, crop_x=args.crop_x, crop_y=args.crop_y,
                zoom=args.zoom, cpu_threads=args.threads,
            )
            print(f"\n=== Whisper {model} ===", file=sys.stderr)
            try:
                result = run_pipeline(args.input, s, cancel, progress, keep_temp=args.keep_temp, label=args.label)
            except KeyboardInterrupt:
                cancel.cancel()
                print("Cancelado.", file=sys.stderr)
                return 130
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            else:
                _print_result(result)
            job_ids.append(result["job_id"])
            exit_code = exit_code or (0 if result["status"] == "completed" else 1)
            gc.collect()  # liberar el modelo antes del siguiente
        if len(job_ids) == 2 and exit_code == 0:
            s = Settings()
            text = compare_jobs(job_ids[0], job_ids[1], s)
            print("\n" + text)
            print(f"Comparación guardada en {save_comparison(text, job_ids[0], job_ids[1], s)}")
        return exit_code
    except StreamClipperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

import argparse
import json
import sys

from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import StreamClipperError
from streamclipper.media.ffmpeg import probe
from streamclipper.pipeline import STAGES, resolve_input, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="streamclipper", description="Clips verticales desde streams de videojuegos")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("process", help="Analiza un MP4 y genera hasta N clips verticales")
    pr.add_argument("input", help="Ruta a un MP4 local")
    pr.add_argument("--model", default="base", help="Modelo faster-whisper (base, small, ...)")
    pr.add_argument("--language", default=None, help="Código de idioma (es, en); por defecto autodetecta")
    pr.add_argument("--max-clips", type=int, default=3)
    pr.add_argument("--min-clip", type=float, default=15.0, help="Duración mínima en s")
    pr.add_argument("--max-clip", type=float, default=60.0, help="Duración máxima en s")
    pr.add_argument("--pre", type=float, default=6.0, help="Contexto previo al momento (s)")
    pr.add_argument("--post", type=float, default=6.0, help="Contexto posterior al momento (s)")
    pr.add_argument("--min-score", type=float, default=0.35)
    pr.add_argument("--layout", choices=["blur-fit", "fill", "crop"], default="blur-fit")
    pr.add_argument("--crop-x", type=float, default=0.5, help="Posición horizontal del recorte (0..1)")
    pr.add_argument("--crop-y", type=float, default=0.5, help="Posición vertical del recorte (0..1)")
    pr.add_argument("--zoom", type=float, default=1.0)
    pr.add_argument("--threads", type=int, default=4)
    pr.add_argument("--keep-temp", action="store_true")
    pr.add_argument("--json", action="store_true", help="Imprime el resultado completo en JSON")

    pb = sub.add_parser("probe", help="Valida un video con FFprobe")
    pb.add_argument("input")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "probe":
            s = Settings()
            print(json.dumps(vars(probe(resolve_input(args.input), s.ffprobe)), indent=2))
            return 0

        s = Settings(
            whisper_model=args.model, language=args.language, max_clips=args.max_clips,
            min_clip=args.min_clip, max_clip=args.max_clip, pre_context=args.pre, post_context=args.post,
            min_score=args.min_score, layout=args.layout, crop_x=args.crop_x, crop_y=args.crop_y,
            zoom=args.zoom, cpu_threads=args.threads,
        )
        cancel = CancelToken()

        def progress(stage: str, frac: float, msg: str) -> None:
            if msg or frac in (0.0, 1.0):
                n = STAGES.index(stage) + 1
                print(f"[{n}/{len(STAGES)}] {stage:<15} {'listo' if frac == 1.0 and not msg else msg}", file=sys.stderr)

        try:
            result = run_pipeline(args.input, s, cancel, progress, keep_temp=args.keep_temp)
        except KeyboardInterrupt:
            cancel.cancel()
            print("Cancelado.", file=sys.stderr)
            return 130

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            print(f"\nTarea {result['job_id']}: {result['status']}")
            if result["error"]:
                print(f"Error: {result['error']}")
            if result.get("note"):
                print(result["note"])
            for c in result["clips"]:
                ok = "OK" if c["validation"]["ok"] else "FALLÓ"
                print(f"  {c['file']}  [{c['start']:.1f}s-{c['end']:.1f}s] puntuación {c['score']:.2f} validación {ok}")
                print(f"    {c['explanation']}")
            print(f"Resultados: {result['job_dir']}")
        return 0 if result["status"] == "completed" else 1
    except StreamClipperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

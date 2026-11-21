import json
from pathlib import Path

from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError


def resolve_job(ref: str, settings: Settings) -> Path:
    """Acepta una ruta a la carpeta del job o un fragmento único de su id (p. ej. 'small-partida1')."""
    p = Path(ref)
    if p.is_dir() and (p / "result.json").is_file():
        return p.resolve()
    base = settings.jobs_dir
    if (base / ref).is_dir() and (base / ref / "result.json").is_file():  # un id exacto siempre gana
        return base / ref
    matches = [d for d in base.iterdir() if d.is_dir() and ref in d.name] if base.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise InvalidInputError(f"No se encontró el job '{ref}' en {base}")
    raise InvalidInputError(f"El id '{ref}' es ambiguo: {', '.join(m.name for m in matches)}")


def load_result(job_dir: Path) -> dict:
    return json.loads((job_dir / "result.json").read_text(encoding="utf-8"))

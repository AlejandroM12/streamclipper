"""Estado persistente de los trabajos de la aplicación (JSON por trabajo, sin base de datos)."""
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

ACTIVE = ("queued", "running")
FINAL = ("completed", "failed", "cancelled", "interrupted")

STAGE_LABELS = {
    "descarga": "Descargar el video",
    "validar": "Validar el video",
    "extraer_audio": "Extraer el audio",
    "transcribir": "Transcribir (Whisper)",
    "analizar_audio": "Analizar el audio",
    "seleccionar": "Elegir los mejores momentos",
    "renderizar": "Generar los clips",
    "verificar": "Verificar los clips",
    "preparar": "Preparar la edición",
}
PROCESS_STAGES = ["validar", "extraer_audio", "transcribir", "analizar_audio", "seleccionar", "renderizar", "verificar"]
EDIT_STAGES = ["preparar", "renderizar", "verificar"]


@dataclass
class Stage:
    key: str
    label: str
    status: str = "pending"          # pending | running | done | failed | cancelled
    fraction: float | None = None    # solo cuando el avance es REAL (p. ej. segundos transcritos); None = sin dato
    detail: str = ""
    started: float | None = None
    seconds: float | None = None


@dataclass
class AppJob:
    id: str
    kind: str                         # process | edit
    status: str = "queued"           # queued | running | completed | failed | cancelled | interrupted
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    title: str = ""
    input: str = ""                   # URL o ruta del MP4
    input_kind: str = ""              # url | file | edit
    params: dict = field(default_factory=dict)
    stages: list[dict] = field(default_factory=list)
    message: str = ""
    error: str | None = None
    hint: str | None = None
    error_detail: str | None = None
    session: str | None = None        # carpeta del job del pipeline (data/jobs/<session>)
    warnings: list[str] = field(default_factory=list)
    dup_key: str = ""
    result: dict = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppJob":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def stage(self, key: str) -> dict | None:
        return next((s for s in self.stages if s["key"] == key), None)


def make_stages(keys: list[str]) -> list[dict]:
    return [asdict(Stage(k, STAGE_LABELS.get(k, k))) for k in keys]


class JobStore:
    """Un JSON por trabajo en data/app/jobs/, con escritura atómica."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save(self, job: AppJob) -> None:
        path = self.root / f"{job.id}.json"
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)

    def load_all(self) -> list[AppJob]:
        jobs = []
        for p in sorted(self.root.glob("*.json")):
            try:
                jobs.append(AppJob.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, TypeError):
                continue   # un JSON dañado no impide arrancar
        return jobs

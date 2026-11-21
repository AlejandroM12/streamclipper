# StreamClipper

Detecta momentos destacados en videos de streams y los exporta como clips verticales (1080×1920, H.264 + AAC) con subtítulos.
Todo corre en local, sobre CPU. Uso personal. **Estado: pipeline CLI (Etapa 2). Todavía no hay interfaz web ni API.**
La calidad de selección con gameplay real **aún no está validada**: ver [docs/real-video-test.md](docs/real-video-test.md).

## Instalación (Windows 11)

Requisitos: Python 3.11+ y el build portable de FFmpeg de <https://www.gyan.dev/ffmpeg/builds/> (`ffmpeg-release-essentials`).

```powershell
# 1. FFmpeg portable: copiá ffmpeg.exe y ffprobe.exe (carpeta bin del zip) a tools\ffmpeg\
# 2. Entorno Python aislado
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

No se modifica el PATH ni se instalan componentes globales. La primera transcripción descarga el modelo Whisper
(`base` ~150 MB, `small` ~480 MB) en la caché de Hugging Face.

## Uso

```powershell
cd backend
.\.venv\Scripts\python.exe -m streamclipper probe   ruta\video.mp4
.\.venv\Scripts\python.exe -m streamclipper process ruta\video.mp4                       # base por defecto
.\.venv\Scripts\python.exe -m streamclipper process ruta\video.mp4 --model base,small    # ambos + comparación
.\.venv\Scripts\python.exe -m streamclipper compare <jobA> <jobB>
.\.venv\Scripts\python.exe -m streamclipper export  <job> --alternatives 3               # sin retranscribir
.\.venv\Scripts\python.exe -m streamclipper export  <job> --ids 5 7
```

Opciones de `process`: `--model`, `--label`, `--language es|en`, `--max-clips`, `--min-clip/--max-clip`, `--pre/--post`
(contexto en s), `--layout blur-fit|fill|crop` (+ `--crop-x/--crop-y/--zoom`), `--min-score`, `--peak-db`,
`--text-trigger`, `--threads`, `--keep-temp`, `--json`.

Cada ejecución crea `data/jobs/<fecha>-<modelo>[-etiqueta]-<id>/` con `out/clip_NN.mp4` (+ `.ass`), `transcript.json`,
`result.json` (estado completo), `report.md` / `report.json` (duración, tiempos por etapa, modelo, RAM aproximada,
candidatos con puntuación y motivos, timestamps, avisos y errores) y `job.log` (JSON por línea). Si el material no
tiene momentos que superen el umbral, **no se genera ningún clip**.

La puntuación es un ranking heurístico explicable dentro de cada video; **no** es una probabilidad de viralización.
Cada clip se verifica con FFprobe (códecs, resolución, duración), se decodifica completo y se comprueba que no
empiece ni termine dentro de una frase o palabra transcrita (si hay indicios de corte semántico, avisa).

## Pruebas

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -m "not integration"   # rápidas (~3 s)
.\.venv\Scripts\python.exe -m pytest                        # todas; incluye FFmpeg + Whisper reales (~90 s)
```

La integración usa un video sintético (`backend/tools/make_synthetic.py`, voz SAPI de Windows). Valida el pipeline
técnico, no la calidad del detector con gameplay real.

## Arquitectura

```
backend/streamclipper/
  media/ (ffprobe, audio, validación)   transcription/ (faster-whisper, oraciones)
  analysis/ (audio, texto, límites)     selection/ (candidatos, CandidateScorer, selector)
  render/ (ASS, vertical, clips)        pipeline.py, export.py, compare.py, report.py, cli.py
  core/ (config, logs, memoria, cancelación)
```

`selection/base.py::CandidateScorer` es el punto de extensión para un scorer basado en Ollama (no se usa en el MVP).

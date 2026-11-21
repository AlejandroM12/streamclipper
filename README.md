# StreamClipper

Detecta momentos destacados en videos de streams y los exporta como clips verticales (1080×1920, H.264 + AAC) con subtítulos.
Todo corre en local, sobre CPU. Uso personal. **Estado: Etapa 2 (pipeline CLI). Todavía no hay interfaz web ni API.**

## Instalación (Windows 11)

Requisitos: Python 3.11+ y el build portable de FFmpeg de <https://www.gyan.dev/ffmpeg/builds/> (`ffmpeg-release-essentials`).

```powershell
# 1. FFmpeg portable: copiá ffmpeg.exe y ffprobe.exe (carpeta bin del zip) a tools\ffmpeg\
# 2. Entorno Python aislado
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

No se modifica el PATH ni se instalan componentes globales. La primera transcripción descarga el modelo
Whisper (`base`, ~150 MB) en la caché de Hugging Face.

## Uso

```powershell
cd backend
.\.venv\Scripts\python.exe -m streamclipper probe  ruta\video.mp4
.\.venv\Scripts\python.exe -m streamclipper process ruta\video.mp4
```

Opciones útiles de `process`: `--model small`, `--language es`, `--max-clips 3`, `--min-clip 15 --max-clip 60`,
`--pre 6 --post 6` (contexto en segundos), `--layout blur-fit|fill|crop` (con `--crop-x/--crop-y/--zoom`),
`--min-score`, `--threads`, `--keep-temp`, `--json`.

Cada tarea crea `data/jobs/<id>/` con `out/clip_NN.mp4` (+ `.ass`), `transcript.json`, `result.json`
(candidatos con puntuación y explicación) y `job.log` (JSON por línea). Si el material no tiene momentos que
superen el umbral, **no se genera ningún clip**.

La puntuación es un ranking heurístico explicable dentro de cada video; **no** es una probabilidad de viralización.

## Pruebas

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -m "not integration"   # rápidas
.\.venv\Scripts\python.exe -m pytest -m integration         # FFmpeg + Whisper reales (~30 s)
```

La integración usa un video sintético (`backend/tools/make_synthetic.py`, voz SAPI de Windows). Valida el pipeline
técnico, no la calidad del detector con gameplay real.

## Arquitectura

```
backend/streamclipper/
  media/ (ffprobe, audio, validación)   transcription/ (faster-whisper, oraciones)
  analysis/ (audio, texto)              selection/ (candidatos, CandidateScorer, selector)
  render/ (ASS, vertical 1080x1920)     pipeline.py, cli.py
```

`selection/base.py::CandidateScorer` es el punto de extensión para un scorer basado en Ollama (no se usa en el MVP).

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
.\.venv\Scripts\python.exe -m streamclipper compare <jobA> <jobB>      # candidatos, correlación, top-3 cruzado, clips
.\.venv\Scripts\python.exe -m streamclipper export  <job> --alternatives 3               # sin retranscribir
.\.venv\Scripts\python.exe -m streamclipper export  <job> --ids 5 7
.\.venv\Scripts\python.exe -m streamclipper rescore <job> --suffix v2 --evaluate         # algoritmo actual, sin retranscribir
.\.venv\Scripts\python.exe -m streamclipper export  <job>-v2 --selected                  # renderiza los clips del re-puntuado
.\.venv\Scripts\python.exe -m streamclipper evaluate <job> <job>-v2                      # antes/después
```

### Descargar desde YouTube (solo contenido propio o autorizado)

```powershell
.\.venv\Scripts\python.exe -m streamclipper download "https://www.youtube.com/watch?v=<id>" --dry-run   # solo valida
.\.venv\Scripts\python.exe -m streamclipper download "https://www.youtube.com/watch?v=<id>"             # descarga
.\.venv\Scripts\python.exe -m streamclipper process  "https://www.youtube.com/watch?v=<id>" --model base,small
```

- Acepta solo URLs **HTTPS** de `youtube.com`, `m.youtube.com`, `youtu.be` y `/shorts/` de **un único video**. Se rechazan
  listas (`/playlist`; en `watch?v=..&list=..` se ignora la lista), canales, búsquedas, directos, credenciales y otros dominios.
- Antes de descargar consulta duración, disponibilidad y tamaño estimado. Límites por defecto: **20 min**, **1500 MB**,
  **1080p** (`--max-minutes`, `--max-size-mb`, `--max-height`). `--dry-run` muestra título, duración, tamaño estimado y licencia.
- Se guarda como `data/samples/youtube-<id>.mp4` (ignorado por git) con un `youtube-<id>.source.json` de procedencia.
  El MP4 se valida con FFprobe. **No sobrescribe**: si existe, falla salvo `--overwrite` (`process <URL>` lo reutiliza).
- No usa cookies ni evade restricciones: videos privados, eliminados, con restricción de edad/región, solo para miembros,
  con DRM o directos se rechazan con un mensaje claro.

Opciones de `process`: `--model`, `--label`, `--language es|en`, `--max-clips`, `--min-clip/--max-clip`, `--pre/--post`
(contexto en s), `--layout blur-fit|fill|crop` (+ `--crop-x/--crop-y/--zoom`), `--min-score`, `--peak-db`,
`--text-trigger`, `--lexicon ruta.json`, `--min-word-prob`, `--repeat-min-prob`, `--threads`, `--keep-temp`, `--json`.

**Léxico y confianza:** las reacciones verbales dependen de la confianza de Whisper y, para palabras ambiguas como
«vamos», de evidencia de énfasis (contexto, repetición confiable, alargamiento, volumen). Reglas y cómo
personalizarlas: [docs/lexicon.md](docs/lexicon.md). Cada explicación indica la ventana temporal usada y la evidencia
aceptada o descartada. El reporte avisa cuando la selección es frágil (candidatos con puntuaciones casi empatadas).

`compare` informa por separado: (1) los conjuntos completos de candidatos (IoU, correlación de ranking), (2) dónde cae el
top-3 de cada job en el ranking del otro y (3) los clips finales. Coincidir en (3) no implica mejor calidad.
`rescore` aplica el algoritmo actual a un job existente reutilizando su transcripción (`evaluate` explica cada cambio de
puntuación y marca posibles regresiones); los jobs anteriores sin estadísticas de Whisper siguen funcionando.

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
  rescore.py, evaluation.py             analysis/lexicon.py, analysis/confidence.py
  core/ (config, logs, memoria, cancelación)
```

`selection/base.py::CandidateScorer` es el punto de extensión para un scorer basado en Ollama (no se usa en el MVP).

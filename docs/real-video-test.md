# Guía: prueba con un video real de gaming

Objetivo: comprobar la calidad de selección con material real y reunir datos para calibrar. **No se cambian
umbrales hasta tener varios videos** (ver "Calibración").

## 1. Preparación

- **Video:** MP4 de 5 a 20 minutos, con voz del streamer, de una fuente que tengas autorizada. Empezá por uno de ~5 min.
- **Idioma:** si hablan en español, agregá `--language es` (más rápido y evita errores de autodetección).
- **RAM:** cerrá navegadores/IDEs pesados. Con menos de 1 GB libres el pipeline avisa; se midió ~0.5–0.6 GB de pico
  con `base`/`small` en el video sintético, pero no se midió aún con video real largo.
- **Espacio:** el pipeline exige 5 GB libres (`min_free_gb`).

## 1b. Obtener el video

Si es tuyo o tenés autorización para usarlo, podés dejar que StreamClipper lo descargue (YouTube, un solo video,
máx. 20 min). Primero validá sin descargar:

```powershell
.\.venv\Scripts\python.exe -m streamclipper download "<URL del video autorizado>" --dry-run
```

Luego `process "<URL>"` (paso 2) lo descarga a `data\samples\` y lo procesa; o descargalo aparte con `download`.
Si ya tenés el MP4, usá su ruta.

## 2. Ejecutar y comparar `base` vs `small`

```powershell
cd C:\Users\axel_\Projects\StreamClipper\backend
.\.venv\Scripts\python.exe -m streamclipper process "C:\ruta\al\video.mp4" --model base,small --language es --label partida1
```

Corre `base` y luego `small` (uno por vez, cada uno en su carpeta `data\jobs\<fecha>-<modelo>-partida1-xxxx\`) y
genera la comparación en `data\comparisons\`. La primera vez con `small` descarga ~480 MB.

Con URL en vez de ruta (descarga una vez y reutiliza el archivo en las siguientes corridas):

```powershell
.\.venv\Scripts\python.exe -m streamclipper process "<URL del video autorizado>" --model base,small --language es --label partida1
```

Por separado, si preferís:

```powershell
.\.venv\Scripts\python.exe -m streamclipper process "C:\ruta\video.mp4" --model base  --language es --label partida1
.\.venv\Scripts\python.exe -m streamclipper process "C:\ruta\video.mp4" --model small --language es --label partida1
.\.venv\Scripts\python.exe -m streamclipper compare partida1-base partida1-small   # o los ids/prefijos completos
```

(`compare` y `export` aceptan el id completo, un fragmento único del id como `small-partida1`, o la ruta de la
carpeta del job. Si el fragmento coincide con varios jobs, avisa y lista las opciones; usá una `--label` distinta
por corrida para evitarlo.)

## 3. Revisar resultados

Cada job tiene `report.md` (legible) y `report.json`: duración, tiempos por etapa, RAM, candidatos con puntuación y
motivos, clips con timestamps, avisos. Para cada clip en `out\clip_NN.mp4`:

1. ¿Es realmente un momento destacado? (sí / parcial / no)
2. ¿Empieza y termina en un lugar natural? Los avisos de `report.md` marcan cortes a mitad de frase.
3. ¿Los subtítulos coinciden con lo que se oye?
4. ¿Se ve bien el encuadre? Probá `--layout fill --crop-x 0.2` o `--layout crop --zoom 1.5` si la cámara no está centrada.

Después, mirá la tabla de candidatos: ¿hay momentos buenos que quedaron fuera (falsos negativos) o clips elegidos que
no lo eran (falsos positivos)? Anotá los minutos reales de los mejores momentos del video como referencia.

## 4. Clips alternativos (sin retranscribir)

```powershell
.\.venv\Scripts\python.exe -m streamclipper export <job> --alternatives 3   # siguientes mejores sin solape
.\.venv\Scripts\python.exe -m streamclipper export <job> --ids 5 7          # por número (#) de report.md
```

Salen como `out\alt_NN.mp4` del mismo job y se agregan a su reporte.

## 5. Calibración (después de ver los resultados)

Registrá por video: modelo, falsos positivos, falsos negativos y `report.md`. Parámetros disponibles (los valores por
defecto no se cambiaron):

| Opción | Defecto | Efecto |
|---|---|---|
| `--peak-db` | 6.0 | dB sobre el nivel habitual para tratar un pico de audio como ancla |
| `--text-trigger` | 0.5 | reacción verbal mínima para considerar una frase |
| `--min-score` | 0.35 | puntuación mínima para generar un clip |
| `--pre` / `--post` | 6 / 6 | contexto en segundos antes/después del momento |
| `--min-clip` / `--max-clip` | 15 / 60 | duración permitida |

Los pesos de la puntuación están en `streamclipper/selection/heuristic.py` y el léxico de reacciones en
`streamclipper/analysis/text.py` (inglés/español; ampliarlo con el vocabulario real del streamer es probablemente la
mejora más rentable). Cambiá una cosa por vez y reprocesá con `export`/`process` para ver el efecto.

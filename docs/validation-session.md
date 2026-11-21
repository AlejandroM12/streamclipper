# Validación de calidad con gameplay real

Objetivo: saber si StreamClipper encuentra momentos interesantes en un video de gaming y si los clips sirven después
de una revisión humana. **La calidad no se da por validada hasta que revises los resultados reales.** Ninguna parte de
la herramienta califica un clip como entretenido o publicable: solo informa hechos y guarda tu criterio.

> Regla de la prueba: se usan los **parámetros por defecto**. No se ajusta ningún umbral, peso ni léxico antes de
> mirar los resultados originales.

## 1. Un comando

Con una URL (video propio o que tengas autorización para usar) o un MP4 local:

```powershell
cd C:\Users\axel_\Projects\StreamClipper\backend
.\.venv\Scripts\python.exe -m streamclipper session "<URL o ruta.mp4>" --label partida1 --language es
```

Corre Whisper `base` y luego `small` (uno por vez; la primera vez descarga `small`, ~480 MB). Para cada modelo:
job completo con **los 3 mejores clips renderizados**, reporte, y la **lista de hasta 10 candidatos alternativos**
(`alternatives.md` / `alternatives.json`, sin renderizar). Al final genera la comparación entre modelos y un informe
por clip. Queda todo en `data\reviews\partida1\` (`session.json`, `report.md`) y en `data\jobs\<job>\`.

Antes de empezar cerrá navegadores/IDEs pesados (RAM). Opciones: `--models base,small`, `--alternatives 10`,
`--max-minutes 20`, `--max-height 1080`. Si la etiqueta ya existe, no sobrescribe.

## 2. Informe por clip (`data\reviews\<etiqueta>\report.md`)

Para cada clip y alternativo: tiempo y duración, puntuación y componentes, motivos (con evidencia aceptada/descartada),
transcripción, confianza de transcripción (probabilidad media, % de palabras < 0.4, estadísticas de Whisper),
señales de audio (ancla, pico, subida, reacción verbal) y **avisos de subtítulos**:

| Aviso | Qué mide |
|---|---|
| primeros/últimos N s sin subtítulos | huecos ≥ 2.5 s en los bordes del clip |
| líneas de baja confianza | ≥ 40 % de las palabras de la línea con prob. < 0.4 |
| caracteres/segundo | líneas con más de 20 cps (difíciles de leer) |
| palabras > 1.5 s | marcas de tiempo dudosas |
| alucinación típica | frases como «gracias por ver», «suscríbete», «subtítulos de…» |
| límites de frase | cortes dentro de frases o indicios semánticos |
| desfase estimado | subtítulos contra la actividad de voz (VAD) del clip: + = tardíos; |lag| ≥ 0.3 s o correlación < 0.3 |

El desfase es una **estimación** (un VAD no es una referencia exacta); los avisos son informativos y no intervienen en la
selección. Un clip «sin reacción verbal aceptada» fue elegido por audio y contexto: revisalo con ese dato en mente.

## 3. Revisar y evaluar

```powershell
.\.venv\Scripts\python.exe -m streamclipper review partida1
```

Abre en el navegador (solo en tu equipo, `127.0.0.1`) una página con todos los clips y candidatos. Cada uno tiene
reproductor, transcripción, avisos y tres botones: **Bueno para publicar**, **Necesita edición**, **No sirve**, más un
cuadro de notas. Atajos: `1` `2` `3` evalúan, `0` limpia, `j`/`k` navegan. Los candidatos sin renderizar se reproducen
directamente desde el video original en su tramo (sin subtítulos). La casilla «ocultar modelo y puntuación» sirve para
evaluar a ciegas, sin que el modelo o el número te influyan.

Las evaluaciones se guardan al instante en `data\reviews\<huella-del-video>.ratings.json` (un JSON legible), junto con
el inicio y fin del momento, el job, el modelo, la puntuación y la versión del algoritmo (commit y parámetros).

## 4. Comparar versiones futuras con tus evaluaciones

```powershell
.\.venv\Scripts\python.exe -m streamclipper reviews <job>
```

Muestra, para un job (de cualquier versión del algoritmo, p. ej. uno re-puntuado con `rescore`): cuántos de sus clips
elegidos caen sobre momentos que marcaste bueno / edición / no sirve, y qué momentos buenos quedaron fuera (y en qué
puesto del ranking aparecen). Los momentos se identifican por solape temporal (IoU ≥ 0.5), no por el job, así que las
evaluaciones se reutilizan. Si el mismo momento recibió evaluaciones distintas (p. ej. de base y de small), se informa
como discrepancia.

## 5. Después de revisar

- `export <job> --ids 5 7` renderiza candidatos alternativos concretos (número `#` de la lista).
- Con tus evaluaciones decidimos qué cambiar (léxico, pesos, umbrales) y se prueba con `rescore` + `reviews`.

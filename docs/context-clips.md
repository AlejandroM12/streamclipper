# Clips con contexto narrativo (1.4)

Hasta 1.3 el recorte automático se ajustaba a la *reacción*: un clip típico duraba 8–20 s y el espectador no sabía qué había
pasado antes ni cómo terminó. Desde 1.4 el clip se **construye** alrededor del momento detectado.

## Dos etapas separadas

| Etapa | Qué hace | Dónde |
|---|---|---|
| **A. Detección** | Elige el momento (ranking Clásico o Smart, sin cambios). | `selection/` (`scorer`, `candidates`) |
| **B. Construcción** | Decide qué fragmento necesita el espectador: dónde empieza la situación y cuándo termina. | `selection/context_clip.py` |

El ranking no cambia: los candidatos, las puntuaciones y los umbrales son los de siempre. Solo cambia el intervalo que se renderiza.

## Cómo se decide el inicio y el final

Es una heurística local, sencilla y medible. **No entiende la escena**: sigue la continuidad de lo que se dice y se oye.

1. **Evento**: la ráfaga de reacción (la ventana corta que ya calculaba `refine_window`).
2. **Inicio**: desde el evento hacia atrás se encadenan las frases separadas por pausas menores de 2,5 s (misma conversación o
   jugada). Una pausa de 2,5 s o más es el comienzo natural de la situación. Tope: 30 s antes del evento.
3. **Final**: igual hacia adelante, y se espera a que el audio vuelva a la calma (menos de +1 dB sobre lo habitual durante 1,5 s)
   para no cortar mientras la reacción sigue. Tope: 30 s después.
4. **Presupuesto del preset**: el preset orienta cuánto contexto cabe (reparto aproximado 55 % antes / 45 % después).
5. **Nunca a mitad de frase**; los límites siempre caen en el borde de una frase (se valida con `check_boundaries`).
6. **Sin relleno**: si la situación está completa en 24 s (empieza y termina en pausas naturales), el clip dura 24 s aunque el preset
   pida 30–60 s. No se agrega silencio ni contenido sin relación.
7. **Excepcionalmente hasta 90 s**: solo si ambos extremos son límites naturales y la conversación es densa (≥ 1,2 palabras/s).

### Evidencia usada y no usada

Usa: marcas de tiempo por palabra de Whisper, segmentación en frases, pausas, energía de audio y continuidad de la conversación.
**No usa** análisis visual (no hay infraestructura reutilizable barata para eso): el motivo del recorte nunca se atribuye a lo que se ve.

### Expansión de respaldo (estimación)

Si no hay evidencia (sin habla antes o después y audio activo) se aplica una expansión conservadora (4–12 s antes, 3–7 s después según
el preset) y el clip queda marcado como **estimación** con confianza **baja**. Si alargar obligaría a cortar una frase, se vuelve al
evento sin ampliar.

### Qué se guarda por clip (`context` en `result.json`)

`original` (intervalo del candidato, nunca se modifica), `event`, `contextual` (propuesto), `final` (renderizado), `start_kind` /
`end_kind` (`pausa`, `material`, `limite`, `calma`, `estimacion`, `sinhabla`), `start_reason` / `end_reason`, `confidence`
(`alta` / `media` / `baja`), `estimated`, `exceptional`, `notes`. Las sesiones anteriores no tienen este campo y se muestran igual que antes.

## Duración (Nuevo → «Duración de los clips»)

| Preset | Orientación | Notas |
|---|---|---|
| Cortos | ≈ 15–30 s | |
| **TikTok / Reels** (por defecto) | ≈ 30–60 s | excepcional hasta 90 s |
| Extendidos | ≈ 45–90 s | |

Orientan los límites; **no fuerzan** una duración. Los clips nuevos de la app usan TikTok / Reels. La configuración interna
(`Settings.clip_preset`) sigue en `legacy` (comportamiento 1.3) para el CLI y las sesiones antiguas.

## Transmisión completa (bloques de 20 min)

* Las marcas son **absolutas del VOD**. El contexto de un clip se reúne de la transcripción de todos los bloques que lo tocan
  (`merge_transcripts`: cada frase pertenece al bloque cuyo núcleo contiene su punto medio, sin duplicados en la zona de 30 s compartida).
* No se vuelve a transcribir ni se descarga el VOD completo: la fase final descarga solo el rango del clip con ±40 s de margen.
* La deduplicación global sigue igual y, **antes del render**, `resolve_overlaps` descarta el clip de menor puntuación cuando dos
  intervalos ya ampliados se pisan o quedan a menos de 3 s, y rellena desde la reserva.
* Los subtítulos se generan sobre todo el intervalo contextual con el desfase correcto del corte parcial (alineación medida ≤ 100 ms).

## Ampliar con contexto (clips existentes)

En la galería de cada clip: **Ampliar con contexto**.

1. Se abre el modal y la propuesta se calcula sola, con TikTok / Reels preseleccionado (si cambiás la duración, se recalcula).
2. Ves «Original 10 s → propuesto 31 s», el motivo del inicio y del final, la confianza y **Ver la propuesta** para previsualizarla.
3. **Generar nuevo clip**: cuando el trabajo se acepta el modal se cierra y un panel de progreso (abajo a la derecha) muestra las etapas.
4. Al terminar, la versión «Con contexto NNN» queda seleccionada en la galería, lista para reproducir y descargar (MP4 y SRT).

Si algo falla (por ejemplo, la fuente ya no existe) el modal lo explica, con una sugerencia y «Reintentar», y no cambia nada.

* Recupera el intervalo original y el origen. Sesiones de archivo/URL usan su video; los clips de «Transmisión completa» descargan
  solo el rango necesario del VOD (`<sesión>/context/`).
* Crea una **versión nueva** («Con contexto 001») con subtítulos. El clip original y sus versiones no se modifican ni se borran.
* Si la fuente ya no está disponible (archivo movido, VOD borrado) lo dice con un mensaje claro y no cambia nada.
* No repite el análisis de 8 h: solo transcribe el rango necesario.

## Señales visuales (1.5)

Los límites del contexto también se apoyan en transiciones visuales (cambios de pregunta/pantalla) cuando las hay; ver [smart-editor.md](smart-editor.md).

## Limitaciones conocidas

* No hay comprensión semántica ni visual de la escena; una pausa larga no siempre es el fin de una jugada.
* Con Whisper en inglés sin puntuación, muchas frases se consideran «dependientes» y el inicio puede quedar en el comienzo de la reacción
  (`confianza media`); el motivo se indica en «Por qué este recorte».
* Un clip puede seguir siendo corto (15–20 s) cuando la situación empieza y termina enseguida; está permitido y se explica.
* Los clips de «Transmisión completa» tardan más en generarse (descargan ≈100 s en alta calidad por clip).
* Cualquier clip debe revisarse antes de publicarlo.

## Resultados reales

Material real, directorios temporales, datos del usuario sin modificar (el hash del clip original no cambió).

**Sesión GTA (archivo, español)** — duración antes → ahora (preset TikTok / Reels), con el contexto verificado visualmente
(planteamiento → reacción → desenlace, subtítulos sincronizados):

| Clip | Antes | Ahora | Inicio | Fin |
|---|---|---|---|---|
| #1 | 16,8 s | 49,5 s | tope de contexto | pausa natural de 3,7 s |
| #2 | 15,0 s | 31,8 s | pausa natural de 7,0 s | tope de contexto |
| #3 | 8,8 s | 37,0 s | tope de contexto | pausa natural de 8,4 s |

**Ampliar con contexto (clip de un VOD, desde el navegador)**: clip de 16,75 s → versión «Con contexto 001» de 31,25 s (402,0–433,25 s del
stream), 18 subtítulos dentro del clip. El clip original conserva el mismo SHA-256.

**Transmisión completa (Kick, 3 bloques, 10 clips pedidos)**: duraciones 15,8–44,2 s (mediana 34,5 s; 5 de 10 entre 30 y 60 s, 5 por debajo de 30 s
porque la situación estaba completa o no había más contexto continuo). Sin solapes entre clips (distancia mínima 7,8 s). Confianza: 3 alta,
7 media, 0 baja. Render de los 10 clips: 13 min (≈100 s de descarga HQ por clip), pico de disco 1,4 GB.

**Suite**: 588 pruebas pasan (2 omitidas), lint limpio.

# Transmisión completa (automático) — StreamClipper 1.3

Pegás la URL de una transmisión de varias horas (YouTube, Twitch o Kick) y StreamClipper la analiza **entera**, bloque a bloque,
compara los momentos de toda la transmisión y genera solo los mejores 10, 15 o 20 clips. Sin elegir tramos y sin descargar el VOD
completo.

## 1. Los tres modos de *Nuevo*

| | Video corto | Seleccionar tramos | Transmisión completa (automático) |
|---|---|---|---|
| Para | videos y clips de hasta 20 min, archivos locales | partes concretas de un VOD de Twitch/Kick | transmisiones largas, sin elegir nada |
| Qué analiza | todo el video | hasta 5 tramos de ≤ 20 min que elegís | **toda** la duración, en bloques de 20 min |
| Clips | los mejores del video (cantidad configurable) | los mejores de **cada** tramo | los mejores **globales** (objetivo 10/15/20) |
| Resultado | una sesión | una sesión por tramo | una galería global; cada clip es una sesión |
| Proveedores | YouTube, Twitch, Kick, archivo | Twitch y Kick (VOD) | YouTube, Twitch y Kick |

La pantalla elige sola «Transmisión completa» cuando el video dura más de 20 min (podés cambiarlo). El detector **Clásico** sigue
siendo el predeterminado; Smart Highlights es opcional.

## 2. Cómo funciona

1. **Plan.** La duración se divide en bloques *núcleo* contiguos de 20 min, sin huecos (8 h = 24 bloques; un último bloque corto
   queda como bloque propio y, si dura menos de 60 s, se absorbe en el anterior). No hay un límite de bloques. Cada bloque se
   descarga con **30 s de contexto a cada lado**, así un momento que cruza un límite se analiza completo.
2. **Análisis en serie** (un único trabajo pesado, un bloque a la vez): descarga solo ese rango (yt-dlp resuelve el flujo y FFmpeg
   corta con `-ss/-t`) **en 480p**, verifica con FFprobe, extrae el audio, transcribe con Whisper, aplica el detector y guarda
   la sesión del bloque (solo análisis: no renderiza). **El video del bloque se borra al terminar**, aunque falle o se cancele.
3. **Selección global.** Cuando todos los bloques están listos se juntan los candidatos de toda la transmisión con tiempos
   **absolutos**. Un candidato pertenece al bloque cuyo núcleo contiene su ancla, así las zonas de contexto solapadas nunca
   generan duplicados; además se deduplica globalmente (solapamiento > 15 % o anclas a menos de 20 s: gana el mejor). Se
   ordenan por puntuación y se toman los mejores del objetivo. **No se toman 3 por bloque.**
4. **Render diferido.** Solo los clips elegidos se renderizan: cada uno se **re-descarga en alta calidad** (su ventana ± 6 s) y se
   procesa con el pipeline existente reutilizando la transcripción y el candidato de su bloque (no se retranscribe ni se detecta de
   nuevo). Cada clip final es una sesión normal: se puede reproducir, valorar, editar y exportar.

**Por qué se re-descarga en vez de guardar los bloques:** conservar 8 h (≈ 24 bloques) ocuparía decenas de GB; los bloques de
análisis ocupan ≈ 210 MB cada uno (480p) y se borran. Re-descargar ≈ 30 s por clip final cuesta ≈ 33 s y permite **reintentar
renderizados sin retranscribir**. La contrapartida: **el VOD tiene que seguir disponible** al renderizar (Kick y Twitch eliminan
los VOD con el tiempo). Las sesiones ya existentes no dependen de esto y siguen funcionando sin conexión.

**Segundos extra por keyframes:** el corte copia flujos, así que arranca en el keyframe/segmento HLS anterior. Se midió comparando
dos descargas del mismo rango: desfase de 0 ms (2 de 3 pruebas) y 100 ms (1 de 3), sobre una resolución de 10 ms. Cada recorte
final tiene ± 6 s de margen y se re-mide con FFprobe; la duración real queda registrada (`measured_cut_s`).

## 3. Umbrales: recomendados y exploratorios

- **Recomendado:** cumple los criterios normales de calidad del detector elegido (puntuación ≥ `min_score` y señal propia).
- **Exploratorio:** el mejor material que **no** alcanza el umbral. Se identifica con una insignia y un aviso; **no se baja el
  umbral en silencio ni se lo presenta como excelente.**
- Si hay menos recomendados que el objetivo, el trabajo lo informa («Solo N momentos cumplen… hay M exploratorios disponibles»).
  Podés marcar *generar también exploratorios* al empezar, o generarlos después con **Generar 5 clips exploratorios** en la
  galería global (sin volver a analizar). Si no hay más candidatos, la transmisión no tiene más momentos que destaquen.
- Vigilancia de concentración: como máximo 3 clips en una ventana de 5 min, **salvo** que el candidato alcance ≥ 95 % del mejor
  puntaje global (no se descarta lo claramente superior). Lo que queda fuera se conserva como reserva.
- **Calibración entre bloques:** la puntuación no se transforma. Ya es relativa al nivel de audio habitual de cada bloque
  (percentil 75) y el resto de señales usan pesos fijos en [0, 1], así que los bloques son comparables. La app guarda la
  distribución de cada bloque (`result.scales`) para detectar anomalías en lugar de ocultarlas.

## 4. Progreso, cancelación y recuperación

La tarjeta del trabajo muestra, con datos reales: bloque actual (p. ej. «bloque 7 de 24» y su rango del stream), etapa
(descarga, transcripción, análisis, render), % de bloques completados, candidatos encontrados, clips finales seleccionados,
avisos y errores. El estado se guarda en `data/app/jobs/` y las sesiones de cada bloque en `data/jobs/`.

- **Cancelar** detiene el bloque en curso (FFmpeg se mata) y conserva lo terminado.
- **Cierre brusco o reinicio:** los bloques completados se conservan; el interrumpido se rehace; al abrir, el trabajo figura
  *Interrumpido* y **Continuar** retoma desde el primer bloque pendiente sin repetir transcripciones ni duplicar candidatos. Si el
  bloque interrumpido ya se había descargado completo, se reutiliza. La selección queda guardada: un reintento del render no
  vuelve a elegir y solo genera los clips que faltan.
- **Un bloque que falla** no frena a los demás, pero **no se eligen clips con la transmisión incompleta**: se reintentan solo los
  bloques fallidos.
- Errores contemplados: VOD eliminado/privado/de suscriptores, URL expirada, directo no finalizado, duración no confiable,
  bloque sin audio, descarga parcial fallida (nunca se cae a descargar el VOD entero), falta de espacio (se comprueba antes de
  cada bloque), corte de internet, fallo de Whisper, diferencia entre duración pedida y real.

## 5. Tiempo y espacio (medido en esta PC: Ryzen 5 5600G, 16 GB, sin GPU)

Estimaciones que muestra la app antes de empezar: **tiempo ≈ 0,25 × la duración + 35 s por clip** (rango hasta × 1,5) y
**disco ≈ 0,4 GB temporales + ≈ 0,06 GB por clip final**.

| Transmisión | Bloques | Tiempo estimado | Disco temporal |
|---|---|---|---|
| 1 h | 3 | 0,4–0,5 h (10 clips) | ≈ 0,4 GB |
| 4 h | 12 | 1,2–1,8 h | ≈ 0,4 GB + clips |
| 8 h | 24 | 2,1–3,2 h (15 clips) | ≈ 0,4 GB + ≈ 0,9 GB |
| 11,9 h | 36 | 3,5–6 h | ≈ 0,4 GB + clips |

## 6. Usarlo (sin comandos)

1. Doble clic en `iniciar_streamclipper.bat`.
2. **Nuevo** → pegá la URL de la transmisión. Se muestran plataforma, título y duración; elige «Transmisión completa (automático)»
   y muestra bloques, tiempo y disco estimados.
3. Elegí el objetivo (10/15/20), el detector (Clásico por defecto), el idioma y, si querés, *exploratorios*. **Procesar**.
4. Seguí el progreso en **Trabajos**. Podés **Cancelar** y **Continuar** (también tras cerrar la aplicación).
5. Al terminar, **Ver la galería global**: ordená por puntuación o por momento del stream, filtrá recomendados/exploratorios,
   reproducí, valorá, editá y exportá (MP4 + SRT). *Sesiones* muestra la transmisión como una sola tarjeta (con *Eliminar
   transmisión*); las sesiones de análisis de los bloques no aparecen sueltas.

## 7. Qué se probó y qué no

**Probado de verdad**
- **Desde la interfaz real (Chrome con clics y escritura), con un VOD público de Kick de 61 min** (3 bloques de 20 min, Whisper
  small, inglés, Clásico, objetivo 10): plan de bloques, análisis, **cierre brusco** (`taskkill /T`) durante el bloque 2 con el
  bloque 1 ya terminado, reinicio con el `.bat`, **Continuar**, selección global, render de 10 clips (1080×1920 con audio),
  galería global (ordenar, valorar, exportar).
  - Por bloque: descarga ≈ 210 MB en 74–80 s; análisis 282–316 s (≈ 0,23–0,26 × la duración); **RAM pico 1,0–1,3 GB**.
  - Total: 144 candidatos (104 tras quedarse con los de su bloque y los 40 mejores), 42 duplicados evitados, 10 recomendados.
    Render de los 10 clips: 320–400 s. **Pico de disco en `samples`: 360 MB.** Sin FFmpeg ni archivos temporales sobrantes tras el
    cierre brusco.
- **Ejecución controlada sobre un VOD de 11,9 h** (Kick): planificó 36 bloques y descargó de verdad un bloque a 11,3 h dentro
  del VOD (217 MB en 55 s), sin descargar el resto.
- Un video de YouTube de 49 min: corte parcial real de 30 s a los 25:00 (H.264 + AAC).
- Pruebas automáticas (566 en total): planificación (8 h = 24 bloques, último bloque parcial, cola absorbida), conversión de tiempos,
  zonas de solape, deduplicación global, ranking entre bloques, umbrales, exploratorios, render diferido, cancelación,
  recuperación tras reinicio, reintentos, compatibilidad con sesiones antiguas, proveedores, interfaz con clics reales, y un
  flujo completo con FFmpeg y Whisper reales sobre un «VOD» con dos momentos conocidos (los encuentra en bloques distintos).

**No probado (pendiente)**
- **Una transmisión de 8 h completa**: no se procesó entera. El tiempo de 8 h es una extrapolación de lo medido (≈ 2,1–3,2 h).
- El modo automático **completo** con Twitch y con YouTube sobre la red (solo se probaron la vista previa, el corte parcial y
  pruebas automáticas con proveedores simulados).
- Candidatos **exploratorios** en una corrida real: en el VOD probado todos los candidatos cumplieron el umbral; se cubren con
  pruebas automáticas.
- Reintento de render cuando el VOD ya fue eliminado de la plataforma, y reinicio real de Windows (solo cierre brusco del proceso).

## 8. Limitaciones

- **Whisper en CPU domina el tiempo**: una transmisión de 8 h son horas de proceso con la computadora encendida.
- **El VOD debe seguir disponible** hasta que se rendericen los clips.
- **El umbral no discriminó en la transmisión probada**: 142 de 144 candidatos lo superaron, así que «recomendados» equivale casi a
  «los mejores por puntuación». No se cambió el umbral (hacerlo sin valoraciones sería arbitrario); sí queda registrado para
  decidirlo con evidencia. Los clips siguen requiriendo revisión humana: la puntuación es un ranking heurístico.
- El análisis usa 480p (suficiente para audio y fotogramas diminutos); los clips finales se renderizan desde una re-descarga en
  alta calidad (hasta 1080p según lo disponible).
- La descarga de YouTube por rangos es más lenta (≈ 22 s para 30 s en la prueba).

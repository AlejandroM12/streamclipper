# Smart Editor (1.5)

Convierte StreamClipper en una herramienta de edición de clips para TikTok, Reels y Shorts, sin salir del procesamiento local
(sin GPU, sin APIs de pago). Cuatro piezas: composición vertical, editor manual, límites narrativos con evidencia visual y
regeneración masiva.

## 1. Composición vertical por regiones

El video horizontal ocupaba una franja chica del 1080×1920 y el resto era fondo desenfocado. Ahora se pueden declarar **regiones**
del video original (fracciones `[x, y, ancho, alto]`) y FFmpeg las recorta, las escala al ancho del lienzo (lanczos: el texto sigue
nítido) y las apila dentro de la zona segura de la plataforma. Los subtítulos van debajo, en su recuadro.

| Preset | Qué hace | Regiones |
|---|---|---|
| Original | Cuadro completo sobre fondo desenfocado (como siempre) | ninguna |
| Gameplay | Prioriza el juego: recorta a la región del juego; sin región, el recorte horizontal de siempre | `main` |
| Reacciones | Cámara(s) arriba, lado a lado, y contenido principal abajo | 1–3 `cams` + `main` |
| Podcast / conversación | Participantes en pila (rejilla 2×2 con 3–4) y, opcional, contenido al final | 1–4 `cams` (+ `main`) |
| Personalizado | Hasta 3 filas de hasta 2 regiones | `rows` |

Decisiones:

* **No hay detección automática de cámaras ni caras.** Una región solo se usa si la persona la declaró (con el selector sobre un
  fotograma). Detectar «dónde está la cámara» con métodos ligeros no es fiable (cada stream arma su pantalla distinto; en el material de
  prueba el diseño incluso cambió entre dos partes del mismo stream). Se priorizó una herramienta manual que funciona.
* **Sin evidencia suficiente, vuelve a Original y lo dice.** «Reacciones» sin cámara o sin contenido, «podcast» sin participantes, etc.
  usan Original y dejan un aviso en la versión («Composición: …»). Nunca se inventa un recorte.
* **No se recorta más de lo declarado**: cada región se muestra completa, conservando su proporción. Si el bloque no cabe en la zona
  segura, todo se reduce proporcionalmente (se informa el porcentaje).
* Se mantienen 1080×1920, el audio original (sin tocar), los subtítulos, la zona segura y la calidad de exportación (`crf`, `preset`).
* Código: `render/compose.py`. Se guarda en la versión (`meta.json`: `composition`, `framing`).

## 2. Editor manual («Editar clip»)

En la galería de cada clip. Abre un editor con:

* El video, y una **línea de tiempo** con la forma de onda de audio, los límites de las frases de la transcripción (franja inferior),
  el **momento principal** detectado (rojo), el intervalo del **clip original** (gris), la **propuesta de contexto** (borde azul), el
  intervalo actual (verde, con asas arrastrables) y las **transiciones visuales** (líneas violetas).
* Inicio y final numéricos, «+5 s de contexto anterior/posterior», «Ajustar a frases», «Usar la propuesta de contexto», marcar inicio/final
  en el instante actual, «Ver el tramo», imantado opcional a los límites de frase y aviso si el intervalo deja fuera el momento principal.
* **Tiempos:** los del clip/editor (segundos del video del editor) están separados de la **hora absoluta del VOD** (se muestra aparte).
* **Composición:** preset, regiones dibujadas arrastrando sobre un fotograma y **previsualización 9:16** (con el recuadro de subtítulos).
* Subtítulos editables (ya existía) y «Restaurar valores originales».
* «Regenerar como versión nueva»: el modal se cierra al aceptarse el trabajo y un panel muestra el progreso; la versión nueva queda
  seleccionada. Las versiones anteriores y el original no se modifican.

El eje del editor es el video **más amplio disponible**: el archivo completo (videos cortos/tramos) o el tramo del VOD ya recuperado por
«Ampliar con contexto» (`<sesión>/context/`). Para ver más allá de un clip del modo automático hay que pulsar antes «Ampliar con
contexto»: recupera **solo** el rango necesario del VOD (clip ± 40 s); nunca se descarga el VOD completo. Las transcripciones se
reutilizan (no se retranscribe).

## 3. Límites narrativos con evidencia visual

Además de pausas, frases y calma del audio (1.4), ahora hay una señal visual ligera (`analysis/transitions.py`):

* Se decodifican fotogramas de 160×90 en gris a 2 por segundo y se mide cuánto cambia la imagen entre muestras.
* Una **transición** es un cambio ≥ 6 veces la mediana de ese mismo video (y ≥ 0,03). Sin OCR, sin modelos, sin dependencias nuevas.
* Si la mediana es alta (la imagen cambia siempre, p. ej. gameplay con mucho movimiento) la señal se descarta (`noisy`): no inventa límites.
* **Uso:** una transición que queda a **más de 4 s** del evento, antes o después, marca donde empieza/termina otra situación (otra
  pregunta): el contexto no la cruza. Más cerca se trata como parte de la misma situación (la pantalla cambia al reaccionar o al
  mostrar el resultado) para no recortar el desenlace. El momento principal nunca se toca.
* **Evidencia vs. estimación:** cada límite queda registrado como *respaldado* (pausa natural, borde del material, calma de audio,
  transición visual) o *estimado* (tope de contexto, expansión de respaldo, sin habla). Se guardan además las señales usadas.
* Coste medido: ≈ 4 s de CPU por cada 90 s de video.
* No se modificó el ranking global de candidatos.
* Alcance: la señal se usa en «Ampliar con contexto», en el editor («Usar la propuesta») y en la regeneración masiva. La generación
  inicial del modo automático sigue usando pausas y audio.

## 4. Regeneración masiva

En la galería de una sesión o de una transmisión completa: casilla **Seleccionar** en cada clip (o «Seleccionar todos») y
**Regenerar seleccionados…**:

1. Elegís si ampliar con contexto, la duración (Cortos / TikTok-Reels / Extendidos) y la composición (sin cambios, Original o la última
   usada en el editor).
2. **Ver propuestas:** se calcula el intervalo de cada clip, de a uno, con progreso individual y general (cancelable).
3. Revisás la tabla (duración antes → después, confianza, evidencia) y desmarcás los que no quieras.
4. **Regenerar:** el modal se cierra y un panel muestra el avance clip por clip. Cada clip es una versión nueva.

Procesamiento secuencial (un clip a la vez), estado persistido en el trabajo, cancelación y **reintento solo de los fallidos**. Un clip que
falla no detiene a los demás. No se repite el análisis del VOD: solo se recupera el tramo de cada clip. Los originales se conservan.

⚠ **Las mismas regiones se aplican a todos los clips**: sirven solo si el stream mantiene el mismo diseño de pantalla. En el material de
prueba el diseño cambió entre dos partes del mismo stream y las regiones de una no servían para la otra. Revisá cada resultado.

## Resultados reales

Material: dos clips de una transmisión de trivia de fútbol (Kick, VOD de varias horas) y un clip de gameplay (GTA). Todos los MP4
generados están en `data/exports/1.5-review/` (ver `LEEME.md`). Todos: 1080×1920, audio y video de igual duración, subtítulos dentro del
clip, validación OK.

**Trivia, clip que mezclaba dos preguntas** (el que ampliaste a 42,3 s):

| Variante | Duración | Qué se ve |
|---|---|---|
| A. Antes (1.4) | 42,3 s | Mezcla la pregunta anterior con la actual; contenido chico, mucho fondo desenfocado |
| B. Límites nuevos, composición Original | 23,5 s | Empieza en 33,5 s: el panel cambió a los 31,0 s (transición visual) → solo la pregunta del momento |
| C. Límites nuevos + Reacciones | 23,5 s | Cámaras arriba, tarjeta de la pregunta legible abajo |
| D. Ajuste manual (+8 s al final, imantado a frase) | 31,9 s | Igual que C con más desenlace |
| E. Podcast (solo participantes) | 23,5 s | Ambos participantes grandes (la pregunta no se muestra) |

**Segunda sesión de trivia** (diseño de pantalla distinto): la transición a los 33,0 s coincide con el inicio ya elegido por la pausa
(8 s antes del evento): corrobora, no cambia el resultado.

**Gameplay (GTA):** Original / Gameplay (recorte horizontal) / Reacciones (cámara grande arriba + juego). Con regiones mal elegidas el
juego queda más chico y recortado: para gameplay con cámara pequeña conviene Original o Gameplay con la región del juego bien definida.

**Rendimiento** (Ryzen 5 5600G, sin GPU, x264 `veryfast`): render de 23,5 s de clip ≈ 35 s; de 31,9 s ≈ 48 s (≈ 1,5× tiempo real).
Pico medido de memoria: ffmpeg ≈ 411 MB y el proceso de render ≈ 70 MB. Un clip a la vez en el trabajo masivo.

## Limitaciones

* Las regiones son manuales y por stream/diseño; no hay detección automática de cámaras.
* La composición apila regiones: con un contenido muy alto (p. ej. un panel vertical) el bloque se reduce para caber en la zona segura.
* La señal visual mide *cuánto cambia la imagen*, no entiende qué pasó: en pantallas con mucha animación (tragamonedas) puede haber
  transiciones «falsas»; por eso se exige distancia al evento y se descarta si la imagen cambia siempre.
* El selector de regiones usa un fotograma: si el diseño cambia dentro del clip, hay que mirar otro instante.
* Un clip nunca es «publicable» por su duración, resolución o puntuación: hay que verlo.

## Cómo probar cada función

1. **Composición / editor:** galería → «Editar clip» → elegí «Reacciones», seleccioná «Cámara 1» y arrastrá sobre el fotograma; repetí
   con «Contenido principal»; mirá la previsualización; ajustá el intervalo en la línea de tiempo; «Regenerar como versión nueva».
2. **Límites con evidencia:** «Ampliar con contexto» en un clip de trivia/presentación; en el editor se ven las líneas violetas.
3. **Masivo:** marcá «Seleccionar» en varios clips (o «Seleccionar todos») → «Regenerar seleccionados…» → «Ver propuestas» → «Regenerar».

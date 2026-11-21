# Aplicación local (MVP 1.0)

Interfaz web que corre **solo en tu equipo** (`127.0.0.1`), sin cuentas ni base de datos. Usa el mismo pipeline, la misma
revisión y los mismos presets que la línea de comandos; no hay nada simulado.

## Requisitos

- Windows 11, Python 3.11 o superior (con «Add python.exe to PATH»).
- Unos 2 GB libres de RAM para Whisper `small` (el modelo `base` usa menos) y espacio en disco.
- Internet solo para: instalar, descargar el modelo la primera vez y bajar videos de YouTube.

## Instalar y abrir

1. Doble clic en `instalar.bat` (una sola vez): crea el entorno en `backend\.venv`, instala dependencias y descarga FFmpeg
   portable en `tools\ffmpeg`. No toca el PATH ni instala nada global. Termina con un chequeo (`streamclipper doctor`).
2. Doble clic en `iniciar_streamclipper.bat`: arranca el servidor y abre el navegador en `http://127.0.0.1:8780/`
   (si el puerto está ocupado usa el siguiente). Cerrá la ventana negra para salir.

Opciones del `.bat`: `--no-open` (no abre el navegador), `--port 8781`.

## Uso

1. **Nuevo**: pegá una URL (YouTube, VOD o clip de Twitch, VOD o clip de Kick) o elegí un MP4 (botón *Examinar…* o un video de `data\samples`). Los valores
   recomendados ya están puestos: Whisper `small`, español, 3 clips, presentación segura (blur-fit).
2. **Procesar**: aparece un trabajo con sus etapas. El porcentaje solo se muestra donde es real (descarga, transcripción y
   render); en el resto se ve el estado de la etapa. Se puede **cancelar** y **reintentar**. Solo corre un trabajo pesado a
   la vez; los demás esperan en la cola.
3. **Abrir clips**: galería con reproductor, transcripción y confianza, motivos del ranking y avisos. Evaluá cada clip
   (*Bueno para publicar* / *Necesita edición* / *No sirve*): se guarda en `data\reviews`.
4. **Editar**: cambiá inicio y final, el texto/tiempos de los subtítulos y la presentación, y pulsá *Regenerar este clip*.
   No se retranscribe: se reutiliza la transcripción de la sesión. Se crea una versión nueva en
   `data\jobs\<sesión>\edits\<clip>\vNNN`; el original y las versiones anteriores no se tocan.
5. **Exportar**: marcá *Exportar* en las versiones que quieras (MP4 1080×1920 + SRT) y pulsá el botón: se copian a
   `data\exports\<sesión>\` con nombres descriptivos. También hay descarga directa de cada MP4 y SRT.
6. **Sesiones**: historial de todo lo procesado (también por la CLI). Si cerrás la aplicación en medio de un trabajo, al
   volver aparece como *Interrumpido* y se puede reintentar.

## VOD de Twitch y Kick por tramos

Un VOD suele durar horas, así que no se descarga entero: se eligen **uno o varios tramos** y solo esos se descargan y procesan.

1. En *Nuevo*, pegá la URL del VOD (`twitch.tv/videos/<id>` o `kick.com/<canal>/videos/<id>`). Se muestran plataforma,
   título y duración total, y aparece el editor de **Tramos a procesar**.
2. Cargá cada tramo como `HH:MM:SS` → `HH:MM:SS` (por ejemplo `01:25:00` → `01:40:00`). Con *+ Agregar tramo* y ✕ se agregan y
   quitan. La página muestra la duración de cada tramo y el tiempo total seleccionado, y no deja procesar si hay errores.
3. Reglas (se validan también en el servidor): hasta **5 tramos** por VOD, de **10 s a 20 min** cada uno, dentro de la duración
   del VOD, sin superponerse (que uno termine donde empieza el siguiente es válido). *Usar todo el VOD* aparece si dura ≤ 20 min.
4. **Procesar**: es **un solo trabajo** que recorre los tramos **uno detrás de otro** (un único proceso pesado a la vez).
   Cada tramo muestra su propio progreso (descarga, transcripción, render…) y termina en **su propia sesión** con sus clips,
   valoración, edición y exportación de siempre. Valores recomendados: Whisper `small`, español, blur-fit, 3 clips por tramo.
5. **Ver los clips de todos los tramos** abre una vista agrupada del VOD (`#/vod/<id>`); en *Sesiones* los tramos de un VOD
   aparecen juntos en una sola tarjeta. Cada sesión guarda un `segment.json` con la URL original, el VOD y el rango
   (inicio/fin), de modo que la relación VOD → tramo → clips no depende de la interfaz.
6. **Cancelar** detiene el tramo en curso (FFmpeg se mata en menos de un segundo) y conserva los tramos terminados. Si se
   cierra la aplicación, al volver el trabajo figura *Interrumpido* y **Continuar con los tramos pendientes y fallidos** no
   repite los tramos ya terminados. Un tramo que falla no frena a los demás; el botón reintenta solo los fallidos.
7. No se admiten dos trabajos idénticos (mismo VOD, mismos tramos y misma configuración) a la vez.

**Cómo se descarga un tramo:** yt-dlp resuelve el flujo HLS del VOD (sin cuentas ni cookies) y FFmpeg descarga *solo ese rango*
(`-ss <inicio> -t <duración>` sobre la entrada, copiando los flujos sin recodificar). No existe una descarga previa del VOD
completo y, si el corte falla, se informa el error: nunca se recurre en silencio a descargar el VOD entero. Como la copia de
flujos arranca en el segmento/keyframe de HLS, **el tramo puede incluir unos segundos adicionales**: en las pruebas reales
fueron de 0,05 a 0,13 s sobre 40 s (Twitch 720p, Kick 1080p60), pero en un VOD con keyframes más separados pueden ser algunos
segundos. La duración real se mide con FFprobe (que además exige video y audio y rechaza duraciones muy distintas de la
pedida) y queda en `data\samples\<archivo>.source.json` (`measured_duration_s`, `extra_seconds`). Los tramos se guardan como
`data\samples\<plataforma>-vod-<id>_<HHMMSS>-<HHMMSS>.mp4`.

## Detección de momentos: Clásica o Smart Highlights

En *Nuevo* se elige el modo de **Detección de momentos**. **Clásica** (por defecto) es el ranking original y la referencia.
**Smart Highlights** (experimental) suma audio contextual, contexto de la transcripción y, opcionalmente, imagen, y resta ruido,
música y cortes pobres. **Todavía no está demostrado que supere a la clásica**: ver [smart-highlights.md](smart-highlights.md).
Un video ya transcrito (mismo modelo e idioma) **no se retranscribe**. En la galería, *Probar con Smart Highlights* procesa el mismo
video con el otro detector para poder compararlos y valorar ambos.

## Seguridad y límites

- Escucha solo en loopback; se rechaza cualquier `Host` que no sea local y las escrituras exigen JSON.
- URLs: la plataforma se detecta por el dominio exacto (YouTube, Twitch o Kick) y antes de procesar se muestra título,
  duración y plataforma. Solo videos públicos de un único video, VOD o clip: se rechazan listas, colecciones, canales,
  directos y dominios parecidos (`twitch.tv.evil.com`, etc.). Límites: 20 min (por video, clip o tramo), 1500 MB y 1080p. No se piden ni guardan
  cuentas, contraseñas ni cookies: el contenido de suscriptores o que exige iniciar sesión se informa y se rechaza.
  Usá contenido propio o con autorización.
- **Los VOD de Twitch y Kick casi siempre duran horas** (en las pruebas, el más corto medía 64 min): se procesan por tramos
  (ver arriba). Los clips (hasta ~60 s) y YouTube se descargan enteros como siempre.
- La puntuación de los clips es un ranking heurístico: **nada se considera publicable sin revisión humana**.
- Edición: mínimo 3 s y máximo 60 s por clip. No es un editor de video completo.

## Problemas frecuentes

`streamclipper doctor` (o la pestaña *Estado del sistema*) indica qué falta. Con poca RAM libre, cerrá el navegador/IDE o
usá el modelo `base`.

Si al abrir la aplicación aparece *«Esta ventana y el servidor de StreamClipper son de versiones distintas»*, hay una ventana negra anterior
abierta: cerrá todas y volvé a abrir iniciar_streamclipper.bat. Si el puerto 8780 está ocupado, la ventana nueva lo avisa y usa el
siguiente (el navegador se abre en la dirección correcta).


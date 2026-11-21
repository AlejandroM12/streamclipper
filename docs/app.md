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

1. **Nuevo**: pegá una URL de YouTube o elegí un MP4 (botón *Examinar…* o un video de `data\samples`). Los valores
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

## Seguridad y límites

- Escucha solo en loopback; se rechaza cualquier `Host` que no sea local y las escrituras exigen JSON.
- YouTube: solo URL públicas de video (sin listas ni directos), hasta 20 min y 1500 MB. Usá contenido propio o autorizado.
- La puntuación de los clips es un ranking heurístico: **nada se considera publicable sin revisión humana**.
- Edición: mínimo 3 s y máximo 60 s por clip. No es un editor de video completo.

## Problemas frecuentes

`streamclipper doctor` (o la pestaña *Estado del sistema*) indica qué falta. Con poca RAM libre, cerrá el navegador/IDE o
usá el modelo `base`.

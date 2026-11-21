# Subtítulos, límites del clip y presentación vertical

## Principios

- **Nunca se inventan ni se reemplazan palabras.** El texto es el que transcribió Whisper. Las palabras de baja confianza
  se **marcan** (aviso en el informe y, si lo pedís, color en el video), no se corrigen.
- **El audio no se toca.** Un subtítulo aparece cuando empieza su primera palabra (según los timestamps por palabra) y solo
  se ajusta cuánto tiempo permanece en pantalla.
- Hay un `.srt` editable junto a cada MP4 para corregir a mano lo que haga falta.

## Cómo se arman los subtítulos (`render/subtitles.py`)

1. Se toman las palabras del clip con sus tiempos. Una palabra no puede «durar» más de 1.2 s (Whisper a veces estira una
   palabra sobre una pausa y el subtítulo quedaría pegado).
2. Se separan frases por puntuación final (`. ? ! …`) y por pausas (> `sub_phrase_gap`).
3. Dentro de cada frase se elige el corte óptimo en subtítulos (programación dinámica) respetando los límites y
   prefiriendo: cortar tras comas, no terminar con artículos/preposiciones/conjunciones, evitar palabras huérfanas y no
   partir pausas largas.
4. Cada subtítulo se reparte en hasta `sub_max_lines` líneas parejas de hasta `sub_line_chars` caracteres (por defecto se deriva
   del tamaño de fuente y los márgenes seguros: ≈ 25 con fuente 68). El salto de línea es explícito, no automático.
5. Tiempos: el subtítulo permanece `sub_tail_hold` tras su última palabra; si el siguiente empieza a menos de
   `sub_bridge_gap` se une sin hueco (evita parpadeos); se garantiza una lectura mínima de `sub_min_duration` s y
   `sub_max_cps` caracteres/s cuando hay espacio antes del siguiente. Un subtítulo inviable (< 0.7 s) se une al vecino aunque
   supere el máximo de palabras en hasta 3.

| Parámetro (Settings / CLI) | Defecto | Qué controla |
|---|---|---|
| `sub_max_words` / `--sub-max-words` | 7 | palabras por subtítulo |
| `sub_line_chars` / `--sub-line-chars` | derivado (25) | caracteres por línea |
| `sub_max_lines` / `--sub-max-lines` | 2 | líneas por subtítulo |
| `sub_min_duration` / `--sub-min-duration` | 1.0 s | lectura mínima |
| `sub_max_cps` / `--sub-max-cps` | 17 | caracteres/s cómodos |
| `sub_bridge_gap`, `sub_tail_hold`, `sub_phrase_gap` | 0.6 / 0.3 / 0.7 s | huecos, cola y pausa de frase |
| `sub_font_size` / `--sub-font-size` | 68 | tamaño de fuente |
| `sub_low_prob`, `sub_mark_low_conf` / `--mark-low-conf` | 0.4, no | baja confianza y su coloreado (naranja) |

**Límite honesto:** un subtítulo no puede ir más lento que el habla. En habla muy rápida (como la del locutor de partida1,
≈ 20–27 caracteres/s mientras habla, medido) una parte de los subtítulos superará los 17 cps aunque estén bien armados. Los
informes muestran el **ritmo del habla** del clip para distinguir esto de un defecto de los subtítulos.

## Ajuste de inicio y final (`selection/refine.py`)

Es un paso **separado de la selección**: se aplica después de elegir el momento y no cambia candidatos, puntuaciones ni
ranking. Reglas (configurables con `refine_*` / `--refine-*` / `--no-refine`):

- **Reacción:** se detecta a partir del pico de audio: la ráfaga de frases encadenadas (pausas ≤ 1.5 s) cuyo volumen alcanza el
  umbral (pico − 6 dB), porque el volumen cae entre palabra y palabra. El inicio de la reacción nunca se recorta; el final se
  limita a 8 s tras el pico (`refine_max_post`) para no arrastrar una tirada larga.
- **Fin:** la reacción termina cuando el volumen cae 6 dB bajo su pico y se sostiene 1 s (o al terminar la ráfaga, lo que ocurra después); el clip termina al final de la
  frase que contiene ese instante, con 0.6 s de margen, y conserva al menos 2.5 s después del momento principal. Si la
  frase quedó incompleta (termina en artículo/preposición/conjunción, o continúa sin pausa) se completa con la siguiente
  (hasta 4 s más allá de la ventana original).
- **Inicio:** empieza en la frase que contiene el momento, con 1 s de margen. El contexto previo se incluye **solo** si esa
  frase depende de la anterior (empieza con «y», «pero», «porque»… o continúa una oración partida sin pausa); una
  exclamación corta no arrastra la frase previa.
- Los cortes se desplazan hasta ±0.35 s al instante más silencioso que no caiga dentro de una frase.
- Se respetan `min_clip` y `max_clip`; si hay que alargar para llegar al mínimo, se apunta al instante exacto (sin
  pasarse) y se prefiere el contexto previo.
- Si el resultado cortara una frase o palabra, se conserva la ventana original y se informa.

## Presentación vertical (`render/layout.py`, `render/vertical.py`)

1080×1920, H.264/AAC. Presets (`layout`, `--layout`):

| Preset | Qué hace | Cuándo |
|---|---|---|
| `blur-fit` (defecto) | cuadro completo, sin recortar, sobre fondo desenfocado | seguro con cualquier fuente |
| `focus` | acerca el gameplay recortando los costados (`--fg-zoom`, `--fg-x`) | gameplay 16:9; con `--fg-x` se deja la facecam a la vista |
| `fill`, `crop` | cubre todo el lienzo recortando | material ya vertical o recorte manual |

- El gameplay y los subtítulos se colocan dentro de la **zona segura**; los subtítulos van **justo debajo del gameplay** (no
  sobre él ni en la franja inferior que cubren los botones y la descripción).
- Márgenes seguros aproximados (px; `--platform`): TikTok arriba 170 / abajo 480; Reels 220 / 400; Shorts 160 / 380; a los
  lados 60 / 120. `universal` toma el máximo de cada lado. **Son aproximados**: las plataformas no publican una cifra única y
  cambian su interfaz.
- Fondo: `--bg-blur` (radio, 0 = sin desenfoque) y `--bg-dim` (oscurecimiento 0–0.5) para que se lean los subtítulos.
- `focus` recorta los costados: con la facecam a la izquierda (`--fg-x 0.1`) se pierde parte del lado derecho (menús, HUD y
  textos propios del juego pueden quedar cortados).
- La facecam no está siempre en el mismo lugar: `focus` no la detecta, se le indica la posición con `--fg-x` (0 = izquierda,
  1 = derecha). No hay logos ni marcas de agua.

## Comparar con la versión anterior

```powershell
streamclipper rerender <job> --tag stage3 --layout focus --fg-x 0.1
```

Genera `out_stage3/` (MP4, `.ass`, `.srt`) sin tocar `out/`, y `rerender_stage3.md` con la comparación por clip: duración, inicio y
final, cobertura y velocidad de subtítulos, avisos, tiempo de render y memoria. `streamclipper review <sesión>` muestra el
original y la nueva versión juntos para evaluarlas. **Menos avisos no significa un clip mejor.**

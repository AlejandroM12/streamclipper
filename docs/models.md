# Modelos Whisper: cuál usar y cuánto cuestan

**Recomendado para gameplay en español: `small`.** En la sesión `partida1` (GTA V, 10 min, español) produjo transcripciones
considerablemente mejores que `base`. `base` sigue disponible para pruebas rápidas y para comparar
(`--model base`, `session`, `compare`).

`process` usa `small` por defecto si el idioma es español o no se indica; usa `base` para otros idiomas, donde todavía no
hay mediciones propias (`streamclipper.core.config.recommended_model`). La configuración por defecto de la API
(`Settings`) sigue siendo `base`.

## Calidad medida (partida1, 607 s, español)

| | base | small |
|---|---|---|
| Palabras con probabilidad < 0.4 | 30.9 % | 13.8 % |
| Oraciones de baja confianza según Whisper | 53 de 217 | 0 de 221 |

Con `base` aparecieron palabras inventadas o deformadas en los clips («ayolos», «mocholos», «habrunes»); `small` también
se equivoca (p. ej. «Experiría consolas»), menos. Ninguno de los dos es perfecto: por eso el informe marca los subtítulos de
baja confianza y no los corrige.

## Costo medido (CPU Ryzen 5 5600G, `int8`, 4 hilos, sin GPU)

Mediciones reales de los reportes de cada ejecución (pico de memoria aproximado: Python + FFmpeg, muestreo cada 0.5 s):

| Video | Modelo | Transcripción | Factor vs. duración | Total del job | Pico de RAM aprox. |
|---|---|---|---|---|---|
| GTA V, 607 s, es | base | 122.5 s | 0.20× | 193.6 s | 538 MB |
| GTA V, 607 s, es | small | 183.6 s | 0.30× | 273.6 s | 1028 MB |
| Vlog de pesca, 1657 s, es | base | 60.7 s | 0.037× | 164.6 s | 713 MB |
| Vlog de pesca, 1657 s, es | small | 180.4 s | 0.109× | 288.9 s | 964 MB |
| Sintético (voz SAPI), 105 s, en | base | 4.6 s | 0.044× | 34.7 s | 512 MB |
| Sintético (voz SAPI), 105 s, en | small | 11.3 s | 0.107× | 48.8 s | 567 MB |

Cómo leerlo:

- **Tiempo:** `base` tardó entre 0.04× y 0.20× la duración del video y `small` entre 0.11× y 0.30×; `small` fue de 1.5× a
  3× más lento que `base` en el mismo video. El factor depende mucho de cuánto se habla y de la carga de la máquina
  (partida1 tiene habla casi continua y se corrió con poca RAM libre); no conviene extrapolar de un video a otro.
- **Memoria:** `base` ≈ 0.5–0.7 GB y `small` ≈ 1.0 GB de pico. Con esta máquina (16 GB, a menudo con 0.5–1 GB libres por
  Chrome/IDE) conviene cerrar programas antes de correr `small`.
- **Render:** el render de los clips es independiente del modelo (≈ 20–40 s por clip de 15–25 s con origen 1080p).
- Costo de una sesión `base,small` completa de partida1: unos 8 minutos (193.6 s + 273.6 s) más las pasadas de render.

Los números son de una sola máquina y de pocos videos; son una orientación, no una garantía.

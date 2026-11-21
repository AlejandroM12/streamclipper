# Comparación: ranking clásico vs Smart Highlights

Misma transcripción y mismo audio para ambos; sin renderizar. Las valoraciones humanas disponibles son pocas y de clips que el clásico ya había elegido (sesgo de selección): ver límites en `docs/smart-highlights.md`.

| Sesión | Video | Split | Modo | Clips | Duplicados | Bordes (err/avisos) | Habla en clip | Valorados (good/edit/bad) | Tiempo | RAM pico |
|---|---|---|---|---|---|---|---|---|---|---|
| `21-170328-small-app-be5a` | youtube-4hxvaBEpgBM.mp | dev | classic | 3 | 0 | 0/3 | 0.84 | 3 (2/1/0) | 0.01 s | 39 MB |
| `21-170328-small-app-be5a` | youtube-4hxvaBEpgBM.mp | dev | smart | 3 | 0 | 0/2 | 0.72 | 2 (1/1/0) | 1.67 s | 117 MB |
| `0144-small-partida1-3717` | youtube-4hxvaBEpgBM.mp | dev | classic | 3 | 0 | 0/2 | 0.79 | 2 (1/1/0) | 0.0 s | 40 MB |
| `0144-small-partida1-3717` | youtube-4hxvaBEpgBM.mp | dev | smart | 3 | 0 | 0/2 | 0.73 | 2 (1/1/0) | 1.17 s | 77 MB |
| `15831-base-partida1-429b` | youtube-4hxvaBEpgBM.mp | dev | classic | 3 | 0 | 0/0 | 0.62 | 2 (1/1/0) | 0.0 s | 40 MB |
| `15831-base-partida1-429b` | youtube-4hxvaBEpgBM.mp | dev | smart | 1 | 0 | 0/0 | 0.63 | 1 (1/0/0) | 2.25 s | 171 MB |
| `mall-gaming-real-01-bb73` | youtube-V98MCYWGxE0.mp | dev | classic | 3 | 0 | 0/1 | 0.69 | 0 (0/0/0) | 0.02 s | 41 MB |
| `mall-gaming-real-01-bb73` | youtube-V98MCYWGxE0.mp | dev | smart | 3 | 0 | 0/3 | 0.63 | 0 (0/0/0) | 15.25 s | 176 MB |
| `base-gaming-real-01-59a1` | youtube-V98MCYWGxE0.mp | dev | classic | 3 | 0 | 0/6 | 0.67 | 0 (0/0/0) | 0.02 s | 41 MB |
| `base-gaming-real-01-59a1` | youtube-V98MCYWGxE0.mp | dev | smart | 3 | 0 | 0/5 | 0.65 | 0 (0/0/0) | 14.84 s | 179 MB |
| `-025453-base-ytwire-ecba` | youtube-aqz-KE-bpKQ.mp | test | classic | 3 | 0 | 0/0 | 0.0 | 0 (0/0/0) | 0.0 s | 42 MB |
| `-025453-base-ytwire-ecba` | youtube-aqz-KE-bpKQ.mp | test | smart | 3 | 0 | 0/0 | 0.0 | 0 (0/0/0) | 5.39 s | 119 MB |
| `-014110-small-rerun-bd80` | synthetic.mp4 | test | classic | 2 | 0 | 0/2 | 0.44 | 0 (0/0/0) | 0.0 s | 41 MB |
| `-014110-small-rerun-bd80` | synthetic.mp4 | test | smart | 2 | 0 | 0/2 | 0.36 | 0 (0/0/0) | 2.17 s | 167 MB |
| `1-014034-base-rerun-158c` | synthetic.mp4 | test | classic | 2 | 0 | 0/0 | 0.28 | 0 (0/0/0) | 0.01 s | 42 MB |
| `1-014034-base-rerun-158c` | synthetic.mp4 | test | smart | 2 | 0 | 0/0 | 0.29 | 0 (0/0/0) | 2.09 s | 168 MB |
| `261121-013743-small-902a` | synthetic.mp4 | test | classic | 2 | 0 | 0/2 | 0.4 | 0 (0/0/0) | 0.0 s | 41 MB |
| `261121-013743-small-902a` | synthetic.mp4 | test | smart | 2 | 0 | 0/2 | 0.36 | 0 (0/0/0) | 2.02 s | 163 MB |
| `0261121-013708-base-1b89` | synthetic.mp4 | test | classic | 2 | 0 | 0/0 | 0.28 | 0 (0/0/0) | 0.02 s | 41 MB |
| `0261121-013708-base-1b89` | synthetic.mp4 | test | smart | 2 | 0 | 0/0 | 0.29 | 0 (0/0/0) | 2.16 s | 168 MB |

## Clips elegidos y por qué

### `20261121-170328-small-app-be5a` (youtube-4hxvaBEpgBM.mp4, 607 s) — coincidencia de top: 2/3
- **classic** (clásico; duplicados evitados: 0)
  - 140.2–156.0 s, puntuación 0.528, valoración humana: edit; señales: reaccion_verbal 0.50, frases 1.00, contexto 1.00
  - 402.1–423.8 s, puntuación 0.533, valoración humana: good; señales: intensidad_audio 0.43, contexto 1.00, calidad_transcripcion 0.76
  - 532.3–552.6 s, puntuación 0.586, valoración humana: good; señales: reaccion_verbal 0.50, frases 1.00, contexto 1.00
- **smart** (smart; duplicados evitados: 0)
  - 140.2–156.0 s, puntuación 0.421, valoración humana: edit; señales: reaccion_verbal 0.50, calidad_transcripcion 0.77, frases 1.00
  - 322.6–338.3 s, puntuación 0.453, valoración humana: sin valorar; señales: reaccion_verbal 0.50, calidad_transcripcion 1.00, energia_contexto 0.62
  - 402.1–423.8 s, puntuación 0.443, valoración humana: good; señales: intensidad_audio 0.43, calidad_transcripcion 0.76, energia_contexto 0.54

### `20261121-120144-small-partida1-3717` (youtube-4hxvaBEpgBM.mp4, 607 s) — coincidencia de top: 2/3
- **classic** (clásico; duplicados evitados: 0)
  - 108.5–126.6 s, puntuación 0.487, valoración humana: sin valorar; señales: reaccion_verbal 0.50, contexto 1.00, calidad_transcripcion 0.76
  - 140.2–156.0 s, puntuación 0.531, valoración humana: edit; señales: reaccion_verbal 0.50, frases 1.00, contexto 1.00
  - 409.8–432.2 s, puntuación 0.572, valoración humana: good; señales: intensidad_audio 0.43, frases 1.00, contexto 1.00
- **smart** (smart; duplicados evitados: 0)
  - 140.2–156.0 s, puntuación 0.421, valoración humana: edit; señales: reaccion_verbal 0.50, calidad_transcripcion 0.80, frases 1.00
  - 366.1–387.0 s, puntuación 0.389, valoración humana: sin valorar; señales: reaccion_verbal 0.50, calidad_transcripcion 0.69, energia_contexto 0.50
  - 409.8–432.2 s, puntuación 0.487, valoración humana: good; señales: intensidad_audio 0.43, calidad_transcripcion 0.81, frases 1.00

### `20261121-115831-base-partida1-429b` (youtube-4hxvaBEpgBM.mp4, 607 s) — coincidencia de top: 1/3
- **classic** (clásico; duplicados evitados: 0)
  - 140.6–157.2 s, puntuación 0.45, valoración humana: edit; señales: reaccion_verbal 0.50, frases 1.00, contexto 1.00
  - 193.2–208.9 s, puntuación 0.456, valoración humana: sin valorar; señales: reaccion_verbal 0.50, contexto 1.00, calidad_transcripcion 0.76
  - 409.0–425.0 s, puntuación 0.533, valoración humana: good; señales: intensidad_audio 0.43, frases 1.00, contexto 1.00
- **smart** (smart; duplicados evitados: 0)
  - 409.0–425.0 s, puntuación 0.365, valoración humana: good; señales: intensidad_audio 0.43, frases 1.00, energia_contexto 0.54

### `20261121-030652-small-gaming-real-01-bb73` (youtube-V98MCYWGxE0.mp4, 1657 s) — coincidencia de top: 1/3
- **classic** (clásico; duplicados evitados: 0)
  - 21.8–38.0 s, puntuación 0.513, valoración humana: sin valorar; señales: intensidad_audio 0.58, contexto 1.00, calidad_transcripcion 0.92
  - 621.0–636.7 s, puntuación 0.511, valoración humana: sin valorar; señales: intensidad_audio 0.63, dinamica_audio 1.00, contexto 1.00
  - 1490.0–1506.6 s, puntuación 0.539, valoración humana: sin valorar; señales: intensidad_audio 0.56, dinamica_audio 1.00, contexto 1.00
- **smart** (smart; duplicados evitados: 0)
  - 514.0–529.6 s, puntuación 0.488, valoración humana: sin valorar; señales: intensidad_audio 0.64, reaccion_vocal 0.83, energia_contexto 1.00
  - 1031.2–1055.3 s, puntuación 0.526, valoración humana: sin valorar; señales: energia_contexto 0.93, reaccion_verbal 0.50, calidad_transcripcion 0.90
  - 1484.8–1501.4 s, puntuación 0.509, valoración humana: sin valorar; señales: energia_contexto 1.00, intensidad_audio 0.56, reaccion_vocal 0.72

### `20261121-030408-base-gaming-real-01-59a1` (youtube-V98MCYWGxE0.mp4, 1657 s) — coincidencia de top: 1/3
- **classic** (clásico; duplicados evitados: 0)
  - 679.9–697.5 s, puntuación 0.501, valoración humana: sin valorar; señales: intensidad_audio 0.57, dinamica_audio 1.00, contexto 1.00
  - 1129.1–1145.1 s, puntuación 0.541, valoración humana: sin valorar; señales: reaccion_verbal 0.50, dinamica_audio 1.00, contexto 1.00
  - 1484.8–1507.2 s, puntuación 0.511, valoración humana: sin valorar; señales: intensidad_audio 0.56, dinamica_audio 1.00, contexto 1.00
- **smart** (smart; duplicados evitados: 0)
  - 579.2–594.9 s, puntuación 0.487, valoración humana: sin valorar; señales: energia_contexto 1.00, intensidad_audio 0.57, reaccion_vocal 0.74
  - 1030.9–1048.4 s, puntuación 0.507, valoración humana: sin valorar; señales: energia_contexto 0.93, reaccion_verbal 0.50, calidad_transcripcion 0.71
  - 1484.8–1507.2 s, puntuación 0.5, valoración humana: sin valorar; señales: energia_contexto 1.00, intensidad_audio 0.56, reaccion_vocal 0.72

### `20261121-025453-base-ytwire-ecba` (youtube-aqz-KE-bpKQ.mp4, 635 s) — coincidencia de top: 2/3
- **classic** (clásico; duplicados evitados: 0)
  - 194.2–209.9 s, puntuación 0.539, valoración humana: sin valorar; señales: intensidad_audio 1.00, dinamica_audio 1.00, contexto 1.00
  - 344.8–360.4 s, puntuación 0.539, valoración humana: sin valorar; señales: intensidad_audio 1.00, dinamica_audio 1.00, contexto 1.00
  - 417.2–432.9 s, puntuación 0.496, valoración humana: sin valorar; señales: intensidad_audio 0.86, dinamica_audio 1.00, contexto 1.00
- **smart** (smart; faltó: contexto de transcripción (no hay palabras transcritas); duplicados evitados: 1)
  - 194.2–209.9 s, puntuación 0.455, valoración humana: sin valorar; señales: intensidad_audio 1.00, energia_contexto 1.00, dinamica_audio 1.00
  - 335.5–351.1 s, puntuación 0.408, valoración humana: sin valorar; señales: intensidad_audio 0.85, energia_contexto 1.00, dinamica_audio 1.00
  - 417.2–432.9 s, puntuación 0.475, valoración humana: sin valorar; señales: intensidad_audio 0.86, energia_contexto 1.00, dinamica_audio 1.00

### `20261121-014110-small-rerun-bd80` (synthetic.mp4, 105 s) — coincidencia de top: 2/2
- **classic** (clásico; duplicados evitados: 0)
  - 26.2–44.3 s, puntuación 0.745, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_verbal 0.50, dinamica_audio 1.00
  - 72.7–94.1 s, puntuación 0.868, valoración humana: sin valorar; señales: intensidad_audio 0.89, reaccion_verbal 1.00, dinamica_audio 1.00
- **smart** (smart; duplicados evitados: 2)
  - 27.8–44.3 s, puntuación 0.698, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_vocal 1.00, energia_contexto 1.00
  - 71.2–86.9 s, puntuación 0.75, valoración humana: sin valorar; señales: reaccion_verbal 1.00, intensidad_audio 0.91, reaccion_vocal 1.00

### `20261121-014034-base-rerun-158c` (synthetic.mp4, 105 s) — coincidencia de top: 2/2
- **classic** (clásico; duplicados evitados: 0)
  - 26.2–44.3 s, puntuación 0.745, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_verbal 0.50, dinamica_audio 1.00
  - 69.7–85.3 s, puntuación 0.734, valoración humana: sin valorar; señales: intensidad_audio 0.90, reaccion_verbal 0.50, dinamica_audio 1.00
- **smart** (smart; duplicados evitados: 2)
  - 27.8–44.3 s, puntuación 0.698, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_vocal 1.00, energia_contexto 1.00
  - 71.2–86.9 s, puntuación 0.62, valoración humana: sin valorar; señales: intensidad_audio 0.91, reaccion_vocal 1.00, energia_contexto 1.00

### `20261121-013743-small-902a` (synthetic.mp4, 105 s) — coincidencia de top: 2/2
- **classic** (clásico; duplicados evitados: 0)
  - 26.2–44.3 s, puntuación 0.745, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_verbal 0.50, dinamica_audio 1.00
  - 72.7–89.4 s, puntuación 0.86, valoración humana: sin valorar; señales: intensidad_audio 0.89, reaccion_verbal 1.00, dinamica_audio 1.00
- **smart** (smart; duplicados evitados: 2)
  - 27.8–44.3 s, puntuación 0.698, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_vocal 1.00, energia_contexto 1.00
  - 71.2–86.9 s, puntuación 0.75, valoración humana: sin valorar; señales: reaccion_verbal 1.00, intensidad_audio 0.91, reaccion_vocal 1.00

### `20261121-013708-base-1b89` (synthetic.mp4, 105 s) — coincidencia de top: 2/2
- **classic** (clásico; duplicados evitados: 0)
  - 26.2–44.3 s, puntuación 0.745, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_verbal 0.50, dinamica_audio 1.00
  - 69.7–85.3 s, puntuación 0.734, valoración humana: sin valorar; señales: intensidad_audio 0.90, reaccion_verbal 0.50, dinamica_audio 1.00
- **smart** (smart; duplicados evitados: 2)
  - 27.8–44.3 s, puntuación 0.698, valoración humana: sin valorar; señales: intensidad_audio 0.92, reaccion_vocal 1.00, energia_contexto 1.00
  - 71.2–86.9 s, puntuación 0.62, valoración humana: sin valorar; señales: intensidad_audio 0.91, reaccion_vocal 1.00, energia_contexto 1.00

## Valoraciones humanas guardadas

NO CONCLUYENTE: valoraciones de Smart = 0 (0 video/s), clásico = 4 (1 video/s). Se necesitan ≥ 10 clips valorados y ≥ 3 videos por modo. No se puede afirmar que Smart Highlights mejore ni que empeore.

Ajuste automático DESHABILITADO: al menos 5 videos con ≥ 10 clips valorados, ≥ 3 «good» y ≥ 3 «bad» (hay 0); valoraciones también de clips NO elegidos por el detector (para medir falsos negativos).

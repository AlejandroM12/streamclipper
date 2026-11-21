# Léxico de reacciones y evidencia de confianza

La señal verbal de la puntuación (`reaccion_verbal`) sale de un léxico es/en y de reglas de contexto. Código:
`analysis/lexicon.py` (datos y reglas) y `analysis/text.py` (evaluación). Todo es configurable.

## Cómo suma una frase

| Tipo | Ejemplos | Aporte | Condición |
|---|---|---|---|
| **fuerte** | `oh my god`, `no puede ser`, `increíble`, `mierda`… | 1.0 | confianza de Whisper suficiente |
| **leve** | `wow`, `loco`, `brutal`, `clutch`… | 0.5 | ídem |
| **ambigua** | `vamos` | 1.0 / 0.5 / 0 según evidencia | ver abajo |

`reacción de la frase = min(1, suma de aportes / 2)`. Cada término suma una vez por frase. Para un candidato se toma
la **frase de mayor reacción dentro de ±4 s del ancla** (no se suman frases distintas).

## Confianza (aplica a todos los términos)

- Un término cuyas palabras tienen confianza media de Whisper **< `min_word_prob` (0.4)** se descarta, con la razón.
- Si Whisper marca la frase como **posible no-voz** (`no_speech_prob > 0.6` **y** `avg_logprob < -1.0`, la regla de
  silencio de Whisper) no cuenta ningún término. Con solo una de las dos condiciones no se infiere nada.
- Sin probabilidades (transcripciones sin marcas por palabra) la confianza se trata como desconocida: no se descarta.

## Palabras ambiguas: «vamos»

«Vamos» es una reacción en «¡VAMOOOS!» y una perífrasis cotidiana en «vamos a revisar las cañas». Se decide con
puntos de evidencia:

| Señal | Puntos | Cómo se detecta |
|---|---|---|
| alargada | 2 | `vamooos` (3 o más «o») — Whisper a veces conserva el alargamiento |
| repetición confiable | 2 | «vamos» repetido, con confianza media ≥ `repeat_min_prob` (0.6) y sin marca de transcripción repetitiva |
| volumen | 2 | exceso de volumen alrededor de la palabra ≥ `audio_peak_db` (6 dB sobre el nivel habitual del video) |
| exclamación | 1 | `!` o `¡` en la frase |
| frase corta | 1 | la frase tiene ≤ 3 palabras y no es «vamos a …» |

- **≥ 4 puntos → reacción fuerte (1.0)**; **2–3 → leve (0.5)**; **< 2 → no cuenta** («sin evidencia de énfasis»).
- La exclamación y la frase corta **nunca bastan solas**: Whisper omite signos con frecuencia, y una sola señal débil
  suma 1 punto.
- **Uso cotidiano:** si «vamos» va seguido de `a, al, para, pa, con, por, hacia, en, de, hasta, tras, sobre`, no cuenta,
  salvo que esté alargada o reúna ≥ 4 puntos (en cuyo caso se limita a leve).
- Sin curva de audio no hay señal de volumen (se indica en la razón).

| Frase | Resultado |
|---|---|
| «Vamos a revisar las cañas», «Vamos a empezar» | no cuenta (uso cotidiano) |
| «¡Vamos, vamos!» (confiable) | fuerte |
| «vamos vamos» (sin signos) | leve |
| «vamooos» | leve; fuerte si además hay volumen |
| «vamos» solo, tono neutro | no cuenta |
| «vamos» ×5 con confianza media 0.51 | no cuenta (bucle de baja confianza) |

## Repeticiones y `frases`

`frases = min(1, (exclamaciones + 0.5·preguntas + repeticiones confiables) / 3)`, tomando la frase de mayor valor del
clip. Una repetición de la misma palabra solo cuenta si su **confianza media ≥ `repeat_min_prob`** y Whisper no marcó
la frase como repetitiva (`compression_ratio > 2.4`). Un grito repetido y bien transcrito («¡no, no, no!») sigue
sumando; un bucle de baja confianza se descarta y queda registrado.

## Explicaciones

Cada componente guarda `window` (la ventana temporal exacta) y `evidence`:
`+` suma al valor, `~` es un término aceptado en otra frase de la ventana (no suma: se toma el máximo),
`-` fue descartado, con su razón. Una nota nunca menciona como motivo algo que no suma.

## Personalizar

`--lexicon ruta.json` (también en `rescore`):

```json
{
  "extend_strong": ["que bestia"],
  "extend_mild": ["groso"],
  "remove": ["joder"],
  "disable_ambiguous": ["vamos"],
  "ambiguous": {
    "dale": {"pattern": "dale", "elongated": "dal{3,}e", "mundane_next": ["que", "a"],
             "strong_points": 4, "mild_points": 2}
  }
}
```

`remove` elimina una expresión por su regex exacta; las demás claves agregan o reemplazan reglas. Las expresiones son
regex sobre texto en minúsculas, sin acentos ni puntuación. Claves desconocidas o regex inválidas dan un error claro.
Ajustes numéricos: `--min-word-prob`, `--repeat-min-prob`, `--peak-db`.

## Límites conocidos

- «Mierda», «joder»… cuentan como reacción aunque se usen como muletilla («me quiero ir a la mierda porque…»): no hay
  análisis semántico.
- La señal acústica es el volumen relativo al propio video; en material de poco rango dinámico casi nada la alcanza.
- Las reglas se escribieron con ejemplos en español; el léxico en inglés no tiene todavía reglas ambiguas.

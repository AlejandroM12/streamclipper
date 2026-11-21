# Smart Highlights v1.2

Segundo detector de momentos, **experimental**, que convive con el ranking clásico. **No está demostrado que sea mejor**:
con los datos disponibles la comparación es *no concluyente* (ver «Resultados reales»). Por eso el detector por defecto sigue
siendo el clásico.

## 1. Arquitectura

```
WAV 16 kHz ─► analysis/audio.py (curva dB, existente) ─► analysis/audio_features.py (señales de audio nuevas)
transcript.json ───────────────────────────────────────► analysis/narrative.py     (contexto de transcripción)
video (opcional) ──────────────────────────────────────► analysis/visual.py        (muestreo ligero de fotogramas)
                      selection/heuristic.py (clásico, intacto) ─┐
                                                                 ├─► selection/smart.py (SmartScorer) ─► ranking + selección
                      anclas clásicas + reacciones vocales/risas ┘
evaluation_smart.py: comparación en memoria, valoraciones humanas, separación dev/test, ajuste deshabilitado
```

- `Settings.detector` = `classic` (por defecto) o `smart`; `smart_visual` activa/desactiva la imagen; `reuse_transcript`
  evita retranscribir un video ya transcrito (mismo video, modelo e idioma, con evidencia de Whisper completa).
- `SmartScorer` cumple la interfaz `CandidateScorer` existente (`pipeline.run_pipeline(scorer=…)`). Reutiliza los componentes
  del clásico, así que **el ranking anterior sigue intacto como línea base**. El render, los subtítulos y los presets no se tocan.
- Si **ninguna** señal nueva está disponible, Smart devuelve *exactamente* el ranking clásico (hay una prueba que lo verifica) y
  lo informa (`result.json → detector.effective = "clásico (degradado…)"`). Si falta solo alguna (p. ej. la imagen), los pesos de
  las demás se renormalizan y la galería muestra qué faltó.
- No se inventan puntuaciones de confianza: la puntuación es un **ranking relativo dentro del video**, igual que la clásica.

## 2. Señales

Suman (pesos fijados a priori; suman 1,0 entre las disponibles):

| Señal | Qué mide | Peso |
|---|---|---|
| `intensidad_audio`, `dinamica_audio` | clásicas: pico y subida de volumen cerca del momento | 0,16 / 0,06 |
| `energia_contexto` | dB sobre la mediana de los 30 s previos (sorpresa respecto del contexto) | 0,10 |
| `reaccion_vocal` | pico sonoro y sorprendente, con habla transcrita en el pico y espectro no plano | 0,12 |
| `risa_probable` | ráfagas regulares de amplitud a 3–8 Hz; solo cuenta si la evidencia es ≥ 0,40 | 0,04 |
| `reaccion_verbal`, `frases` | clásicas: léxico de reacción con confianza de Whisper, exclamaciones, repeticiones confiables | 0,18 / 0,06 |
| `arco_narrativo` | planteamiento (pregunta o marcador de anticipación) **y** desenlace posterior | 0,08 |
| `contexto`, `duracion`, `calidad_transcripcion` | clásicas | 0,05 / 0,02 / 0,08 |
| `visual` | cambio visual del momento frente a su contexto y novedad (percentil entre candidatos); solo suma con corroboración de audio o texto, y no suma si el movimiento es constante | 0,05 |

Restan (cada penalización es un componente de peso negativo con su evidencia):

| Penalización | Causa | Peso |
|---|---|---|
| `musica_o_ruido` | pico fuerte con espectro plano, volumen constante, o fuerte sin habla transcrita | 0,20 |
| `audio_repetitivo` | picos a intervalos regulares (disparos, explosiones) | 0,15 |
| `aire_muerto` | poco habla y gran parte del clip por debajo del nivel habitual | 0,20 |
| `transcripcion_dudosa` | banderas de Whisper, palabras poco confiables, frase repetida en bucle | 0,12 |
| `frase_incompleta` | el clip empieza o termina a mitad de oración (una frase con puntuación final cierra) | 0,08 |

Selección: mismo umbral (`min_score`) y reglas del clásico, más **anti-duplicados**: no se eligen dos clips con solapamiento
> 15 %, texto casi igual o anclas a menos de 20 s. Además de las anclas clásicas, Smart agrega anclas por reacción vocal y risa.

**Lo que se descubrió midiendo audio real (y por qué el diseño es así):**
- La proporción de energía en la «banda de voz» **no separa** habla de juego (0,73 frente a 0,63 en partida1), así que no se usa;
  la presencia de voz se decide con las palabras transcritas.
- El análisis visual con fotogramas clave es barato (1–9 s) pero la «novedad» saturaba en ~1,0 para todo (la cámara de un juego
  siempre cambia): se normaliza como percentil entre los candidatos del propio video.
- `risa_probable` **no se activó en ~50 min de audio real** (partida1, vlog de pesca, tramos de Kick y Twitch). Con audio
  sintético sí detecta ráfagas regulares (0,44) y no se activa con habla (0,0). No está validada con risas reales: tiene peso 0,04.

## 3. Cómo se comparó con el clásico

`streamclipper smart-eval` corre **ambos detectores en memoria** sobre la misma transcripción y el mismo audio de las sesiones
existentes (sin renderizar, sin retranscribir) y mide duplicados, bordes de frase, habla en el clip, silencio, tiempo, RAM y la
coincidencia con las valoraciones humanas. Reproducible: `docs/eval/comparison-v1.2.md` y `docs/eval/baseline-v1.2.json`.

**Separación ajuste/evaluación:** no se ajustó nada con valoraciones. Los pesos son priores definidos antes de ver resultados.
Cada video se asigna a `dev` o `test` por su huella (par/impar, regla previa); un futuro ajuste solo podría usar `dev` y toda
afirmación de mejora debe medirse en `test`. El ajuste automático está **deshabilitado** (`smart-eval --tune` explica por qué).

> Transparencia: una corrección de `frase_incompleta` (ya no penaliza frases con puntuación final como «No, no, no.») se hizo
> al inspeccionar por qué Smart descartaba un clip valorado «good». Se corrigió por principio y no cambió el resultado, pero esa
> sesión (partida1) **ya no cuenta como evidencia independiente**.

## 4. Resultados reales y limitaciones

Material: 10 sesiones de 4 videos distintos (gameplay GTA 10 min ×3 sesiones, vlog de pesca 27 min ×2, Big Buck Bunny sin habla,
video sintético ×4). **Valoraciones humanas: 4, de un solo video, todas sobre clips que el clásico ya había elegido.**

| Medida | Clásico | Smart | Lectura |
|---|---|---|---|
| Coincidencia del top-3 | — | partida1: 2/3, 2/3, 1/3; pesca: 1/3, 1/3; Big Buck Bunny: 2/3; sintético: 2/2 | Smart cambia 1–2 de 3 momentos en material real |
| Clips valorados en su top-3 (good/edit/bad) | 3 (2/1/0), 2 (1/1/0), 2 (1/1/0) | 2 (1/1/0), 2 (1/1/0), 1 (1/0/0) | **Smart no recupera un clip «good» que el clásico sí tenía**; sus clips nuevos no tienen valoración |
| Duplicados en el top-3 | 0 | 0 | **la línea base no tenía duplicados: no hay mejora que medir** |
| Errores de bordes de frase | 0 | 0 | igual |
| Avisos de bordes (suma) | 3, 2, 0, 1, 6 | 2, 2, 0, 3, 5 | sin diferencia clara |
| Cantidad de clips | 3 | 3, salvo 1 en partida1 con `base` | Smart devolvió **menos clips** en un caso (umbral calibrado para el clásico) |

Conclusión: **no se puede afirmar que Smart Highlights mejore ni que empeore**. No hay valoraciones suficientes (se necesitan
≥ 10 clips valorados y ≥ 3 videos por modo) y las existentes tienen sesgo de selección. La única evidencia a favor es sobre
**audio sintético con verdad conocida**: Smart penaliza explosiones repetitivas que el clásico puntúa alto y deja la reacción
real por encima con más margen (`tests/test_smart.py`). Eso demuestra que el mecanismo funciona, no que mejore con material real.

**Los falsos negativos no se pueden medir:** una valoración solo existe para clips que alguien vio; los momentos que ningún
detector eligió casi nunca están valorados.

## 5. Consumo (medido)

| | Clásico | Smart |
|---|---|---|
| Tiempo de la etapa de selección (video de 10 min) | ≈ 0,03 s | 1,2–2,3 s |
| Tiempo con imagen sobre un video de 27 min (20 ventanas) | ≈ 0,02 s | ≈ 15 s (≈ 9 s son fotogramas) |
| Segundos de video muestreados | — | 84–240 s de 607–1657 s (tope 900 s) |
| RAM del análisis (monitor en proceso) | ≈ 40 MB | ≈ 77–180 MB (+40–140 MB) |
| Ejecución real desde la app (partida1, transcripción reutilizada) | — | selección 1,2 s; trabajo total 60 s (58 s de render); RAM pico 471 MB |
| Referencia: el trabajo original con Whisper `small` | transcribir 183 s; total 274 s; RAM pico 1028 MB | — |

En esta máquina (Ryzen 5 5600G, 16 GB, sin GPU dedicada) Smart agrega segundos, no minutos; el costo dominante sigue siendo
Whisper y el render. El estado del sistema avisa si quedan menos de 2 GB libres.

## 6. Cómo usarlo

1. **Nuevo → Detección de momentos → Smart Highlights (experimental)**. Opciones avanzadas: desactivar el análisis visual.
2. En la **galería** se muestran el modo, la puntuación y, en «Por qué se eligió este momento», lo que suma y lo que resta con su
   evidencia. Las señales ausentes se informan.
3. **Comparar:** en una sesión, *Probar con Smart Highlights* (o con la clásica) procesa el **mismo video** con el otro detector
   **reutilizando la transcripción** (no retranscribe) y crea una sesión nueva.
4. **Valorar** los clips de ambos modos del mismo video. En *Estado del sistema* aparece la evaluación por detector.
5. Línea de comandos: `streamclipper smart-eval [jobs…] --out informe.md --json datos.json` (compara sin renderizar).

## 7. Próximas mejoras, justificadas por evidencia

- **Más valoraciones, en ambos modos y también de clips no elegidos** (≥ 5 videos con ≥ 10 clips, ≥ 3 «good» y ≥ 3 «bad»): hoy es
  el cuello de botella; sin eso no hay forma honesta de ajustar ni de concluir. El botón *Probar con…* existe para generarlas.
- **Calibrar `min_score` para Smart cuando haya valoraciones**: devolvió 1 clip en vez de 3 en un caso, porque su escala difiere
  de la del clásico y el umbral se heredó sin cambios.
- **Validar o eliminar `risa_probable`**: 0 activaciones en audio real; necesita ejemplos etiquetados de risa.
- **Revisar el peso visual (0,05)**: 3 de los 5 primeros candidatos de partida1 mostraban «movimiento constante», así que casi
  nunca suma; solo se justifica con evidencia de que ayuda.
- **Medir CPU, no solo tiempo**, y la memoria de pico exacta del proceso (hoy es un muestreo cada 0,5 s).

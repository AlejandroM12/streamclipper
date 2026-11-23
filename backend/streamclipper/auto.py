"""Transmisión completa (automático): planificación en bloques, tiempos absolutos y selección global de highlights.

Esquema (ver docs/auto-vod.md):
1. `plan_blocks`: la duración total se cubre con bloques «núcleo» contiguos de hasta BLOCK_S segundos, sin huecos. Cada bloque se
   descarga con un margen de contexto a cada lado (MARGIN_S) para no perder momentos que cruzan un límite.
2. Cada bloque se analiza (descarga parcial, Whisper, detector) SIN renderizar y se guarda como una sesión de solo análisis.
3. `pool_from_block`: un candidato pertenece al bloque cuyo NÚCLEO contiene su ancla; así los márgenes solapados nunca generan
   candidatos duplicados. Los tiempos pasan a ser absolutos respecto del VOD.
4. `select_global`: se comparan los candidatos de TODA la transmisión, se deduplica globalmente y se separan «recomendados»
   (criterios normales de calidad) de «exploratorios» (mejores del resto, de menor confianza, que el usuario decide si genera).
5. Solo los clips elegidos se renderizan, a partir de una re-descarga corta de cada uno (ver runners.py).

La puntuación no se transforma entre bloques: ya es relativa al nivel habitual de audio de cada bloque (percentil 75) y el resto de
las señales usan pesos fijos en [0, 1], así que los bloques son comparables sin recalibrar. `block_scales` reporta la distribución
de cada bloque para detectar anomalías en vez de ocultarlas.
"""
import math
from dataclasses import asdict, dataclass

from streamclipper.core.errors import InvalidInputError
from streamclipper.media.download import format_hms
from streamclipper.models import Segment, Transcript, Word

BLOCK_S = 1200            # duración del núcleo de cada bloque (20 min)
MARGIN_S = 30             # contexto descargado a cada lado de un bloque
MIN_TAIL_S = 60           # un último bloque más corto que esto se absorbe en el anterior
MIN_TOTAL_S = 60          # duración mínima para el modo automático
ANALYSIS_HEIGHT = 480     # el análisis no necesita 1080p: ahorra descarga y disco
TARGETS = (10, 15, 20)
KEEP_PER_BLOCK = 40       # candidatos que se conservan por bloque para la selección global
ANCHOR_GAP_S = 20.0       # dos clips con anclas más cerca son el mismo momento
MAX_OVERLAP = 0.15
CROWD_WINDOW_S = 300.0    # ventana para vigilar la concentración de clips
CROWD_MAX = 3             # clips por ventana antes de exigir superioridad clara
CROWD_EXCEPTION = 0.95    # un candidato con ≥ 95 % del mejor puntaje global siempre pasa (no se descarta lo claramente superior)
EXPLORATORY_CAP = 60
ANALYZE_RTF = 0.25        # medido en esta PC (61 min de Kick): descarga + Whisper small + análisis de un bloque ≈ 0.25 × su duración
RENDER_PER_CLIP_S = 35    # medido: re-descarga en alta calidad + render de un clip final ≈ 33 s
CUT_PAD_S = 6.0           # margen de la re-descarga de cada clip final (keyframes, fades y refinado de límites)


@dataclass(frozen=True)
class Block:
    index: int
    start: int            # núcleo, segundos absolutos del VOD
    end: int
    dl_start: int         # rango realmente descargado (con contexto)
    dl_end: int
    last: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def plan_blocks(duration: float, block_s: int = BLOCK_S, margin_s: int = MARGIN_S) -> list[Block]:
    """Cubre [0, duración) con bloques núcleo contiguos; el último bloque muy corto se absorbe en el anterior."""
    d = int(duration)
    if d < MIN_TOTAL_S:
        raise InvalidInputError(f"La transmisión dura {format_hms(d)}: es demasiado corta para el modo automático "
                                f"(mínimo {MIN_TOTAL_S} s). Usá «Video corto».")
    cores = [[s, min(s + block_s, d)] for s in range(0, d, block_s)]
    if len(cores) > 1 and cores[-1][1] - cores[-1][0] < MIN_TAIL_S:
        tail = cores.pop()
        cores[-1][1] = tail[1]                     # (el índice se evalúa DESPUÉS del pop)
    return [Block(i, a, b, max(0, a - margin_s), min(d, b + margin_s), i == len(cores) - 1) for i, (a, b) in enumerate(cores)]


def max_download_s(block_s: int = BLOCK_S, margin_s: int = MARGIN_S) -> int:
    """Duración máxima de una descarga de bloque (núcleo + tail absorbido + márgenes)."""
    return block_s + MIN_TAIL_S + 2 * margin_s


def estimates(duration: float, target: int = 15, block_s: int = BLOCK_S, margin_s: int = MARGIN_S) -> dict:
    """Estimaciones para mostrar ANTES de empezar, según lo medido en esta PC (Whisper small); es una guía, no una promesa."""
    blocks = plan_blocks(duration, block_s, margin_s)
    dl_total = sum(b.dl_end - b.dl_start for b in blocks)
    low_h = (duration * ANALYZE_RTF + target * RENDER_PER_CLIP_S) / 3600
    return {"blocks": len(blocks), "block_minutes": block_s // 60, "analyze_seconds": int(duration), "download_seconds": dl_total,
            "est_hours_low": round(low_h, 1), "est_hours_high": round(low_h * 1.5, 1), "analyze_rtf": ANALYZE_RTF,
            "disk_peak_gb": round((max_download_s(block_s, margin_s) * 2e6 / 8 + 45e6) / 1e9, 2),     # un bloque a 480p + su audio
            "final_clips_gb": round(target * 0.06, 2), "analysis_height": ANALYSIS_HEIGHT}


def _owns(block: Block, anchor: float) -> bool:
    return block.start <= anchor < block.end or (block.last and anchor == block.end)


def pool_from_block(block: Block, session: str, result: dict, keep: int = KEEP_PER_BLOCK) -> list[dict]:
    """Candidatos de un bloque en tiempo ABSOLUTO. Solo los que tienen su ancla en el núcleo del bloque."""
    eligible = set(result.get("eligible", []))
    out = []
    for c in result.get("candidates", [])[:keep]:
        a = block.dl_start + c["anchor"]
        if not _owns(block, a):
            continue
        out.append({"id": f"{block.index}:{c['rank']}", "block": block.index, "session": session, "rank": c["rank"],
                    "start": round(block.dl_start + c["start"], 2), "end": round(block.dl_start + c["end"], 2), "anchor": round(a, 2),
                    "score": c["score"], "eligible": c["rank"] in eligible, "triggers": c.get("triggers", []),
                    "text": c.get("text", "")[:240],
                    "top": [f"{x['name']} {x['value']:.2f}" for x in sorted(c["components"], key=lambda x: abs(x["contribution"]), reverse=True)[:3]]})
    return out


def block_scales(pool: list[dict], blocks: int) -> list[dict]:
    """Distribución de puntajes por bloque (para detectar bloques con escala anómala; no se transforma nada)."""
    out = []
    for i in range(blocks):
        s = sorted((c["score"] for c in pool if c["block"] == i), reverse=True)
        out.append({"block": i, "n": len(s), "max": s[0] if s else None, "median": s[len(s) // 2] if s else None,
                    "eligible": sum(1 for c in pool if c["block"] == i and c["eligible"])})
    return out


def _dup(c: dict, o: dict) -> bool:
    inter = max(0.0, min(c["end"], o["end"]) - max(c["start"], o["start"]))
    short = min(c["end"] - c["start"], o["end"] - o["start"])
    return (short > 0 and inter / short > MAX_OVERLAP) or abs(c["anchor"] - o["anchor"]) < ANCHOR_GAP_S


def select_global(pool: list[dict], target: int, explore: bool = False) -> dict:
    """Selección sobre la transmisión completa.

    Devuelve {plan, recommended, exploratory, reserve, duplicates_avoided, crowd_skipped, notes}. `plan` es lo que se renderiza:
    los recomendados (hasta `target`) y, solo si `explore`, exploratorios hasta completar el objetivo (marcados como tales).
    Nunca se baja el umbral: lo que no lo cumple queda como «exploratorio» con su propio rótulo."""
    ordered = sorted(pool, key=lambda c: (-c["score"], c["start"]))
    unique: list[dict] = []
    dups = 0
    for c in ordered:
        if any(_dup(c, o) for o in unique):
            dups += 1                                   # mismo momento visto dos veces (p. ej. zona de solape): gana el mejor
            continue
        unique.append(c)
    best = unique[0]["score"] if unique else 0.0
    chosen: list[dict] = []
    reserve: list[dict] = []
    crowd = 0
    for c in (x for x in unique if x["eligible"]):
        if len(chosen) >= target:
            reserve.append(c)
            continue
        near = sum(1 for o in chosen if abs(o["anchor"] - c["anchor"]) <= CROWD_WINDOW_S)
        if near >= CROWD_MAX and c["score"] < CROWD_EXCEPTION * best:
            crowd += 1                                  # concentración sin superioridad clara: se deja como reserva
            reserve.append(c)
            continue
        chosen.append(c)
    exploratory = [x for x in unique if not x["eligible"]][:EXPLORATORY_CAP]
    plan = [dict(c, kind="recommended") for c in chosen]
    notes: list[str] = []
    if len(chosen) < target:
        notes.append(f"Solo {len(chosen)} momento(s) cumplen los criterios normales de calidad (se pedían {target}). "
                     + (f"Hay {len(exploratory)} candidato(s) exploratorios de menor confianza disponibles." if exploratory
                        else "No hay más candidatos: la transmisión no contiene más momentos que destaquen."))
        if explore:
            plan += [dict(c, kind="exploratory") for c in exploratory[:target - len(chosen)]]
            exploratory = exploratory[target - len(chosen):]
            notes.append("Se incluyeron candidatos EXPLORATORIOS para completar el objetivo: son de menor confianza (no alcanzan el umbral).")
    plan.sort(key=lambda c: -c["score"])
    for i, c in enumerate(plan, 1):
        c["global_rank"] = i
    return {"plan": plan, "recommended": len(chosen), "exploratory": exploratory, "reserve": reserve, "duplicates_avoided": dups,
            "crowd_skipped": crowd, "notes": notes, "candidates_total": len(pool), "unique_total": len(unique)}


def shift_candidate(c: dict, shift: float) -> dict:
    """Candidato de un bloque llevado al eje de tiempo de otro archivo (restando `shift`), incluidas las ventanas de sus componentes."""
    out = dict(c, start=c["start"] - shift, end=c["end"] - shift, anchor=c["anchor"] - shift)
    out["components"] = [dict(x, window=[x["window"][0] - shift, x["window"][1] - shift] if x.get("window") else None) for x in c["components"]]
    return out


def clip_range(item: dict, duration: float, pad: float = CUT_PAD_S) -> tuple[int, int]:
    """Rango absoluto a re-descargar para renderizar un clip final: su ventana con margen, entero y dentro del VOD."""
    a = max(0, int(math.floor(item["start"] - pad)))
    b = min(int(duration), int(math.ceil(item["end"] + pad)))
    return a, b


def shift_transcript(tr: Transcript, shift: float, lo: float, hi: float) -> Transcript:
    """Transcripción recortada a [lo, hi] (en el eje del bloque, tras restar `shift`) y llevada al eje del recorte.
    Solo se conservan frases completas dentro del rango: nunca se parte una frase."""
    segs = []
    for s in tr.segments:
        a, b = s.start - shift, s.end - shift
        if a >= lo - 0.01 and b <= hi + 0.01:
            segs.append(Segment(a, b, s.text, [Word(w.start - shift, w.end - shift, w.text, w.prob) for w in s.words],
                                s.no_speech_prob, s.avg_logprob, s.compression_ratio))
    return Transcript(tr.language, segs, [], tr.evidence_version)

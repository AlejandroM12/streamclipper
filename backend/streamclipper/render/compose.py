"""Composición vertical por regiones (1.5): cámara(s) arriba, contenido abajo, participantes en pila, etc.

Una composición es un dict con regiones del video ORIGINAL expresadas como fracciones [x, y, w, h] (0..1):
    {"preset": "reacciones", "cams": [[x,y,w,h], ...], "main": [x,y,w,h]}
Presets:
  original      cuadro completo sobre fondo desenfocado (blur-fit; sin regiones).
  gameplay      prioriza el juego: si hay región `main`, se recorta a ella; si no, el recorte horizontal «focus» de siempre.
  reacciones    cámara(s) arriba (lado a lado) y contenido principal abajo.
  podcast       participantes en pila (1 por fila, o 2 por fila con 3-4) y, si se indica, el contenido principal al final.
  personalizado filas de regiones definidas a mano: {"rows": [[r, r], [r]]} (de arriba hacia abajo).

El sistema NO detecta cámaras ni caras: una región se usa solo si la persona la declaró. Sin regiones suficientes el preset
vuelve a «original» y lo dice (nunca se inventa un recorte). Las filas se escalan al ancho del lienzo y, si no caben en la
zona segura de la plataforma, todo el bloque se reduce proporcionalmente: nunca se recorta más de lo declarado.
"""
import math
import subprocess
from pathlib import Path

from streamclipper.core.errors import InvalidInputError
from streamclipper.render.layout import _GAP, H, SAFE_MARGINS, W, Layout, _even, line_chars

COMPOSITIONS = ("original", "gameplay", "reacciones", "podcast", "personalizado")
MIN_REGION = 0.05          # lado mínimo de una región (fracción del cuadro)
MAX_REGIONS = 6
ROW_GAP = 12               # px entre filas y entre regiones de una misma fila
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _region(r) -> list[float]:
    try:
        x, y, w, h = (float(v) for v in r)
    except (TypeError, ValueError):
        raise InvalidInputError("Cada región debe ser [x, y, ancho, alto] con valores entre 0 y 1.") from None
    if not (0 <= x < 1 and 0 <= y < 1 and w >= MIN_REGION and h >= MIN_REGION and x + w <= 1 + 1e-6 and y + h <= 1 + 1e-6):
        raise InvalidInputError(f"Región fuera del cuadro o demasiado pequeña (mínimo {MIN_REGION:.0%} por lado): {[x, y, w, h]}")
    return [round(x, 4), round(y, 4), round(min(w, 1 - x), 4), round(min(h, 1 - y), 4)]


def parse_composition(d) -> dict | None:
    """Valida y normaliza una composición recibida del cliente. None = sin composición (se usa la presentación de siempre)."""
    if d in (None, "", {}):
        return None
    if not isinstance(d, dict):
        raise InvalidInputError("Composición inválida.")
    preset = d.get("preset") or "original"
    if preset not in COMPOSITIONS:
        raise InvalidInputError(f"Composición desconocida: {preset}")
    out: dict = {"preset": preset, "cams": [_region(r) for r in (d.get("cams") or [])],
                 "main": _region(d["main"]) if d.get("main") else None,
                 "rows": [[_region(r) for r in row] for row in (d.get("rows") or [])]}
    n = len(out["cams"]) + (1 if out["main"] else 0) + sum(len(r) for r in out["rows"])
    if n > MAX_REGIONS or any(len(r) > 3 for r in out["rows"]):
        raise InvalidInputError(f"Demasiadas regiones (máximo {MAX_REGIONS}, hasta 3 por fila).")
    return out


def effective_rows(comp: dict) -> tuple[list[list[list[float]]] | None, str]:
    """(filas de regiones, motivo del respaldo). filas=None: no hay regiones utilizables -> el preset vuelve a `original`
    (o a `focus` para gameplay sin región). El motivo explica por qué."""
    p, cams, main, rows = comp["preset"], comp.get("cams") or [], comp.get("main"), comp.get("rows") or []
    if p == "original":
        return None, ""
    if p == "gameplay":
        return ([[main]], "") if main else (None, "gameplay sin región del juego: se usa el recorte horizontal de siempre")
    if p == "reacciones":
        if cams and main:
            return [cams, [main]], ""
        return None, "reacciones necesita al menos una cámara y el contenido principal: se usa Original"
    if p == "podcast":
        if not cams:
            return None, "podcast necesita al menos un participante: se usa Original"
        per = 2 if len(cams) >= 3 else 1
        out = [cams[i:i + per] for i in range(0, len(cams), per)]
        return (out + ([[main]] if main else []), "")
    if rows:
        return rows, ""
    flat = ([cams] if cams else []) + ([[main]] if main else [])
    return (flat, "") if flat else (None, "personalizado sin regiones: se usa Original")


def settings_overrides(comp: dict | None) -> tuple[dict, str]:
    """Ajustes de Settings para una composición y, si hubo respaldo, el motivo. Sin composición: sin cambios."""
    if comp is None:
        return {}, ""
    rows, why = effective_rows(comp)
    if rows is not None:
        return {"layout": "stack", "composition": comp}, ""
    return {"layout": "focus" if comp["preset"] == "gameplay" else "blur-fit"}, why


def compute_stack(src_w: int, src_h: int, cfg) -> Layout:
    """Distribución de las filas de regiones en el lienzo 1080x1920 dentro de la zona segura (subtítulos debajo)."""
    comp = cfg.composition or {}
    rows, _ = effective_rows(comp) if comp else (None, "")
    if not rows:
        raise InvalidInputError("La composición no tiene regiones utilizables.")
    m = SAFE_MARGINS[cfg.safe_platform]
    chars = cfg.sub_line_chars or line_chars(cfg.sub_font_size, m["left"], m["right"])
    block_h = math.ceil(cfg.sub_font_size * 1.18 * cfg.sub_max_lines) + 12
    avail_h = H - m["top"] - m["bottom"] - _GAP - block_h
    sizes, geo = [], []
    for row in rows:
        px = []
        for x, y, w, h in row:
            cw, ch = _even(w * src_w), _even(h * src_h)
            cx, cy = min(int(x * src_w) // 2 * 2, src_w - cw), min(int(y * src_h) // 2 * 2, src_h - ch)
            px.append((cw, ch, cx, cy))
        gaps = ROW_GAP * (len(px) - 1)
        h_row = (W - gaps) / sum(cw / ch for cw, ch, _, _ in px)       # altura común: todas las regiones de la fila llenan el ancho
        geo.append(px)
        sizes.append(h_row)
    total = sum(sizes) + ROW_GAP * (len(rows) - 1)
    k = min(1.0, 0.995 * avail_h / total)                               # holgura para el redondeo a pares
    stack_h = total * k
    y = m["top"] + max(0, (H - m["top"] - m["bottom"] - (stack_h + _GAP + block_h)) // 2)
    panels, y_cur = [], float(y)
    for px, h_row in zip(geo, sizes):
        hh = _even(h_row * k)
        widths = [_even(hh * cw / ch) for cw, ch, _, _ in px]
        x_cur = (W - (sum(widths) + ROW_GAP * (len(px) - 1))) // 2
        for (cw, ch, cx, cy), pw in zip(px, widths):
            panels.append({"crop": (cw, ch, cx, cy), "size": (pw, hh), "pos": (int(x_cur), int(round(y_cur)))})
            x_cur += pw + ROW_GAP
        y_cur += hh + ROW_GAP
    end_y = max(p["pos"][1] + p["size"][1] for p in panels)
    framing = {"layout": "stack", "preset": comp.get("preset"), "scale_pct": round(100 * k), "regions": len(panels),
               "panels": [{"crop": list(p["crop"]), "size": list(p["size"]), "pos": list(p["pos"])} for p in panels],
               "note": ("composición por regiones: cada región se muestra completa (sin recortar más de lo declarado)"
                        + (f"; se redujo al {round(100 * k)} % para que quepa en la zona segura" if k < 0.995 else ""))}
    return Layout("stack", None, None, 8, end_y + _GAP, m["left"], m["right"], chars, block_h, framing, tuple(panels))


def stack_filter(layout: Layout, settings, tail: str) -> str:
    """Grafo FFmpeg: fondo desenfocado + cada región recortada, escalada (lanczos: texto nítido) y superpuesta."""
    n = len(layout.panels)
    blur = f",boxblur={settings.bg_blur}:4" if settings.bg_blur > 0 else ""
    dim = f",eq=brightness=-{settings.bg_dim:g}" if settings.bg_dim > 0 else ""
    labels = "".join(f"[p{i}]" for i in range(n))
    parts = [f"[0:v]split={n + 1}[bgs]{labels}",
             f"[bgs]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}{blur}{dim}[o0]"]
    for i, p in enumerate(layout.panels):
        cw, ch, cx, cy = p["crop"]
        pw, ph = p["size"]
        x, y = p["pos"]
        parts.append(f"[p{i}]crop={cw}:{ch}:{cx}:{cy},scale={pw}:{ph}:flags=lanczos[q{i}]")
        last = f"[o{i + 1}]" if i < n - 1 else ""
        parts.append(f"[o{i}][q{i}]overlay={x}:{y}{(',' + tail + '[v]') if i == n - 1 else last}")
    return ";".join(parts)


def _run_frame(cmd: list, timeout: int = 60) -> bytes:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise InvalidInputError(f"No se pudo generar la imagen de previsualización: {e}") from None
    if r.returncode != 0 or not r.stdout:
        raise InvalidInputError("No se pudo generar la imagen de previsualización: " + r.stderr.decode("utf-8", "replace").strip()[-200:])
    return r.stdout


def source_frame(src: Path, t: float, ffmpeg, max_w: int = 960) -> bytes:
    """Un fotograma del video original (JPEG) para elegir regiones a mano."""
    return _run_frame([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
                       "-frames:v", "1", "-vf", f"scale={max_w}:-2", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"])


def composed_frame(src: Path, t: float, layout: Layout, cfg, ffmpeg, filter_graph: str, width: int = 360) -> bytes:
    """Previsualización de la composición (JPEG pequeño) con el recuadro donde irán los subtítulos. `filter_graph` termina en [v]."""
    box = f"drawbox=x=60:y={layout.sub_margin_v}:w={W - 120}:h={layout.sub_block_h}:color=white@0.28:t=fill"
    graph = f"{filter_graph};[v]{box},scale={width}:-2[pv]"
    return _run_frame([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
                       "-filter_complex", graph, "-map", "[pv]", "-frames:v", "1", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"])

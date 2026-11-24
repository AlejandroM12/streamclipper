"""Composición vertical 1080x1920: dónde va el gameplay y dónde los subtítulos, respetando márgenes seguros.

Los márgenes seguros son APROXIMADOS (las plataformas no publican una cifra única y cambian su interfaz): dejan
libres las zonas que suelen tapar la barra superior, la descripción/botones inferiores y los íconos de la derecha.
Son configurables y `universal` toma el máximo de cada lado.
"""
import math
from dataclasses import dataclass, field

from streamclipper.core.config import parse_region

W, H = 1080, 1920

SAFE_MARGINS: dict[str, dict[str, int]] = {
    "tiktok": {"top": 170, "bottom": 480, "left": 60, "right": 120},
    "reels": {"top": 220, "bottom": 400, "left": 60, "right": 120},
    "shorts": {"top": 160, "bottom": 380, "left": 60, "right": 120},
}
SAFE_MARGINS["universal"] = {k: max(m[k] for m in SAFE_MARGINS.values()) for k in ("top", "bottom", "left", "right")}

_GAP = 40  # separación entre el gameplay y los subtítulos


@dataclass(frozen=True)
class Layout:
    mode: str                          # blur-fit | focus | fill | crop
    fg: tuple[int, int, int, int] | None   # (ancho, alto, x, y) del gameplay en el lienzo
    crop: tuple[int, int, int, int] | None  # (ancho, alto, x, y) del recorte en el video original
    sub_alignment: int                 # ASS: 8 = arriba-centro (bajo el gameplay), 2 = abajo-centro
    sub_margin_v: int                  # distancia vertical desde el borde (arriba para 8, abajo para 2)
    margin_l: int
    margin_r: int
    sub_line_chars: int
    sub_block_h: int
    framing: dict = field(default_factory=dict, compare=False)   # qué se recorta y si se respetan las regiones a conservar
    panels: tuple = ()                 # stack: regiones recortadas y su lugar en el lienzo (ver render/compose.py)


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def line_chars(font_size: int, margin_l: int, margin_r: int) -> int:
    """Caracteres por línea que caben en el ancho útil (Arial Bold: ≈0.52·tamaño por carácter)."""
    return max(10, int((W - margin_l - margin_r) / (0.52 * font_size)))


def fit_crop(src_w: int, zoom: float, fg_x: float, regions: list[tuple[float, float, float, float]]
             ) -> tuple[int, int, float, bool | None]:
    """Recorte horizontal (ancho, x, zoom usado, ¿se conservan las regiones?).

    Sin regiones: zoom y posición manuales (fg_zoom, fg_x). Con regiones declaradas a mano: se limita el zoom para que
    todas quepan juntas y el recorte se centra en ellas. Si no caben ni con zoom 1 (cuadro completo) el resultado es el
    cuadro completo. El sistema no detecta nada: solo respeta lo que se declara."""
    even = src_w - src_w % 2
    if not regions:
        cw = min(_even(src_w / zoom), even)
        return cw, int((src_w - cw) * fg_x) // 2 * 2, zoom, None
    x0 = min(r[0] for r in regions) * src_w
    x1 = max(r[0] + r[2] for r in regions) * src_w
    zoom_used = max(1.0, min(zoom, src_w / max(1.0, x1 - x0)))
    cw = min(_even(src_w / zoom_used), even)
    cx = int(min(max((x0 + x1) / 2 - cw / 2, 0), src_w - cw)) // 2 * 2
    kept = cx <= x0 + 2 and x1 <= cx + cw + 2
    return cw, cx, zoom_used, kept


def _framing(layout: str, src_w: int, src_h: int, cw: int, cx: int, zoom_req: float, zoom_used: float,
             regions: list[str], kept: bool | None) -> dict:
    out = {"layout": layout, "zoom_requested": round(zoom_req, 2), "zoom_used": round(zoom_used, 2),
           "crop_left_pct": round(100 * cx / src_w, 1), "crop_right_pct": round(100 * (src_w - cx - cw) / src_w, 1),
           "keep_regions": regions, "regions_kept": kept}
    if layout == "blur-fit":
        out["note"] = "cuadro completo, sin recortar"
    elif not regions:
        out["note"] = ("recorte manual sin regiones declaradas: el sistema no detecta la facecam ni el HUD; "
                       "indicalos con --keep x,y,w,h para protegerlos")
    elif kept:
        out["note"] = "las regiones declaradas quedan completas" + (
            f" (zoom reducido de {zoom_req:g} a {zoom_used:.2f} para que quepan)" if zoom_used < zoom_req - 0.005 else "")
    else:
        out["note"] = "las regiones declaradas no caben: se usa el cuadro completo"
    return out


def compute_layout(src_w: int, src_h: int, cfg) -> Layout:
    if cfg.layout == "stack":
        from streamclipper.render.compose import compute_stack
        return compute_stack(src_w, src_h, cfg)
    m = SAFE_MARGINS[cfg.safe_platform]
    chars = cfg.sub_line_chars or line_chars(cfg.sub_font_size, m["left"], m["right"])
    block_h = math.ceil(cfg.sub_font_size * 1.18 * cfg.sub_max_lines) + 12
    if cfg.layout in ("fill", "crop"):
        zoom = 1.0 if cfg.layout == "fill" else cfg.zoom
        visible = min(1.0, (W / H) / (src_w / src_h) / zoom)   # fracción del ancho original que se ve
        fr = {"layout": cfg.layout, "visible_width_pct": round(100 * visible, 1), "keep_regions": list(cfg.keep_regions),
              "regions_kept": None, "note": "cubre todo el lienzo recortando: no se conservan regiones declaradas"}
        return Layout(cfg.layout, None, None, 2, m["bottom"], m["left"], m["right"], chars, block_h, fr)

    regions = [parse_region(r) for r in cfg.keep_regions]
    zoom_req = cfg.fg_zoom if cfg.layout == "focus" else 1.0
    if cfg.layout == "focus":
        cw, cx, zoom, kept = fit_crop(src_w, zoom_req, cfg.fg_x, regions)
    else:  # blur-fit: cuadro completo
        cw, cx, zoom, kept = min(_even(src_w), src_w - src_w % 2), 0, 1.0, None
    ch = src_h - src_h % 2
    framing = _framing(cfg.layout, src_w, src_h, cw, cx, zoom_req, zoom, list(cfg.keep_regions), kept)
    fw = W
    fh = _even(fw * ch / cw)
    avail = H - m["top"] - m["bottom"] - _GAP - block_h
    if fh > avail:  # fuente muy alta: se reduce para que quepan gameplay y subtítulos dentro de la zona segura
        k = avail / fh
        fw, fh = _even(fw * k), _even(avail)
    total = fh + _GAP + block_h
    y0 = m["top"] + max(0, (H - m["top"] - m["bottom"] - total) // 2)
    x0 = (W - fw) // 2
    return Layout(cfg.layout, (fw, fh, x0, y0), (cw, ch, cx, 0), 8, y0 + fh + _GAP, m["left"], m["right"],
                  chars, block_h, framing)

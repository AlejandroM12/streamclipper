"""Composición vertical 1080x1920: dónde va el gameplay y dónde los subtítulos, respetando márgenes seguros.

Los márgenes seguros son APROXIMADOS (las plataformas no publican una cifra única y cambian su interfaz): dejan
libres las zonas que suelen tapar la barra superior, la descripción/botones inferiores y los íconos de la derecha.
Son configurables y `universal` toma el máximo de cada lado.
"""
import math
from dataclasses import dataclass

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


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def line_chars(font_size: int, margin_l: int, margin_r: int) -> int:
    """Caracteres por línea que caben en el ancho útil (Arial Bold: ≈0.52·tamaño por carácter)."""
    return max(10, int((W - margin_l - margin_r) / (0.52 * font_size)))


def compute_layout(src_w: int, src_h: int, cfg) -> Layout:
    m = SAFE_MARGINS[cfg.safe_platform]
    chars = cfg.sub_line_chars or line_chars(cfg.sub_font_size, m["left"], m["right"])
    block_h = math.ceil(cfg.sub_font_size * 1.18 * cfg.sub_max_lines) + 12
    if cfg.layout in ("fill", "crop"):
        return Layout(cfg.layout, None, None, 2, m["bottom"], m["left"], m["right"], chars, block_h)

    zoom = cfg.fg_zoom if cfg.layout == "focus" else 1.0
    cw = _even(src_w / zoom)
    cw = min(cw, src_w - src_w % 2)
    ch = src_h - src_h % 2
    cx = int((src_w - cw) * cfg.fg_x) // 2 * 2
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
                  chars, block_h)

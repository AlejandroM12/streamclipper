"""Descarga de videos públicos de YouTube, Twitch y Kick con yt-dlp (API de Python, nunca shell).

Alcance deliberadamente estrecho: solo URLs HTTPS de un único video de YouTube/Shorts, un VOD (grabación) o un clip de
Twitch, o un VOD o un clip de Kick. Se valida la URL, se consulta la metadata (duración, tamaño estimado,
disponibilidad) ANTES de descargar y el archivo resultante se verifica con FFprobe. No hay transmisiones en vivo, no se
usan cuentas, contraseñas ni cookies y no se intenta saltar restricciones de acceso, DRM o controles de la plataforma:
si el video no es accesible públicamente, se informa y se aborta.
"""
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from streamclipper.core.cancel import CancelToken
from streamclipper.core.config import Settings
from streamclipper.core.errors import CancelledError, InvalidInputError, MediaError
from streamclipper.core.log import log_event
from streamclipper.media.ffmpeg import probe

ALLOWED_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"})
TWITCH_HOSTS = frozenset({"twitch.tv", "www.twitch.tv", "m.twitch.tv", "go.twitch.tv", "clips.twitch.tv"})
KICK_HOSTS = frozenset({"kick.com", "www.kick.com"})
PLATFORM_LABELS = {"youtube": "YouTube", "twitch": "Twitch", "kick": "Kick"}
KIND_LABELS = {"watch": "video", "shorts": "short", "short-link": "video", "vod": "grabación (VOD)", "clip": "clip"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_TWITCH_VOD_ID = re.compile(r"^\d{1,12}$")
_TWITCH_CLIP_SLUG = re.compile(r"^[A-Za-z0-9_-]{5,120}$")
_KICK_CHANNEL = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_KICK_VOD_ID = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
_KICK_CLIP_ID = re.compile(r"^clip_[A-Za-z0-9_-]{5,60}$")
_MAX_URL_LEN = 300
_BLOCKED_AVAILABILITY = {
    "private": "es privado",
    "premium_only": "es solo para suscriptores Premium",
    "subscriber_only": "es solo para miembros del canal",
    "needs_auth": "requiere iniciar sesión",
}


def looks_like_url(text: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text.strip()))


@dataclass(frozen=True)
class DownloadLimits:
    max_duration_s: float = 20 * 60
    max_bytes: int = 1500 * 1024 * 1024
    max_height: int = 1080


@dataclass(frozen=True)
class VideoRef:
    video_id: str
    canonical_url: str
    kind: str                  # YouTube: watch | shorts | short-link; Twitch y Kick: vod | clip
    ignored_playlist: bool = False
    platform: str = "youtube"  # youtube | twitch | kick

    @property
    def label(self) -> str:
        return PLATFORM_LABELS[self.platform]

    @property
    def kind_label(self) -> str:
        return KIND_LABELS[self.kind]

    @property
    def file_prefix(self) -> str:
        """Prefijo del archivo en data/samples. YouTube conserva `youtube-<id>`; el resto distingue VOD de clip."""
        return "youtube" if self.platform == "youtube" else f"{self.platform}-{self.kind}"


@dataclass
class DownloadResult:
    path: Path
    video_id: str
    title: str | None = None
    duration: float | None = None
    size_bytes: int | None = None
    estimated_bytes: int | None = None
    license: str | None = None
    reused: bool = False
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)
    platform: str = "youtube"


# --- validación de la URL ----------------------------------------------------------------------
def _split_url(url: str):
    """Validaciones comunes a todas las plataformas. Devuelve (partes, host en minúsculas)."""
    raw = url.strip()
    if not raw:
        raise InvalidInputError("La URL está vacía")
    if len(raw) > _MAX_URL_LEN:
        raise InvalidInputError("La URL es demasiado larga")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise InvalidInputError("La URL contiene espacios o caracteres de control")
    try:
        parts = urlsplit(raw)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError:
        raise InvalidInputError("URL mal formada") from None
    if parts.scheme.lower() != "https":
        raise InvalidInputError("Solo se aceptan URLs HTTPS (https://...)")
    if parts.username or parts.password:
        raise InvalidInputError("La URL no puede incluir credenciales")
    if port not in (None, 443):
        raise InvalidInputError("La URL no puede usar un puerto distinto de 443")
    return parts, host or raw


def parse_media_url(url: str) -> VideoRef:
    """Detecta la plataforma por el dominio exacto y valida la URL. Solo un video, un VOD o un clip: nada de listas,
    canales ni directos. Lanza InvalidInputError con un mensaje claro si no se acepta."""
    parts, host = _split_url(url)
    if host in ALLOWED_HOSTS:
        return parse_youtube_url(url)
    if host in TWITCH_HOSTS:
        return _parse_twitch(parts, host)
    if host in KICK_HOSTS:
        return _parse_kick(parts)
    raise InvalidInputError(f"Dominio no permitido: '{host}'. Se aceptan YouTube (youtube.com, youtu.be), "
                            "Twitch (twitch.tv, clips.twitch.tv) y Kick (kick.com)")


_NOT_A_VOD = ("Esa URL no es un VOD ni un clip. Se aceptan: {fmt}. "
              "No se aceptan canales, listas, colecciones ni transmisiones en vivo.")


def _parse_twitch(parts, host: str) -> VideoRef:
    segs = [s for s in parts.path.split("/") if s]
    query = parse_qs(parts.query, keep_blank_values=True)
    fmt = "twitch.tv/videos/<id> y clips.twitch.tv/<clip>"
    ignored = "collection" in query

    def vod(vid: str) -> VideoRef:
        return VideoRef(vid, f"https://www.twitch.tv/videos/{vid}", "vod", ignored, "twitch")

    def clip(slug: str) -> VideoRef:
        return VideoRef(slug, f"https://clips.twitch.tv/{slug}", "clip", False, "twitch")

    if host == "clips.twitch.tv":
        slug = None
        if segs == ["embed"] and len(query.get("clip", [])) == 1:
            slug = query["clip"][0]
        elif len(segs) in (1, 2) and segs[0] != "embed":
            slug = segs[-1]
        if slug is None:
            raise InvalidInputError(_NOT_A_VOD.format(fmt=fmt))
        if not _TWITCH_CLIP_SLUG.match(slug):
            raise InvalidInputError("El identificador del clip de Twitch no es válido")
        return clip(slug)
    if len(segs) == 2 and segs[0] == "videos" and segs[1].isdigit() or (len(segs) == 3 and segs[1] in ("v", "video")):
        vid = segs[-1]
        if not _TWITCH_VOD_ID.match(vid):
            raise InvalidInputError("El identificador del VOD de Twitch no es válido (se esperan solo dígitos)")
        return vod(vid)
    if (len(segs) == 3 and segs[1] == "clip") or (len(segs) == 2 and segs[0] == "clip"):
        if not _TWITCH_CLIP_SLUG.match(segs[-1]):
            raise InvalidInputError("El identificador del clip de Twitch no es válido")
        return clip(segs[-1])
    if segs[:1] in (["collections"], ["directory"]) or segs[-1:] in (["videos"], ["clips"], ["collections"]):
        raise InvalidInputError("No se aceptan listas, colecciones ni páginas de canal: indicá la URL de un VOD o de un clip")
    if len(segs) == 1:
        raise InvalidInputError("Esa URL es de un canal de Twitch (transmisión en vivo). No se aceptan directos: "
                                "usá la URL de un VOD (twitch.tv/videos/<id>) o de un clip")
    raise InvalidInputError(_NOT_A_VOD.format(fmt=fmt))


def _parse_kick(parts) -> VideoRef:
    segs = [s for s in parts.path.split("/") if s]
    query = parse_qs(parts.query, keep_blank_values=True)
    fmt = "kick.com/<canal>/videos/<id> y kick.com/<canal>/clips/<clip>"
    if not segs or not _KICK_CHANNEL.match(segs[0]) or segs[0].lower() in ("video", "categories", "search", "auth", "browse"):
        raise InvalidInputError(_NOT_A_VOD.format(fmt=fmt))
    chan = segs[0]
    if len(segs) == 3 and segs[1] == "videos":
        if not _KICK_VOD_ID.match(segs[2]):
            raise InvalidInputError("El identificador del VOD de Kick no es válido")
        vid = segs[2].lower()
        return VideoRef(vid, f"https://kick.com/{chan}/videos/{vid}", "vod", False, "kick")
    clip_id = segs[2] if len(segs) == 3 and segs[1] == "clips" else None
    if len(segs) == 1 and len(query.get("clip", [])) == 1:
        clip_id = query["clip"][0]
    if clip_id is not None:
        if not _KICK_CLIP_ID.match(clip_id):
            raise InvalidInputError("El identificador del clip de Kick no es válido")
        return VideoRef(clip_id, f"https://kick.com/{chan}/clips/{clip_id}", "clip", False, "kick")
    if segs[1:] in (["videos"], ["clips"]):
        raise InvalidInputError("No se aceptan listas de videos o clips de un canal: indicá la URL de un VOD o de un clip")
    if len(segs) == 1:
        raise InvalidInputError("Esa URL es de un canal de Kick (transmisión en vivo). No se aceptan directos: "
                                "usá la URL de un VOD (kick.com/<canal>/videos/<id>) o de un clip")
    raise InvalidInputError(_NOT_A_VOD.format(fmt=fmt))


def parse_youtube_url(url: str) -> VideoRef:
    """Valida y normaliza la URL. Lanza InvalidInputError con un mensaje claro si no se acepta."""
    parts, host = _split_url(url)
    if host not in ALLOWED_HOSTS:
        raise InvalidInputError(
            f"Dominio no permitido: '{host}'. Por ahora solo YouTube (youtube.com, youtu.be)")

    query = parse_qs(parts.query, keep_blank_values=True)
    has_list = "list" in query
    segs = [s for s in parts.path.split("/") if s]

    if host == "youtu.be":
        if len(segs) != 1:
            raise InvalidInputError("Enlace youtu.be inválido: se espera https://youtu.be/<id>")
        return _ref(segs[0], "short-link", has_list)

    if segs[:1] == ["playlist"]:
        raise InvalidInputError("No se aceptan listas de reproducción: indicá la URL de un único video")
    if segs[:1] == ["watch"] and len(segs) == 1:
        ids = query.get("v", [])
        if len(ids) != 1:
            raise InvalidInputError("La URL de watch debe incluir exactamente un parámetro v=<id>")
        return _ref(ids[0], "watch", has_list)
    if segs[:1] == ["shorts"] and len(segs) == 2:
        return _ref(segs[1], "shorts", has_list)
    if segs[:1] == ["live"]:
        raise InvalidInputError("No se aceptan transmisiones en vivo")
    raise InvalidInputError(
        "URL de YouTube no soportada: se aceptan /watch?v=<id>, /shorts/<id> y youtu.be/<id> "
        "(no canales, búsquedas ni listas)")


def _ref(video_id: str, kind: str, has_list: bool) -> VideoRef:
    if not _VIDEO_ID.match(video_id):
        raise InvalidInputError("El identificador de video no es válido (se esperan 11 caracteres [A-Za-z0-9_-])")
    return VideoRef(video_id, f"https://www.youtube.com/watch?v={video_id}", kind, has_list)


# --- validación de la metadata (antes de descargar) --------------------------------------------
def check_info(info: dict, limits: DownloadLimits) -> int | None:
    """Rechaza videos no aptos. Devuelve el tamaño estimado en bytes, o None si no se conoce."""
    if info.get("_type") == "playlist" or info.get("entries") is not None:
        raise InvalidInputError("La URL corresponde a una lista de reproducción, no a un único video")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        raise InvalidInputError("No se aceptan transmisiones en vivo (ni estrenos/directos recién terminados)")
    blocked = _BLOCKED_AVAILABILITY.get(info.get("availability") or "")
    if blocked:
        raise InvalidInputError(f"El video no es público: {blocked}")
    duration = info.get("duration")
    if not duration:
        raise InvalidInputError("No se pudo determinar la duración del video; se rechaza por seguridad")
    if duration > limits.max_duration_s:
        raise InvalidInputError(
            f"El video dura {duration / 60:.1f} min; el máximo permitido es {limits.max_duration_s / 60:.0f} min")

    fmts = info.get("requested_formats") or [info]
    sizes = [f.get("filesize") or f.get("filesize_approx") for f in fmts]
    estimated = sum(sizes) if sizes and all(sizes) else None
    if estimated is None:   # HLS (Twitch, Kick) no informa el tamaño: se estima con la tasa de bits y la duración
        rates = [f.get("tbr") for f in fmts]
        if rates and all(rates):
            estimated = sum(rates) * 125 * duration
    if estimated and estimated > limits.max_bytes:
        raise InvalidInputError(
            f"Tamaño estimado {estimated / 1e6:.0f} MB; el máximo permitido es {limits.max_bytes / 1e6:.0f} MB")
    return int(estimated) if estimated else None


_ERROR_HINTS = [
    (r"private video", "El video es privado"),
    (r"video unavailable|has been removed|no longer available|does not exist|account.*terminated|"
     r"not_found|resource not found|http error 404",
     "El video no está disponible (eliminado, vencido, o la cuenta fue cerrada)"),
    (r"members[- ]only|join this channel", "El video es solo para miembros del canal"),
    (r"subscriber-only|requires a channel subscription|logged into an account|login required|log in|"
     r"authentication|unauthorized|vod_manifest_restricted",
     "El contenido es solo para suscriptores o requiere iniciar sesión. StreamClipper no usa cuentas, "
     "contraseñas ni cookies, así que no se puede descargar"),
    (r"cloudflare|just a moment|impersonat|http error 403",
     "La plataforma bloqueó la consulta automática (protección anti-bot); probá más tarde. No se evade el control"),
    (r"sign in to confirm your age|age[- ]restricted|inappropriate for some users",
     "El video tiene restricción de edad y requiere iniciar sesión (no soportado)"),
    (r"available in your country|blocked.{0,40}country|geo.?restrict|geo.?block",
     "El video no está disponible en tu región (no se intenta evadir la restricción)"),
    (r"copyright", "El video fue bloqueado por derechos de autor"),
    (r"sign in to confirm you.re not a bot|confirm you.re not a bot",
     "YouTube pidió verificación (anti-bot); intentá más tarde. No se usan cookies ni se evade el control"),
    (r"premieres in|live event will begin", "El video es un estreno/directo aún no disponible"),
    (r"drm|protected", "El video está protegido (DRM) y no se puede descargar"),
    (r"unable to download webpage|timed out|connection|getaddrinfo|network is unreachable",
     "No se pudo conectar con la plataforma; revisá tu conexión"),
    (r"requested format is not available|no video formats",
     "No hay un formato compatible disponible para este video"),
]


def explain_download_error(message: str) -> str:
    low = message.lower()
    for pattern, text in _ERROR_HINTS:
        if re.search(pattern, low):
            return text
    clean = re.sub(r"\x1b\[[0-9;]*m", "", message).strip()
    return f"No se pudo descargar el video: {clean[-300:]}"


# --- descarga ----------------------------------------------------------------------------------
def _format_selector(max_height: int) -> str:
    h = f"[height<={max_height}]"
    # H.264 primero (decodifica y renderiza más rápido en CPU que AV1/VP9), con alternativas
    u = f"[height<=?{max_height}]"   # «?»: acepta formatos sin altura informada (clips de Kick); los que la informan se filtran igual
    return (f"bv*{h}[vcodec^=avc1]+ba[ext=m4a]/bv*{h}[ext=mp4]+ba[ext=m4a]/b{h}[ext=mp4]/b{h}/b{u}[ext=mp4]/b{u}")


def _default_factory(opts: dict):
    import yt_dlp
    return yt_dlp.YoutubeDL(opts)


def samples_dir(settings: Settings) -> Path:
    return settings.data_dir / "samples"


def target_path(settings: Settings, video_id: str, prefix: str = "youtube") -> Path:
    return samples_dir(settings) / f"{prefix}-{video_id}.mp4"  # el nombre sale solo del id y del prefijo ya validados


def download_youtube(url: str, *args, **kwargs) -> DownloadResult:
    """Igual que download_media pero solo acepta YouTube (compatibilidad con el código y la CLI anteriores)."""
    parse_youtube_url(url)
    return download_media(url, *args, **kwargs)


def download_media(
    url: str,
    settings: Settings,
    limits: DownloadLimits | None = None,
    *,
    on_exists: str = "error",          # error | reuse | overwrite
    dry_run: bool = False,
    cancel: CancelToken | None = None,
    progress: Callable[[str], None] | None = None,
    ydl_factory: Callable[[dict], object] | None = None,
) -> DownloadResult:
    limits = limits or DownloadLimits()
    cancel = cancel or CancelToken()
    progress = progress or (lambda _m: None)
    ydl_factory = ydl_factory or _default_factory
    if on_exists not in ("error", "reuse", "overwrite"):
        raise ValueError(on_exists)

    ref = parse_media_url(url)
    warnings = []
    if ref.ignored_playlist:
        warnings.append("La URL incluía una lista o colección: se ignora y se procesa solo el video indicado")
    final = target_path(settings, ref.video_id, ref.file_prefix)

    if final.exists() and not dry_run:
        if on_exists == "error":
            raise InvalidInputError(
                f"Ya existe {final}. Usá --overwrite para reemplazarlo o procesá ese archivo directamente")
        if on_exists == "reuse":
            info = probe(final, settings.ffprobe)
            log_event("download_reused", path=str(final), video_id=ref.video_id)
            return DownloadResult(final, ref.video_id, duration=info.duration, size_bytes=info.size_bytes,
                                  reused=True, warnings=warnings)

    tmp = samples_dir(settings) / f".downloading-{ref.file_prefix}-{ref.video_id}"
    state = {"last": -1.0}

    def hook(d: dict) -> None:
        if cancel.cancelled:
            raise CancelledError("Descarga cancelada")
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = d["downloaded_bytes"] / total * 100
                if pct - state["last"] >= 10:
                    state["last"] = pct
                    progress(f"descargando {pct:.0f}% de {total / 1e6:.0f} MB")

    opts = {
        "format": _format_selector(limits.max_height),
        "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
        "outtmpl": str(tmp / "%(id)s.%(ext)s"), "merge_output_format": "mp4",
        "restrictfilenames": True, "max_filesize": limits.max_bytes,
        "socket_timeout": 30, "retries": 3, "fragment_retries": 3,
        "progress_hooks": [hook], "ffmpeg_location": str(settings.ffmpeg.parent),
        "ignoreconfig": True, "usenetrc": False,   # sin config, netrc, cuentas ni cookies: solo contenido público
    }

    try:
        progress("consultando metadata")
        with ydl_factory(opts) as ydl:
            info = ydl.extract_info(ref.canonical_url, download=False)
            estimated = check_info(info, limits)
            meta = DownloadResult(final, ref.video_id, title=info.get("title"), duration=info.get("duration"),
                                  estimated_bytes=estimated, license=info.get("license"),
                                  dry_run=dry_run, warnings=warnings, platform=ref.platform)
            if estimated is None:
                warnings.append(f"{ref.label} no informó el tamaño: se aplica el límite durante la descarga")
            if dry_run:
                return meta
            samples_dir(settings).mkdir(parents=True, exist_ok=True)
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            progress(f"descargando '{meta.title}' ({meta.duration / 60:.1f} min)")
            ydl.process_ie_result(info, download=True)

        cancel.check()
        found = sorted(tmp.glob("*.mp4"))   # el nombre lo decide yt-dlp (el id real puede diferir del de la URL en Twitch)
        if len(found) != 1:
            raise MediaError("No se generó el MP4 (¿superó el tamaño máximo o no hay formato compatible?)")
        produced = found[0]
        size = produced.stat().st_size
        if size > limits.max_bytes:
            raise MediaError(f"El archivo descargado pesa {size / 1e6:.0f} MB y supera el máximo permitido")
        info_f = probe(produced, settings.ffprobe)  # FFprobe valida que sea un video real y legible
        if not info_f.has_audio:
            raise MediaError("El video descargado no tiene pista de audio")
        if info_f.duration > limits.max_duration_s + 5:
            raise MediaError(f"El archivo dura {info_f.duration / 60:.1f} min, más del máximo permitido")

        if final.exists() and on_exists != "overwrite":
            raise InvalidInputError(f"{final} apareció mientras se descargaba; no se sobrescribe")
        os.replace(produced, final)
        meta.size_bytes = info_f.size_bytes
        meta.duration = info_f.duration
        _write_sidecar(final, ref, info, info_f.size_bytes)
        log_event("download_ok", video_id=ref.video_id, path=str(final), bytes=info_f.size_bytes)
        return meta
    except CancelledError:
        raise
    except (InvalidInputError, MediaError):
        raise
    except Exception as e:  # yt_dlp.utils.DownloadError, ExtractorError, etc.
        if cancel.cancelled:
            raise CancelledError("Descarga cancelada") from e
        msg = explain_download_error(str(e))
        log_event("download_failed", level=40, video_id=ref.video_id, error=str(e)[-300:])
        raise MediaError(msg) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # sin restos ante error, cancelación o éxito


def _write_sidecar(final: Path, ref: VideoRef, info: dict, size: int) -> None:
    """Registro de procedencia junto al video, para poder revisar de dónde salió y bajo qué licencia."""
    data = {
        "source_url": ref.canonical_url, "platform": ref.platform, "kind": ref.kind, "video_id": ref.video_id,
        "title": info.get("title"),
        "uploader": info.get("uploader"), "duration_s": info.get("duration"), "license": info.get("license"),
        "upload_date": info.get("upload_date"), "size_bytes": size,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Descargado con yt-dlp para uso personal; el usuario es responsable de contar con autorización.",
    }
    final.with_suffix(".source.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def inspect_media(url: str, limits: DownloadLimits | None = None, ydl_factory: Callable[[dict], object] | None = None) -> dict:
    """Plataforma, título y duración de una URL ANTES de procesarla, sin descargar nada. Si la URL es válida pero el video
    no cumple los límites (o no es accesible), se devuelve igualmente lo que se sepa y el motivo en `reason`."""
    limits = limits or DownloadLimits()
    ref = parse_media_url(url)           # una URL inválida o de otro dominio lanza InvalidInputError: nunca llega a la red
    out = {"platform": ref.platform, "platform_label": ref.label, "kind": ref.kind, "kind_label": ref.kind_label,
           "video_id": ref.video_id, "canonical_url": ref.canonical_url, "title": None, "duration": None, "uploader": None,
           "estimated_mb": None, "acceptable": False, "reason": None, "warnings": []}
    if ref.ignored_playlist:
        out["warnings"].append("La URL incluía una lista o colección: se ignora y se procesa solo el video indicado")
    opts = {"format": _format_selector(limits.max_height), "noplaylist": True, "quiet": True, "no_warnings": True,
            "socket_timeout": 30, "retries": 2, "ignoreconfig": True, "usenetrc": False, "skip_download": True}
    try:
        with (ydl_factory or _default_factory)(opts) as ydl:
            info = ydl.extract_info(ref.canonical_url, download=False)
    except Exception as e:
        out["reason"] = explain_download_error(str(e))
        return out
    out.update(title=info.get("title"), duration=info.get("duration"), uploader=info.get("uploader") or info.get("channel"))
    try:
        est = check_info(info, limits)
        out["estimated_mb"] = round(est / 1e6) if est else None
        out["acceptable"] = True
    except InvalidInputError as e:
        out["reason"] = str(e)
    return out

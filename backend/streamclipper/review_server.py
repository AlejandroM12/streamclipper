"""Herramienta local de revisión: sirve los clips y guarda tus evaluaciones en un JSON.

Solo escucha en 127.0.0.1, valida el encabezado Host (contra DNS rebinding), exige JSON en las escrituras y solo
sirve archivos de las carpetas de los jobs revisados (y el video original de cada uno). La interfaz construye el DOM
con textContent: las transcripciones son datos de terceros y nunca se insertan como HTML.
"""
import json
import re
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from streamclipper.core.config import Settings
from streamclipper.core.errors import InvalidInputError
from streamclipper.jobs import load_result
from streamclipper.review import RATINGS, ReviewStore, algorithm_info, build_items

_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
          ".ass": "text/plain; charset=utf-8"}
_MAX_BODY = 20_000


@dataclass
class ReviewState:
    jobs: list[dict] = field(default_factory=list)   # {dir, result, algorithm}
    items: list[dict] = field(default_factory=list)
    store: ReviewStore | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def load_state(job_dirs: list[Path], settings: Settings, n_alternatives: int = 10, with_sync: bool = True) -> ReviewState:
    state = ReviewState()
    fps = set()
    for idx, d in enumerate(job_dirs):
        result = load_result(d)
        fps.add(result.get("source_fingerprint"))
        state.jobs.append({"dir": d, "result": result, "algorithm": algorithm_info(result)})
        for it in build_items(d, settings, n_alternatives, with_sync):
            it["job_index"] = idx
            state.items.append(it)
    if len(fps) != 1 or None in fps:
        raise InvalidInputError("Los jobs deben ser del mismo video (misma huella de archivo) para compartir evaluaciones")
    first = state.jobs[0]["result"]
    state.store = ReviewStore.for_source(settings, fps.pop(), first["source"])
    return state


def _html() -> bytes:
    return _PAGE.encode("utf-8")


def make_server(state: ReviewState, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "StreamClipperReview/1"

        def log_message(self, *a):  # silencioso
            pass

        # --- utilidades ---
        def _host_ok(self) -> bool:
            h = (self.headers.get("Host") or "").lower()
            return h in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, msg: str):
            self._json({"error": msg}, status)

        def _file(self, path: Path):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            rng = self.headers.get("Range")
            if rng and size:
                m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
                if not m or not (m.group(1) or m.group(2)):
                    return self._range_error(size)
                if m.group(1) == "":
                    start = max(0, size - int(m.group(2)))
                else:
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                if start > end or start >= size:
                    return self._range_error(size)
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", _TYPES.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # el navegador cerró la conexión (adelantó el video)

        def _range_error(self, size: int):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        # --- rutas ---
        def do_GET(self):
            if not self._host_ok():
                return self._error(403, "host no permitido")
            path = unquote(urlsplit(self.path).path)
            if path == "/":
                body = _html()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                with state.lock:
                    return self._json({
                        "labels": RATINGS, "items": state.items, "ratings": state.store.ratings,
                        "summary": state.store.summary(), "source": state.store.data["source"],
                        "jobs": [{"job_id": j["result"]["job_id"], "model": j["result"]["settings"]["whisper_model"]}
                                 for j in state.jobs]})
            m = re.fullmatch(r"/media/(\d+)/(.+)", path)
            if m and int(m.group(1)) < len(state.jobs):
                root = state.jobs[int(m.group(1))]["dir"].resolve()
                f = (root / m.group(2)).resolve()
                if f.is_relative_to(root) and f.suffix.lower() in (".mp4", ".ass") and f.is_file():
                    return self._file(f)
                return self._error(404, "no encontrado")
            m = re.fullmatch(r"/source/(\d+)", path)
            if m and int(m.group(1)) < len(state.jobs):
                f = Path(state.jobs[int(m.group(1))]["result"]["source"])
                if f.is_file() and f.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm"):
                    return self._file(f)
                return self._error(404, "el video original no está disponible")
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self._error(404, "no encontrado")

        def do_POST(self):
            if not self._host_ok():
                return self._error(403, "host no permitido")
            if urlsplit(self.path).path != "/api/rating":
                return self._error(404, "no encontrado")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._error(415, "se espera application/json")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._error(400, "Content-Length inválido")
            if not 0 < n <= _MAX_BODY:
                return self._error(413, "cuerpo vacío o demasiado grande")
            try:
                data = json.loads(self.rfile.read(n))
                item_id, rating, note = data["id"], data.get("rating"), str(data.get("note") or "")
            except (json.JSONDecodeError, KeyError, TypeError):
                return self._error(400, "JSON inválido: se esperan id, rating y note")
            item = next((i for i in state.items if i["id"] == item_id), None)
            if item is None:
                return self._error(404, "elemento desconocido")
            try:
                with state.lock:
                    state.store.set(item, rating, note, state.jobs[item["job_index"]]["algorithm"])
                    self._json({"ok": True, "rating": state.store.ratings.get(item_id), "summary": state.store.summary()})
            except InvalidInputError as e:
                self._error(400, str(e))

    return ThreadingHTTPServer((host, port), Handler)


def serve(state: ReviewState, port: int = 8765, open_browser: bool = True) -> None:
    server = None
    for p in range(port, port + 20):
        try:
            server = make_server(state, "127.0.0.1", p)
            break
        except OSError:
            continue
    if server is None:
        raise InvalidInputError(f"No hay puertos libres entre {port} y {port + 19}")
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Revisión disponible en {url}\nEvaluaciones: {state.store.path}\nCtrl+C para terminar.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido. Las evaluaciones ya están guardadas.")
    finally:
        server.server_close()


_PAGE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revisión de clips</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--mut:#5d6778;--line:#d9dee7;--good:#1f8a4c;--edit:#b7791f;--bad:#c0392b;--acc:#2f6fed;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e8ecf3;--mut:#9aa6b8;--line:#2d3544;--good:#4cc582;--edit:#e0a93c;--bad:#ef6b5b;--acc:#7da5ff;--warn:#e0b45a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
header h1{font-size:16px;margin:0}.sum{color:var(--mut);font-size:13px}
select,button,textarea{font:inherit;color:inherit}select{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;display:grid;grid-template-columns:minmax(220px,300px) 1fr;gap:16px}
.card.cur{outline:2px solid var(--acc)}
@media (max-width:760px){.card{grid-template-columns:1fr}}
video{width:100%;max-height:70vh;background:#000;border-radius:8px}
h2{font-size:15px;margin:0 0 4px}.meta{color:var(--mut);font-size:13px;margin:0 0 8px}
.rate{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.rate button{border:2px solid var(--line);background:var(--bg);border-radius:8px;padding:6px 12px;cursor:pointer}
.rate button.on.good{border-color:var(--good);color:var(--good);font-weight:600}.rate button.on.edit{border-color:var(--edit);color:var(--edit);font-weight:600}.rate button.on.bad{border-color:var(--bad);color:var(--bad);font-weight:600}
.rate .clear{border-style:dashed;color:var(--mut)}
textarea{width:100%;min-height:48px;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px}
details{margin:6px 0}summary{cursor:pointer;color:var(--acc)}
.tx{white-space:pre-wrap;background:var(--bg);border-radius:6px;padding:8px;font-size:14px}
.iss{color:var(--warn);margin:2px 0}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:0 8px;font-size:12px;color:var(--mut);margin-right:4px}
.ev{font-size:13px;color:var(--mut);margin:2px 0}.note{font-size:12px;color:var(--mut)}
.empty{padding:40px;text-align:center;color:var(--mut)}
</style>
</head>
<body>
<header>
  <h1>Revisión de clips</h1>
  <span class="sum" id="sum"></span>
  <label>Mostrar <select id="fkind"><option value="all">todo</option><option value="clip">clips elegidos</option><option value="alt">alternativos renderizados</option><option value="candidate">candidatos sin renderizar</option><option value="pending">solo pendientes</option></select></label>
  <label>Modelo <select id="fmodel"></select></label>
  <label><input type="checkbox" id="blind"> ocultar modelo y puntuación</label>
  <span class="sum">Atajos: 1 bueno · 2 edición · 3 no sirve · 0 limpiar · j/k siguiente/anterior</span>
</header>
<main id="list"></main>
<script>
"use strict";
let S = null, cur = 0;
const $ = (id) => document.getElementById(id);
function el(tag, props, ...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(props||{})){if(v==null||v===false)continue;if(k==="class")e.className=v;else if(k.startsWith("on"))e.addEventListener(k.slice(2),v);else e.setAttribute(k,v);}for(const c of kids.flat()){if(c==null)continue;e.append(c instanceof Node?c:document.createTextNode(String(c)));}return e;}
function mmss(t){const m=Math.floor(t/60),s=(t%60).toFixed(1).padStart(4,"0");return m+":"+s;}
function load(k,d){try{return localStorage.getItem(k)??d}catch(e){return d}}
function save(k,v){try{localStorage.setItem(k,v)}catch(e){}}
async function api(url,opts){const r=await fetch(url,opts);if(!r.ok)throw new Error((await r.json().catch(()=>({}))).error||r.status);return r.json();}
async function init(){
  S = await api("/api/state");
  const fm=$("fmodel");fm.append(el("option",{value:"all"},"todos"));
  for(const m of [...new Set(S.jobs.map(j=>j.model))])fm.append(el("option",{value:m},"Whisper "+m));
  $("blind").checked = load("blind","0")==="1";
  for(const id of ["fkind","fmodel","blind"])$(id).addEventListener("change",()=>{save("blind",$("blind").checked?"1":"0");render();});
  render();
}
function visible(){
  const k=$("fkind").value,m=$("fmodel").value;
  return S.items.filter(i=>(m==="all"||i.model===m)&&(k==="all"||(k==="pending"?!(S.ratings[i.id]&&S.ratings[i.id].rating):i.kind===k)));
}
function summary(){
  const c={good:0,edit:0,bad:0};let n=0;
  for(const i of S.items){const r=S.ratings[i.id];if(r&&r.rating){c[r.rating]++;n++;}}
  $("sum").textContent=`${n} de ${S.items.length} evaluados · ${S.labels.good}: ${c.good} · ${S.labels.edit}: ${c.edit} · ${S.labels.bad}: ${c.bad}`;
}
async function rate(item,rating,note){
  const r=await api("/api/rating",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:item.id,rating,note})});
  if(r.rating)S.ratings[item.id]=r.rating;else delete S.ratings[item.id];
  summary();render(true);
}
function render(keepScroll){
  const y=window.scrollY;const list=$("list");list.replaceChildren();
  const blind=$("blind").checked,items=visible();summary();
  if(!items.length){list.append(el("div",{class:"empty"},"No hay elementos para este filtro."));return;}
  items.forEach((it,idx)=>list.append(card(it,idx,blind,items)));
  if(keepScroll)window.scrollTo(0,y);
}
function card(it,idx,blind,items){
  const rt=S.ratings[it.id]||{};
  const kind={clip:"clip elegido",alt:"alternativo renderizado",candidate:"candidato sin renderizar"}[it.kind];
  let video;
  if(it.kind==="candidate"){
    video=el("video",{controls:"",preload:"metadata",src:`/source/${it.job_index}#t=${it.start},${it.end}`});
    video.addEventListener("timeupdate",()=>{if(video.currentTime>=it.end){video.pause();video.currentTime=it.start;}});
  } else if(it.media_exists===false){
    video=el("div",{class:"empty"},"El MP4 ya no está en la carpeta del job.");
  } else {
    video=el("video",{controls:"",preload:"metadata",src:`/media/${it.job_index}/${it.media}`});
  }
  const buttons=Object.entries(S.labels).map(([k,label])=>el("button",{class:`${k} ${rt.rating===k?"on":""}`,onclick:()=>rate(it,rt.rating===k?null:k,note.value)},label));
  buttons.push(el("button",{class:"clear",onclick:()=>{note.value="";rate(it,null,"");}},"Limpiar"));
  const note=el("textarea",{placeholder:"Notas (opcional): qué cambiarías, por qué no sirve…"});note.value=rt.note||"";
  note.addEventListener("change",()=>rate(it,rt.rating||null,note.value));
  const c=it.confidence,sub=it.subtitles,a=it.audio;
  const meta=[`${mmss(it.start)}–${mmss(it.end)} (${it.duration} s)`];
  if(!blind){meta.push(`Whisper ${it.model}`,`#${it.rank}`,`puntuación ${it.score}${it.below_min_score?" (bajo el umbral)":""}`);}
  const body=el("div",{},
    el("h2",{},`${it.name} · ${kind}`),el("p",{class:"meta"},meta.join(" · ")),
    el("div",{class:"rate"},buttons),note,
    el("details",{open:""},el("summary",{},"Transcripción"),
      el("div",{class:"tx"},it.transcript||"(sin habla transcrita)"),
      el("div",{class:"ev"},`Confianza: ${c.words} palabras, prob. media ${c.mean_word_prob}, ${c.words_below_0_4_pct} % con prob < 0.4`+(c.flags?` · Whisper: ${JSON.stringify(c.flags)}`:" · sin estadísticas de Whisper")),
      c.low_confidence_words.length?el("div",{class:"ev"},"Baja confianza: "+c.low_confidence_words.join(", ")):null),
    sub?el("details",{open:sub.issues&&sub.issues.length?"":null},el("summary",{},`Subtítulos${sub.issues&&sub.issues.length?" ("+sub.issues.length+" avisos)":""}`),
      sub.available?el("div",{class:"ev"},`${sub.lines} líneas, cobertura ${sub.coverage_pct} %`+(sub.sync&&sub.sync.available?`, desfase estimado ${sub.sync.lag_s>0?"+":""}${sub.sync.lag_s} s (+ = tardíos)`:"")):null,
      ...(sub.issues||[]).map(x=>el("div",{class:"iss"},"⚠ "+x))):null,
    el("details",{},el("summary",{},"Audio y motivos de selección"),
      a&&a.anchor_in_clip_s!==undefined?el("div",{class:"ev"},`Ancla a ${a.anchor_in_clip_s} s del inicio · señales: ${(a.triggers||[]).join(", ")}`):null,
      a&&a.intensity?el("div",{class:"ev"},a.intensity.note):null,a&&a.dynamics?el("div",{class:"ev"},a.dynamics.note):null,
      a&&a.verbal?el("div",{class:"ev"},"Reacción verbal: "+a.verbal.note+(a.no_accepted_verbal_reaction?" (sin reacción verbal aceptada)":"")):null,
      ...(blind?[]:it.components.map(x=>el("div",{class:"ev"},`${x.name} ${x.value} × ${x.weight} = ${x.contribution} — ${x.note}`))),
      ...(blind?[]:it.components.flatMap(x=>x.evidence.map(e=>el("div",{class:"note"},`${x.name}: ${e}`))))));
  const node=el("section",{class:"card"+(idx===cur?" cur":""),onclick:()=>{cur=idx;}} ,el("div",{},video),body);
  node._item=it;return node;
}
document.addEventListener("keydown",(e)=>{
  if(!S||["TEXTAREA","SELECT","INPUT"].includes(document.activeElement.tagName))return;
  const cards=[...document.querySelectorAll(".card")];if(!cards.length)return;
  if(e.key==="j"||e.key==="k"){cur=Math.max(0,Math.min(cards.length-1,cur+(e.key==="j"?1:-1)));cards.forEach((c,i)=>c.classList.toggle("cur",i===cur));cards[cur].scrollIntoView({block:"center"});return;}
  const map={"1":"good","2":"edit","3":"bad","0":null};
  if(e.key in map){const it=cards[cur]._item;const note=cards[cur].querySelector("textarea").value;rate(it,map[e.key],note);}
});
init().catch(e=>{$("list").append(el("div",{class:"empty"},"No se pudo cargar: "+e.message));});
</script>
</body>
</html>
"""

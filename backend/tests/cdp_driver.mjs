// Conductor mínimo de Chrome real para las pruebas de interfaz: abre una URL, ejecuta pasos (JS + espera) y devuelve el resultado.
// Uso: node cdp_driver.mjs <chrome> <spec.json>   con spec = { url, steps: [{ js, wait }], read }
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [, , chromePath, specPath] = process.argv;
const spec = JSON.parse(readFileSync(specPath, "utf-8"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const dbg = 9600 + Math.floor(Math.random() * 300);
const prof = mkdtempSync(join(tmpdir(), "sc-cdp-"));
const chrome = spawn(chromePath, [`--remote-debugging-port=${dbg}`, "--headless=new", "--disable-gpu", `--user-data-dir=${prof}`, "--no-first-run", "--window-size=1200,1400", "about:blank"], { stdio: "ignore" });
let ws, nextId = 1;
const pending = new Map();
const send = (method, params = {}) => new Promise((res) => { const id = nextId++; pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
const evalJs = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 300));
  return r.result.result.value;
};
const waitFor = async (expr, ms = 30000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { try { if (await evalJs(expr)) return true; } catch (e) { /* DOM aún no listo */ } await sleep(400); }
  throw new Error("timeout esperando: " + expr);
};
try {
  for (let i = 0; i < 60 && !ws; i++) {
    try {
      const page = (await (await fetch(`http://127.0.0.1:${dbg}/json/list`)).json()).find((t) => t.type === "page");
      if (page) { ws = new WebSocket(page.webSocketDebuggerUrl); await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; }); }
    } catch (e) { await sleep(300); }
  }
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
  await send("Page.enable");
  await send("Page.navigate", { url: spec.url });
  await sleep(1500);
  for (const st of spec.steps || []) {
    if (st.js) await evalJs(st.js);
    if (st.wait) await waitFor(st.wait);
  }
  console.log(JSON.stringify({ ok: true, value: await evalJs(spec.read) }));
} catch (e) {
  console.log(JSON.stringify({ ok: false, error: String(e.message) }));
} finally {
  try { ws.close(); } catch (e) { /* nada */ }
  chrome.kill();
  try { rmSync(prof, { recursive: true, force: true }); } catch (e) { /* nada */ }
}

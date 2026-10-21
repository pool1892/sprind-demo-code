// The scripts `fleetlab.audit.isolate` runs in Codex's `exec` isolate.
// Sections start with a line "// ==== name"; @@NAME@@ is filled in per run.
// Every attempt ends as "absent" (nothing came back), "refused: <message>" or
// "RETURNED: <what>". Only the last is a reach. The login path is answered
// with "OPENED" or "not opened", never with content. The lists of what is tried
// (@@GLOBALS@@, @@PATHS@@, @@TOOLS@@) are those of isolate.py, which expects one
// report for each entry.

// ==== globals
const names = new Set();
let o = globalThis;
while (o) {
  for (const n of Reflect.ownKeys(o)) names.add(String(n));
  o = Object.getPrototypeOf(o);
}
const out = {};
for (const n of [...names].sort()) {
  let t;
  try { t = typeof globalThis[n]; } catch (e) { t = "throws"; }
  out[n] = t;
}
text(JSON.stringify(out));

// ==== reach
const out = {};
function shown(v) {
  try { return (typeof v === "string" ? v : JSON.stringify(v)) + ""; }
  catch (e) { return String(v); }
}
async function attempt(name, f) {
  try {
    const v = await f();
    out[name] = v === undefined ? "absent" : "RETURNED: " + shown(v).slice(0, 160);
  } catch (e) {
    out[name] = "refused: " + String(e).slice(0, 120);
  }
}
async function opened(name, f) {
  try { const v = await f(); out[name] = v === undefined ? "not opened" : "OPENED"; }
  catch (e) { out[name] = "not opened"; }
}
const globals = @@GLOBALS@@;
for (const n of globals) {
  await attempt("global " + n, async () =>
    typeof globalThis[n] === "undefined" ? undefined : typeof globalThis[n]);
}
const paths = @@PATHS@@;
for (const p of paths) {
  await attempt("file by import " + p, async () => Object.keys(await import(p)));
  await attempt("file by load " + p, async () => load(p));
  await attempt("file by image " + p, async () => { image(p); return "accepted"; });
  await attempt("file by audio " + p, async () => { audio(p); return "accepted"; });
  await attempt("file by generatedImage " + p, async () => {
    generatedImage({ image_url: p });
    return "accepted";
  });
}
await opened("login by import", async () => Object.keys(await import("@@LOGIN@@")));
await opened("login by import as a file url", async () =>
  Object.keys(await import("file://@@LOGIN@@")));
await opened("login by load", async () => load("@@LOGIN@@"));
await opened("login by image", async () => { image("@@LOGIN@@"); return 1; });
await opened("login by audio", async () => { audio("@@LOGIN@@"); return 1; });
await opened("login by generatedImage", async () => {
  generatedImage({ image_url: "@@LOGIN@@" });
  return 1;
});
await attempt("module fs", async () => Object.keys(await import("fs")));
await attempt("module node:fs", async () => Object.keys(await import("node:fs")));
await attempt("module data url", async () =>
  Object.keys(await import("data:text/javascript,export const a=1")));
const url = "http://127.0.0.1:@@PORT@@/@@CONNECT@@/";
await attempt("connection by import", async () => Object.keys(await import(url + "m.js")));
await attempt("connection by image", async () => { image(url + "a.png"); return "accepted"; });
await attempt("connection by audio", async () => { audio(url + "a.wav"); return "accepted"; });
await attempt("connection by generatedImage", async () => {
  generatedImage({ image_url: url + "g.png" });
  return "accepted";
});
await attempt("environment by import.meta", async () => {
  const k = Object.keys(import.meta);
  return k.length ? k : undefined;
});
await attempt("environment by Function", async () => {
  const g = Function("return this")();
  return g === globalThis ? undefined : Object.getOwnPropertyNames(g);
});
await attempt("tools listed", async () => {
  const k = Reflect.ownKeys(tools).map(String);
  return k.length ? k : undefined;
});
await attempt("tools in ALL_TOOLS", async () =>
  ALL_TOOLS.length ? ALL_TOOLS.map(t => t.name) : undefined);
const tool_names = @@TOOLS@@;
for (const n of tool_names) {
  await attempt("tool " + n, async () => {
    const f = tools[n];
    if (typeof f !== "function") return undefined;
    return await f({
      sql: "SELECT 1", cmd: "cat /proc/self/mountinfo", path: "@@CWD@@/@@FILE@@", part: "fleet",
      command: ["cat", "/proc/self/mountinfo"],
    });
  });
}
out["info clock"] = String(Date.now());
out["info time zone"] = shown(Intl.DateTimeFormat().resolvedOptions());
out["info stack"] = String(new Error("x").stack).split("\n").slice(0, 2).join(" | ");
text(JSON.stringify(out));

// ==== store
const before = load("isolate-probe");
store("isolate-probe", "stored-@@MARK@@");
text(JSON.stringify({
  before: before === undefined ? "absent" : String(before),
  after: String(load("isolate-probe")),
}));

// ==== recall
const v = load("isolate-probe");
text(JSON.stringify({ seen_by_a_later_call: v === undefined ? "absent" : String(v) }));

// ==== load
const v = load("isolate-probe");
text(JSON.stringify({ seen_by_another_process: v === undefined ? "absent" : String(v) }));

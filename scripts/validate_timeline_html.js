// Headless smoke test for output/run_timeline.html -- the JS half of
// scripts/validate_timeline.py.  Extracts the page's own <script> and runs it in
// a vm context with a stubbed DOM + canvas, drives the interactions, and checks
// the data logic.  Everything asserted here is DERIVED FROM THE PAYLOAD, never
// hard-coded, so changing config/site.yaml (cases, worker count) does not break
// the test.
//
// Exit code 0 = all checks passed, 1 = a check failed or the page would not boot.
//
// What only this file can catch (build_run_timeline.py --check cannot):
//   * a JS syntax error or a throw during init()  -> the page is dead
//   * ancestry()/childrenOf() walking a parent cycle -> the browser hangs
//   * a filter that leaks spans it should have hidden
//   * pick() returning an out-of-range span index
//   * self_s + direct-children wall != wall (the profiler's time accounting)

const fs = require("fs");
const vm = require("vm");

const HTML = process.argv[2] || "output/run_timeline.html";
const html = fs.readFileSync(HTML, "utf8");

const payloadM = html.match(
  /<script id="payload" type="application\/json">([\s\S]*?)<\/script>/);
const scripts = [...html.matchAll(/<script(?![^>]*id="payload")[^>]*>([\s\S]*?)<\/script>/g)];
if (!payloadM) { console.error("no <script id=\"payload\"> block in " + HTML); process.exit(1); }
if (scripts.length !== 1) {
  console.error("expected exactly 1 app <script>, got " + scripts.length);
  process.exit(1);
}
const appJs = scripts[0][1];

// A concrete 2d context. Do NOT swap this for a Proxy: every property access
// then allocates a closure, which dominated the runtime of this suite.
const NOOP = () => {};
const counter = { r: 0, clip: 0, save: 0, restore: 0, fill: 0, stroke: 0 };
/* The GEOMETRY of every fillRect, not just how many.  A selection wash is a single
 * call, so counting calls cannot see it: one plot-height-wide rect and no wash are
 * indistinguishable by count.  Recording the rects is what lets a check say "the
 * annotation must not paint anything taller than a cap". */
const rects = [];
/* every draw call the range can make, so a check can assert the WHOLE annotation is
 * absent when the range is off-axis rather than only its fillRects - the range
 * draws with fillRect, stroke() and fill(), and counting one kind would have let the
 * other two through untested. */
const drawn = () => counter.r + counter.fill + counter.stroke;
const ctx2d = {
  canvas: null,
  fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "",
  textBaseline: "", globalAlpha: 1, lineCap: "", lineJoin: "", shadowBlur: 0,
  /* The GEOMETRY and the ALPHA of every fillRect, not just how many.  Two things
   * are only observable here: a selection wash is a single call, so counting calls
   * cannot see it, and the held-back bars are told apart from the full-contrast
   * ones purely by the globalAlpha in force at the moment of the call - they are
   * the same rects, in the same place, the same number of times. */
  fillRect: (x, y, w, h) => { counter.r++; rects.push([x, y, w, h, ctx2d.globalAlpha]); },
  strokeRect: NOOP, clearRect: NOOP, fillText: NOOP, strokeText: NOOP,
  beginPath: NOOP, closePath: NOOP, moveTo: NOOP, lineTo: NOOP,
  arc: NOOP, ellipse: NOOP, rect: NOOP, roundRect: NOOP,
  quadraticCurveTo: NOOP, bezierCurveTo: NOOP, arcTo: NOOP,
  fill: () => { counter.fill++; }, stroke: () => { counter.stroke++; },
  /* save/restore/clip are counted so a check can prove the range is FENCED to the
   * plot.  A clip is invisible in the fillRect count - the page still issues the
   * call with its full, unclipped coordinates and the canvas does the trimming - so
   * "the wash was clipped" is only observable here.  The counts must stay in step:
   * an unbalanced save/restore would leak state into every later draw. */
  save: () => { counter.save++; }, restore: () => { counter.restore++; },
  translate: NOOP, scale: NOOP, rotate: NOOP, setTransform: NOOP,
  transform: NOOP, clip: () => { counter.clip++; }, setLineDash: NOOP, getLineDash: () => [],
  measureText: (s) => ({ width: String(s).length * 6.2, actualBoundingBoxAscent: 8 }),
  createLinearGradient: () => ({ addColorStop: NOOP }),
  createPattern: () => null,
  putImageData: NOOP, drawImage: NOOP,
};

// Elements are memoised per id, so a test can write .innerHTML on one lookup and
// read it back on the next -- exactly how the page itself behaves.
//
// addEventListener is REAL here, not a no-op.  The page's zoom/drag handling
// lives entirely in window/canvas listeners (pointerdown on the canvas,
// pointermove/pointerup on the window, blur on the window), so with a stubbed
// DOM those handlers were never even registered and the whole drag path went
// untested -- which is how a rubber band that survived a lost release could pass
// 27 checks.  Which is why a window/canvas listener registry is needed to test
// any of it.
const byId = new Map();
const listeners = new Map();
/* what the page did to pointer capture, and the timers it scheduled.  Both are
 * observable because both are load-bearing: capture is what makes a release
 * outside the window arrive, and the watchdog timer is the only thing that can
 * clear a band when the browser delivers no event at all. */
const capture = { log: [], held: null };
const timers = { pending: new Map(), seq: 1 };
function on(target, type, fn) {
  const k = target + ":" + type;
  if (!listeners.has(k)) listeners.set(k, []);
  listeners.get(k).push(fn);
}
function fire(target, type, ev) {
  const fns = listeners.get(target + ":" + type) || [];
  for (const f of fns) f(ev || {});
  return fns.length;
}
/* Pointer events, and only pointer events.  The page listens for pointerdown /
 * pointermove / pointerup / pointercancel, so a check that fired "mousedown"
 * would quietly arm nothing and still "pass" -- the vacuous-pass trap this suite
 * already fell into once.  `mouse()` is kept as an alias so the older checks
 * read the same, and `buttons` defaults to 1 (button held) because every
 * pointermove during a real drag reports it. */
const mouse = (x, y, buttons) => ({
  clientX: x, clientY: y, buttons: buttons === undefined ? 1 : buttons,
  button: 0, pointerId: 1, pointerType: "mouse", isPrimary: true,
  preventDefault() {}, stopPropagation() {},
});
const pointer = mouse;
/* CONTENT pixel -> CLIENT pixel, for aiming a pointer at something a check found
 * with pick(), hitBoxes or tToPx().  `getBoundingClientRect().left` is
 * `-scrollLeft` in the stub, so a content pixel X sits at client X - scrollLeft.
 *
 * This is the FOURTH time this file has been bitten by aiming at the wrong
 * coordinate system, and the reason it went unnoticed for so long is that the two
 * numbers used to be equal: every check scrolled back to 0 first, so content ==
 * client.  The drag now magnifies, and magnifying centres the range, so the
 * scroll offset is no longer 0 after a drag - and a check that fired a pointer at
 * a content coordinate silently clicked somewhere else entirely.  It reported
 * "ok" for the wrong reason: the click that was supposed to land on empty plot
 * space landed on a bar instead, so the range survived, and the check blamed the
 * page.  Every pointer aimed at a discovered pixel goes through here. */
const clientX = (contentX) => contentX - scrolled("scroller").left;

/* The scroll model.  The page puts the canvas in an `overflow:auto` pane and
 * sizes the canvas to the VIRTUAL extent, so a check that aims at a pixel has to
 * be able to scroll first.  This is the third time this file has been bitten by a
 * stub that could not express what the page actually does: `getBoundingClientRect`
 * was a CONSTANT {left:0, top:0}, so the moment the pane scrolled, every
 * pointer-derived check would still have aimed at the old content coordinates and
 * reported "ok" while testing nothing.  So the rect is derived from the scroll
 * offsets, and the offsets clamp exactly as a real box does. */
const VIEW_W = 1400, VIEW_H = 620;
const scroll = new Map();               // id -> {left, top}
function scrolled(id) {
  if (!scroll.has(id)) scroll.set(id, { left: 0, top: 0 });
  return scroll.get(id);
}
/* Dispatch to the page's own `scroll` listener, on the element that scrolled.
 * Synchronous rather than deferred: a real browser fires it before the next
 * frame, so firing it now cannot make a check pass that would otherwise fail -
 * it only removes a race. */
function fireScroll() {
  const fns = listeners.get("el:scroller:scroll") || [];
  for (const f of fns) f({ target: byId.get("scroller") });
  return fns.length;
}
function El(id) {
  if (id && byId.has(id)) return byId.get(id);
  const key = id || "el";
  const e = {
    id: key, textContent: "", innerHTML: "", value: "", checked: false,
    title: "", href: "", download: "", dataset: {},
    // a real width matters: draw() derives every pixel from it, and with
    // clientWidth undefined the plot geometry is NaN, which would make any
    // pixel-position test (drag zoom, gutter clicks) vacuous
    clientWidth: VIEW_W, clientHeight: VIEW_H,
    width: VIEW_W, height: VIEW_H,
    style: {},
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener(type, fn) { on("el:" + key, type, fn); },
    removeEventListener() {}, appendChild() {},
    setAttribute() {}, getAttribute: () => null, click() {},
    /* the canvas's own box, in viewport coordinates: scrolling the pane moves the
     * canvas against the window, which is why `e.clientX - rect.left` is a
     * CONTENT coordinate and needs no scroll term on the page side */
    get scrollLeft() { return scrolled(key).left; },
    set scrollLeft(v) {
      const s = scrolled(key);
      const max = Math.max(0, e.scrollWidth - e.clientWidth);
      const next = Math.min(max, Math.max(0, Number(v) || 0));
      if (next === s.left) return;        // unchanged: a real scroll event is not fired
      s.left = next;
      /* A browser fires `scroll` when scrollTop/scrollLeft is assigned.  The stub
       * has to, or the page's S.sx/S.sy would keep the value it last set itself
       * and the sticky gutter would be pinned to a position the pane is not at -
       * which is precisely what the scroll checks below exist to catch. */
      if (key === "scroller") fireScroll();
    },
    get scrollTop() { return scrolled(key).top; },
    set scrollTop(v) {
      const s = scrolled(key);
      const max = Math.max(0, e.scrollHeight - e.clientHeight);
      const next = Math.min(max, Math.max(0, Number(v) || 0));
      if (next === s.top) return;
      s.top = next;
      if (key === "scroller") fireScroll();
    },
    getBoundingClientRect() {
      /* The CANVAS IS A CHILD OF THE SCROLL PANE, so its box moves with the pane
       * and not with any scroll of its own - and the canvas never scrolls itself.
       * Giving it its own entry made `rect.left` a permanent 0, which quietly
       * redefined what the page was being asked: the page computes
       * `e.clientX - rect.left`, and against a frozen 0 that is just `clientX`, so
       * every pointer-driven check was aiming at what it THOUGHT were content
       * coordinates while the page read them as viewport ones.  The two agreed only
       * while the pane sat at 0 - which is why the mistake survived: it surfaced the
       * moment the drag started centring the range and moving the scroll offset. */
      const own = scrolled(key);
      const pane = (key === "cv") ? scrolled("scroller") : own;
      return { left: -pane.left, top: -pane.top, width: e.width, height: e.height,
               right: e.width - pane.left, bottom: e.height - pane.top };
    },
    querySelector: () => El("q"), querySelectorAll: () => [],
    getContext: () => ctx2d,
    /* the page captures the pointer on pointerdown, so the stub has to be able
     * to record it -- otherwise the "does it ask for capture" check would be
     * vacuous, and a page that stopped capturing would still pass */
    setPointerCapture(id) { capture.log.push(["set", id]); },
    releasePointerCapture(id) { capture.log.push(["release", id]); },
    hasPointerCapture() { return capture.log.some(([k, i]) => k === "set" && i === capture.held); },
  };
  if (key === "scroller"){
    /* #scroller is the scroll viewport and its content is exactly the canvas, so
     * its scroll extent IS the canvas size.  Without this the offsets would clamp
     * against width==clientWidth (always 0 of range) and every scroll check would
     * silently pass on a pane that could not move.  Its client box follows the
     * height/width the page set on style, for the same reason.
     *
     * The extent must come from the canvas's CSS **layout box** (`style.width`),
     * NOT from `cv.width`.  A browser scrolls the layout box; `cv.width` is the
     * backing store, which is `round(w * dpr)` and therefore a dpr multiple of it.
     * Reading the backing store made the stub model a scroll range dpr times too
     * wide, so every offset assertion passed on a display it had never actually
     * modelled - Windows 125%/150% being the common case that was silently
     * untested. */
    const dpr = () => sandbox.window.devicePixelRatio || 1;
    const layoutW = () => { const cv = El("cv");
      return cv.style.width ? (parseFloat(cv.style.width) || 0) : cv.width / dpr(); };
    const layoutH = () => { const cv = El("cv");
      return cv.style.height ? (parseFloat(cv.style.height) || 0) : cv.height / dpr(); };
    Object.defineProperties(e, {
      clientWidth:  { get: () => (e.style.width ? parseInt(e.style.width, 10) || VIEW_W : VIEW_W) },
      clientHeight: { get: () => (e.style.height ? parseInt(e.style.height, 10) || VIEW_H : VIEW_H) },
      scrollWidth:  { get: () => layoutW() },
      scrollHeight: { get: () => layoutH() },
    });
  }
  if (id) byId.set(id, e);
  return e;
}

const payloadEl = El("payload");
payloadEl.textContent = payloadM[1];

const sandbox = {
  document: {
    getElementById: (id) => (id === "payload" ? payloadEl : El(id)),
    querySelector: () => El("q"),
    querySelectorAll: () => [],
    createElement: () => El("new"),
    addEventListener(type, fn) { on("document", type, fn); },
  },
  /* innerWidth/innerHeight belong on `window` as well as at the top level: the
   * page reads `window.innerHeight` to cap the pane at 70vh, while the off-window
   * watchdog reads the bare global.  With only the global defined, the page saw
   * undefined, fell back to its 900 default and sized the pane to 630 - which
   * silently changed every vertical clamp. */
  window: { addEventListener(type, fn) { on("window", type, fn); }, devicePixelRatio: 1,
            innerWidth: 1400, innerHeight: 620 },
  requestAnimationFrame: (f) => f(0),
  /* a real timer queue, NOT `f => f(0)`: the page arms a watchdog that has to
   * stay pending while the button is down, and an immediate-call stub would
   * cancel every drag the instant it started, so every drag check would pass
   * for the wrong reason.  `tick()` fires whatever is due. */
  setTimeout: (f) => { const id = timers.seq++; timers.pending.set(id, f); return id; },
  clearTimeout: (id) => { timers.pending.delete(id); },
  URL: { createObjectURL: () => "blob:stub" },
  Blob: function Blob(parts) { this.parts = parts; },
  performance: { now: () => 0 },
  Intl, JSON, Math, Date, console, Set, Map, Array, Object, Number, String,
  // harness handles, reachable from the check strings (which are evaluated
  // INSIDE the vm, so Node scope is not visible to them)
  fire, mouse, pointer, counter, capture, clientX,
  /* the page source, so a check can assert on the real markup: the stubbed DOM
   * has no parentNode, so "is the canvas actually inside the scroll pane" can
   * only be answered by looking at the HTML */
  html,
  /* pane control for the scroll model above: RESET must zero it, or a check that
   * scrolled would leak that scroll into every later check's coordinates */
  zeroScroll: () => { for (const [, s] of scroll) { s.left = 0; s.top = 0; } return scroll.size; },
  setScroll: (id, l, t) => {
    const e = El(id); e.scrollLeft = l; e.scrollTop = t;
    return [e.scrollLeft, e.scrollTop];
  },
  getScroll: (id) => { const e = El(id); return [e.scrollLeft, e.scrollTop]; },
  /* the page's off-window watchdog asks how big the window is; without these the
   * guard is silently false and the watchdog never arms */
  innerWidth: 1400, innerHeight: 620,
  /* fire every pending timer, as if its delay had elapsed */
  tickTimers: () => { const due = [...timers.pending.entries()]; timers.pending.clear();
                       for (const [, f] of due) f(); return due.length; },
  pendingTimers: () => timers.pending.size,
  /* check bodies are run with vm.runInContext, so they see ONLY what is on the
   * sandbox - a module-level helper has to be handed over explicitly or the check
   * throws "not defined" and reports a harness error, not a page fault. */
  drawnCalls: () => drawn(),
  pageCtx: () => ctx2d,
  rectLog: () => rects,
};
sandbox.globalThis = sandbox;
sandbox.window.document = sandbox.document;

const ctx = vm.createContext(sandbox);
let pass = 0, fail = 0;

const tBoot = Date.now();
let boot = "ok";
try {
  vm.runInContext(appJs, ctx, { filename: "run_timeline.js" });
} catch (e) {
  boot = "THREW: " + e.message;
}
console.log("boot: " + boot + "  (" + (Date.now() - tBoot) + " ms, " +
            counter.r + " fillRect calls during init)");
if (boot !== "ok") {
  console.log("\npass 0  fail 1");
  process.exit(1);
}

/* The state init() hands the page.  Captured here, at boot, because every later
 * check that touches the mode leaks into the next one - asking "does it open on
 * the merged view?" after a focus test just reads that test's leftovers.  The
 * button's hidden state is deliberately NOT in here: the stub has no HTML
 * attribute, so `.hidden` is undefined until the page writes it.  The merged
 * button is asserted by the focus round-trip check instead. */
const OPENED = vm.runInContext(
  'JSON.stringify({mode: S.mode, nRows: rows().length, ' +
  'labels: rows().map(function(r){return r.label}).join(",")})', ctx);
sandbox.OPENED_JSON = OPENED;      // readable by later checks

function chk(name, code, want) {
  const t = Date.now();
  let got;
  try {
    got = vm.runInContext(code, ctx);
  } catch (e) {
    fail++;
    console.log("  FAIL  " + name + "  -> threw: " + e.message);
    return;
  }
  const ok = want === undefined ? !!got : JSON.stringify(got) === JSON.stringify(want);
  const tail = want === undefined ? "" : "  = " + JSON.stringify(got);
  if (ok) { pass++; console.log("  ok    " + name + tail + "   [" + (Date.now() - t) + " ms]"); }
  else { fail++; console.log("  FAIL  " + name + "  got " + JSON.stringify(got) +
                            " want " + JSON.stringify(want)); }
}

// Put the state back the way init() left it before every check.
// RESET pins S.mode="case": the checks below this line were written against the
// single-case view and must keep testing it.  The merged view gets its own
// prelude so the two never leak into each other.  Both also drop a half-finished
// press: a drag left armed by the previous check would paint its band into the
// next check's `draw()` and make the fillRect counts a lie.
const RESET = `S.mode="case"; S.lane="thread"; S.minDur=0; S.query=""; S.hiddenBranch=new Set();
  S.isoThread=null; S.zoom=1; S.sel=null; S.treeRoot=-1; S.rowH=null; S.sx=0; S.sy=0; S.range=null;
  zeroScroll(); drag=null; pan=null; press=null; tickTimers(); buildOrder(); resize(); syncRowHControl();
  syncRangeUI();`;
const RESET_ALL = `S.mode="all"; S.lane="thread"; S.minDur=0; S.query=""; S.hiddenBranch=new Set();
  S.isoThread=null; S.zoom=1; S.sel=null; S.treeRoot=-1; S.rowH=null; S.sx=0; S.sy=0; S.range=null;
  zeroScroll(); drag=null; pan=null; press=null; tickTimers(); buildOrder(); resize(); syncRowHControl();
  syncRangeUI();`;
// first span at nesting depth 2, used by the selection checks
const DEEP = `var d=-1; for (var i=0;i<C().n;i++){ if (C().dp[i]===2){ d=i; break; } }`;

/* -------------------------------------------------------------- payload shape */
chk("payload has at least one case", "DATA.cases.length >= 1", true);
chk("case labels are unique and non-empty", `
  var l = DATA.cases.map(c => c.label);
  l.every(x => typeof x === "string" && x.length > 0) && new Set(l).size === l.length
    ? "ok" : "labels " + JSON.stringify(l)`);
chk("case picker lists every case", `
  var sel = document.getElementById("caseSel").innerHTML;
  DATA.cases.every(c => sel.indexOf(c.label) >= 0) ? "ok" : "picker missing a label"`);
chk("every span array is n long", `
  ${RESET}
  var keys = ["id","pid","st","ph","th","dp","t0","t1","dur","self"];
  var bad = [];
  for (var ci=0; ci<DATA.cases.length; ci++){ S.ci=ci; var c=C();
    keys.forEach(k => { if (c[k].length !== c.n) bad.push(c.label+"."+k); }); }
  bad.length ? bad.join(",") : "ok"`, "ok");
chk("span ids are 0..n-1 (parents indexable)", `
  ${RESET} var bad=0;
  for (var ci=0; ci<DATA.cases.length; ci++){ S.ci=ci; var c=C();
    for (var i=0;i<c.n;i++) if (c.id[i] !== i) bad++; }
  bad===0`, true);

/* ------------------------------------------------------------------- drawing */
chk("draw every case in both lane modes", `
  for (var ci=0; ci<DATA.cases.length; ci++){ ${RESET} S.ci=ci;
    S.lane="thread"; draw(); S.lane="depth"; draw(); }
  "no errors"`, "no errors");
chk("thread row labels equal the payload lane order", `
  ${RESET} S.ci=0;
  var want = C().lane_name.join(","), got = rows().map(r => r.label).join(",");
  got === want ? "ok" : "rows " + got + " != lane_name " + want`, "ok");
chk("depth row count = max_depth + 1", `
  ${RESET} S.ci=0; S.lane="depth"; rows().length === C().max_depth + 1`, true);
chk("rowOf maps into a real row (depth mode)", `
  ${RESET} S.ci=0; S.lane="depth"; var bad=0;
  for (var i=0;i<C().n;i++) if (rows()[rowOf(C(),i)] === undefined) bad++; bad===0`, true);
chk("rowOf maps into a real row (thread mode)", `
  ${RESET} S.ci=0; var bad=0;
  for (var i=0;i<C().n;i++) if (C().lane_name[rowOf(C(),i)] === undefined) bad++; bad===0`, true);

/* ------------------------------------------------------------------- filters */
chk("min-duration filter drops short spans", `
  ${RESET} S.ci=0; S.minDur=50; var bad=0, cnt=0;
  for (var i=0;i<C().n;i++) if (visible(i)){ cnt++; if (C().dur[i] < 50) bad++; }
  bad===0 && cnt>0 ? "ok" : "bad="+bad+" visible="+cnt`, "ok");
chk("search filter matches only the stage name", `
  ${RESET} S.ci=0; S.query="reflect"; var bad=0, cnt=0;
  for (var i=0;i<C().n;i++) if (visible(i)){ cnt++;
    if (C().nm[C().st[i]].toLowerCase().indexOf("reflect") < 0) bad++; }
  bad===0 && cnt>0 ? "ok" : "bad="+bad+" visible="+cnt`, "ok");
chk("hiding every function branch hides every span", `
  ${RESET} S.ci=0; var cm = colorModel(C());
  S.hiddenBranch = new Set(cm.branches.map(b => b.name));
  var bad=0, left=0;
  for (var i=0;i<C().n;i++) if (visible(i)){ bad++; left++; }
  /* nothing may survive, and the set must have covered every branch in the
   * payload - otherwise this would pass on a payload with no branches at all */
  (left===0 && S.hiddenBranch.size === cm.branches.length && cm.branches.length >= 2)
    ? "ok" : bad + " of " + C().n + " visible across " + S.hiddenBranch.size
             + " of " + cm.branches.length + " branches"`, "ok");
chk("isolating a worker keeps it and main only", `
  ${RESET} S.ci=0; S.isoThread = 4; var bad=0, lanes = {};
  for (var i=0;i<C().n;i++) if (visible(i)){ lanes[C().th[i]] = 1;
    if (C().th[i] !== 4 && C().th[i] !== 0) bad++; }
  bad===0 ? "ok" : "leaked lanes " + Object.keys(lanes).join(",")`, "ok");

/* -------------------------------------------------- function colour model */
/* These are the checks that make the palette a CONTRACT rather than a look.
 * The page is self-contained, so nothing else would notice a ramp collapsing to
 * one shade, two branches sharing a hue, or the red stopping meaning "error" -
 * a wrong colour is not a crash, it is a lie the reader believes. */
chk("every span under one sub-function shares one colour", `
  ${RESET} S.ci=0; var cm = colorModel(C()), seen={}, bad=0, groups=0;
  for (var i=0;i<C().n;i++){
    if (cm.bad[i]) continue;
    var k = cm.nameOf(i) + "\\u0000" + cm.keyNameOf(i);
    if (!(k in seen)){ seen[k] = cm.col[i]; groups++; }
    else if (seen[k] !== cm.col[i]) bad++;
  }
  (bad===0 && groups >= 3) ? "ok"
    : bad + " mismatched colours over " + groups + " (branch, sub-function) pairs"`, "ok");
chk("shades inside a branch step apart in lightness", `
  ${RESET} S.ci=0; var cm = colorModel(C()), bad=0, worst=9, pairs=0, steps=[];
  for (var b=0;b<cm.branches.length;b++){
    var r = cm.branches[b].ramp;
    for (var k=1;k<r.length;k++){
      var d = Math.abs(hexL(r[k]) - hexL(r[k-1]));
      pairs++; if (d < worst){ worst = d; steps = [cm.branches[b].name, r[k-1], r[k]]; }
      if (d < RAMP_MIN_STEP - 1e-9) bad++;
    }
  }
  (bad===0 && pairs > 0) ? "ok" : bad + " of " + pairs + " steps under "
    + RAMP_MIN_STEP + " (worst " + worst.toFixed(4) + " in " + steps.join(" ")+")"`, "ok");
chk("no ramp's bright end falls below the target, and none runs off the top", `
  ${RESET_ALL} var cm = colorModel(V()), dark=[], n=0;
  for (var b=0;b<cm.branches.length;b++){
    if (!cm.branches[b].keys.length) continue;
    n++;
    var hi = Math.max.apply(null, cm.branches[b].ramp.map(hexL));
    /* the bright end may RISE above the target - a wide branch needs the room -
     * but it may never fall below it, and never past RAMP_L_MAX.  The cap is
     * compared with the page's own 8-bit tolerance, because hexL measures a
     * quantised colour, not the float that asked for it. */
    if (hi < RAMP_L_HI - QUANT_TOL) dark.push(cm.branches[b].name + " " + hi.toFixed(3));
    if (hi > RAMP_L_MAX + QUANT_TOL) dark.push(cm.branches[b].name + " " + hi.toFixed(3) + " over the cap");
  }
  (dark.length===0 && n>=2) ? "ok" : dark.join(", ") + " of " + n + " ramps"`, "ok");
chk("no function family takes the reserved error red", `
  ${RESET_ALL} var cm = colorModel(V()), tooClose=[], gap=9, clash=0;
  for (var b=0;b<cm.branches.length;b++){
    var g = hueGap(cm.branches[b].hue, ERR_HUE);
    if (g < gap) gap = g;
    if (g < HUE_GUARD) tooClose.push(cm.branches[b].name + "@" + cm.branches[b].hue);
  }
  /* and no span may be painted the error colour without being an error */
  for (var i=0;i<V().n;i++) if (!cm.bad[i] && cm.col[i] === ERR_COLOR) clash++;
  (tooClose.length===0 && clash===0) ? "ok"
    : tooClose.join(",") + " (closest " + gap.toFixed(1) + " deg); "
      + clash + " spans painted the error red"`, "ok");
chk("every assigned colour is readable on the canvas", `
  ${RESET_ALL} var cm = colorModel(V()), seen={}, worst=99, n=0, dark=null;
  var push = function(col){
    if (col in seen) return; seen[col] = 1; n++;
    var r = contrastRatio(col, CANVAS_BG);
    if (r < worst){ worst = r; dark = col; }
  };
  for (var i=0;i<V().n;i++) if (!cm.bad[i]) push(cm.col[i]);
  for (var b=0;b<cm.branches.length;b++)
    for (var k=0;k<cm.branches[b].ramp.length;k++) push(cm.branches[b].ramp[k]);
  push(ERR_COLOR);
  (worst >= 3) ? "ok" : "worst " + worst.toFixed(2) + ":1 (" + dark + ") over " + n + " colours"`, "ok");
chk("the legend lists exactly the branches the payload contains", `
  ${RESET_ALL} renderLegend();
  var cm = colorModel(V()), h = document.getElementById("legend").innerHTML, missing=[];
  for (var b=0;b<cm.branches.length;b++)
    if (h.indexOf(cm.branches[b].name) < 0) missing.push(cm.branches[b].name);
  var chips = (h.match(/class="li[^"]*" data-b="/g) || []).length;
  (missing.length===0 && chips === cm.branches.length)
    ? "ok" : "missing [" + missing.join(",") + "]; " + chips
             + " chips vs " + cm.branches.length + " branches"`, "ok");
chk("the real payload has no error spans, so red stays unused", `
  ${RESET_ALL} var cm = colorModel(V()), why = null;
  for (var i=0;i<V().n && !why;i++) if (cm.bad[i]) why = cm.nameOf(i) + ": " + cm.whyOf(i);
  (cm.badCount === 0) ? "ok" : cm.badCount + " flagged, e.g. " + why`, "ok");
chk("a span that breaks an invariant is red, counted, and unfilterable", `
  (function(){
    ${RESET_ALL}
    var c = V(), cm0 = colorModel(c), pa = c.pidg, i = -1;
    /* corrupt a DRAWN span that has direct children, so the break is
     * self+children==wall - the invariant the profiler closes - and not some
     * other one that would not exercise the accounting path */
    for (var k=0;k<MERGED.n;k++){
      if (MERGED.kind[k] === KIND.NONE || cm0.bad[k]) continue;
      for (var j=0;j<MERGED.n;j++) if (pa[j] === k){ i = k; break; }
      if (i >= 0) break;
    }
    if (i < 0) return "no drawn span with children to corrupt";
    var keep = c.dur[i], branch = cm0.nameOf(i);
    try {
      c.dur[i] = keep + 5000; c._cm = null;          /* force a rebuild */
      var cm = colorModel(c);
      var red    = cm.bad[i] === 1 && cm.col[i] === ERR_COLOR;
      var whyOk  = (cm.whyOf(i) || "").indexOf("self + children") >= 0;
      /* an error is a fact about the data, not a preference: hiding its own
       * branch must not be able to hide it */
      S.hiddenBranch = new Set([branch]);
      var survives = visible(i);
      renderStats();
      var counted = /error spans/.test(document.getElementById("stats").innerHTML)
                 && document.getElementById("stats").innerHTML.indexOf(">1<") >= 0;
      renderLegend();
      var lg = document.getElementById("legend").innerHTML;
      var chip = lg.indexOf('id="errChip"') >= 0 && /class="li err"/.test(lg);
      return (red && whyOk && survives && counted && chip) ? "ok"
        : "red=" + red + " whyOk=" + whyOk + " survives=" + survives
          + " counted=" + counted + " chip=" + chip;
    } finally {
      c.dur[i] = keep; c._cm = null; S.hiddenBranch = new Set(); draw();
    }
  })()`, "ok");
chk("the tooltip and CSV name the function and the error", `
  (function(){
    ${RESET_ALL} draw();
    var cm = colorModel(V()), i = -1;
    for (var k=0;k<MERGED.n;k++) if (MERGED.kind[k] !== KIND.NONE && visible(k)){ i = k; break; }
    if (i < 0) return "no drawn span in the merged view";
    showTip(i, 100, 100);
    var t = document.getElementById("tip").innerHTML;
    var hasFn = t.indexOf(cm.nameOf(i)) >= 0 && t.indexOf(cm.keyNameOf(i)) >= 0;
    var csv = buildCsv();                     // an array of lines, not a string
    var head = csv[0] || "", row = csv[1] || "";
    return (hasFn && /^case,span_id,parent_id,stage,function,error,/.test(head) && row.indexOf(cm.nameOf(i)) >= 0)
      ? "ok" : "tip=" + hasFn + " head=" + head + " row=" + row.slice(0, 90);
  })()`, "ok");

/* ---------------------------------------------------------- zoom and select */
/* The axis invariant these tests exist to protect: zoom is a SCALE on the whole
 * axis, so `tToPx(sp.t1) - tToPx(sp.t0)` must equal the content plot width at
 * every zoom.  If that ever stops holding, some part of the run has become
 * unreachable and no other check would notice - the scroll checks only ask
 * whether the pane can move. */
chk("zoom is a scale: the whole axis is always mapped across the content", `
  (function(){
    ${RESET_ALL}
    /* V(), not C(): in the merged view the axis belongs to the whole sweep,
     * while C() is still the case at S.ci - a slice about 1/6.7 of it, so asking
     * tToPx to span C() measures a fraction of the axis and looks like a mapping
     * bug that is not there. */
    var bad = [], sp = caseSpan(V());
    [1, 2, 4, 8, 64, 200].forEach(function(z){
      setZoom(z, (sp.t0 + sp.t1)/2);
      var spanPx = tToPx(sp.t1) - tToPx(sp.t0);
      if (Math.abs(spanPx - plotWPx()) > 0.5)
        bad.push("z" + z + ": axis occupies " + spanPx.toFixed(2) + " px of a "
                 + plotWPx().toFixed(2) + " px plot");
      if (tToPx(sp.t0) < geom.padL - 0.5)
        bad.push("z" + z + ": axis starts before the plot");
    });
    setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
/* The report that prompted the whole change, written as the reader would do it:
 * sweep to a band over case 2, then scroll sideways and check case 5 is still
 * there.  Under the old window model the answer was no - the drag had moved
 * `S.view` to the dragged range and the rest of the sweep was simply not on the
 * axis. */
chk("zoom in on one case, then scroll on to another", `
  (function(){
    ${RESET_ALL}
    var c2 = DATA.cases[1], c5 = DATA.cases[4];
    if (!c2 || !c5) return "needs at least 5 cases";
    var pane = document.getElementById("scroller");
    /* sweep a band over case 2's slice, exactly as a drag would.  The pixel
     * positions have to come from the PRE-drag mapping and the answers from the
     * POST-drag one, because the drag resizes the canvas - reading geom before
     * the drag and its W after it reported the pre-drag width and made a correct
     * page look like it had thrown case 5 away. */
    var x0 = tToPx(c2.t_off_ms), x1 = tToPx(c2.t_off_ms + c2.dur_ms);
    /* the expected times, taken from the mapping as it is BEFORE the drag.  Asking
     * pxToT for them afterwards would compare the range against a mapping that the
     * zoom has already changed, and blame the range for having moved. */
    var t0 = pxToT(x0 + 12), t1 = pxToT(x1 - 12);
    fire("el:cv", "pointerdown", pointer(x0 + 12, 300, 1));
    fire("window", "pointermove", pointer(x1 - 12, 300, 1));
    fire("window", "pointerup", pointer(x1 - 12, 300, 1));
    if (!S.range) return "the drag selected no range";
    /* the drag itself is what magnifies now, so there is no second call here to make:
     * asking the page to zoom as well would let this check pass while the GESTURE
     * was broken, which is the one thing under test */
    if (S.zoom <= 1.01) return "the drag did not magnify (zoom " + S.zoom.toFixed(3) + ")";
    var g = geom;
    /* case 5 must still be ON the content, not merely somewhere off the end */
    var x5 = tToPx(c5.t_off_ms), x5end = tToPx(c5.t_off_ms + c5.dur_ms);
    if (!(x5 > g.padL && x5end < g.W))
      return "case 5 left the content: " + x5.toFixed(0) + ".." + x5end.toFixed(0)
             + " of 0.." + g.W.toFixed(0);
    /* and scrolling must be able to BRING it into the pane.  Asking for the whole
     * of case 5 to be visible at once would be the wrong requirement - at a deep
     * zoom no two cases fit on screen, which is the point of zooming - so the test
     * is the gesture instead: scroll the band under the pointer and confirm the
     * target lands in the pane. */
    var midPx = tToPx(c5.t_off_ms + c5.dur_ms/2);
    pane.scrollLeft = midPx - paneW()/2; syncScroll();
    var vis = pxToT(S.sx), visEnd = pxToT(S.sx + paneW());
    var midT = c5.t_off_ms + c5.dur_ms/2;
    if (!(vis <= midT && visEnd >= midT))
      return "scrolled to " + fmtT(vis) + ".." + fmtT(visEnd)
             + " but case 5's middle is at " + fmtT(midT);
    /* and the scroll must not have been clamped short of it */
    if (Math.abs(pane.scrollLeft - (midPx - paneW()/2)) > 1)
      return "asked for offset " + (midPx - paneW()/2).toFixed(0) + ", got " + pane.scrollLeft;
    /* the range must have survived all of it, unchanged and still absolute */
    if (!S.range) return "the range was lost by zooming or scrolling";
    if (Math.abs(S.range.t0 - t0) > 1 || Math.abs(S.range.t1 - t1) > 1)
      return "the range moved with the view: " + S.range.t0 + ".." + S.range.t1;
    setZoom(1, null, 0);
    return "ok";
  })()`, "ok");
/* ---------------------------------------------------------------- the range
 *
 * A drag selects a range AND magnifies it, so `S.range` is a new piece of state and
 * the contract around it has two separate halves, which the checks below keep
 * apart on purpose:
 *   - the GESTURE: drag = select + zoom, and losing the release must do neither.
 *   - the SELECTION: the range is an ANNOTATION.  Unselected data must still be on
 *     the chart - drawn at full contrast, countable and reachable - after a
 *     selection and after any amount of scrolling.  Magnifying takes it off the
 *     AXIS; it must never take it off the CHART.
 * The drag-driven checks set the zoom as a side effect, so anything asserting the
 * "never filters" half has to arm `S.range` directly rather than through a gesture.
 */
chk("a range does not remove a single bar from the chart", `
  (function(){
    ${RESET_ALL}
    /* draw() FIRST, so the drawn count is this view's own and not whatever the
     * previous check left in geom - the prelude resizes but does not repaint, and
     * reading a stale geom.drawn here reports a 1259 -> 7560 jump that has nothing
     * to do with the range */
    draw();
    var pane = document.getElementById("scroller");
    var c2 = DATA.cases[1];
    if (!c2) return "needs at least 2 cases";
    var before = geom.drawn, beforeW = geom.W;
    /* armed directly, not by dragging: the drag zooms, and a zoom legitimately
     * changes both the drawn count and the content width, so going through the
     * gesture would test "does zooming change the count" and call it a filter. */
    S.range = {t0: c2.t_off_ms + c2.dur_ms * 0.2, t1: c2.t_off_ms + c2.dur_ms * 0.8};
    syncRangeUI(); draw();
    if (geom.drawn !== before) return "selecting changed the drawn count " + before + " -> " + geom.drawn;
    /* now walk the scrollbar right across the whole sweep and confirm the count
     * holds at every stop.  A range implemented as a filter would show up here as a
     * dip in the count, and a range drawn as a dim overlay would not - which is why
     * the count, not the pixels, is the assertion. */
    var dips = [];
    for (var f = 0; f <= 1.0001; f += 0.1){
      pane.scrollLeft = f * Math.max(0, pane.scrollWidth - pane.clientWidth);
      syncScroll(); draw();
      if (geom.drawn !== before) dips.push("at " + (f*100).toFixed(0) + "%: " + geom.drawn + " of " + before);
    }
    /* the range is unchanged by any of that scrolling - it is time, not a viewport */
    if (S.range.t0 !== c2.t_off_ms + c2.dur_ms * 0.2) return "the range's start drifted to " + S.range.t0;
    pane.scrollLeft = 0; syncScroll();
    if (geom.W !== beforeW) return "selecting resized the content " + beforeW + " -> " + geom.W;
    S.range = null; syncRangeUI();
    return dips.length ? dips.join(" | ") : "ok";
  })()`, "ok");
chk("the range is absolute sweep time and survives every view switch", `
  (function(){
    ${RESET_ALL}
    var c2 = DATA.cases[1], c5 = DATA.cases[4];
    if (!c2 || !c5) return "needs at least 5 cases";
    /* select across case 2's slice, where a RELATIVE range (fractions of the
     * current axis) would be indistinguishable from an absolute one */
    var x0 = tToPx(c2.t_off_ms) + 30, x1 = tToPx(c2.t_off_ms + c2.dur_ms) - 30;
    fire("el:cv", "pointerdown", pointer(x0, 300, 1));
    fire("window", "pointermove", pointer(x1, 300, 1));
    fire("window", "pointerup", pointer(x1, 300, 1));
    var want = S.range ? S.range.t0 + ".." + S.range.t1 : null;
    if (!want) return "no range to begin with";
    var bad = [];
    /* focus a DIFFERENT case: the axis is now that case's own slice, so a range
     * that survived here is proving it is stored in sweep time and not as a
     * position on the current axis */
    focusCase(4);
    if (!S.range) bad.push("focusing a case dropped the range");
    else if (S.range.t0 + ".." + S.range.t1 !== want) bad.push("focusing rewrote the range to " + S.range.t0 + ".." + S.range.t1);
    if (S.zoom !== 1) bad.push("focusing kept the zoom at " + S.zoom);
    /* the annotation has to be drawn, and CLIPPED: this focused case does not
     * contain the range, so the marker must land outside the plot or not at all,
     * never as a long rule across the toolbar */
    draw();
    showAllCases();
    if (!S.range || S.range.t0 + ".." + S.range.t1 !== want) bad.push("going back to the merged view lost the range");
    S.lane = "depth"; buildOrder(); draw();
    if (!S.range || S.range.t0 + ".." + S.range.t1 !== want) bad.push("switching lane mode lost the range");
    S.lane = "thread"; buildOrder(); draw();
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("a range outside the current axis is clipped away, not drawn across the page", `
  (function(){
    ${RESET_ALL}
    var bad = [], c2 = DATA.cases[1];
    if (!c2) return "needs at least 2 cases";
    /* select case 2's region in the MERGED view, then focus a case that does not
     * contain it.  S.range is absolute, so it is still set - and now it is almost
     * entirely outside the focused axis, which is the case that would paint a rule
     * thousands of pixels wide straight across the toolbar without a fence. */
    var x0 = tToPx(c2.t_off_ms) + 10, x1 = tToPx(c2.t_off_ms + c2.dur_ms) - 10;
    fire("el:cv", "pointerdown", pointer(x0, 300, 1));
    fire("window", "pointermove", pointer(x1, 300, 1));
    fire("window", "pointerup", pointer(x1, 300, 1));
    if (!S.range) return "no range to start with";
    focusCase(0);
    if (!S.range) return "focusing dropped the range";
    var ax0 = caseSpan(C()).t0, ax1 = caseSpan(C()).t1;
    var off = S.range.t1 < ax0 || S.range.t0 > ax1;
    if (!off) return "case 1 overlaps case 2 in the payload, so this cannot be tested";
    /* Measure EVERY branch of the annotation against a same-view baseline.  Two
     * traps: draw() itself issues clips and draw calls (so "zero clips" is the
     * wrong assertion - the delta is the assertion), and the counters are
     * cumulative, so they must be zeroed immediately before each measurement or
     * the reading includes the previous branch's draw. */
    var measure = function(){
      counter.r = counter.fill = counter.stroke = counter.clip = 0;
      rectLog().length = 0;
      draw();
      return {calls: drawnCalls(), clip: counter.clip, rects: rectLog().slice()};
    };
    /* a rect at least half the plot tall is a wash by definition: the annotation is
     * two thin rules and four small caps, and bars are ~15 px in a 62 px row */
    var tall = function(m){ return m.rects.filter(function(r){ return r[3] >= geom.plotH/2; }).length; };
    var keep = S.range;
    S.range = null;   var base = measure();
    S.range = keep;    var off  = measure();
    if (off.calls !== base.calls)
      bad.push("an off-axis range still issued " + (off.calls - base.calls) + " draw call(s)");
    if (off.clip !== base.clip)
      bad.push("an off-axis range still issued " + (off.clip - base.clip) + " clip(s)");
    /* straddling the edge: drawn, and fenced.  An unfenced rule would run off the
     * plot; a fence with no matching restore would leak into every later draw. */
    S.range = {t0: ax0 - 500, t1: ax0 + 5000};
    counter.save = 0; counter.restore = 0;
    pageCtx().globalAlpha = 1;
    var straddle = measure();
    if (straddle.clip <= base.clip) bad.push("a straddling range was drawn without a clip");
    if (counter.save !== counter.restore)
      bad.push("save/restore unbalanced drawing the range: " + counter.save + "/" + counter.restore);
    /* the range itself may hold bars back now, but it must not LEAK.  The post-loop
     * reset is what stops the sticky axis, the caret and the label pass inheriting
     * the last bar's alpha, and it only became load-bearing when the selection
     * started dimming - before that, alpha was always 1 here and a missing reset
     * would have been harmless. */
    if (pageCtx().globalAlpha !== 1)
      bad.push("after drawing the range, globalAlpha is still " + pageCtx().globalAlpha + ", so it leaked");
    /* and it still must not paint a wash: holding bars back is per-span, but a
     * plot-height-tall rect would sit ON TOP of all of them. */
    S.range = {t0: ax0, t1: ax1};
    var full = measure();
    if (tall(full) > tall(base))
      bad.push("a whole-plot range painted " + (tall(full) - tall(base)) +
               " plot-height-tall rect(s), i.e. a wash over every bar");
    S.range = null;
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("a range is a focus, never a filter: nothing is dimmed, hidden or un-counted", `
  (function(){
    ${RESET_ALL}
    var bad = [], c0 = DATA.cases[0];
    if (!c0) return "needs at least 1 case";
    /* A narrow slice of case 1 in the merged view, so there are bars on BOTH sides
     * of it - otherwise a page that ignored the range would pass. */
    var at = function(f){ return {t0: c0.t_off_ms + c0.dur_ms * f[0], t1: c0.t_off_ms + c0.dur_ms * f[1]}; };
    var measure = function(){
      counter.r = 0; rectLog().length = 0; hitBoxes.length = 0;
      draw();
      return {rects: rectLog().slice(), drawn: geom.drawn, hits: hitBoxes.length};
    };
    /* The range is set HERE, in code, and not with a drag.  The drag also zooms, and
     * zooming legitimately changes geom.drawn and the content width, so driving the
     * selection through the gesture would make every count below move for a reason
     * that has nothing to do with filtering.  Setting the state directly is what
     * isolates the "a range must not remove data" contract from "a drag zooms" - the
     * two are separate promises and they deserve separate checks. */
    S.range = null;
    var none = measure();
    S.range = at([0.30, 0.40]);
    var sel = measure();
    if (sel.drawn !== none.drawn)
      bad.push("selecting changed geom.drawn " + none.drawn + " -> " + sel.drawn);
    if (sel.hits !== none.hits)
      bad.push("selecting changed the pick targets " + none.hits + " -> " + sel.hits);
    if (sel.rects.length !== none.rects.length)
      bad.push("selecting changed the rect count " + none.rects.length + " -> " + sel.rects.length);
    /* AND nothing may be dimmed: every bar at the canvas's own alpha, in BOTH
     * states.  Scoped to bar-height rects rather than the whole log, because other
     * things on the canvas (row bands, the search dim) may legitimately carry their
     * own alpha - and the baseline is measured with no range precisely so that a
     * permanently-dim element cannot satisfy this by accident. */
    var bars = function(rs){ return rs.filter(function(r){ return r[3] <= 20; }); };
    var notFull = function(rs){ return rs.filter(function(r){ return r[4] !== 1; }).length; };
    var bn = bars(none.rects), bs = bars(sel.rects);
    if (!bn.length) return "no bar-height rects found, so this cannot be tested";
    if (bn.length !== bs.length)
      bad.push("the bar count changed with a range " + bn.length + " -> " + bs.length);
    if (notFull(bs))
      bad.push(notFull(bs) + " of " + bs.length + " bar(s) were drawn below full alpha with a range");
    if (notFull(bn) !== notFull(bs))
      bad.push("bar alpha differs with and without a range (" + notFull(bn) + " vs " + notFull(bs) + "), so the range is tinting bars");
    /* the range has to actually SPLIT what is on screen, or a page that ignored it
     * would pass every line above - the reference is the page's own pick list, not a
     * second guess at the cull conditions */
    var outside = 0, model = V();
    hitBoxes.forEach(function(hb){
      if (model.t1[hb.i] < S.range.t0 || model.t0[hb.i] > S.range.t1) outside++;
    });
    if (outside < 5)
      bad.push("only " + outside + " span(s) lie outside the range; too few for this to mean anything");
    /* clearing it must leave the chart exactly as it was */
    S.range = null;
    var cleared = measure();
    if (cleared.drawn !== none.drawn)
      bad.push("clearing the range did not restore geom.drawn " + none.drawn + " -> " + cleared.drawn);
    if (cleared.rects.length !== none.rects.length)
      bad.push("clearing the range changed the rect count back " + cleared.rects.length);
    /* and the no-wash rule still holds: a range paints rules and caps, never a slab */
    S.range = at([0.30, 0.40]);
    counter.r = 0; rectLog().length = 0; draw();
    var tallBars = rectLog().filter(function(r){ return r[3] > geom.plotH * 0.5; });
    if (tallBars.length) bad.push(tallBars.length + " plot-height-tall rect(s), i.e. a wash over the bars");
    S.range = null;
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("Reset clears the range, a click on a bar keeps it", `
  (function(){
    ${RESET_ALL}
    var py = geom.padT + geom.rowH/2;
    fire("el:cv", "pointerdown", pointer(geom.padL + 40, py));
    fire("window", "pointermove", pointer(geom.padL + 200, py));
    fire("window", "pointerup", pointer(geom.padL + 200, py));
    if (!S.range) return "no range to test with";
    var want = S.range.t0 + ".." + S.range.t1, z0 = S.zoom;
    if (!(z0 > 1.01)) return "the drag did not magnify, so this check proves nothing (zoom " + z0 + ")";
    var btn = document.getElementById("btnZoomSel");
    if (btn.disabled) return "the zoom-to-selection button stayed disabled with a range selected";
    /* Click a bar, at a point taken FROM the hit boxes rather than computed from
     * the row geometry.  A bar is inset inside its lane row - a 4 px _analyze_single
     * tick is 15 px tall in a 62 px row - so the row's mid-line is not guaranteed
     * to be over any bar at all, and an earlier version of this check used it and
     * then concluded there was nothing to click. */
    if (!hitBoxes.length) return "the chart has no hit boxes after a draw";
    var hb = hitBoxes[Math.floor(hitBoxes.length/2)];
    var hx = (hb.x0 + hb.x1)/2, hy = (hb.y0 + hb.y1)/2;
    fire("el:cv", "pointerdown", pointer(clientX(hx), hy));
    fire("window", "pointerup", pointer(clientX(hx), hy, 0));
    if (S.sel === null) return "the click selected no span";
    if (!S.range || S.range.t0 + ".." + S.range.t1 !== want) return "clicking a bar cleared the range";
    /* ...and it must not MOVE the view either.  Clicking a bar is how a reader
     * finds out which stage a region was, so a click that ejected them from the
     * region they had just dived into would break the one workflow the selection
     * exists for - and it is invisible to a range-only assertion, because the
     * range survives a toggle perfectly well. */
    if (S.zoom !== z0) return "clicking a bar changed the zoom " + z0 + " -> " + S.zoom;
    /* and the reset step does clear it */
    resetAll();
    if (S.range !== null) return "Reset kept the range";
    if (!document.getElementById("btnZoomSel").disabled) return "the button stayed enabled after Reset";
    return "ok";
  })()`, "ok");
chk("a click on empty plot area clears the range", `
  (function(){
    ${RESET_ALL}
    var py = geom.padT + geom.rowH/2;
    fire("el:cv", "pointerdown", pointer(geom.padL + 40, py));
    fire("window", "pointermove", pointer(geom.padL + 200, py));
    fire("window", "pointerup", pointer(geom.padL + 200, py));
    if (!S.range) return "no range to clear";
    /* find plot space with no bar under it, so this really is the empty case and
     * not a bar that pick() found under the last pixel of the row.  The y comes
     * from a real bar's hit box: at a row's mid-line there is often no bar at all,
     * and then "the click found nothing" would be true for a reason that has
     * nothing to do with the click landing on empty space. */
    if (!hitBoxes.length) return "the chart has no hit boxes after a draw";
    var hb = hitBoxes[Math.floor(hitBoxes.length/2)];
    var my = (hb.y0 + hb.y1)/2, mx = null;
    /* search only the VISIBLE window.  A content coordinate outside it is not a
     * point a reader can click, and firing one anyway tests a state no UI can
     * reach: pointerdown decides "gutter" from the visible left edge, so a click
     * scrolled off the left edge arrives as a GUTTER press, which isolates a lane
     * and deliberately leaves the range alone.  That is correct behaviour being
     * reported as a broken click - the fifth variant of the same trap. */
    var lo = S.sx + geom.padL + 4, hi = S.sx + geom.padL + geom.plotW - 4;
    for (var x = lo; x < hi; x += 2){
      if (pick(x, my) === null){ mx = x; break; }
    }
    if (mx === null) return "no empty spot inside the visible window of a row that has bars"
                       + " (minDur is 0, so every row is full) [" + lo.toFixed(0) + ".." + hi.toFixed(0) + "]";
    /* the page must SEE the point we picked as empty, i.e. the client->content
     * round trip has to land on the same pixel; without this the message below
     * cannot tell "the click missed" from "the click hit a bar" */
    var paneEl = document.getElementById("scroller"), paneX = paneEl.scrollLeft;
    var seen = clientX(mx) + paneX;
    if (Math.abs(seen - mx) > 1e-6)
      return "the coordinate round trip moved the click: fired at content " + mx
           + ", the page will see " + seen;
    fire("el:cv", "pointerdown", pointer(clientX(mx), my));
    fire("window", "pointerup", pointer(clientX(mx), my, 0));
    if (S.range !== null)
      return "clicking empty space left the range at " + S.range.t0
           + " [clicked content " + mx.toFixed(0) + " (x" + seen.toFixed(0) + ", pane "
           + paneX.toFixed(0) + "), padL " + geom.padL
           + ", picked " + JSON.stringify(pick(mx, my)) + ", sel " + JSON.stringify(S.sel) + "]";
    if (S.sel !== null) return "the empty click selected span " + S.sel;
    return "ok";
  })()`, "ok");
chk("Enter and the button are the way back out of a selection, and back in", `
  (function(){
    ${RESET_ALL}
    var c2 = DATA.cases[1];
    if (!c2) return "needs at least 2 cases";
    /* the middle HALF of case 2, not all of it.  A brush that covers a slice
     * already on screen has nothing to magnify - the factor clamps to 1 and the
     * control looks dead - so a test that brushes the whole case would be asserting
     * nothing while reading as though it were testing the magnification. */
    var c0 = tToPx(c2.t_off_ms), c1 = tToPx(c2.t_off_ms + c2.dur_ms);
    var x0 = c0 + (c1 - c0)*0.25, x1 = c0 + (c1 - c0)*0.75;
    fire("el:cv", "pointerdown", pointer(x0, 300, 1));
    fire("window", "pointermove", pointer(x1, 300, 1));
    fire("window", "pointerup", pointer(x1, 300, 1));
    var want = S.range;
    if (!want) return "the drag selected no range";
    var span = want.t1 - want.t0;
    if (span <= 0) return "degenerate range";
    var vis = paneW() - padL();
    var expect = Math.max(1, Math.min(maxZoom(), vis / (x1 - x0)));
    var bad = [];
    /* the drag got us here, so the FIRST press is the way out - this is the check
     * that matters most now, because a control that can only go one way would strand
     * every reader who has just dived in with a single drag */
    if (S.zoom < 1.01) bad.push("the drag did not magnify (zoom " + S.zoom + ")");
    else if (Math.abs(S.zoom - expect) > 0.03 * expect)
      bad.push("the drag zoomed to " + S.zoom.toFixed(2) + "x, geometry says " + expect.toFixed(2) + "x");
    /* the range is centred in the pane, and is still the SAME range */
    var midT = (want.t0 + want.t1)/2, m = tToPx(midT) - S.sx;
    if (Math.abs(m - (paneW() + padL())/2) > 2) bad.push("the range's middle landed at x" + m.toFixed(0) + ", not centred");
    fire("document", "keydown", {key:"Enter"});
    if (S.zoom !== 1) bad.push("Enter did not zoom back out (zoom " + S.zoom + ")");
    if (S.range !== want) bad.push("zooming out lost or rewrote the range");
    /* and it goes back in again, because "undo" that can only be done once is a
     * dead end rather than a control */
    fire("document", "keydown", {key:"Enter"});
    if (S.zoom < 1.01) bad.push("the second Enter did not zoom back in (zoom " + S.zoom + ")");
    /* the button is the same action, not a second implementation: fired
     * through the real listener, because calling zoomToRange() directly would pass
     * even if the button were wired to nothing */
    if (document.getElementById("btnZoomSel").disabled) bad.push("the button is disabled with a range");
    fire("el:btnZoomSel", "click", {});
    if (S.zoom !== 1) bad.push("the button did not zoom back out (zoom " + S.zoom + ")");
    setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("a second drag zooms to the new slice, replacing the first range", `
  (function(){
    ${RESET_ALL}
    draw();
    var a = geom.padL + 20, b = geom.padL + 300, py = geom.padT + geom.rowH/2;
    fire("el:cv", "pointerdown", pointer(a, py));
    fire("window", "pointermove", pointer(b, py));
    fire("window", "pointerup", pointer(b, py, 0));
    var first = S.range;
    if (!first) return "the first drag selected nothing";
    if (!(S.zoom > 1.01)) return "the first drag did not magnify (zoom " + S.zoom + ")";
    /* drill down: a second drag, taken INSIDE the first selection, because that is
     * what a reader does when the region they landed in is still too big.  It has
     * to replace the range and re-zoom, not stack a second window on top. */
    var g = geom;                                  /* re-read: the zoom resized it */
    var vis = paneW() - padL();
    /* The centre comes from the PANE, not from the content.  The first drag centred
     * its range in the viewport, which is a statement about S.sx - and a content
     * coordinate taken from the first range's own midpoint is only the same number
     * by coincidence.  Ask the pane instead, and then check the drag landed: a
     * point left of the visible gutter arrives as a gutter press, arms no drag, and
     * "the second drag" silently measures the first one a second time. */
    var m0 = S.sx + g.padL + vis/2;
    /* 240 px, not 40: the axis is already magnified ~4.5x, and since the zoom is a
     * scale that MULTIPLIES, filling the pane now needs ~24x on top of it.  Asking
     * for 120 px would want 47x, which is above maxZoom() - and the cap is doing its
     * documented job there (a browser asked to paint more than 32767 px of canvas
     * paints NOTHING), so "the range fills the pane" would become a contract the page
     * must refuse.  The test picks a magnification the cap allows; the cap keeps its
     * own check. */
    var x0 = m0 - 120, x1 = m0 + 120;
    if (!(x0 - S.sx > g.padL + 4 && x1 - S.sx < g.padL + g.plotW - 4))
      return "the second drag's window " + x0.toFixed(0) + ".." + x1.toFixed(0)
           + " is not inside the visible plot (pane at " + S.sx.toFixed(0) + ")";
    /* clientX, not the content pixel: the drag above moved the scroll offset, so
     * the two coordinate systems have come apart */
    fire("el:cv", "pointerdown", pointer(clientX(x0), py));
    fire("window", "pointermove", pointer(clientX(x1), py));
    fire("window", "pointerup", pointer(clientX(x1), py, 0));
    var second = S.range;
    var bad = [];
    if (!second) bad.push("the second drag cleared the range");
    else {
      if (second === first)
        bad.push("the second drag changed nothing: range is still " + first.t0 + ".." + first.t1);
      if (!(second.t0 >= first.t0 - 1 && second.t1 <= first.t1 + 1))
        bad.push("the second range " + second.t0 + ".." + second.t1 + " is not inside the first " + first.t0 + ".." + first.t1);
    }
    if (!(S.zoom > 1.01))
      bad.push("the second drag zoomed out to " + S.zoom);
    var wantVis = second ? tToPx(second.t1) - tToPx(second.t0) : 0;
    if (second && wantVis > 0 && Math.abs(wantVis - vis) > 0.05 * vis)
      bad.push("the new range occupies " + wantVis.toFixed(0) + " px of a " + vis.toFixed(0)
             + " px plot, not filled [zoom " + S.zoom.toFixed(2) + "x, cap " + maxZoom().toFixed(2)
             + "x, range " + (second.t1 - second.t0).toFixed(0) + " ms, axis " + (caseSpan(V()).t1 - caseSpan(V()).t0).toFixed(0) + " ms]");
    setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("a drag across the whole axis is a selection that zooms nowhere", `
  (function(){
    ${RESET_ALL}
    draw();
    var g = geom, py = geom.padT + geom.rowH/2;
    var x0 = g.padL + 2, x1 = g.padL + g.plotW - 2;
    fire("el:cv", "pointerdown", pointer(x0, py));
    fire("window", "pointermove", pointer(x1, py));
    fire("window", "pointerup", pointer(x1, py, 0));
    var bad = [];
    if (!S.range) bad.push("a full-width drag selected no range");
    /* Magnifying something that is already the whole visible axis has no answer
     * except "stay where you are".  A factor of 0.98 here means the pan landed the
     * reader somewhere arbitrary, and a factor of 20 would mean the pane width was
     * divided by a quantity that was never a pixel count. */
    if (!(S.zoom >= 1 && S.zoom <= 1.01))
      bad.push("a drag covering the visible axis moved the zoom to " + S.zoom);
    setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("a brush of a third of the axis magnifies, not just the extremes", `
  (function(){
    ${RESET_ALL}
    draw();
    var g = geom, py = geom.padT + geom.rowH/2, vis = paneW() - padL();
    /* The two ends of the range are checked elsewhere - a whole-axis brush zooms
     * nowhere, and a small slice is checked below.  This is the MIDDLE, and it is
     * the one a reader actually performs: two cases out of six.  A deadband wide
     * enough to swallow it would still pass both of those, and a cap or a ratio
     * that only works when the scale is 1 would look fine on the first drag. */
    var x0 = g.padL + vis*0.30, x1 = g.padL + vis*0.60;
    fire("el:cv", "pointerdown", pointer(x0, py));
    fire("window", "pointermove", pointer(x1, py));
    fire("window", "pointerup", pointer(x1, py, 0));
    var bad = [];
    if (!S.range) bad.push("a third of the axis selected no range");
    var want = vis / (x1 - x0);                     /* 1/0.30 of the axis */
    if (Math.abs(S.zoom - want) > 0.03 * want)
      bad.push("a 30% brush zoomed to " + S.zoom.toFixed(3) + "x, geometry says " + want.toFixed(3) + "x");
    if (S.range){
      var filled = tToPx(S.range.t1) - tToPx(S.range.t0);
      if (Math.abs(filled - vis) > 0.05 * vis)
        bad.push("the selection fills " + filled.toFixed(0) + " px of a " + vis.toFixed(0) + " px plot");
    }
    setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("a sliver too small to fit stops at the canvas cap", `
  (function(){
    ${RESET_ALL}
    draw();
    var g = geom, py = geom.padT + geom.rowH/2, vis = paneW() - padL();
    /* An 8 px brush asks for ~150x.  The canvas cannot hold that: a browser asked
     * to paint more than 32767 px of width paints NOTHING at all, so the zoom has to
     * be refused rather than granted.  This is the only path that exercises the cap
     * through the SELECTION - the cap's own check calls setZoom directly, so a
     * selection that dropped the cap would leave both of them passing while every
     * deep dive came back as a blank chart. */
    var x0 = g.padL + vis/2 - 4, x1 = g.padL + vis/2 + 4;
    fire("el:cv", "pointerdown", pointer(x0, py));
    fire("window", "pointermove", pointer(x1, py));
    fire("window", "pointerup", pointer(x1, py, 0));
    var bad = [];
    if (!S.range) bad.push("an 8 px brush selected no range");
    var cap = maxZoom();
    if (S.zoom > cap + 1e-6)
      bad.push("an 8 px brush asked for " + S.zoom.toFixed(1) + "x, past the " + cap.toFixed(1) + "x canvas cap");
    if (S.zoom < cap - 0.02 * cap)
      bad.push("it stopped at " + S.zoom.toFixed(1) + "x instead of the " + cap.toFixed(1) + "x cap");
    setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("a cancelled drag leaves an existing range alone", `
  (function(){
    ${RESET_ALL}
    var py = geom.padT + geom.rowH/2;
    fire("el:cv", "pointerdown", pointer(geom.padL + 40, py));
    fire("window", "pointermove", pointer(geom.padL + 200, py));
    fire("window", "pointerup", pointer(geom.padL + 200, py));
    var want = S.range.t0 + ".." + S.range.t1;
    /* every one of the four lost-release nets, plus a plain cancelDrag */
    ["pointercancel", "blur", "contextmenu"].forEach(function(ev){
      fire("el:cv", "pointerdown", pointer(geom.padL + 40, py));
      fire("window", "pointermove", pointer(geom.padL + 260, py));
      fire("el:cv", ev);
    });
    cancelDrag();
    /* An ABORTED drag must not destroy a range the reader already had.  The two
     * are different things - "I am choosing" versus "I have chosen" - and
     * cancelDrag() clears one of them, so conflating them loses work silently. */
    if (!S.range || S.range.t0 + ".." + S.range.t1 !== want)
      return "a cancelled drag changed the range to " + JSON.stringify(S.range);
    if (bandRange() !== null) return "a band survived the cancel";
    return "ok";
  })()`, "ok");
chk("a zoom keeps its anchor under the same pixel", `
  (function(){
    ${RESET_ALL}
    var bad = [], sp = caseSpan(C()), pane = document.getElementById("scroller");
    /* zoom in about a point well away from both ends, and once more from a
     * scrolled position, so the anchor cannot pass by sitting at x=0 */
    [[0, 300], [0, 1200]].forEach(function(st){
      pane.scrollLeft = st[0]; syncScroll();
      var mx = st[1], t = pxToT(mx), before = mx - S.sx;
      setZoom(6, t, mx - S.sx);
      var after = tToPx(t) - S.sx;
      if (Math.abs(after - before) > 1)
        bad.push("from x" + st[0] + ": anchor moved " + before.toFixed(1) + " -> " + after.toFixed(1));
    });
    setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("the zoom is clamped to what a browser canvas can hold", `
  (function(){
    ${RESET_ALL}
    /* a browser silently refuses to paint past 32767 px or 268435456 px^2 - it
     * does not throw, it just draws nothing, so the cap has to be ours.  The
     * check recomputes the cap from the SAME two limits rather than expecting it
     * to move: at 6 rows it is the SIDE limit that binds, and an earlier version
     * of this check asserted the cap shrank for a taller view, which is only
     * true once the AREA limit is the binding one. */
    var bad = [], h = Math.round(contentH());
    var want = Math.max(1, (Math.min(32767, 268435456/h) - padL() - PAD.R) / Math.max(1, plotW0()));
    setZoom(1e9, (caseSpan(V()).t0 + caseSpan(V()).t1)/2);
    var g = geom;
    if (Math.abs(S.zoom - want) > 1e-6 * want)
      bad.push("asked 1e9, got " + S.zoom.toFixed(1) + " but the cap is " + want.toFixed(1));
    if (g.W > 32767) bad.push("canvas " + g.W + " px wide, over the 32767 px side limit");
    if (g.W * g.H > 268435456) bad.push("canvas " + g.W + "x" + g.H + " px2, over the area limit");
    /* and the AREA term has to be real, not decoration: a view tall enough for
     * 268435456/32767 = 8193 px puts the area limit below the side limit, and the
     * cap must then fall below the side-only value */
    S.rowH = 2000; buildOrder(); resize();
    var tall = maxZoom(), sideOnly = (32767 - padL() - PAD.R) / Math.max(1, plotW0());
    if (!(tall < sideOnly - 1e-6))
      bad.push("a " + Math.round(contentH()) + " px tall view still allows " + tall.toFixed(1)
               + "x, the side limit alone allows " + sideOnly.toFixed(1) + "x");
    S.rowH = null; buildOrder(); setZoom(1, null, 0);
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("the arrow keys pan the axis instead of clipping it", `
  (function(){
    ${RESET_ALL} S.zoom = 4; resize(); draw();
    var pane = document.getElementById("scroller"), sp = caseSpan(C());
    var ax0 = tToPx(sp.t0), ax1 = tToPx(sp.t1), w0 = ax1 - ax0;
    pane.scrollLeft = 0; syncScroll();
    fire("document", "keydown", {key:"ArrowRight"}); fire("document", "keydown", {key:"ArrowRight"});
    var moved = pane.scrollLeft;
    fire("document", "keydown", {key:"ArrowLeft"});
    var back = pane.scrollLeft;
    /* the axis must still be whole after any amount of panning */
    var stillWhole = Math.abs((tToPx(sp.t1) - tToPx(sp.t0)) - w0) < 0.5;
    /* and panning must not have shrunk the content */
    var bad = [];
    if (!(moved > 0)) bad.push("ArrowRight did not pan (scrollLeft " + moved + ")");
    if (back >= moved) bad.push("ArrowLeft did not pan back (" + moved + " -> " + back + ")");
    if (!stillWhole) bad.push("the axis stopped spanning the content");
    /* panning at the very ends is a no-op rather than an error */
    pane.scrollLeft = 0; syncScroll(); fire("document", "keydown", {key:"ArrowLeft"});
    if (pane.scrollLeft !== 0) bad.push("panned left past the start: " + pane.scrollLeft);
    setZoom(1, null, 0);    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");
chk("draw while zoomed", `${RESET} S.ci=0; setZoom(4, (caseSpan(C()).t0+caseSpan(C()).t1)/2); draw(); "ok"`, "ok");
chk("zooming out restores the full range", `
  (function(){
    ${RESET} S.ci=0;
    /* the fit case: zoom 1 puts the whole axis on the pane with no horizontal
     * scrollbar, and it is the only state in which scrollLeft can be non-zero
     * by accident, so it is the one worth pinning */
    setZoom(1, null, 0);
    var sp = caseSpan(C()), g = geom;
    return (S.zoom === 1 && document.getElementById("scroller").scrollLeft === 0
      && Math.abs(tToPx(sp.t1) - tToPx(sp.t0) - g.plotW) < 0.5)
      ? "ok" : "zoom " + S.zoom + ", scroll " + document.getElementById("scroller").scrollLeft;
  })()`, "ok");
chk("selecting a depth-2 span renders a call tree", `
  ${RESET} S.ci=0; ${DEEP}
  d >= 0 ? (select(d),
    document.getElementById("treeBody").innerHTML.length > 0
      ? "ok" : "tree empty for " + C().nm[C().st[d]])
        : "no depth-2 span in this payload"`);
chk("ancestry terminates at a root (no parent cycle)", `
  ${RESET} S.ci=0; ${DEEP}
  // ancestry() returns ancestors root-first and EXCLUDES the span itself, so
  // a[0] is the root (depth 0) and a[a.length-1] is the immediate parent.
  var a = ancestry(d);
  (d < 0 || C().dp[a[0]] === 0) ? "ok"
    : "chain of " + a.length + " from depth " + C().dp[d] +
      " starts at depth " + C().dp[a[0]] + ", not a root"`, "ok");
chk("drilling into a worker subtree renders", `
  ${RESET} S.ci=0; ${DEEP}
  S.treeRoot = C().pid[C().pid[d]]; renderTree();
  var h = document.getElementById("treeBody").innerHTML; S.treeRoot = -1;
  h.length > 0 ? "ok" : "empty"`, "ok");
chk("hover tooltip populates", `
  ${RESET} S.ci=0; ${DEEP}
  showTip(d, 300, 200);
  document.getElementById("tip").innerHTML.length > 0 ? "ok" : "empty"`, "ok");
chk("pick() never returns an out-of-range span", `
  ${RESET} S.ci=0; draw(); var bad=0;
  for (var x=0;x<1400;x+=53) for (var y=0;y<620;y+=31){ var i=pick(x,y);
    if (i !== null && (i < 0 || i >= C().n)) bad++; } bad===0`, true);
chk("inventory / stats / legend render", `
  ${RESET} S.ci=0; renderInv(); renderStats(); renderLegend();
  document.getElementById("invCount").textContent.indexOf("function") >= 0
    ? "ok" : "inventory count " + document.getElementById("invCount").textContent`, "ok");

/* ------------------------------------------- integrity of the profiler spans */
chk("parents fully contain their children", `
  ${RESET} S.ci=0; var bad=0;
  for (var i=0;i<C().n;i++){ var p = C().pid[i]; if (C().dp[i] === 0) continue;
    if (C().t0[p] > C().t0[i] + 1e-6 || C().t1[p] < C().t1[i] - 1e-6) bad++; }
  bad===0 ? "ok" : bad+" violations"`, "ok");
chk("a parent shares its child's lane", `
  ${RESET} S.ci=0; var bad=0;
  for (var i=0;i<C().n;i++){ if (C().dp[i] === 0) continue;
    if (C().th[C().pid[i]] !== C().th[i]) bad++; } bad===0`, true);
chk("depth == parent depth + 1", `
  ${RESET} S.ci=0; var bad=0;
  for (var i=0;i<C().n;i++){ if (C().dp[i] === 0) continue;
    if (C().dp[C().pid[i]] !== C().dp[i] - 1) bad++; } bad===0`, true);
chk("self_s + direct-children wall == wall", `
  // run_spans.csv rounds t0/t1/self_s to 1 us independently, and the identity
  // mixes three rounded quantities: the parent's dur (t1-t0, 1 us), the
  // parent's own self_s (0.5 us) and each child's dur (t1-t0, 1 us).  So the
  // residual is bounded by 1.5 us + 1 us per child, i.e. it scales with the
  // child count at 1 us, not 0.5 us.
  var bad=0, worst=0, tot=0;
  for (var ci=0; ci<DATA.cases.length; ci++){ S.ci=ci; var c=C();
    var kids={}, nk={};
    for (var i=0;i<c.n;i++){ var p=c.pid[i];
      if (p>=0){ kids[p]=(kids[p]||0)+c.dur[i]; nk[p]=(nk[p]||0)+1; } }
    for (var i=0;i<c.n;i++){ tot++;
      var tol=(1.5+(nk[i]||0))*1e-3;
      var e=Math.abs((kids[i]||0) + c.self[i]*1000 - c.dur[i]);
      if (e/tol > worst) worst = e/tol;
      if (e > tol) bad++; } }
  console.log("        ("+tot+" spans, worst residual "+(worst*100).toFixed(0)
    +"% of its 1.5 us + 1 us/child bound)");
  bad===0 ? "ok" : bad+" of "+tot+" spans exceed the rounding bound"`, "ok");

/* ================================ the merged all-cases view ==================
 * Six cases on one canvas, one row each.  Everything below is derived from the
 * payload: the row rule's assumptions are asserted against the data rather than
 * assumed, and the global<->local translation is checked for every span. */
chk("merged view has one row per case, labelled", `
  ${RESET_ALL}
  rows().length === DATA.cases.length &&
  rows().map(r => r.label).join("|") === DATA.cases.map(c => c.label).join("|")
    ? "ok" : rows().map(r => r.label).join("|")`, "ok");
chk("merged index spans every span of every case", `
  ${RESET_ALL}
  var want = DATA.cases.reduce((s,c) => s + c.n, 0);
  MERGED.n === want && V().n === want
    ? "ok" : MERGED.n + " != " + want`, "ok");
chk("global <-> local translation is exact for every span", `
  ${RESET_ALL}
  var bad = [];
  for (var k=0; k<DATA.cases.length; k++){
    var off = MERGED.caseStart[k], c = DATA.cases[k];
    for (var i=0; i<c.n; i++){
      var g = off + i;
      if (MERGED.gCase[g] !== k) bad.push("case@"+g);
      else if (MERGED.local(g) !== i) bad.push("local@"+g);
      else if (MERGED.label(g) !== c.label) bad.push("label@"+g);
      else if (MERGED.view.pidg[g] !== (c.pid[i] >= 0 ? off + c.pid[i] : -1)) bad.push("pid@"+g);
      if (bad.length > 4) break;
    }
    if (bad.length > 4) break;
  }
  bad.length ? bad.join(",") : "ok"`, "ok");
chk("every span resolves to a real case label", `
  ${RESET_ALL}
  var labels = {}, bad = 0;
  for (var i=0; i<MERGED.n; i++){ var l = caseOf(i);
    if (DATA.cases.every(c => c.label !== l)) bad++; else labels[l] = 1; }
  bad===0 && Object.keys(labels).length === DATA.cases.length
    ? "ok" : bad + " spans unnamed, " + Object.keys(labels).length + " labels seen"`, "ok");
chk("a merged row contains all of its own case and nothing else", `
  ${RESET_ALL}
  var r = rows(), bad = 0, cnt = {};
  for (var i=0; i<MERGED.n; i++){ var k = r[rowOf(V(), i)];
    if (k === undefined) bad++; else cnt[k.label] = (cnt[k.label]||0) + 1; }
  var labels = DATA.cases.map(c => c.label);
  labels.every(l => cnt[l] === DATA.cases[labels.indexOf(l)].n) && bad === 0
    ? "ok" : JSON.stringify(cnt)`, "ok");
/* the payload's parent_id is a per-case LOCAL row index, so the merged walk
 * must use the globalised copy - otherwise a span parents onto an unrelated
 * span of another case that happens to sit at the same offset */
chk("merged parent walk never crosses a case boundary", `
  ${RESET_ALL}
  var pa = parentArray(), bad = 0;
  for (var i=0; i<MERGED.n; i++){ var p = pa[i];
    if (p >= 0 && MERGED.gCase[p] !== MERGED.gCase[i]) bad++; }
  bad===0 ? "ok" : bad + " cross-case parents"`, "ok");
chk("merged ancestry terminates at a root (no cross-case cycle)", `
  ${RESET_ALL}
  var pa = parentArray(), bad = 0, worst = 0;
  for (var i=0; i<MERGED.n; i++){
    var p = pa[i], hops = 0;
    while (p >= 0 && hops < 50){ p = pa[p]; hops++; }
    if (p >= 0) bad++;
    if (hops > worst) worst = hops;
  }
  bad===0 ? "ok" : bad + " of " + MERGED.n + " chains do not terminate (deepest " + worst + ")"`, "ok");
/* the row rule, asserted against the data instead of assumed */
chk("work-type rule: analyze_pair is a main-lane block, sub-panel ticks are depth-0 workers", `
  ${RESET_ALL}
  var badPair = 0, badSub = 0, mainIx = {};
  DATA.cases.forEach((c,k) => { mainIx[k] = c.lane_name.indexOf("main"); });
  for (var i=0; i<MERGED.n; i++){
    /* i indexes the MERGED arrays, so the per-case arrays need the LOCAL
     * index - reading c.st[i] would be correct for case 0 and garbage for the
     * other five, which is exactly how this check first failed (5/6 wrong). */
    var k = MERGED.gCase[i], l = MERGED.gLocal[i], c = DATA.cases[k];
    var nm = c.nm[c.st[l]], kind = MERGED.kind[i];
    var onMain = (c.th[l] === mainIx[k]);
    if (kind === KIND.PAIR && !(nm === "analyze_pair" && onMain && c.dp[l] <= 1)) badPair++;
    if (kind === KIND.SUB && !(nm === "_analyze_single" && !onMain && c.dp[l] === 0)) badSub++;
  }
  (badPair === 0 && badSub === 0) ? "ok" : badPair + " bad blocks, " + badSub + " bad ticks"`, "ok");
chk("merged draw covers every classified span exactly once", `
  ${RESET_ALL} draw();
  var want = MERGED.kindCount[KIND.PAIR] + MERGED.kindCount[KIND.SUB] + MERGED.kindCount[KIND.MAIN];
  var got = MERGED.n - MERGED.kindCount[KIND.NONE];
  want === got && geom.drawn === got
    ? "ok" : "classified " + got + " of " + MERGED.n + ", drew " + geom.drawn`, "ok");
chk("the merged x axis is one continuous sweep clock", `
  (function(){
    ${RESET_ALL}
    /* the cases were run back to back, so the axis must be the sum of the
     * per-case durations with no overlap and no invented gap: case k's offset
     * is exactly the sum of the durations before it, and the axis ends on the
     * last case's end.  A per-case-0 layout would fail this. */
    var off = 0, bad = [];
    DATA.cases.forEach(function(c, k){
      if (Math.abs(c.t_off_ms - off) > 1e-3) bad.push("case " + k + " at " + c.t_off_ms + " want " + off);
      var lo = Infinity, hi = -Infinity;
      for (var i = 0; i < c.n; i++){ if (c.t0[i] < lo) lo = c.t0[i]; if (c.t1[i] > hi) hi = c.t1[i]; }
      if (lo < c.t_off_ms - 1e-3) bad.push("case " + k + " span starts before its slice");
      if (hi > c.t_off_ms + c.dur_ms + 1e-3) bad.push("case " + k + " span ends after its slice");
      off += c.dur_ms;
    });
    if (Math.abs(V().total_ms - off) > 1e-3) bad.push("axis " + V().total_ms + " vs sweep " + off);
    if (Math.abs(V().sum_ms - off) > 1e-3) bad.push("sum_ms " + V().sum_ms + " vs sweep " + off);
    return bad.length ? bad.join("; ") : "ok";
  })()`, "ok");
chk("every merged span lies in its own case's slice of the clock", `
  (function(){
    ${RESET_ALL}
    var bad = 0, T0 = MERGED.view.t0, T1 = MERGED.view.t1;
    for (var i = 0; i < MERGED.n; i++){
      var c = DATA.cases[MERGED.gCase[i]];
      if (T0[i] < c.t_off_ms - 1e-3) bad++;
      if (T1[i] > c.t_off_ms + c.dur_ms + 1e-3) bad++;
      if (T0[i] > T1[i]) bad++;
    }
    return bad === 0 ? "ok" : bad + " spans outside their case slice";
  })()`, "ok");
chk("focusing a case keeps that case's own slice as the axis", `
  (function(){
    ${RESET_ALL}
    var c = DATA.cases[2], sp = caseSpan(c);
    /* a focused case is about its own internals: stretching 100 s of lanes
     * across the 668 s sweep axis would squeeze them to a sixth of the width */
    if (Math.abs(sp.t0 - c.t_off_ms) > 1e-3) return "t0 " + sp.t0;
    if (Math.abs(sp.t1 - (c.t_off_ms + c.dur_ms)) > 1e-3) return "t1 " + sp.t1;
    if (Math.abs((sp.t1 - sp.t0) - c.dur_ms) > 1e-3) return "extent " + (sp.t1 - sp.t0) + " vs " + c.dur_ms;
    return (sp.t1 <= V().total_ms) ? "ok" : "case slice past the sweep axis";
  })()`, "ok");
chk("pick() never returns an out-of-range merged span", `
  ${RESET_ALL} draw(); var bad=0;
  for (var x=0;x<1400;x+=53) for (var y=0;y<620;y+=31){ var i=pick(x,y);
    if (i !== null && (i < 0 || i >= MERGED.n)) bad++; } bad===0`, true);
chk("the function legend hides only its own branch", `
  (function(){
    ${RESET_ALL} draw();
    var cm = colorModel(V());
    /* pick the branch that owns the sub-panel ticks - the one the old work-type
     * legend hid - so this is the same check as before, re-pointed at the filter
     * that replaced it */
    var target = null;
    for (var bi=0; bi<cm.branches.length; bi++){
      if (cm.branches[bi].keys.some(k => k.name === "_analyze_single")) target = cm.branches[bi].name;
    }
    if (!target) return "no branch keys _analyze_single; branches: "
      + cm.branches.map(b => b.name).join(",");
    var beforeVis = [], i;
    for (i=0; i<MERGED.n; i++) if (visible(i)) beforeVis.push(i);
    var before = geom.drawn, tSpans = 0, oSpans = 0;
    for (i=0; i<MERGED.n; i++)
      (cm.nameOf(i) === target ? tSpans++ : oSpans++);
    S.hiddenBranch = new Set([target]); draw();
    var mid = geom.drawn, leaked = 0, lost = 0;
    var still = {};
    for (i=0; i<MERGED.n; i++) if (visible(i)){
      still[i] = 1;
      if (cm.nameOf(i) === target) leaked++;      /* the branch did not go away */
    }
    /* the exact expectation: everything survives except the hidden branch, and
     * nothing of any other branch is collaterally filtered */
    for (i=0; i<MERGED.n; i++){
      if (cm.nameOf(i) === target) continue;
      if (beforeVis.indexOf(i) >= 0 && !still[i]) lost++;
    }
    S.hiddenBranch = new Set(); draw();
    var ok = leaked === 0 && lost === 0 && tSpans > 0 && oSpans > 0
             && mid < before && geom.drawn === before;
    return ok ? "ok"
      : leaked + " spans of " + target + " survived, " + lost
        + " spans of other branches were collaterally hidden; "
        + before + " -> " + mid + " bars, " + tSpans + "/" + oSpans + " spans";
  })()`, "ok");
chk("merged inventory aggregates per stage and names its cases", `
  ${RESET_ALL} renderInv();
  var h = document.querySelector("#inv tbody").innerHTML, cnt = document.getElementById("invCount").textContent;
  (cnt.indexOf("function") >= 0 && cnt.indexOf("case") >= 0 &&
   h.indexOf('title="') >= 0 && h.indexOf("<td") >= 0)
    ? "ok" : "count " + cnt`, "ok");
chk("inventory case cell is a count in the merged view, a label when focused", `
  ${RESET_ALL} renderInv();
  var mergedCell = /<td title="([^"]+)">/.exec(document.querySelector("#inv tbody").innerHTML);
  ${RESET} S.ci=0; renderInv();
  var caseCell = /<td title="az[^"]*">az/.exec(document.querySelector("#inv tbody").innerHTML);
  (mergedCell && caseCell) ? "ok"
    : "merged " + (mergedCell && mergedCell[0]) + " / focused " + (caseCell && caseCell[0])`, "ok");
chk("merged tooltip names the case and the work type", `
  ${RESET_ALL} draw();
  var g = MERGED.caseStart[1];
  showTip(g, 300, 200);
  var h = document.getElementById("tip").innerHTML;
  (h.indexOf(DATA.cases[1].label) >= 0 && h.indexOf("work type") >= 0)
    ? "ok" : h.slice(0, 120)`, "ok");
chk("call tree names the case of the selected span", `
  ${RESET_ALL} draw();
  var g = MERGED.caseStart[2];
  select(g);
  var t = document.getElementById("treeInfo").textContent;
  t.indexOf(DATA.cases[2].label) >= 0 ? "ok" : t`, "ok");
chk("exported CSV leads with the case column and tags every row", `
  ${RESET_ALL}
  var L = buildCsv(), head = L[0].split(",")[0];
  var bad = L.length - 1, okAll = true;
  for (var r=1; r<L.length; r++){
    var lab = L[r].split(",")[0].replace(/"/g, "");
    if (DATA.cases.every(c => c.label !== lab)){ okAll = false; break; }
  }
  (head === "case" && okAll && bad > 0) ? "ok" : head + ", " + bad + " rows"`, "ok");
/* focus round-trip: merged -> one case -> back */
chk("clicking a case name focuses it", `
  ${RESET_ALL} draw();
  var my = geom.padT + geom.rowH*2 + 4;
  onGutterClick(my);
  (S.mode === "case" && S.ci === 2 && rows().length !== DATA.cases.length &&
   document.getElementById("btnAll").hidden === false && document.getElementById("caseSel").value === "2")
    ? "ok" : S.mode + "/" + S.ci + "/" + rows().length`, "ok");
chk("the all-cases button returns to the merged view", `
  ${RESET_ALL} focusCase(4);
  fire("el:btnAll", "click", {});
  (S.mode === "all" && rows().length === DATA.cases.length &&
   document.getElementById("btnAll").hidden === true) ? "ok" : S.mode`, "ok");
chk("focusing another case keeps the filters but drops the zoom", `
  ${RESET_ALL} draw();
  S.query = "reflect"; setZoom(4, (caseSpan(C()).t0+caseSpan(C()).t1)/2);
  focusCase(3);
  (S.query === "reflect" && S.sel === null && S.mode === "case" && S.zoom === 1)
    ? "ok" : "query " + S.query + " sel " + S.sel + " zoom " + S.zoom`, "ok");
chk("the page opens on the merged view, one labelled row per case", `
  var o = JSON.parse(OPENED_JSON);
  o.mode === "all" && o.nRows === DATA.cases.length &&
  o.labels === DATA.cases.map(function(c){return c.label}).join(",")
    ? "ok" : JSON.stringify(o)`, "ok");

/* ==================== the rubber band (the bug this suite now covers) =====
 * The band used to be painted imperatively by the move handler and cleared only
 * by a release the window received, so a release that never arrived (mouse let
 * go outside the window) left `drag` armed and every later move re-striped the
 * canvas.  Nothing could clear it, not even Reset.  These checks drive the real
 * listeners; with the previous stubbed DOM they were not even registered.
 *
 * They fire POINTER events, which is what the page listens for.  Firing
 * "mousedown" here would arm nothing and every check would still report "ok". */
const DRAG = `
  draw();
  var px = geom.padL + 40, py = geom.padT + geom.rowH/2;
  fire("el:cv", "pointerdown", pointer(px, py));
  fire("window", "pointermove", pointer(px + 80, py));`;
chk("dragging paints the band", `
  ${RESET}${DRAG}
  bandRange() !== null ? "ok" : "no band while dragging"`, "ok");
chk("the band is drawn by draw(), from state", `
  ${RESET}
  var plain = (function(){ var a = counter.r; draw(); return counter.r - a; })();
  drag = {x0: geom.padL + 10, x1: geom.padL + 90, t0: 0, moved: true, buttons: 1, id: 1};
  var banded = (function(){ var a = counter.r; draw(); return counter.r - a; })();
  drag = null;
  banded > plain ? "ok" : "banded draw " + banded + " fillRects vs plain " + plain`, "ok");
chk("a window blur cancels the band", `
  ${RESET}${DRAG}
  fire("window", "blur", {});
  (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range)`, "ok");
chk("Escape cancels the band", `
  ${RESET}${DRAG}
  fire("document", "keydown", {key:"Escape"});
  (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range)`, "ok");
chk("Reset cancels the band", `
  ${RESET}${DRAG}
  fire("el:btnReset", "click", {});
  (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range)`, "ok");
chk("handing the tab away cancels the band", `
  ${RESET}${DRAG}
  document.hidden = true;                       /* the browser sets this; the stub must too */
  fire("document", "visibilitychange", {});
  document.hidden = false;
  (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range)`, "ok");
chk("a click that drifts does not select a range", `
  ${RESET}
  draw();
  var px = geom.padL + 60, py = geom.padT + geom.rowH/2;
  fire("el:cv", "pointerdown", pointer(px, py));
  fire("window", "pointermove", pointer(px + 3, py));
  fire("window", "pointerup", pointer(px + 3, py, 0));
  /* 3 px is under BAND_MIN, so this is a click, not a brush.  A click that drifts
   * a few pixels is a real gesture - the hand moves while the button is down - and
   * turning it into a 3 px selection would make the control feel broken. */
  (S.zoom === 1 && bandRange() === null && drag === null && S.range === null) ? "ok"
    : "zoom " + S.zoom + " range " + JSON.stringify(S.range)`, "ok");
/* The load-bearing one: a drag is a BRUSH that takes you THERE - it records the
 * range and magnifies it in one gesture.  The magnification is what the gesture is
 * for, so this asserts the zoom, not its absence, and pins the factor to the
 * geometry: a px/ms mix-up or a missing clamp here would move the reader to a
 * scale that has nothing to do with the region they chose. */
chk("a real drag selects a range and magnifies it", `
  ${RESET}${DRAG}
  var beforeW = geom.W;
  fire("window", "pointerup", pointer(geom.padL + 120, geom.padT + geom.rowH/2, 0));
  var bad = [];
  if (drag !== null) bad.push("drag still armed");
  if (!S.range) bad.push("no range was recorded");
  else {
    var w = S.range.t1 - S.range.t0;
    if (!(w > 0)) bad.push("degenerate range " + w);
    /* 80 px of a ~1256 px plot at fit is a real slice of the axis, and asking for
     * it to be most of the case's span would catch a unit mix-up (px vs ms) */
    var c0 = C().t_off_ms, c1 = c0 + C().dur_ms;
    if (!(S.range.t0 > c0 && S.range.t1 < c1)) bad.push("range " + S.range.t0 + ".." + S.range.t1 + " left the case " + c0 + ".." + c1);
    if (w > (c1 - c0) * 0.5) bad.push("an 80 px brush covered " + (100*w/(c1-c0)).toFixed(0) + "% of the case");
  }
  /* the zoom itself: deeper than fit, and equal to the factor the geometry asks
   * for.  80 px were drawn, so "make those 80 px fill the plot" is the only sensible
   * answer, and the canvas clamp applies to it exactly as it does to a wheel tick. */
  var vis = paneW() - padL();
  var expect = Math.max(1, Math.min(maxZoom(), vis / 80));
  if (!(S.zoom > 1.01)) bad.push("the drag did not magnify (zoom " + S.zoom + ")");
  else if (expect > 1.01 && Math.abs(S.zoom - expect) > 0.03 * expect)
    bad.push("zoomed to " + S.zoom.toFixed(2) + "x, geometry says " + expect.toFixed(2) + "x");
  /* and the content really did get wider - a zoom that only changed the number
   * would be a lie the reader cannot see through */
  if (S.zoom > 1.01 && geom.W <= beforeW)
    bad.push("the zoom did not widen the content (" + beforeW + " -> " + geom.W + ")");
  if (bad.length) bad.join(" | "); else "ok"`, "ok");
chk("dragging right to left selects the same range as left to right", `
  ${RESET}
  draw();
  var a = geom.padL + 40, b = geom.padL + 140, py = geom.padT + geom.rowH/2;
  fire("el:cv", "pointerdown", pointer(a, py));
  fire("window", "pointermove", pointer(b, py));
  fire("window", "pointerup", pointer(b, py, 0));
  var lr = JSON.stringify(S.range);
  ${RESET}
  draw();
  fire("el:cv", "pointerdown", pointer(b, py));
  fire("window", "pointermove", pointer(a, py));
  fire("window", "pointerup", pointer(a, py, 0));
  /* an inverted brush is the same region, not a negative one: the two ends are
   * sorted on release, so this is free - but a reader dragging right-to-left is
   * completely ordinary, and an inverted range would silently invert the zoom. */
  (lr === JSON.stringify(S.range) && S.range.t0 < S.range.t1) ? "ok"
    : "left-to-right " + lr + " but right-to-left " + JSON.stringify(S.range)`, "ok");
chk("a press in the label gutter does not start a drag", `
  ${RESET}
  draw();
  fire("el:cv", "pointerdown", pointer(geom.padL - 20, geom.padT + geom.rowH*2 + 4));
  fire("window", "pointermove", pointer(400, geom.padT + geom.rowH*2 + 4));
  drag === null ? "ok" : "drag started in the gutter"`, "ok");
chk("a gutter click still isolates a worker lane", `
  ${RESET}
  draw();
  var my = geom.padT + geom.rowH*3 + 4;
  fire("el:cv", "pointerdown", pointer(geom.padL - 20, my));
  fire("window", "pointerup", pointer(geom.padL - 20, my, 0));
  S.isoThread === 3 ? "ok" : "isoThread " + S.isoThread`, "ok");

/* ---- the four nets that stop a lost release from leaving a band behind ----
 * Symptom: press on the chart, drag out of the browser window, let go there.
 * No pointerup reaches the page and no blur fires either (the window is still
 * focused - blur only happens on an app switch), so `drag` stayed armed and the
 * next move inside the window re-striped the canvas.
 *
 * Since the drag also MAGNIFIES, every one of these nets is now about the VIEW and
 * not just about a stripe: an armed `drag` that is later completed by some stray
 * release would select a range the reader never chose and move them into it.  So
 * each check asserts all three - disarmed, no band, zoom still 1, no range. */
chk("pointerdown captures the pointer, and the release is given back", `
  ${RESET} capture.log.length = 0;
  ${DRAG}
  var set = capture.log.filter(function(e){ return e[0] === "set" && e[1] === 1; }).length;
  fire("window", "pointerup", pointer(geom.padL + 120, geom.padT + geom.rowH/2, 0));
  var rel = capture.log.filter(function(e){ return e[0] === "release" && e[1] === 1; }).length;
  (set === 1 && rel === 1) ? "ok" : "set " + set + ", release " + rel
    + " / " + JSON.stringify(capture.log)`, "ok");
chk("a release outside the window still completes the drag, cleanly", `
  (function(){
    ${RESET}${DRAG}
    /* the pointer comes up far off-canvas, as it would after leaving the window.  A
     * release that ARRIVES is a release the reader made: setPointerCapture
     * deliberately retargets it here, and the band was showing them the span all
     * along, so completing is right.  What must not happen is anything left over -
     * an armed drag, a stripe, or a range that runs past the data. */
    fire("window", "pointerup", pointer(9999, geom.padT + geom.rowH/2, 0));
    var sp = axisSpan();
    if (drag !== null) return "the drag is still armed";
    if (bandRange() !== null) return "a band survived the release";
    if (!S.range) return "the release selected nothing at all";
    /* the drag STARTED inside the plot, so the range starts there; the end went off
     * the canvas, and pxToT extrapolates, so THAT end is what must be clamped -
     * an unclamped range ran to 1157 s of a 670 s sweep */
    if (S.range.t0 < sp.t0 - 1)
      return "the range starts at " + S.range.t0 + ", before the axis at " + sp.t0;
    if (Math.abs(S.range.t1 - sp.t1) > 1)
      return "the off-canvas release was not clamped: the range ends at " + S.range.t1
           + " but the axis ends at " + sp.t1;
    if (S.zoom !== 1)
      return "a selection covering " + (100*(S.range.t1 - S.range.t0)/(sp.t1 - sp.t0)).toFixed(1)
           + "% of the axis moved the zoom to " + S.zoom;
    return "ok";
  })()`, "ok");
chk("a move that reports no button held cancels the band by itself", `
  ${RESET}${DRAG}
  /* the self-heal: whatever missed the release, the next move says buttons = 0 */
  fire("window", "pointermove", pointer(geom.padL + 60, geom.padT + geom.rowH/2, 0));
  (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range)`, "ok");
chk("pointercancel cancels the band", `
  ${RESET}${DRAG}
  fire("window", "pointercancel", {});
  (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range)`, "ok");
chk("the watchdog clears a band whose release never arrived at all", `
  (function(){
    ${RESET} draw();
    /* the hardest case: the pointer leaves the window, is released there, and
     * the mouse is never moved again - no release, no move, no blur, no cancel.
     * Only the passage of time can clear it, and only because the pointer is
     * off-window (a quiet drag INSIDE the window must be left alone). */
    var px = geom.padL + 40, py = geom.padT + geom.rowH/2;
    fire("el:cv", "pointerdown", pointer(px, py));
    fire("window", "pointermove", pointer(innerWidth + 200, py));
    if (pendingTimers() === 0) return "no watchdog armed off-window";
    tickTimers();
    return (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range);
  })()`, "ok");
chk("a quiet drag inside the window keeps its band (no watchdog)", `
  (function(){
    ${RESET} draw();
    var px = geom.padL + 40, py = geom.padT + geom.rowH/2;
    fire("el:cv", "pointerdown", pointer(px, py));
    fire("window", "pointermove", pointer(px + 80, py));
    tickTimers();
    /* holding still to read the axis must NOT lose the pending zoom */
    return (drag !== null && bandRange() !== null) ? "ok" : "band eaten by the watchdog";
  })()`, "ok");
chk("the watchdog tracks the real row, not row 0", `
  (function(){
    ${RESET} draw();
    /* a lower row, dragged past the BOTTOM of the window.  A watchdog that
     * guessed row 0's mid-line for y would call this inside and never arm. */
    var px = geom.padL + 40, py = geom.padT + geom.rowH*3;
    fire("el:cv", "pointerdown", pointer(px, py));
    fire("window", "pointermove", pointer(px + 80, innerHeight + 40));
    if (pendingTimers() === 0) return "no watchdog armed below the window";
    tickTimers();
    return (drag === null && bandRange() === null && S.zoom === 1 && S.range === null) ? "ok"
    : "drag " + (drag ? "armed" : "null") + " band " + (bandRange() ? "shown" : "none")
      + " zoom " + S.zoom + " range " + JSON.stringify(S.range);
  })()`, "ok");
chk("a right-button release cannot end a left drag and zoom with it", `
  (function(){
    ${RESET}${DRAG}
    var ev = pointer(geom.padL + 500, geom.padT + geom.rowH/2, 0); ev.button = 2;
    fire("window", "pointerup", ev);
    /* the left button is still down, so the drag is still legitimately armed -
     * what must not happen is the chart zooming to the right button's position, or
     * a range being recorded for a drag the reader has not finished */
    if (S.zoom > 1.01) return "zoomed to " + S.zoom.toFixed(3) + "x";
    if (S.range !== null) return "recorded a range " + JSON.stringify(S.range) + " from a right-button release";
    fire("window", "contextmenu", {});
    return (drag === null && S.zoom === 1 && S.range === null) ? "ok"
      : "context menu left drag " + (drag ? "armed" : "null") + " zoom " + S.zoom + " range " + JSON.stringify(S.range);
  })()`, "ok");
chk("a case-mode click selects a span and a merged-mode click does not move the focus", `
  (function(){
    /* pick() is topmost-wins and hitBoxes is in painter's order, so a thin bar
     * painted later can cover the one we aim at.  Laying the cases on the sweep
     * clock made every bar 1/6 as wide, which is enough for the crowded merged
     * rows to be mutually occluding - so search backwards for a span the hit test
     * genuinely resolves to itself, and click THAT.  The contract under test is
     * "a click selects the span under the cursor", not "this corner is free". */
    var usable = function(){ return hitBoxes.filter(function(h){
      return h.y1 - h.y0 > 3 && h.x1 - h.x0 > 3; }); };
    var probe = function(t){
      var mx = (t.x0 + t.x1)/2, my = (t.y0 + t.y1)/2;
      var cand = [[mx,my],[t.x0+1,t.y0+1],[t.x1-1,t.y1-1],[mx,t.y0+1],[t.x0+1,my]];
      for (var i = 0; i < cand.length; i++){
        var x = Math.round(cand[i][0]), y = Math.round(cand[i][1]);
        if (x >= t.x0 && x <= t.x1 && y >= t.y0 && y <= t.y1 && pick(x, y) === t.i) return [x, y];
      }
      return null;
    };
    /* the last span that is not fully buried under later bars */
    var topmost = function(){
      var b = usable();
      for (var i = b.length - 1; i >= 0 && i > b.length - 60; i--){
        var p = probe(b[i]);
        if (p) return {box: b[i], at: p};
      }
      return null;
    };
    var click = function(at){
      fire("el:cv", "pointerdown", pointer(clientX(at[0]), at[1]));
      fire("window", "pointerup", pointer(clientX(at[0]), at[1], 0));
    };
    ${RESET} draw();
    var t = topmost();
    if (!t) return "no clickable bar in the case view";
    click(t.at);
    if (S.sel !== t.box.i) return "sel " + S.sel + " want " + t.box.i;
    var ci = S.ci;
    ${RESET_ALL} draw();
    var u = topmost();
    if (!u) return "no clickable bar in the merged view";
    click(u.at);
    /* selecting in the merged view is fine and useful - the tree and tooltip
     * name the case.  What must NOT happen is the click dragging the view out of
     * the merged mode or re-pointing the focus, which is what the row gutter is
     * for. */
    return (S.mode === "all" && S.ci === ci && S.sel === u.box.i)
      ? "ok" : S.mode + "/" + S.ci + "/" + S.sel + " want " + u.box.i;
  })()`, "ok");

/* ------------------------------------------------- virtual canvas, scroll, pan */
/* Everything here follows from one change: the canvas is now bigger than the pane
 * that shows it.  The y axis is CATEGORICAL, so "bigger" can only mean taller
 * rows and taller bars - and once the rows exceed the pane there has to be a way
 * to reach the bottom one.  There are two, and both are checked here: the pane's
 * own scrollbars and a middle-drag pan.
 *
 * The invariant that makes the rest of these checks meaningful: bars live in
 * CONTENT coordinates, and pointer events recover content coordinates from
 * getBoundingClientRect() for free.  So a check can scroll first and then aim at
 * a row it can name.  The harness rect used to be a CONSTANT {left:0, top:0},
 * which would have been the third time this file let a check pass while aiming
 * at the wrong pixels.
 *
 * Two harness traps these checks walk straight into, both of which they now
 * avoid rather than trip over:
 *   - chk() runs its code at GLOBAL scope in the vm, so a `var rows` in a check
 *     would shadow the page's own `rows()` function for every check after it.
 *     Local variable names here are prefixed to stay out of the page's way.
 *   - a check body that uses `return` has to be an IIFE, or it is a syntax error
 *     rather than a failed assertion. */
chk("the chart lives in a scroll pane", `
  /* a stubbed DOM has no parentNode, so this is asserted on the source: the
   * scroller must be inside #chartWrap and the canvas inside the scroller, or
   * the scroll offsets would not clip the canvas at all */
  /<div id="chartWrap">[\\s\\S]{0,80}<div id="scroller"><canvas id="cv"><\\/canvas><\\/div>/.test(html)
    ? "ok" : "the canvas is not inside #scroller in the source"`, "ok");

chk("row height defaults per mode (one 62 px value for all three)", `
  ${RESET}
  var th = rowH(); S.lane = "depth"; var dp = rowH(); S.lane = "thread";
  ${RESET_ALL}
  var mg = rowH();
  (th === 62 && dp === 62 && mg === 62) ? "ok"
    : "thread " + th + " depth " + dp + " merged " + mg`, "ok");

chk("the pane scrolls when the rows overflow it", `
  ${RESET}
  var pane = document.getElementById("scroller");
  /* the y axis is categorical, so the content is rows * rowH tall and it must
   * exceed the pane or there is nothing to scroll to */
  (contentH() > pane.clientHeight + 1) ? "ok"
    : "content " + contentH() + " fits pane " + pane.clientHeight`, "ok");

chk("the content is exactly as tall as padT + rows*rowH + padB", `
  ${RESET}
  var wantH = 20 + rows().length * rowH() + 20, gotH = virtualSize().h;
  (gotH === wantH) ? "ok" : "virtual h " + gotH + " want " + wantH`, "ok");

chk("at zoom 1 the content fits the pane (no x scrollbar)", `
  ${RESET}
  var w0 = virtualSize().w, pane0 = document.getElementById("scroller");
  /* a strict "<=" and not "==": clientWidth excludes the pane's own border, so
   * the content may legitimately be a pixel narrower.  What must hold is that it
   * does not OVERFLOW - a wider canvas at zoom 1 would put a horizontal
   * scrollbar on a chart with nothing zoomed. */
  (w0 <= pane0.clientWidth) ? "ok"
    : "virtual w " + w0 + " vs pane " + pane0.clientWidth`, "ok");

chk("zooming makes the content wider than the pane", `
  ${RESET}
  var sp = caseSpan(C());
  setZoom(4, (sp.t0 + sp.t1)/2);
  var wz = virtualSize().w, pz = document.getElementById("scroller");
  (wz > pz.clientWidth + 100) ? "ok"
    : "virtual w " + wz + " vs pane " + pz.clientWidth`, "ok");

chk("a taller row slider makes the content taller", `
  ${RESET}
  var beforeH = virtualSize().h, nRows = rows().length;
  var sl = document.getElementById("rowH"); sl.value = "100";
  fire("el:rowH", "input", {target: sl});
  var afterH = virtualSize().h;
  (S.rowH === 100 && afterH > beforeH + 10 * nRows) ? "ok"
    : "rowH " + S.rowH + ", " + beforeH + " -> " + afterH`, "ok");

chk("taller rows draw taller bars", `
  ${RESET}
  var barHAt = function(rh){
    S.rowH = rh; resize(); draw();
    var tallest = 0;
    for (var i = 0; i < hitBoxes.length; i++){
      tallest = Math.max(tallest, hitBoxes[i].y1 - hitBoxes[i].y0);
    }
    return Math.round(tallest * 10) / 10;
  };
  var smallBar = barHAt(20), bigBar = barHAt(120);
  /* the old cap held a bar at 3.4 px no matter how tall the row was, so the ratio
   * between the smallest and the largest row setting IS the change.  2.5 px is
   * the deliberate floor at the smallest setting. */
  (bigBar > smallBar * 1.5 && smallBar >= 2.4) ? "ok"
    : "bar h " + smallBar + " -> " + bigBar`, "ok");

chk("a bar is at least 5 px tall at the default row height", `
  ${RESET} draw();
  var tall2 = 0;
  for (var i = 0; i < hitBoxes.length; i++){
    if (hitBoxes[i].y1 - hitBoxes[i].y0 > 0) tall2 = Math.max(tall2, hitBoxes[i].y1 - hitBoxes[i].y0);
  }
  tall2 >= 5 ? "ok" : "tallest bar " + tall2 + " px"`, "ok");

chk("rows outside the pane are culled, and the ones inside are all drawn", `
  ${RESET}
  draw();
  var atTop = geom.drawn;
  var pane3 = document.getElementById("scroller");
  pane3.scrollTop = 99999; draw();            /* clamped to the bottom */
  var atBottom = geom.drawn;
  (atTop > 0 && atBottom > 0 && atBottom < atTop) ? "ok"
    : "drawn at top " + atTop + ", at bottom " + atBottom
      + ", rows " + rows().length`, "ok");

chk("the bar cull is exactly the rows the pane is showing", `
  ${RESET}
  var pane4 = document.getElementById("scroller");
  pane4.scrollTop = 137; draw();
  var c4 = V(), hh = geom.rowH, seen = {};
  for (var i = 0; i < hitBoxes.length; i++) seen[rowOf(c4, hitBoxes[i].i)] = 1;
  var inPane = function(k){
    var y0 = geom.padT + hh*k, y1 = y0 + hh;
    return y1 >= S.sy && y0 <= S.sy + paneH();
  };
  var vis = 0, drew = 0, over = 0;
  for (var k = 0; k < rows().length; k++){
    if (inPane(k)) vis++;
    if (seen[k]){ drew++; if (!inPane(k)) over++; }
  }
  /* both halves matter: a cull that is too loose draws off-pane rows (over), one
   * that is too tight drops rows the reader can see (drew !== vis) */
  (drew === vis && over === 0) ? "ok"
    : "pane rows " + vis + ", drew " + drew + ", off-pane " + over
      + " (sy " + S.sy + ", paneH " + paneH() + ")"`, "ok");

chk("a middle drag pans the pane on both axes", `
  ${RESET}
  (function(){
    var sp = caseSpan(C());
    setZoom(4, (sp.t0 + sp.t1)/2);
    var pane5 = document.getElementById("scroller");
    pane5.scrollLeft = 200; pane5.scrollTop = 120; syncScroll();
    var was = [pane5.scrollLeft, pane5.scrollTop];
    var ev0 = pointer(700, 300, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    if (!pan) return "no pan gesture armed";
    /* drag the content right and up, so the offsets follow LEFT and DOWN.
     * Getting this sign backwards is the classic middle-pan bug. */
    fire("window", "pointermove", pointer(660, 260, 4));
    var now = [pane5.scrollLeft, pane5.scrollTop];
    fire("window", "pointerup", pointer(660, 260, 4));
    return (now[0] === was[0] + 40 && now[1] === was[1] + 40) ? "ok" : was + " -> " + now;
  })()`, "ok");

chk("a middle drag clamps at the scrollable range", `
  ${RESET}
  (function(){
    var pane6 = document.getElementById("scroller");
    /* two directions, because the two ends are different code: dragging the
     * content DOWN-RIGHT from the origin must not produce a negative offset, and
     * dragging it far UP must stop at the last row rather than running off. */
    pane6.scrollTop = 0; pane6.scrollLeft = 0; syncScroll();
    var ev0 = pointer(700, 300, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    fire("window", "pointermove", pointer(900, 500, 4));
    var lowEnd = [pane6.scrollLeft, pane6.scrollTop];

    /* re-arm at the other end and drag far past it */
    fire("window", "pointerup", pointer(900, 500, 4));
    pane6.scrollTop = 99999; syncScroll();
    var atEnd = pane6.scrollTop;
    var ev1 = pointer(700, 300, 4); ev1.button = 1;
    fire("el:cv", "pointerdown", ev1);
    fire("window", "pointermove", pointer(700, -900, 4));
    var highEnd = pane6.scrollTop;
    fire("window", "pointerup", pointer(700, -900, 4));

    if (lowEnd[0] !== 0 || lowEnd[1] !== 0) return "origin clamp: " + lowEnd;
    if (atEnd <= 0) return "could not reach the end (scrollTop " + atEnd + ")";
    if (highEnd !== atEnd) return "far end " + highEnd + " want " + atEnd;
    return "ok";
  })()`, "ok");

chk("a middle drag accumulates EVERY move, not just the last one", `
  ${RESET}
  (function(){
    /* THE trap this file has now hit four times: a check that fires ONE
     * pointermove cannot see a pan whose base never advances, because with a
     * single move the constant base and the cumulative base agree.  A real drag
     * fires dozens of moves, so this fires five and demands the sum. */
    var sp = caseSpan(C());
    setZoom(4, (sp.t0 + sp.t1)/2);
    var pane7 = document.getElementById("scroller");
    pane7.scrollLeft = 300; pane7.scrollTop = 200; syncScroll();
    var was = [pane7.scrollLeft, pane7.scrollTop];
    var ev0 = pointer(900, 400, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    if (!pan) return "no pan gesture armed";
    var steps = [[-30,-20],[-30,-20],[-30,-20],[-30,-20],[-30,-20]];
    var x = 900, y = 400;
    steps.forEach(function(d){ x += d[0]; y += d[1];
      fire("window", "pointermove", pointer(x, y, 4)); });
    var now = [pane7.scrollLeft, pane7.scrollTop];
    fire("window", "pointerup", pointer(x, y, 4));
    /* the content follows the CURSOR, so a cursor that walks left makes the pane
     * scroll RIGHT: a 150 px leftward drag is +150 of scroll, not -150 */
    var wantX = was[0] + 5*30, wantY = was[1] + 5*20;
    return (now[0] === wantX && now[1] === wantY) ? "ok"
      : "5 steps of (-30,-20) gave " + now + ", want [" + wantX + "," + wantY + "]"
        + "  (moved " + (now[0]-was[0]) + "/" + (now[1]-was[1]) + " of 150/100)";
  })()`, "ok");

chk("a drag that hit the end responds the instant it comes back", `
  ${RESET}
  (function(){
    /* Reading the offset back (rather than carrying a base that was captured at
     * pointerdown) is what makes this work: after clamping at the end, the very
     * next pixel of reverse travel has to move the pane.  A base that never
     * advances leaves a dead zone the width of the last step. */
    var sp = caseSpan(C());
    setZoom(4, (sp.t0 + sp.t1)/2);
    var pane8 = document.getElementById("scroller");
    pane8.scrollTop = 99999; syncScroll();
    var atEnd = pane8.scrollTop;
    var ev0 = pointer(700, 300, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    fire("window", "pointermove", pointer(700, -900, 4));    // slam into the far end
    var slammed = pane8.scrollTop;
    fire("window", "pointermove", pointer(700, -880, 4));    // 20 px back, same direction out
    var back = pane8.scrollTop;
    fire("window", "pointerup", pointer(700, -880, 4));
    if (slammed !== atEnd) return "did not reach the end: " + slammed + " vs " + atEnd;
    return (back === atEnd - 20) ? "ok"
      : "reversed 20 px from the end moved " + (atEnd - back) + " px";
  })()`, "ok");

chk("the WHOLE content is reachable - the scroll range is not short", `
  ${RESET}
  (function(){
    /* "It scrolls" is not the same as "you can see all of it".  Every earlier
     * check accepted a pane that could move, so a range that stopped a few px
     * short of the content - leaving the scrollbar thumb unable to reach the end -
     * passed all of them.  This measures the shortfall in px, at four zooms. */
    var bad = [];
    [1, 2, 8, 64].forEach(function(z){
      var sp = caseSpan(C());
      setZoom(z, (sp.t0 + sp.t1)/2);
      var pane = document.getElementById("scroller");
      pane.scrollLeft = 999999; pane.scrollTop = 999999; syncScroll();
      var reachX = pane.scrollLeft, reachY = pane.scrollTop;
      var g = geom;
      /* at the maximum offset the content's trailing edge must land exactly on
       * the pane's trailing edge, on both axes */
      var shortX = (g.padL + g.plotW) - (reachX + paneW());
      var shortY = g.H - (reachY + paneH());
      if (shortX > 0.5 || shortY > 0.5)
        bad.push("z" + z + ": short " + shortX.toFixed(2) + "x/" + shortY.toFixed(2) + "y"
                 + " (content " + g.W + "x" + g.H + ", reached " + reachX + "," + reachY
                 + ", range " + (pane.scrollWidth-pane.clientWidth) + "x"
                 + (pane.scrollHeight-pane.clientHeight) + "y)");
    });
    S.zoom = 1; resize(); draw();
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");

chk("the same is true at a fractional devicePixelRatio", `
  ${RESET_ALL}
  (function(){
    /* Windows 125%/150% is dpr 1.25/1.5, and resize() rounds the BACKING store
     * (Math.round(w * dpr)) while the browser scrolls the CSS LAYOUT box.  At
     * dpr 1 the two agree to the pixel, so this whole class of bug is invisible
     * unless the ratio is fractional - which is the setting the stub used to
     * mis-model on top of everything else. */
    var bad = [];
    [1.25, 1.5].forEach(function(dpr){
      /* the bare window global, not the harness's own object name: the body
       * below runs inside the page's VM context, where the sandbox's "window"
       * key is what the page and this check both see.  Referring to the outer
       * object by name is a ReferenceError there, not a silent no-op. */
      window.devicePixelRatio = dpr;
      S.zoom = 1; resize(); draw();
      var cvEl = document.getElementById("cv");
      /* what draw() ACTUALLY used, not what the backing store divided by dpr
       * happens to be - the check has to read geom.W or it is testing its own
       * arithmetic and passes for either implementation */
      var painted = geom.W;
      var box = parseFloat(cvEl.style.width);         // what the browser scrolls
      if (Math.abs(painted - box) > 0.01)
        bad.push("dpr " + dpr + ": drawn " + painted.toFixed(4)
                 + " vs layout box " + box.toFixed(4));
      var pane = document.getElementById("scroller");
      /* The harness's own fidelity contract, and it has to catch a range that is
       * too LARGE as well as one that is too short: reading the extent from the
       * backing store makes the modelled range dpr times too wide, and a check
       * that only asks "is anything unreachable?" sails straight through it. */
      if (Math.abs(pane.scrollWidth - box) > 0.01)
        bad.push("dpr " + dpr + ": modelled scroll extent " + pane.scrollWidth
                 + " is not the canvas layout box " + box.toFixed(4));
      pane.scrollLeft = 999999; pane.scrollTop = 999999; syncScroll();
      var g = geom;
      var shortX = (g.padL + g.plotW) - (pane.scrollLeft + paneW());
      if (shortX > 0.01) bad.push("dpr " + dpr + ": short " + shortX.toFixed(4) + "x");
    });
    window.devicePixelRatio = 1;
    ${RESET_ALL}
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");

chk("a middle release ends the pan, and only the pan", `
  ${RESET}
  (function(){
    var pane7 = document.getElementById("scroller");
    pane7.scrollTop = 100; syncScroll();
    var ev0 = pointer(700, 300, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    fire("window", "pointermove", pointer(680, 280, 4));
    var rel = pointer(680, 280, 0); rel.button = 1;
    fire("window", "pointerup", rel);
    /* the OLD guard rejected every non-primary release, which is exactly wrong now
     * that the middle button arms something: the pan would have run on forever */
    return (pan === null && drag === null) ? "ok" : "pan " + !!pan + " drag " + !!drag;
  })()`, "ok");

chk("a middle drag cannot select a bar or start a band", `
  ${RESET}
  (function(){
    draw();
    var t = hitBoxes.length ? hitBoxes[Math.floor(hitBoxes.length/2)] : null;
    if (!t) return "no bar to aim at";
    var mx = (t.x0 + t.x1)/2, my = (t.y0 + t.y1)/2;
    var ev0 = pointer(mx, my, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    fire("window", "pointermove", pointer(mx + 60, my, 4));
    fire("window", "pointerup", pointer(mx + 60, my, 4));
    return (S.sel === null && drag === null && !bandRange()) ? "ok"
      : "sel " + S.sel + " drag " + !!drag + " band " + !!bandRange();
  })()`, "ok");

chk("a pan is cancelled by the self-heal when the button is already up", `
  ${RESET}
  (function(){
    var pane8 = document.getElementById("scroller");
    pane8.scrollTop = 100; syncScroll();
    var ev0 = pointer(700, 300, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    if (!pan) return "no pan armed";
    /* buttons === 0, not & 4: the button is provably up, so the next move must
     * heal it.  The old code tested bit 1, which a pan never sets. */
    fire("window", "pointermove", pointer(690, 290, 0));
    return pan === null ? "ok" : "pan still armed";
  })()`, "ok");

chk("a pan does not survive blur", `
  ${RESET}
  (function(){
    var ev0 = pointer(700, 300, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    if (!pan) return "no pan armed";
    fire("window", "blur", {});
    return pan === null ? "ok" : "pan still armed";
  })()`, "ok");

chk("a pan is cancelled by the off-window watchdog", `
  ${RESET}
  (function(){
    var ev0 = pointer(700, 300, 4); ev0.button = 1;
    fire("el:cv", "pointerdown", ev0);
    if (!pan) return "no pan armed";
    /* the watchdog deliberately does NOT key off the button state - that is the
     * thing that stopped arriving - it only arms while the pointer is outside
     * the window, so the move has to go off-window first.  buttons stays 4 so
     * the self-heal does not fire first and hide what is being tested. */
    fire("window", "pointermove", pointer(700, innerHeight + 40, 4));
    if (pendingTimers() < 1) return "watchdog did not arm (" + pendingTimers() + " pending)";
    tickTimers();
    return pan === null ? "ok" : "pan still armed";
  })()`, "ok");

chk("a middle click asks the browser not to autoscroll or paste", `
  ${RESET}
  (function(){
    var n = 0;
    var ev = pointer(700, 300, 4); ev.button = 1;
    ev.preventDefault = function(){ n++; };
    fire("el:cv", "pointerdown", ev);
    var m = 0;
    var ev2 = pointer(700, 300, 0); ev2.button = 1;
    ev2.preventDefault = function(){ m++; };
    fire("el:cv", "auxclick", ev2);
    return (n === 1 && m === 1) ? "ok" : "pointerdown " + n + " auxclick " + m;
  })()`, "ok");

chk("a plain wheel is left to the pane, and does not zoom", `
  ${RESET}
  (function(){
    var was = S.zoom;
    var ev = {deltaY: 120, clientX: geom.padL + 200, clientY: 200};
    var n = 0; ev.preventDefault = function(){ n++; };
    fire("el:cv", "wheel", ev);
    /* preventDefault here is what would have killed native scrolling, and the old
     * handler called it unconditionally */
    return (n === 0 && S.zoom === was) ? "ok"
      : "preventDefault " + n + ", zoom " + S.zoom;
  })()`, "ok");

chk("ctrl+wheel zooms about the cursor", `
  ${RESET}
  (function(){
    var w0 = geom.W, bad = [];
    var ev = {deltaY: -120, ctrlKey: true, clientX: geom.padL + geom.plotW/2, clientY: 200};
    var n = 0; ev.preventDefault = function(){ n++; };
    fire("el:cv", "wheel", ev);
    if (n !== 1) bad.push("preventDefault called " + n + " times");
    /* wheel UP must magnify.  It used to work by shrinking a window, so the
     * factor was below 1; under a scale model the same factor would have made
     * "zoom in" mean "zoom out", and clamping at 1 made the gesture a no-op
     * from the fit state - which is exactly where a reader starts. */
    if (!(S.zoom > 1.01)) bad.push("wheel up did not magnify: zoom " + S.zoom.toFixed(3));
    if (!(geom.W > w0 * 1.2)) bad.push("content " + w0 + " -> " + geom.W + " px");
    /* and wheel DOWN must come back */
    fire("el:cv", "wheel", {deltaY: 120, ctrlKey: true,
                            clientX: geom.padL + geom.plotW/2, clientY: 200,
                            preventDefault: function(){}});
    if (!(S.zoom < 1.01)) bad.push("wheel down did not come back: zoom " + S.zoom.toFixed(3));
    return bad.length ? bad.join(" | ") : "ok";
  })()`, "ok");

chk("cmd+wheel zooms too (the Mac idiom)", `
  ${RESET}
  (function(){
    var ev = {deltaY: -120, metaKey: true, clientX: geom.padL + geom.plotW/2, clientY: 200};
    var n = 0; ev.preventDefault = function(){ n++; };
    fire("el:cv", "wheel", ev);
    return (n === 1 && S.zoom > 1.01) ? "ok"
      : "preventDefault " + n + ", zoom " + S.zoom;
  })()`, "ok");

chk("a zoomed-in chart redraws against the CONTENT width, not the pane's", `
  ${RESET}
  (function(){
    var sp = caseSpan(C());
    setZoom(4, (sp.t0 + sp.t1)/2);
    var w = virtualSize().w;
    return Math.abs(geom.plotW - (w - geom.padL - geom.padR)) < 1 ? "ok"
      : "plotW " + geom.plotW + " for a " + w + " px canvas";
  })()`, "ok");

chk("the sticky gutter is pinned to the pane, not to the content", `
  ${RESET}
  (function(){
    /* zoom in first: at zoom 1 the content is exactly as wide as the pane, so
     * scrollLeft clamps to 0 and there is nothing for the gutter to cover */
    var sp = caseSpan(C());
    setZoom(4, (sp.t0 + sp.t1)/2);
    var pane9 = document.getElementById("scroller");
    pane9.scrollLeft = 400; syncScroll();
    var sx = S.sx;
    counter.r = 0; drawStickyGutter();
    var painted = counter.r;
    /* without S.sx > 0 the function returns early and paints nothing at all */
    return (sx > 0 && painted > 0) ? "ok"
      : "sx " + sx + ", gutter painted " + painted + " rects";
  })()`, "ok");

chk("the sticky gutter is OPAQUE and bounded, not a second copy of the chart", `
  ${RESET}
  (function(){
    var sp = caseSpan(C());
    setZoom(4, (sp.t0 + sp.t1)/2);
    var paneA = document.getElementById("scroller");
    /* at scroll 0 there is nothing passing under the gutter, so it must paint
     * NOTHING; once it is pinned it must paint a small, CONSTANT amount whatever
     * the scroll - a count that tracked the chart size would mean it was
     * repainting the whole plot a second time on every scroll frame */
    paneA.scrollLeft = 0; syncScroll(); counter.r = 0; drawStickyGutter();
    var at0 = counter.r;
    var counts = [];
    [100, 400, 2000].forEach(function(x){
      paneA.scrollLeft = x; syncScroll(); counter.r = 0; drawStickyGutter();
      counts.push(counter.r);
    });
    var same = counts.every(function(n){ return n === counts[0]; });
    return (at0 === 0 && counts[0] > 0 && same && counts[0] < 10) ? "ok"
      : "at 0: " + at0 + ", scrolled: " + counts.join("/") + " rects";
  })()`, "ok");

chk("the sticky overlays are skipped at scroll origin (no needless paint)", `
  ${RESET}
  (function(){
    var paneB = document.getElementById("scroller");
    paneB.scrollLeft = 0; paneB.scrollTop = 0; syncScroll();
    counter.r = 0; drawStickyGutter();
    var g = counter.r;
    counter.r = 0; drawStickyAxis(1000, fmtT);
    var a = counter.r;
    return (g === 0 && a === 0) ? "ok" : "gutter " + g + ", axis " + a + " rects at origin";
  })()`, "ok");

chk("the sticky overlays DO paint once the pane is scrolled", `
  ${RESET}
  (function(){
    var paneC = document.getElementById("scroller");
    paneC.scrollTop = 120; syncScroll();
    counter.r = 0; drawStickyAxis(1000, fmtT);
    var a = counter.r;
    counter.r = 0; drawStickyGutter();
    var g = counter.r;
    return (a > 0 && g === 0) ? "ok"
      : "axis " + a + " (want > 0), gutter " + g + " (want 0, no x scroll)";
  })()`, "ok");

chk("pick() still resolves content coordinates after a scroll", `
  ${RESET}
  (function(){
    var paneD = document.getElementById("scroller");
    paneD.scrollTop = 137; syncScroll(); draw();
    var c = V(), hh = geom.rowH, k = -1;
    /* the row under the MIDDLE of the pane, not the first row that fits: at
     * sy=137 the first row is already scrolled out above the top edge, so asking
     * for it would (correctly) find no bars */
    var midY = S.sy + paneH()/2;
    for (var r = 0; r < rows().length; r++){
      if (geom.padT + hh*r <= midY && geom.padT + hh*(r+1) >= midY){ k = r; break; }
    }
    if (k < 0) return "no row at the pane's mid-line";
    var yc = geom.padT + hh*(k + .5);
    var boxes = hitBoxes.filter(function(b){ return rowOf(c, b.i) === k; });
    if (!boxes.length) return "row " + k + " drew no bars";
    var b = boxes[0];
    return pick(Math.round((b.x0 + b.x1)/2), Math.round(yc)) >= 0 ? "ok"
      : "pick missed inside row " + k;
  })()`, "ok");

chk("the gutter click target follows the pinned gutter, not the content x", `
  ${RESET}
  (function(){
    var paneE = document.getElementById("scroller");
    paneE.scrollLeft = 300; syncScroll(); draw();
    /* x = padL - 20 in VIEWPORT space is inside the pinned gutter.  Compared
     * against the raw content x - as the old code did - the same click would have
     * landed 300 px to the right of the gutter and started a band instead. */
    var my = geom.padT + geom.rowH*2 + 4;
    fire("el:cv", "pointerdown", pointer(geom.padL - 20, my));
    var guttered = press !== null && press.gutter === true;
    fire("window", "pointerup", pointer(geom.padL - 20, my, 0));
    return (guttered && S.isoThread === 2) ? "ok"
      : "gutter " + guttered + ", iso " + S.isoThread;
  })()`, "ok");

chk("Reset returns the row height, the scroll and the zoom to their defaults", `
  (function(){
    S.rowH = 96;
    var paneF = document.getElementById("scroller");
    paneF.scrollLeft = 250; paneF.scrollTop = 90; syncScroll();
    setZoom(400, caseSpan(C()).t0 + 1);
    fire("el:btnReset", "click", {});
    var sc = getScroll("scroller");
    return (rowH() === 62 && sc[0] === 0 && sc[1] === 0 && S.zoom === 1) ? "ok"
      : "rowH " + rowH() + ", scroll " + sc + ", zoom " + S.zoom;
  })()`, "ok");

chk("the row slider shows the row height actually in force", `
  ${RESET}
  (function(){
    var sl = document.getElementById("rowH"), lbl = document.getElementById("rowHLbl");
    if (+sl.value !== rowH()) return "slider " + sl.value + " vs rowH " + rowH();
    /* All three modes now share one default, so switching modes cannot prove the
     * control followed anything.  Move the depth default instead: if the sync
     * read a cached value rather than the mode's own default, this would not
     * follow. */
    ROWH_DEFAULT.caseDepth = 40;
    S.lane = "depth"; S.rowH = null; syncRowHControl();
    if (+sl.value !== 40 || lbl.textContent !== "40 px")
      return "after a depth default of 40: slider " + sl.value + ", label " + lbl.textContent;
    ROWH_DEFAULT.caseDepth = 62;
    S.lane = "thread"; S.rowH = null; syncRowHControl();
    if (+sl.value !== 62 || lbl.textContent !== "62 px")
      return "after restoring 62: slider " + sl.value + ", label " + lbl.textContent;
    S.rowH = 88;
    fire("el:rowH", "input", { target: { value: "88" } });
    /* the input path deliberately does NOT write back to the control - the user
     * dragged it, so it already reads 88 in a real browser - but the label and
     * the effective row height must both follow */
    return (lbl.textContent === "88 px" && rowH() === 88) ? "ok"
      : "label " + lbl.textContent + ", rowH " + rowH();
  })()`, "ok");

console.log("\npass " + pass + "  fail " + fail);
process.exit(fail ? 1 : 0);

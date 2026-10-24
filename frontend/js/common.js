/* Shared layout, settings and API helpers for every page. */
const ICONS = {
  real: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
  predict: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
  players: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17l5-6 4 4 7-8"/><path d="M14 7h5v5"/></svg>',
  coach: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3a7 7 0 0 0-4 12.7V19h8v-3.3A7 7 0 0 0 12 3z"/><path d="M10 22h4"/><circle cx="12" cy="10" r="2"/></svg>',
  scorecard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 15v-3M12 15V9M17 15v-5"/></svg>',
  batch: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
};
const PAGES = [
  ["index.html", "predict", "Predict"],
  ["players.html", "players", "Player Analysis"],
  ["coach.html", "coach", "AI Coach"],
  ["scorecard.html", "scorecard", "Model Scorecard"],
  ["batch.html", "batch", "Batch Scoring"],
  ["realdata.html", "real", "Real Data"],
];

const store = {
  get(k, d) { try { const v = localStorage.getItem("dv_" + k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("dv_" + k, v); } catch { /* storage unavailable */ } },
};
const settings = { model: store.get("model", ""), unit: store.get("unit", "norm") };
const listeners = [];
function onSettings(fn) { listeners.push(fn); }
function setSetting(k, v) { settings[k] = v; store.set(k, v); listeners.forEach(f => f(settings)); }

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const res = await fetch(path, opt);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}

function showError(msg) {
  const e = document.getElementById("err");
  if (!e) return;
  e.innerHTML = msg;
  e.style.display = "block";
}

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const pct = (v, d = 1) => (v === null || v === undefined) ? "–" : (v * 100).toFixed(d) + "%";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const $ = id => document.getElementById(id);
const labelTag = l => `<span class="tag ${l && l[0] === "T" ? "t" : (["IB", "OB"].includes(l) ? "b" : "")}">${esc(l)}</span>`;

let MODELS = [];
async function initShell(active) {
  if (!document.querySelector('link[rel="icon"]')) {          // browser-tab icon on every page
    const fav = document.createElement("link");
    fav.rel = "icon"; fav.type = "image/png"; fav.href = "favicon.png";
    document.head.appendChild(fav);
  }
  const side = document.querySelector(".sidebar");
  side.innerHTML = `
        <div class="brand"><img class="logo" src="logo-icon.png" alt="AI DartVision logo"><div>AI Dart<span>Vision</span></div></div>
    <div class="tagline">Coordinates in. Scoring regions out.</div>
    <div><div class="side-label">Modules</div><nav class="nav">
      ${PAGES.map(([href, key, name]) => `<a href="${href}" class="${key === active ? "active" : ""}">${ICONS[key]}${name}</a>`).join("")}
    </nav></div>
    <div><div class="side-label">Classifier engine</div><select class="side-select" id="modelSel"><option>Loading…</option></select></div>
    <div><div class="side-label">Coordinate units</div><div class="seg" id="unitSeg">
      <button data-u="norm">Normalised</button><button data-u="mm">Millimetres</button></div></div>
    <div class="side-foot" id="sideFoot"><span class="dot"></span>Connecting…</div>`;
  document.querySelectorAll("#unitSeg button").forEach(b => {
    b.classList.toggle("on", b.dataset.u === settings.unit);
    b.onclick = () => {
      document.querySelectorAll("#unitSeg button").forEach(x => x.classList.toggle("on", x === b));
      setSetting("unit", b.dataset.u);
    };
  });
  try {
    const m = await api("/api/models");
    MODELS = m.models;
    if (!MODELS.length) throw new Error("no models");
    if (!MODELS.find(x => x.key === settings.model)) settings.model = m.best || MODELS[0].key;
    const sel = $("modelSel");
    sel.innerHTML = MODELS.map(x => `<option value="${x.key}" ${x.key === settings.model ? "selected" : ""}>${x.name} (${x.feature_set})${x.key === m.best ? " ★" : ""}</option>`).join("");
    sel.onchange = () => setSetting("model", sel.value);
    $("sideFoot").innerHTML = `<span class="dot"></span>API connected · ${MODELS.length} models`;
    const s = $("apiChip"); if (s) { s.className = "chip"; s.innerHTML = '<span class="dot"></span>Engine ready'; }
  } catch (e) {
    $("sideFoot").innerHTML = `<span class="dot red"></span>API offline`;
    const s = $("apiChip"); if (s) { s.className = "chip red"; s.innerHTML = '<span class="dot red"></span>Engine offline'; }
    showError(`Cannot reach the DartVision API or no models are trained. In the project folder run <code>python run_pipeline.py</code> then <code>uvicorn api:app --reload</code>, and open <code>http://127.0.0.1:8000</code>.`);
    throw e;
  }
}
function modelName(key) { const m = MODELS.find(x => x.key === key); return m ? `${m.name} (${m.feature_set})` : key; }

/* Simple SVG line chart. series: [{name, values, color, axis:'left'|'right', fmt}] */
function lineChart(el, labels, series) {
  const W = 640, H = 260, pl = 46, pr = 46, pt = 20, pb = 34;
  const axes = { left: series.filter(s => s.axis !== "right"), right: series.filter(s => s.axis === "right") };
  const range = arr => { const v = arr.flatMap(s => s.values); if (!v.length) return [0, 1]; let lo = Math.min(...v), hi = Math.max(...v); const pad = (hi - lo) * .15 || 1; return [lo - pad, hi + pad]; };
  const rl = range(axes.left), rr = range(axes.right);
  const X = i => pl + (labels.length === 1 ? (W - pl - pr) / 2 : i * (W - pl - pr) / (labels.length - 1));
  const Y = (v, r) => pt + (H - pt - pb) * (1 - (v - r[0]) / (r[1] - r[0]));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;font-family:IBM Plex Mono,monospace">`;
  for (let k = 0; k <= 4; k++) {
    const y = pt + k * (H - pt - pb) / 4;
    s += `<line x1="${pl}" x2="${W - pr}" y1="${y}" y2="${y}" stroke="#E6DED0"/>`;
    const lv = rl[1] - k * (rl[1] - rl[0]) / 4;
    s += `<text x="${pl - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#8A8F99">${(axes.left[0]?.fmt || (v => v.toFixed(0)))(lv)}</text>`;
    if (axes.right.length) { const rv = rr[1] - k * (rr[1] - rr[0]) / 4; s += `<text x="${W - pr + 8}" y="${y + 4}" font-size="11" fill="#8A8F99">${(axes.right[0].fmt || (v => v.toFixed(0)))(rv)}</text>`; }
  }
  labels.forEach((l, i) => s += `<text x="${X(i)}" y="${H - 10}" text-anchor="middle" font-size="11" fill="#6B6F78">${esc(l)}</text>`);
  series.forEach(se => {
    const r = se.axis === "right" ? rr : rl;
    const pts = se.values.map((v, i) => `${X(i)},${Y(v, r)}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="${se.color}" stroke-width="2.5" ${se.dash ? 'stroke-dasharray="6 4"' : ""}/>`;
    se.values.forEach((v, i) => s += `<circle cx="${X(i)}" cy="${Y(v, r)}" r="4" fill="${se.color}"><title>${esc(se.name)}: ${(se.fmt || (x => x.toFixed(1)))(v)}</title></circle>`);
  });
  s += "</svg>";
  el.innerHTML = s;
}

function downloadText(name, text, type = "text/csv") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}


/* =====================================================================
   ANIMATIONS  (paste at the end of common.js)
   ===================================================================== */
(function () {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 1. Staggered entrance for the blocks on the page */
  const blocks = document.querySelectorAll(".main > .topbar, .page-head, .main > .card, .main > .grid > *, .main > .grid > .stack > *, .tile");
  blocks.forEach((el, i) => { el.style.setProperty("--d", Math.min(i, 14)); el.classList.add("anim-in"); });

  /* 2. Ripple effect on buttons */
  document.addEventListener("pointerdown", e => {
    const b = e.target.closest(".btn");
    if (!b || b.disabled || reduce) return;
    const r = b.getBoundingClientRect(), s = Math.max(r.width, r.height);
    const span = document.createElement("span");
    span.className = "ripple";
    span.style.cssText = `width:${s}px;height:${s}px;left:${e.clientX - r.left - s / 2}px;top:${e.clientY - r.top - s / 2}px`;
    b.appendChild(span);
    setTimeout(() => span.remove(), 650);
  });

  /* 3. Fade-out when moving to another page (browsers without view transitions) */
  if (!("startViewTransition" in document)) {
    document.addEventListener("click", e => {
      const a = e.target.closest(".nav a, a.btn");
      if (!a || e.ctrlKey || e.metaKey || a.target === "_blank") return;
      e.preventDefault();
      document.body.classList.add("leaving");
      setTimeout(() => { location.href = a.href; }, 170);
    });
    window.addEventListener("pageshow", () => document.body.classList.remove("leaving"));
  }

  /* 4. Count-up for big numbers (e.g. 0 -> 60, 0 -> 114.8%) */
  const NUM = /^(\s*)(-?\d+(?:\.\d+)?)(%?\s*)$/;
  function countUp(el) {
    const tn = [...el.childNodes].find(n => n.nodeType === 3 && NUM.test(n.textContent));
    if (!tn) { el._prev = undefined; return; }
    const [, pre, num, post] = tn.textContent.match(NUM);
    const target = parseFloat(num), dec = (num.split(".")[1] || "").length;
    const from = el._prev ?? 0;
    el._prev = target;
    if (from === target || reduce) return;
    el._anim = true;
    const t0 = performance.now(), dur = 750;
    (function step(t) {
      const p = Math.min(1, (t - t0) / dur), k = 1 - Math.pow(1 - p, 3);
      tn.textContent = pre + (from + (target - from) * k).toFixed(dec) + post;
      if (p < 1) requestAnimationFrame(step); else el._anim = false;
    })(t0);
  }
  function flash(tile) {
    if (!tile || reduce) return;
    tile.classList.remove("flash"); void tile.offsetWidth; tile.classList.add("flash");
  }

  /* 5. Watch the page for new content and animate it */
  const obs = new MutationObserver(muts => {
    const bigs = new Set(), tiles = new Set();
    for (const m of muts) {
      const el = m.target.nodeType === 3 ? m.target.parentElement : m.target;
      if (!el) continue;
      const big = el.closest(".big");
      if (big && !big._anim) { bigs.add(big); tiles.add(big.closest(".tile")); }
      if (m.type !== "childList") continue;
      // stagger table rows, confusion-matrix cells and darts
      if (el.matches("table") || el.matches("tbody")) el.querySelectorAll("tbody tr").forEach((tr, i) => tr.style.setProperty("--i", Math.min(i, 20)));
      if (el.matches(".cm")) [...el.children].forEach((c, i) => c.style.setProperty("--i", i));
      if (el.matches(".darts")) [...el.children].forEach((c, i) => c.style.setProperty("--i", Math.min(i, 120)));
      if (el.matches("svg")) el.querySelectorAll(".darts > *").forEach((c, i) => c.style.setProperty("--i", Math.min(i, 120)));
      // keep the chat scrolled to the newest message
      if (el.matches(".chat") && el.lastElementChild) el.lastElementChild.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    bigs.forEach(countUp);
    tiles.forEach(t => t && t.querySelector(".big") && flash(t));
  });
  obs.observe(document.body, { childList: true, subtree: true, characterData: true });
})();
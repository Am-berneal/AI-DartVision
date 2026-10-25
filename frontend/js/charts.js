/* Reusable SVG charts for the analytics sections. Load after board.js and common.js.
   Every chart draws only the values it is given; null values are shown as gaps or "–". */
const ZONE_COL = { treble: "#B83A2B", double: "#2E6F73", single: "#A9B4C2", bull: "#D4A64A", miss: "#D9CDB6" };
const ZONE_NAME = { treble: "Treble", double: "Double", single: "Single", bull: "Bull", miss: "Miss" };
const CH = { ink: "#1B2536", muted: "#6B6F78", grid: "#E6DED0", axis: "#B9B2A4", red: "#B83A2B", teal: "#2E6F73", gold: "#D4A64A", navy: "#1B2536" };
const FONT = 'font-family="IBM Plex Mono, monospace"';

function noData(el, msg = "Not enough data for this view.") {
  if (el) el.innerHTML = `<div class="empty-note">${esc(msg)}</div>`;
}
function zoneLegend(el, keys = ["treble", "double", "single", "bull", "miss"]) {
  if (el) el.innerHTML = keys.map(k => `<span><i style="background:${ZONE_COL[k]}"></i>${ZONE_NAME[k]}</span>`).join("");
}
const pctTxt = (v, d = 0) => (v === null || v === undefined) ? "–" : (v * 100).toFixed(d) + "%";

/* 100% stacked bars, one per label. stacks: [{key, name, color, values: [share 0-1 | null]}] */
function stackedBars(el, labels, stacks, opts = {}) {
  if (!labels.length) return noData(el);
  const W = 560, H = 250, pl = 44, pr = 12, pt = 14, pb = 30;
  const bw = Math.min(56, (W - pl - pr) / labels.length * 0.62), step = (W - pl - pr) / labels.length;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:720px" ${FONT}>`;
  [0, .25, .5, .75, 1].forEach(t => {
    const y = pt + (H - pt - pb) * (1 - t);
    s += `<line x1="${pl}" x2="${W - pr}" y1="${y}" y2="${y}" stroke="${CH.grid}"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end" font-size="12.5" fill="${CH.muted}">${t * 100}%</text>`;
  });
  labels.forEach((lab, i) => {
    const x = pl + i * step + (step - bw) / 2;
    let acc = 0;
    stacks.forEach(st => {
      const v = st.values[i] || 0, h = (H - pt - pb) * v, y = pt + (H - pt - pb) * (1 - acc - v);
      if (v > 0) s += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" fill="${st.color}"><title>${esc(lab)} · ${esc(st.name)}: ${pctTxt(v, 1)}</title></rect>`;
      if (v >= .09) s += `<text x="${x + bw / 2}" y="${y + h / 2 + 4}" text-anchor="middle" font-size="12" fill="${st.key === "single" || st.key === "miss" || st.key === "bull" ? CH.ink : "#fff"}">${Math.round(v * 100)}</text>`;
      acc += v;
    });
    s += `<text x="${x + bw / 2}" y="${H - 10}" text-anchor="middle" font-size="12.5" fill="${CH.muted}">${esc(lab)}</text>`;
  });
  el.innerHTML = s + "</svg>";
}

/* Vertical bars. items: [{label, value, sub?, color?, title?}] */
function barChart(el, items, opts = {}) {
  const vals = items.map(i => i.value).filter(v => v !== null && v !== undefined);
  if (!vals.length) return noData(el);
  const W = 560, H = opts.height || 230, pl = 44, pr = 12, pt = 22, pb = items.some(i => i.sub) ? 46 : 30;
  const max = opts.max || Math.max(...vals) * 1.15 || 1, step = (W - pl - pr) / items.length, bw = Math.min(70, step * .58);
  const fmt = opts.fmt || (v => v.toFixed(1));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:720px" ${FONT}>`;
  for (let k = 0; k <= 4; k++) {
    const y = pt + (H - pt - pb) * k / 4, v = max * (1 - k / 4);
    s += `<line x1="${pl}" x2="${W - pr}" y1="${y}" y2="${y}" stroke="${CH.grid}"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end" font-size="12.5" fill="${CH.muted}">${(opts.axisFmt || (x => x.toFixed(0)))(v)}</text>`;
  }
  items.forEach((it, i) => {
    const x = pl + i * step + (step - bw) / 2, v = it.value;
    if (v !== null && v !== undefined) {
      const h = (H - pt - pb) * v / max, y = H - pb - h;
      s += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="3" fill="${it.color || CH.teal}"><title>${esc(it.title || `${it.label}: ${fmt(v)}`)}</title></rect>`;
      s += `<text x="${x + bw / 2}" y="${y - 6}" text-anchor="middle" font-size="13" font-weight="600" fill="${CH.ink}">${fmt(v)}</text>`;
    } else {
      s += `<text x="${x + bw / 2}" y="${H - pb - 6}" text-anchor="middle" font-size="12.5" fill="${CH.muted}">–</text>`;
    }
    s += `<text x="${x + bw / 2}" y="${H - pb + 16}" text-anchor="middle" font-size="13" fill="${CH.ink}">${esc(it.label)}</text>`;
    if (it.sub) s += `<text x="${x + bw / 2}" y="${H - pb + 32}" text-anchor="middle" font-size="12" fill="${CH.muted}">${esc(it.sub)}</text>`;
  });
  el.innerHTML = s + "</svg>";
}

/* Grouped bars. series: [{name, color, values: [number|null]}], one group per label */
function groupedBars(el, labels, series, opts = {}) {
  const vals = series.flatMap(s => s.values).filter(v => v !== null && v !== undefined);
  if (!vals.length) return noData(el);
  const W = 560, H = opts.height || 240, pl = 44, pr = 12, pt = 20, pb = 30;
  const max = opts.max || Math.max(...vals) * 1.15 || 1, step = (W - pl - pr) / labels.length;
  const bw = Math.min(30, step * .7 / series.length), fmt = opts.fmt || (v => v.toFixed(1));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:720px" ${FONT}>`;
  for (let k = 0; k <= 4; k++) {
    const y = pt + (H - pt - pb) * k / 4;
    s += `<line x1="${pl}" x2="${W - pr}" y1="${y}" y2="${y}" stroke="${CH.grid}"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end" font-size="12.5" fill="${CH.muted}">${(opts.axisFmt || fmt)(max * (1 - k / 4))}</text>`;
  }
  labels.forEach((lab, i) => {
    const gx = pl + i * step + (step - bw * series.length) / 2;
    series.forEach((se, j) => {
      const v = se.values[i];
      if (v === null || v === undefined) return;
      const h = (H - pt - pb) * v / max, x = gx + j * bw, y = H - pb - h;
      s += `<rect x="${x + 1}" y="${y}" width="${bw - 2}" height="${h}" rx="2" fill="${se.color}"><title>${esc(lab)} · ${esc(se.name)}: ${fmt(v)}</title></rect>`;
      if (opts.values !== false) s += `<text x="${x + bw / 2}" y="${y - 5}" text-anchor="middle" font-size="12" fill="${CH.ink}">${fmt(v)}</text>`;
    });
    s += `<text x="${pl + i * step + step / 2}" y="${H - 10}" text-anchor="middle" font-size="13" fill="${CH.ink}">${esc(lab)}</text>`;
  });
  el.innerHTML = s + "</svg>";
}

/* Histogram. bins: [{lo, hi, count}] */
function histogram(el, bins, opts = {}) {
  if (!bins.length || !bins.some(b => b.count)) return noData(el);
  const W = 560, H = 240, pl = 48, pr = 12, pt = 20, pb = 34, max = Math.max(...bins.map(b => b.count)) * 1.12;
  const step = (W - pl - pr) / bins.length, hl = opts.highlightFrom ?? Infinity;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:720px" ${FONT}>`;
  for (let k = 0; k <= 4; k++) {
    const y = pt + (H - pt - pb) * k / 4;
    s += `<line x1="${pl}" x2="${W - pr}" y1="${y}" y2="${y}" stroke="${CH.grid}"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end" font-size="12.5" fill="${CH.muted}">${Math.round(max * (1 - k / 4)).toLocaleString()}</text>`;
  }
  bins.forEach((b, i) => {
    const h = (H - pt - pb) * b.count / max, x = pl + i * step + 2, y = H - pb - h, col = b.lo >= hl ? CH.red : "#9AA4B5";
    s += `<rect x="${x}" y="${y}" width="${step - 4}" height="${h}" fill="${col}"><title>${b.lo}–${b.hi} points: ${b.count.toLocaleString()}</title></rect>`;
    if (b.count) s += `<text x="${x + (step - 4) / 2}" y="${y - 5}" text-anchor="middle" font-size="12" fill="${CH.ink}">${b.count.toLocaleString()}</text>`;
    s += `<text x="${x + (step - 4) / 2}" y="${H - 18}" text-anchor="middle" font-size="12" fill="${CH.muted}">${b.lo}–${b.hi}</text>`;
  });
  s += `<text x="${pl + (W - pl - pr) / 2}" y="${H - 2}" text-anchor="middle" font-size="12" fill="${CH.muted}">${esc(opts.xLabel || "points")}</text>`;
  el.innerHTML = s + "</svg>";
}

/* Radar. axes: [{label}], series: [{name, color, values: [0-1 | null], dash?}] */
function radarChart(el, axes, series) {
  if (!axes.length) return noData(el);
  const W = 420, H = 330, cx = W / 2, cy = 168, R = 100, n = axes.length;
  const pt = (i, v) => { const a = -Math.PI / 2 + i * 2 * Math.PI / n; return [cx + R * v * Math.cos(a), cy + R * v * Math.sin(a)]; };
  let s = `<svg viewBox="-70 0 ${W + 140} ${H}" width="100%" style="display:block;max-width:480px;margin:0 auto" ${FONT}>`;
  [.25, .5, .75, 1].forEach(r => s += `<polygon points="${axes.map((_, i) => pt(i, r).join(",")).join(" ")}" fill="none" stroke="${CH.grid}"/>`);
  axes.forEach((ax, i) => {
    const [x, y] = pt(i, 1), [lx, ly] = pt(i, 1.2);
    s += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="${CH.grid}"/>`;
    s += `<text x="${lx}" y="${ly + 4}" text-anchor="${Math.abs(lx - cx) < 8 ? "middle" : (lx > cx ? "start" : "end")}" font-size="14" fill="${CH.ink}">${esc(ax.label)}</text>`;
  });
  series.forEach(se => {
    const p = se.values.map((v, i) => pt(i, v === null ? 0 : Math.max(.02, v)));
    s += `<polygon points="${p.map(q => q.join(",")).join(" ")}" fill="${se.color}" fill-opacity="${se.dash ? 0 : .18}" stroke="${se.color}" stroke-width="2" ${se.dash ? 'stroke-dasharray="5 4"' : ""}/>`;
    if (!se.dash) p.forEach((q, i) => s += `<circle cx="${q[0]}" cy="${q[1]}" r="3.5" fill="${se.color}"><title>${esc(axes[i].label)}: ${se.values[i] === null ? "not available" : Math.round(se.values[i] * 100) + "% of roster range"}</title></circle>`);
  });
  el.innerHTML = s + "</svg>";
}

/* Polar bars around a board: darts per number. sectors: [{number, count}] in board order */
function sectorChart(svg, data) {
  const sectors = data.sectors || [], max = Math.max(1, ...sectors.map(s => s.count)), total = data.darts || 1;
  svg.setAttribute("viewBox", "-1.25 -1.25 2.5 2.5");
  let s = `<circle r="1.17" fill="#1B2536"/><circle r=".36" fill="#243045"/>`;
  sectors.forEach((sec, i) => {
    const c = 90 - i * 18, r1 = .38 + .62 * (sec.count / max), op = sec.count ? .25 + .75 * sec.count / max : 0;
    s += `<path d="${annulus(.38, 1, c - 8.6, c + 8.6)}" fill="#2E3B52"/>`;
    if (sec.count) s += `<path d="${annulus(.38, r1, c - 8.6, c + 8.6)}" fill="#E0533F" fill-opacity="${op.toFixed(2)}"><title>${sec.number}: ${sec.count.toLocaleString()} darts (${(sec.count / total * 100).toFixed(1)}%)</title></path>`;
    const [x, y] = P(1.09, c);
    s += `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central" font-size=".085" font-weight="700" fill="#C9D0DB" font-family="Space Grotesk, sans-serif">${sec.number}</text>`;
  });
  s += `<text x="0" y="-.05" text-anchor="middle" font-size=".07" fill="#8FD3D6" ${FONT}>BULL ${data.bull ?? 0}</text>`;
  s += `<text x="0" y=".08" text-anchor="middle" font-size=".07" fill="#C9D0DB" ${FONT}>MISS ${data.miss ?? 0}</text>`;
  svg.innerHTML = s;
}

/* Line chart that allows gaps (null) and an optional reference line. series: [{name, color, values, dash?}] */
function trendChart(el, labels, series, opts = {}) {
  const vals = series.flatMap(s => s.values).filter(v => v !== null && v !== undefined);
  if (!vals.length || labels.length < 2) return noData(el, opts.empty || "Needs at least two sessions.");
  const W = 560, H = opts.height || 240, pl = 48, pr = 14, pt = 24, pb = 30;
  let lo = Math.min(...vals, opts.ref ? opts.ref.value : Infinity), hi = Math.max(...vals, opts.ref ? opts.ref.value : -Infinity);
  if (opts.zero) lo = 0;
  const pad = (hi - lo) * .12 || 1; lo = opts.zero ? 0 : lo - pad; hi += pad;
  const X = i => pl + (labels.length === 1 ? 0 : i * (W - pl - pr) / (labels.length - 1)), Y = v => pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo));
  const fmt = opts.fmt || (v => v.toFixed(1));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:720px" ${FONT}>`;
  for (let k = 0; k <= 4; k++) {
    const v = hi - k * (hi - lo) / 4, y = Y(v);
    s += `<line x1="${pl}" x2="${W - pr}" y1="${y}" y2="${y}" stroke="${CH.grid}"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end" font-size="12.5" fill="${CH.muted}">${fmt(v)}</text>`;
  }
  const longest = Math.max(...labels.map(l => String(l).length)), every = Math.max(1, Math.ceil(labels.length / (longest > 6 ? 6 : 10)));
  labels.forEach((l, i) => { if (i % every === 0 || i === labels.length - 1) s += `<text x="${X(i)}" y="${H - 10}" text-anchor="middle" font-size="12" fill="${CH.muted}">${esc(l)}</text>`; });
  if (opts.ref) s += `<line x1="${pl}" x2="${W - pr}" y1="${Y(opts.ref.value)}" y2="${Y(opts.ref.value)}" stroke="${CH.axis}" stroke-dasharray="5 4"/><text x="${W - pr}" y="${pt - 6}" text-anchor="end" font-size="12" fill="${CH.muted}">- - ${esc(opts.ref.label)}</text>`;
  series.forEach(se => {
    let d = "", pen = false;
    se.values.forEach((v, i) => {
      if (v === null || v === undefined) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(i)},${Y(v)} `; pen = true;
    });
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2.4" ${se.dash ? 'stroke-dasharray="6 4"' : ""}/>`;
    se.values.forEach((v, i) => { if (v !== null && v !== undefined) s += `<circle cx="${X(i)}" cy="${Y(v)}" r="3.6" fill="${se.color}"><title>${esc(se.name)} · ${esc(labels[i])}: ${fmt(v)}</title></circle>`; });
  });
  el.innerHTML = s + "</svg>";
}

/* Small horizontal 100% bar of zone shares, for table cells */
function zoneMiniBar(shares) {
  if (!shares) return "–";
  return `<div class="zone-mini" title="${ZONE_KEYS.map(k => `${ZONE_NAME[k]} ${pctTxt(shares[k], 0)}`).join(" · ")}">` +
    ZONE_KEYS.map(k => shares[k] ? `<i style="width:${shares[k] * 100}%;background:${ZONE_COL[k]}"></i>` : "").join("") + "</div>";
}
const ZONE_KEYS = ["treble", "double", "single", "bull", "miss"];
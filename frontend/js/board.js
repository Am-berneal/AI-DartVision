/* Dartboard drawing in SVG. Board coordinates: centre (0,0), x right, y up, double ring outer edge = 1. */
const SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5];
const R = { ib: 6.35 / 170, ob: 15.9 / 170, ti: 99 / 170, to: 107 / 170, di: 162 / 170, do: 1 };
const SVGNS = "http://www.w3.org/2000/svg";
const COL = { dark: "#1E2433", light: "#F3EAD7", red: "#B83A2B", green: "#2E6F73", rim: "#1B2536" };

function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const P = (r, deg) => { const a = deg * Math.PI / 180; return [r * Math.cos(a), -r * Math.sin(a)]; };

function annulus(r0, r1, a1, a2) {
  const [x1, y1] = P(r1, a1), [x2, y2] = P(r1, a2), [x3, y3] = P(r0, a2), [x4, y4] = P(r0, a1);
  return `M${x1},${y1} A${r1},${r1} 0 0 0 ${x2},${y2} L${x3},${y3} A${r0},${r0} 0 0 1 ${x4},${y4} Z`;
}

/* style: "classic" (coloured board) or "schematic" (dark line drawing) */
function drawBoard(svg, opts = {}) {
  const { style = "classic", numbers = true, highlight = null } = opts;
  svg.innerHTML = "";
  svg.setAttribute("viewBox", "-1.22 -1.22 2.44 2.44");
  const g = svgEl("g", {}, svg);
  if (style === "classic") {
    svgEl("circle", { r: 1.17, fill: COL.rim }, g);
    SECTORS.forEach((num, i) => {
      const c = 90 - i * 18, a1 = c - 9, a2 = c + 9, dark = i % 2 === 0;
      [[R.ob, R.ti, dark ? COL.dark : COL.light], [R.ti, R.to, dark ? COL.red : COL.green],
       [R.to, R.di, dark ? COL.dark : COL.light], [R.di, R.do, dark ? COL.red : COL.green]]
        .forEach(([r0, r1, f]) => svgEl("path", { d: annulus(r0, r1, a1, a2), fill: f, stroke: "#B9B2A4", "stroke-width": .004 }, g));
    });
    svgEl("circle", { r: R.ob, fill: COL.green, stroke: "#B9B2A4", "stroke-width": .004 }, g);
    svgEl("circle", { r: R.ib, fill: COL.red, stroke: "#B9B2A4", "stroke-width": .004 }, g);
    svgEl("line", { x1: -1.2, y1: 0, x2: 1.2, y2: 0, stroke: "#8FA3B8", "stroke-width": .004, "stroke-dasharray": ".02 .02", opacity: .6 }, g);
    svgEl("line", { x1: 0, y1: -1.2, x2: 0, y2: 1.2, stroke: "#8FA3B8", "stroke-width": .004, "stroke-dasharray": ".02 .02", opacity: .6 }, g);
  } else {
    svgEl("circle", { r: 1.17, fill: "#1B2536" }, g);
    if (highlight) {
      const i = SECTORS.indexOf(highlight), c = 90 - i * 18;
      svgEl("path", { d: annulus(R.ob, R.do, c - 9, c + 9), fill: "#2E3B52" }, g);
      svgEl("path", { d: annulus(R.ti, R.to, c - 9, c + 9), fill: "#B83A2B", opacity: .75 }, g);
      svgEl("path", { d: annulus(R.di, R.do, c - 9, c + 9), fill: "#2E6F73", opacity: .75 }, g);
    }
    [R.ob, R.ti, R.to, R.di, R.do].forEach(r => svgEl("circle", { r, fill: "none", stroke: "#4A5873", "stroke-width": .006 }, g));
    SECTORS.forEach((_, i) => {
      const a = 90 - i * 18 + 9, [x1, y1] = P(R.ob, a), [x2, y2] = P(R.do, a);
      svgEl("line", { x1, y1, x2, y2, stroke: "#36435C", "stroke-width": .005 }, g);
    });
    svgEl("circle", { r: R.ob, fill: "#2E6F73" }, g);
    svgEl("circle", { r: R.ib, fill: "#B83A2B" }, g);
  }
  if (numbers) {
    SECTORS.forEach((num, i) => {
      const [x, y] = P(1.085, 90 - i * 18);
      const t = svgEl("text", { x, y, "text-anchor": "middle", "dominant-baseline": "central", fill: style === "classic" ? "#fff" : "#8F99AA", "font-size": .085, "font-weight": 700, "font-family": "Space Grotesk, sans-serif" }, g);
      t.textContent = num;
    });
  }
  const layer = svgEl("g", { class: "darts" }, svg);
  return layer;
}

function addDart(layer, x, y, o = {}) {
  const { r = .028, fill = "#F2C14E", stroke = "#1B2536", cls = "", title = "" } = o;
  const c = svgEl("circle", { cx: x, cy: -y, r, fill, stroke, "stroke-width": .008, class: cls }, layer);
  if (title) { const t = svgEl("title", {}, c); t.textContent = title; }
  return c;
}

function addMarker(layer, x, y, text) {
  layer.innerHTML = "";
  svgEl("circle", { cx: x, cy: -y, r: .045, fill: "none", stroke: "#F2C14E", "stroke-width": .012 }, layer);
  svgEl("circle", { cx: x, cy: -y, r: .018, fill: "#F2C14E", stroke: "#1B2536", "stroke-width": .006 }, layer);
  if (text) {
    const w = text.length * .042 + .06, bx = Math.min(Math.max(x + .06, -1.18), 1.18 - w), by = Math.max(-y - .16, -1.18);
    svgEl("rect", { x: bx, y: by, width: w, height: .1, rx: .02, fill: "#1B2536", opacity: .92 }, layer);
    const t = svgEl("text", { x: bx + .03, y: by + .05, "dominant-baseline": "central", fill: "#F2C14E", "font-size": .058, "font-family": "IBM Plex Mono, monospace" }, layer);
    t.textContent = text;
  }
}

/* Density "heatmap": blurred translucent discs per throw */
function addDensity(svg, layer, pts, colour = "#E0533F") {
  const id = "blur" + Math.random().toString(36).slice(2, 7);
  const defs = svgEl("defs", {}, svg);
  const f = svgEl("filter", { id, x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
  svgEl("feGaussianBlur", { stdDeviation: .035 }, f);
  const g = svgEl("g", { filter: `url(#${id})` }, layer);
  const op = Math.max(.05, Math.min(.35, 12 / pts.length));
  pts.forEach(p => svgEl("circle", { cx: p.x, cy: -p.y, r: .04, fill: colour, opacity: op }, g));
  return g;
}

function boardPointFromEvent(svg, evt) {
  const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return { x: p.x, y: -p.y };
}

/* Centre of a scoring region, same rule as dartvision/board.py region_center */
function regionCenter(label) {
  if (label === "IB") return { x: 0, y: 0 };
  if (label === "OB") return { x: 0, y: (R.ib + R.ob) / 2 };
  if (label === "MISS") return { x: 0, y: 1.1 };
  const k = label[0], n = +label.slice(1);
  const r = { T: (R.ti + R.to) / 2, D: (R.di + R.do) / 2, S: (R.to + R.di) / 2 }[k];
  const th = (90 - SECTORS.indexOf(n) * 18) * Math.PI / 180;
  return { x: r * Math.cos(th), y: r * Math.sin(th) };
}
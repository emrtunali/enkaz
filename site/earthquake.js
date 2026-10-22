/* Animated Marmara earthquake. Data: data/anim/m*.json|bin from scripts/istanbul_animation.py,
   cell squares from data/scenario/cells.json. Rendering: MapLibre basemap + deck.gl overlay. */
"use strict";

const { MapboxOverlay, PolygonLayer, ScatterplotLayer, PathLayer, DataFilterExtension } = deck;
const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("en");
const WARM = [[241, 230, 216], [233, 201, 173], [220, 154, 114], [176, 74, 48], [110, 32, 25]];
const MMI_STOPS = [5, 6, 6.7, 7.3, 8];
const ROMAN = (m) => ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][Math.max(0, Math.min(9, Math.round(m) - 1))];
const KM_LAT = 110.57;
const PLACES = {
  Fatih: [28.95, 41.015], "Kadıköy": [29.03, 40.99], "Avcılar": [28.72, 40.98], "Beylikdüzü": [28.64, 41.0],
  "Üsküdar": [29.02, 41.025], "Bağcılar": [28.85, 41.04], "Sarıyer": [29.05, 41.17], Silivri: [28.25, 41.075],
  Pendik: [29.24, 40.88],
};
const DEPTH_KM = 10;

function mmiColor(m) {
  if (m <= MMI_STOPS[0]) return WARM[0];
  for (let i = 1; i < MMI_STOPS.length; i++) {
    if (m <= MMI_STOPS[i]) {
      const f = (m - MMI_STOPS[i - 1]) / (MMI_STOPS[i] - MMI_STOPS[i - 1]);
      return WARM[i - 1].map((v, k) => Math.round(v + f * (WARM[i][k] - v)));
    }
  }
  return WARM[4];
}
const pgaFromMmi = (m) => (m <= 1.78 + 1.55 * 1.57 ? 10 ** ((m - 1.78) / 1.55) : 10 ** ((m + 1.6) / 3.7)) / 980.665;

/* ------------------------------------------------------------------ state */

const st = { t: 0, playing: false, speed: 1, mag: null, data: null, cells: null, idx: null, last: null };

async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(u); return r.json(); }

async function loadMag(tag) {
  const [meta, buf] = await Promise.all([
    getJSON(`data/anim/${tag}.json`),
    fetch(`data/anim/${tag}.bin`).then((r) => r.arrayBuffer()),
  ]);
  const n = new Uint32Array(buf, 0, 1)[0];
  const lon = new Int32Array(buf, 4, n);
  const lat = new Int32Array(buf, 4 + 4 * n, n);
  const t = new Uint16Array(buf, 4 + 8 * n, n);
  const cls = new Uint8Array(buf, 4 + 10 * n, n);
  const [lon0, lat0] = st.idx.origin;
  const pos = new Float32Array(2 * n);
  const time = new Float32Array(n);
  const col = new Uint8Array(4 * n);
  const cumC = new Uint32Array(n + 1);
  const cumE = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) {
    pos[2 * i] = lon0 + lon[i] / 1e5;
    pos[2 * i + 1] = lat0 + lat[i] / 1e5;
    time[i] = t[i] / 10;
    const c = cls[i] === 4 ? [255, 59, 47, 220] : [240, 162, 60, 190];
    col.set(c, 4 * i);
    cumC[i + 1] = cumC[i] + (cls[i] === 4 ? 1 : 0);
    cumE[i + 1] = cumE[i] + (cls[i] === 3 ? 1 : 0);
  }
  // fault geometry in km around the hypocentre
  const [hx, hy] = meta.hypo;
  const kmLon = 111.32 * Math.cos((hy * Math.PI) / 180);
  const tr = meta.trace.map(([x, y]) => [(x - hx) * kmLon, (y - hy) * KM_LAT]);
  const s = [0];
  for (let i = 1; i < tr.length; i++) s.push(s[i - 1] + Math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1]));
  const hi = Math.round(tr.length / 2) - 1;
  const C = meta.cells;
  const base = C.mmi.map(mmiColor);
  const seis = synthSeis(meta);
  st.data = { meta, n, pos, time, col, cumC, cumE, kmLon, trace: meta.trace, s, hi, base, C, seis,
    bins: { length: n, attributes: { getPosition: { value: pos, size: 2 }, getFillColor: { value: col, size: 4 }, getFilterValue: { value: time, size: 1 } } } };
  $("scrub").max = meta.end_s;
  $("calF").textContent = meta.calibration.factor.toFixed(1);
  placesTable();
  summary();
}

/* ------------------------------------------------------------------ geometry helpers */

function circle(r, steps = 160) {
  const { meta, kmLon } = st.data;
  const [hx, hy] = meta.hypo;
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const a = (2 * Math.PI * i) / steps;
    pts.push([hx + (r * Math.cos(a)) / kmLon, hy + (r * Math.sin(a)) / KM_LAT]);
  }
  return pts;
}

function rupturedPath(t) {
  const { trace, s, hi, meta } = st.data;
  const reach = meta.speeds_kms.rupture * t;
  const s0 = s[hi];
  const pts = [];
  for (let i = 0; i < trace.length; i++) if (Math.abs(s[i] - s0) <= reach) pts.push(trace[i]);
  return pts.length > 1 ? pts : [];
}

function frontRadius(v, t) {
  const d = v * t;
  return d > DEPTH_KM ? Math.sqrt(d * d - DEPTH_KM * DEPTH_KM) : 0;
}

function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}

/* ------------------------------------------------------------------ synthetic seismogram */

function synthSeis(meta) {
  // illustrative accelerogram: band-limited noise under P and S envelopes, scaled to the
  // modelled peak acceleration at each place (not a simulated record)
  const dt = 0.02;
  const out = {};
  let seed = Math.round(meta.mag * 1000);
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (const [name, p] of Object.entries(meta.places)) {
    const pga = pgaFromMmi(p.mmi);
    const N = Math.ceil(meta.end_s / dt);
    const comps = Array.from({ length: 14 }, () => ({ f: 1 + 9 * rand(), ph: 2 * Math.PI * rand(), a: 0.5 + rand() }));
    const a = new Float32Array(N);
    let peak = 0;
    for (let i = 0; i < N; i++) {
      const t = i * dt;
      let x = 0;
      for (const c of comps) x += c.a * Math.sin(2 * Math.PI * c.f * t + c.ph);
      const eP = t < p.tp ? 0 : Math.min(1, (t - p.tp) / 0.4) * Math.exp(-(t - p.tp) / 4) * 0.18;
      const eS = t < p.ts ? 0 : Math.min(1, (t - p.ts) / 1.2) * Math.exp(-Math.max(0, t - p.ts - 0.35 * p.dur) / (0.35 * p.dur));
      a[i] = x * (eP + eS);
      peak = Math.max(peak, Math.abs(a[i]));
    }
    for (let i = 0; i < N; i++) a[i] *= pga / peak;
    out[name] = { a, dt, pga };
  }
  return out;
}

function drawSeis() {
  const cv = $("seis");
  const W = (cv.width = cv.clientWidth * devicePixelRatio);
  const H = (cv.height = 64 * devicePixelRatio);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const place = $("seisPlace").textContent;
  const S = st.data.seis[place];
  const scaleG = 0.6;
  ctx.strokeStyle = "rgba(255,255,255,.12)";
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  const n = Math.min(S.a.length, Math.floor(st.t / S.dt));
  const total = S.a.length;
  ctx.strokeStyle = "#e8643a";
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / total) * W;
    const y = H / 2 - (S.a[i] / scaleG) * (H / 2 - 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#a9a395";
  ctx.font = `${10 * devicePixelRatio}px Inter, sans-serif`;
  ctx.fillText(`peak ≈ ${S.pga.toFixed(2)} g`, W - 78 * devicePixelRatio, 12 * devicePixelRatio);
}

/* ------------------------------------------------------------------ layers */

function layers() {
  const D = st.data;
  if (!D) return [];
  const t = st.t;
  const C = D.C;
  const tq = Math.round(t * 20) / 20;
  const L = [];
  // the 1 km intensity squares fade out at street level, where single buildings matter
  const z = st.map ? st.map.getZoom() : 9;
  const cellOpacity = z < 11 ? 1 : Math.max(0.12, 1 - (z - 11) / 3);
  L.push(new PolygonLayer({
    id: "cells", data: st.cells.corners, getPolygon: (c) => c, stroked: false, filled: true, opacity: cellOpacity,
    getFillColor: (_, { index: i }) => {
      if (t < C.tp[i]) return [0, 0, 0, 0];
      if (t < C.ts[i]) return [109, 167, 236, 45];
      const b = D.base[i];
      const end = C.ts[i] + C.dur[i];
      if (t < end) {
        const x = t - C.ts[i];
        const env = Math.min(1, x / 0.8) * Math.exp((-1.4 * x) / C.dur[i]);
        const flick = 0.78 + 0.22 * Math.sin(2 * Math.PI * 2.3 * t + i);
        const a = 95 + 150 * env * flick;
        const lift = 30 * env;
        return [Math.min(255, b[0] + lift), Math.min(255, b[1] + lift * 0.4), b[2], a];
      }
      return [b[0], b[1], b[2], 120];
    },
    updateTriggers: { getFillColor: [tq, st.mag] },
  }));
  L.push(new PathLayer({ id: "fault", data: [{ path: st.fault }], getPath: (d) => d.path, getColor: [150, 150, 160, 160], getWidth: 2, widthUnits: "pixels" }));
  const rp = rupturedPath(t);
  if (rp.length) {
    L.push(new PathLayer({ id: "rupture-glow", data: [{ path: rp }], getPath: (d) => d.path, getColor: [255, 90, 54, 70], getWidth: 12, widthUnits: "pixels", capRounded: true }));
    L.push(new PathLayer({ id: "rupture", data: [{ path: rp }], getPath: (d) => d.path, getColor: [255, 90, 54, 255], getWidth: 4, widthUnits: "pixels", capRounded: true }));
  }
  const rupEnd = D.meta.rupture_s;
  const pulse = t < rupEnd + 2 ? 1 : 0;
  L.push(new ScatterplotLayer({
    id: "hypo", data: [D.meta.hypo], getPosition: (d) => d, getRadius: 1200 + 900 * Math.abs(Math.sin(t * 4)) * pulse,
    getFillColor: [255, 90, 54, pulse ? 200 : 120], stroked: true, getLineColor: [255, 255, 255, 220], lineWidthMinPixels: 1.5,
    radiusUnits: "meters", updateTriggers: { getRadius: tq },
  }));
  const fronts = [];
  const rP = frontRadius(D.meta.speeds_kms.P, t);
  const rS = frontRadius(D.meta.speeds_kms.S, t);
  if (rP > 0 && rP < 220) fronts.push({ path: circle(rP), color: [109, 167, 236, 220], w: 2 });
  if (rS > 0 && rS < 220) fronts.push({ path: circle(rS), color: [232, 100, 58, 240], w: 3.5 });
  if (fronts.length) {
    L.push(new PathLayer({ id: "fronts", data: fronts, getPath: (d) => d.path, getColor: (d) => d.color, getWidth: (d) => d.w, widthUnits: "pixels", updateTriggers: { getPath: tq } }));
  }
  const ext = [new DataFilterExtension({ filterSize: 1 })];
  L.push(new ScatterplotLayer({
    id: "dust", data: D.bins, getRadius: 45, radiusUnits: "meters", radiusMinPixels: 2, radiusMaxPixels: 26,
    getFillColor: [225, 214, 196, 60], filled: true, stroked: false,
    extensions: ext, filterRange: [t - 3, t], filterSoftRange: [t - 0.9, t], filterTransformSize: true, filterTransformColor: true,
    parameters: { depthTest: false },
  }));
  L.push(new ScatterplotLayer({
    id: "buildings", data: D.bins, getRadius: 9, radiusUnits: "meters", radiusMinPixels: 0.7, radiusMaxPixels: 7,
    extensions: ext, filterRange: [-1, t], filterSoftRange: [-1, t - 0.5], filterTransformSize: true, filterTransformColor: true,
    parameters: { depthTest: false },
  }));
  return L;
}

/* ------------------------------------------------------------------ panels */

function placesTable() {
  const P = st.data.meta.places;
  $("places").innerHTML = Object.entries(P).map(([n, p]) =>
    `<tr data-p="${n}"><td><span class="st" data-st="${n}"></span>${n}</td><td class="n">P ${p.tp.toFixed(1)} s · S ${p.ts.toFixed(1)} s · ${ROMAN(p.mmi)}</td></tr>`).join("");
  $("places").querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", () => {
    $("seisPlace").textContent = tr.dataset.p;
    drawSeis();
    const [lon, lat] = PLACES[tr.dataset.p];
    st.map.flyTo({ center: [lon, lat], zoom: 14.2, duration: 1800 });
  }));
}

function events() {
  const m = st.data.meta;
  const f = m.places.Fatih;
  const warn = f.ts - f.tp;
  return [
    [0, `Mw ${m.mag}: the fault starts to break ${DEPTH_KM} km below the sea floor, south of Istanbul.`],
    [0.3, `The rupture runs both ways along the fault at ${m.speeds_kms.rupture} km/s.`],
    [f.tp, `The first jolt (P wave) reaches Fatih. An early-warning system would have about ${warn.toFixed(0)} s before strong shaking.`],
    [f.ts, `Strong shaking (S wave) reaches Fatih: intensity ${ROMAN(f.mmi)}.`],
    [m.rupture_s, `The fault stops breaking after ${m.rupture_s.toFixed(0)} s: ${m.length_km.toFixed(0)} km have ruptured.`],
    [f.ts + f.dur, "Shaking in central Istanbul dies down. Further out it is still going on."],
    [m.end_s - 3, "The shaking is over."],
  ].sort((a, b) => a[0] - b[0]);
}

function updatePanels() {
  const D = st.data;
  const t = st.t;
  $("clock").textContent = `${t.toFixed(1)} s`;
  $("scrub").value = t;
  const k = lowerBound(D.time, t);
  $("sCollapsed").textContent = fmt(D.cumC[k]);
  $("sHeavy").textContent = fmt(D.cumE[k]);
  const C = D.C;
  let shaking = 0, deaths = 0;
  for (let i = 0; i < C.ts.length; i++) {
    const x = (t - C.ts[i]) / C.dur[i];
    if (x >= 0 && x < 1 && C.mmi[i] >= 6) shaking += st.cells.pop[i];
    deaths += C.deaths[i] * Math.min(1, Math.max(0, x));
  }
  $("sShaking").textContent = shaking >= 1e6 ? `${(shaking / 1e6).toFixed(1)}M` : fmt(shaking);
  $("sDeaths").textContent = fmt(deaths);
  const ev = events().filter(([et]) => et <= t + 1e-6).pop();
  $("phase").textContent = ev ? ev[1] : "The fault is locked.";
  for (const [n, p] of Object.entries(D.meta.places)) {
    const el = document.querySelector(`[data-st="${n}"]`);
    if (!el) continue;
    el.className = `st ${t >= p.ts + p.dur ? "done" : t >= p.ts ? "s" : t >= p.tp ? "p" : ""}`;
  }
  drawSeis();
}

function summary() {
  const m = st.data.meta;
  const e = m.expected;
  $("sumTitle").textContent = `After the shaking: Mw ${m.mag}`;
  $("sumText").textContent = `In this simulated earthquake ${fmt(m.drawn.complete)} buildings collapse or are damaged beyond repair ` +
    `and ${fmt(m.drawn.extensive)} more are heavily damaged; the model expects about ${fmt(e.deaths)} deaths at night ` +
    `(90 % of simulated earthquakes: ${fmt(e.deaths_p5_p95[0])}–${fmt(e.deaths_p5_p95[1])}). It is one plausible outcome, ` +
    `drawn from each building's chance of failing; the same magnitude can shake considerably harder or softer.`;
  $("magTable").innerHTML = `<tr><th>Magnitude</th><th>Rupture</th><th class="n">Collapsed</th><th class="n">Heavily damaged</th>` +
    `<th class="n">Deaths</th><th class="n">Fatih: warning</th></tr>` +
    st.idx.magnitudes.map((r) => `<tr><td>Mw ${r.mag}</td><td>${r.length_km.toFixed(0)} km</td><td class="n">${fmt(r.complete)}</td>` +
      `<td class="n">${fmt(r.extensive)}</td><td class="n">${fmt(r.deaths)}</td><td class="n">${r.warning_s.toFixed(1)} s</td></tr>`).join("");
}

/* ------------------------------------------------------------------ main */

(async function main() {
  const [idx, cells, sum] = await Promise.all([
    getJSON("data/anim/index.json"), getJSON("data/scenario/cells.json"), getJSON("data/scenario/summary.json"),
  ]);
  st.idx = idx;
  st.cells = cells;
  st.fault = sum.scenario.trace;
  const map = new maplibregl.Map({
    container: "quakemap", style: "https://tiles.openfreemap.org/styles/dark",
    bounds: [[27.95, 40.62], [29.55, 41.3]], fitBoundsOptions: { padding: { top: 20, bottom: 40, left: 20, right: 20 } },
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  st.map = map;
  map.on("zoomend", () => { if (!st.playing) overlay.setProps({ layers: layers() }); });
  const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
  map.addControl(overlay);
  const render = () => { overlay.setProps({ layers: layers() }); updatePanels(); };

  $("magSeg").innerHTML = idx.magnitudes.map((m) => `<button data-v="${m.tag}" ${m.mag === 7.5 ? 'class="on"' : ""}>${m.mag}</button>`).join("");
  const choose = async (tag) => {
    st.playing = false;
    $("play").textContent = "▶ Play";
    st.mag = tag;
    await loadMag(tag);
    st.t = 0;
    render();
  };
  $("magSeg").querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    $("magSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    await choose(b.dataset.v);
  }));
  $("speedSeg").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $("speedSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    st.speed = +b.dataset.v;
  }));
  $("play").addEventListener("click", () => {
    if (!st.data) return;
    if (st.t >= st.data.meta.end_s) st.t = 0;
    st.playing = !st.playing;
    $("play").textContent = st.playing ? "❚❚ Pause" : "▶ Play";
    st.last = performance.now();
  });
  $("restart").addEventListener("click", () => { st.t = 0; render(); });
  $("overview").addEventListener("click", () => map.fitBounds([[27.95, 40.62], [29.55, 41.3]], { duration: 1500 }));
  $("scrub").addEventListener("input", (e) => { st.t = +e.target.value; render(); });
  const tick = (now) => {
    if (st.playing && st.data) {
      st.t = Math.min(st.data.meta.end_s, st.t + ((now - st.last) / 1000) * st.speed);
      if (st.t >= st.data.meta.end_s) { st.playing = false; $("play").textContent = "▶ Replay"; }
      render();
    }
    st.last = now;
    requestAnimationFrame(tick);
  };
  await choose(idx.magnitudes.find((m) => m.mag === 7.5).tag);
  requestAnimationFrame(tick);
  window.addEventListener("resize", drawSeis);
})();

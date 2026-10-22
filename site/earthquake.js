/* Animated Marmara earthquake. Data: data/anim/* from scripts/istanbul_animation.py, cell squares
   from data/scenario/cells.json. Rendering: MapLibre basemap + deck.gl overlay (GPU time filtering). */
"use strict";

const { MapboxOverlay, PolygonLayer, ScatterplotLayer, PathLayer, LineLayer, DataFilterExtension } = deck;
const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("en");
const short = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}K` : fmt(n));
const WARM = [[241, 230, 216], [233, 201, 173], [220, 154, 114], [176, 74, 48], [110, 32, 25]];
const MMI_STOPS = [5, 6, 6.7, 7.3, 8];
const ROMAN = (m) => ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][Math.max(0, Math.min(9, Math.round(m) - 1))];
const KM_LAT = 110.57;
const DEPTH_KM = 10;
const CUT = [143, 124, 240];
const PLACES = {
  Fatih: [28.95, 41.015], "Kadıköy": [29.03, 40.99], "Avcılar": [28.72, 40.98], "Beylikdüzü": [28.64, 41.0],
  "Üsküdar": [29.02, 41.025], "Bağcılar": [28.85, 41.04], "Sarıyer": [29.05, 41.17], Silivri: [28.25, 41.075],
  Pendik: [29.24, 40.88],
};
const VIEW = [[27.95, 40.62], [29.55, 41.3]];

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

const st = {
  t: 0, playing: false, speed: 1, mag: null, data: null, cells: null, idx: null, last: null, place: "Fatih",
  show: { shaking: true, buildings: true, streets: true, access: false, hospitals: false },
};

async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(u); return r.json(); }
const getBin = (u) => fetch(u).then((r) => r.arrayBuffer());

/* ------------------------------------------------------------------ data */

function parseBuildings(buf) {
  const n = new Uint32Array(buf, 0, 1)[0];
  const lon = new Int32Array(buf, 4, n), lat = new Int32Array(buf, 4 + 4 * n, n);
  const t = new Uint16Array(buf, 4 + 8 * n, n), cls = new Uint8Array(buf, 4 + 10 * n, n);
  const [lon0, lat0] = st.idx.origin;
  const pos = new Float32Array(2 * n), time = new Float32Array(n), col = new Uint8Array(4 * n);
  const cumC = new Uint32Array(n + 1), cumE = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) {
    pos[2 * i] = lon0 + lon[i] / 1e5;
    pos[2 * i + 1] = lat0 + lat[i] / 1e5;
    time[i] = t[i] / 10;
    col.set(cls[i] === 4 ? [255, 59, 47, 220] : [240, 162, 60, 190], 4 * i);
    cumC[i + 1] = cumC[i] + (cls[i] === 4 ? 1 : 0);
    cumE[i + 1] = cumE[i] + (cls[i] === 3 ? 1 : 0);
  }
  return { n, time, cumC, cumE, bins: { length: n, attributes: { getPosition: { value: pos, size: 2 }, getFillColor: { value: col, size: 4 }, getFilterValue: { value: time, size: 1 } } } };
}

function parseRoads(buf) {
  const n = new Uint32Array(buf, 0, 1)[0];
  const c = [0, 1, 2, 3].map((k) => new Int32Array(buf, 4 + 4 * k * n, n));
  const len = new Uint16Array(buf, 4 + 16 * n, n);
  const t = new Uint16Array(buf, 4 + 18 * n, n);
  const cls = new Uint8Array(buf, 4 + 20 * n, n);
  const major = new Set(st.idx.major.map((m) => st.idx.road_classes.indexOf(m)));
  const [lon0, lat0] = st.idx.origin;
  const src = new Float32Array(2 * n), dst = new Float32Array(2 * n), time = new Float32Array(n);
  const col = new Uint8Array(4 * n), width = new Float32Array(n);
  const cumKm = new Float32Array(n + 1), cumMajor = new Float32Array(n + 1);
  for (let i = 0; i < n; i++) {
    src[2 * i] = lon0 + c[0][i] / 1e5; src[2 * i + 1] = lat0 + c[1][i] / 1e5;
    dst[2 * i] = lon0 + c[2][i] / 1e5; dst[2 * i + 1] = lat0 + c[3][i] / 1e5;
    time[i] = t[i] / 10;
    const m = major.has(cls[i]);
    col.set(m ? [255, 59, 47, 255] : [240, 162, 60, 210], 4 * i);
    width[i] = m ? 4 : 1.8;
    cumKm[i + 1] = cumKm[i] + len[i] / 1000;
    cumMajor[i + 1] = cumMajor[i] + (m ? len[i] / 1000 : 0);
  }
  return { n, time, cumKm, cumMajor, bins: { length: n, attributes: {
    getSourcePosition: { value: src, size: 2 }, getTargetPosition: { value: dst, size: 2 },
    getColor: { value: col, size: 4 }, getWidth: { value: width, size: 1 }, getFilterValue: { value: time, size: 1 } } } };
}

async function loadMag(tag) {
  const [meta, bbuf, rbuf] = await Promise.all([getJSON(`data/anim/${tag}.json`), getBin(`data/anim/${tag}.bin`), getBin(`data/anim/${tag}_roads.bin`)]);
  const [hx, hy] = meta.hypo;
  const kmLon = 111.32 * Math.cos((hy * Math.PI) / 180);
  const tr = meta.trace.map(([x, y]) => [(x - hx) * kmLon, (y - hy) * KM_LAT]);
  const s = [0];
  for (let i = 1; i < tr.length; i++) s.push(s[i - 1] + Math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1]));
  const C = meta.cells;
  const cut = C.t0.map((t0, i) => t0 <= 30 && C.t1[i] > 30);
  st.data = { meta, kmLon, s, hi: Math.round(tr.length / 2) - 1, C, cut, base: C.mmi.map(mmiColor),
    B: parseBuildings(bbuf), R: parseRoads(rbuf), seis: synthSeis(meta) };
  $("scrub").max = meta.end_s;
  $("calF").textContent = meta.calibration.factor.toFixed(1);
  placesTable();
  marks();
  summary();
}

/* ------------------------------------------------------------------ geometry */

function circle(r, steps = 160) {
  const { meta, kmLon } = st.data;
  const [hx, hy] = meta.hypo;
  return Array.from({ length: steps + 1 }, (_, i) => {
    const a = (2 * Math.PI * i) / steps;
    return [hx + (r * Math.cos(a)) / kmLon, hy + (r * Math.sin(a)) / KM_LAT];
  });
}
function rupturedPath(t) {
  const { meta, s, hi } = st.data;
  const reach = meta.speeds_kms.rupture * t;
  const pts = meta.trace.filter((_, i) => Math.abs(s[i] - s[hi]) <= reach);
  return pts.length > 1 ? pts : [];
}
const frontRadius = (v, t) => { const d = v * t; return d > DEPTH_KM ? Math.sqrt(d * d - DEPTH_KM * DEPTH_KM) : 0; };
function upperIndex(arr, x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; } return lo; }

/* ------------------------------------------------------------------ seismogram (illustrative) */

function synthSeis(meta) {
  const dt = 0.02, out = {};
  let seed = Math.round(meta.mag * 1000);
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (const [name, p] of Object.entries(meta.places)) {
    const pga = pgaFromMmi(p.mmi), N = Math.ceil(meta.end_s / dt);
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
  const W = (cv.width = cv.clientWidth * devicePixelRatio), H = (cv.height = 56 * devicePixelRatio);
  const ctx = cv.getContext("2d");
  const S = st.data.seis[st.place];
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(255,255,255,.12)";
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  const n = Math.min(S.a.length, Math.floor(st.t / S.dt));
  ctx.strokeStyle = "#e8643a";
  ctx.lineWidth = devicePixelRatio;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / S.a.length) * W, y = H / 2 - (S.a[i] / 0.6) * (H / 2 - 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  $("cardPga").textContent = `${S.pga.toFixed(2)} g`;
}

/* ------------------------------------------------------------------ layers */

function layers() {
  const D = st.data;
  if (!D) return [];
  const t = st.t, C = D.C, tq = Math.round(t * 20) / 20, L = [];
  const z = st.map ? st.map.getZoom() : 9;
  const fade = z < 11 ? 1 : Math.max(0.12, 1 - (z - 11) / 3);
  if (st.show.shaking || st.show.access) {
    L.push(new PolygonLayer({
      id: "cells", data: st.cells.corners, getPolygon: (c) => c, stroked: false, filled: true, opacity: fade,
      getFillColor: (_, { index: i }) => {
        const end = C.ts[i] + C.dur[i];
        if (st.show.access && D.cut[i] && t >= end) return [...CUT, 210];
        if (!st.show.shaking) return [0, 0, 0, 0];
        if (t < C.tp[i]) return [0, 0, 0, 0];
        if (t < C.ts[i]) return [109, 167, 236, 45];
        const b = D.base[i];
        if (t < end) {
          const x = t - C.ts[i], env = Math.min(1, x / 0.8) * Math.exp((-1.4 * x) / C.dur[i]);
          const a = 95 + 150 * env * (0.78 + 0.22 * Math.sin(2 * Math.PI * 2.3 * t + i)), lift = 30 * env;
          return [Math.min(255, b[0] + lift), Math.min(255, b[1] + lift * 0.4), b[2], a];
        }
        return [b[0], b[1], b[2], st.show.access ? 60 : 120];
      },
      updateTriggers: { getFillColor: [tq, st.mag, st.show.shaking, st.show.access] },
    }));
  }
  L.push(new PathLayer({ id: "fault", data: [{ path: st.fault }], getPath: (d) => d.path, getColor: [150, 150, 160, 150], getWidth: 2, widthUnits: "pixels" }));
  const rp = rupturedPath(t);
  if (rp.length) {
    L.push(new PathLayer({ id: "rup-glow", data: [{ path: rp }], getPath: (d) => d.path, getColor: [255, 90, 54, 70], getWidth: 12, widthUnits: "pixels", capRounded: true }));
    L.push(new PathLayer({ id: "rup", data: [{ path: rp }], getPath: (d) => d.path, getColor: [255, 90, 54, 255], getWidth: 4, widthUnits: "pixels", capRounded: true }));
  }
  const pulse = t < D.meta.rupture_s + 2;
  L.push(new ScatterplotLayer({
    id: "hypo", data: [D.meta.hypo], getPosition: (d) => d, radiusUnits: "meters",
    getRadius: 1200 + (pulse ? 900 * Math.abs(Math.sin(t * 4)) : 0), getFillColor: [255, 90, 54, pulse ? 200 : 120],
    stroked: true, getLineColor: [255, 255, 255, 220], lineWidthMinPixels: 1.5, updateTriggers: { getRadius: tq },
  }));
  const fronts = [];
  const rP = frontRadius(D.meta.speeds_kms.P, t), rS = frontRadius(D.meta.speeds_kms.S, t);
  if (rP > 0 && rP < 220) fronts.push({ path: circle(rP), color: [109, 167, 236, 220], w: 2 });
  if (rS > 0 && rS < 220) fronts.push({ path: circle(rS), color: [232, 100, 58, 240], w: 3.5 });
  if (fronts.length) L.push(new PathLayer({ id: "fronts", data: fronts, getPath: (d) => d.path, getColor: (d) => d.color, getWidth: (d) => d.w, widthUnits: "pixels", updateTriggers: { getPath: tq } }));
  const ext = [new DataFilterExtension({ filterSize: 1 })];
  if (st.show.streets) {
    L.push(new LineLayer({
      id: "streets-glow", data: D.R.bins, widthUnits: "pixels", getColor: [255, 120, 60, 90], getWidth: 9,
      extensions: ext, filterRange: [t - 2.5, t], filterSoftRange: [t - 0.8, t], filterTransformColor: true,
    }));
    L.push(new LineLayer({ id: "streets", data: D.R.bins, widthUnits: "pixels", widthMinPixels: 1, extensions: ext, filterRange: [-1, t] }));
  }
  if (st.show.buildings) {
    L.push(new ScatterplotLayer({
      id: "dust", data: D.B.bins, getRadius: 45, radiusUnits: "meters", radiusMinPixels: 2, radiusMaxPixels: 26,
      getFillColor: [225, 214, 196, 60], extensions: ext, filterRange: [t - 3, t], filterSoftRange: [t - 0.9, t],
      filterTransformSize: true, filterTransformColor: true, parameters: { depthTest: false },
    }));
    L.push(new ScatterplotLayer({
      id: "buildings", data: D.B.bins, getRadius: 9, radiusUnits: "meters", radiusMinPixels: 0.7, radiusMaxPixels: 7,
      extensions: ext, filterRange: [-1, t], filterSoftRange: [-1, t - 0.5], filterTransformSize: true, filterTransformColor: true,
      parameters: { depthTest: false },
    }));
  }
  if (st.show.hospitals) {
    L.push(new ScatterplotLayer({
      id: "hospitals", data: st.hospitals, getPosition: (d) => [d[0], d[1]], getRadius: 4, radiusUnits: "pixels",
      getFillColor: [255, 255, 255, 235], stroked: true, getLineColor: [143, 124, 240, 255], lineWidthMinPixels: 2, pickable: true,
    }));
  }
  L.push(new ScatterplotLayer({
    id: "place", data: [PLACES[st.place]], getPosition: (d) => d, getRadius: 7, radiusUnits: "pixels", filled: false, stroked: true,
    getLineColor: [255, 255, 255, 230], lineWidthMinPixels: 2,
  }));
  return L;
}

/* ------------------------------------------------------------------ panels */

function placeState(p, t) { return t >= p.ts + p.dur ? "done" : t >= p.ts ? "s" : t >= p.tp ? "p" : ""; }
const STATE_TEXT = { "": "waiting", p: "first jolt", s: "strong shaking", done: "over" };

function placesTable() {
  const P = st.data.meta.places;
  $("places").innerHTML = Object.entries(P).map(([n, p]) =>
    `<tr data-p="${n}" class="${n === st.place ? "sel" : ""}"><td><span class="st" data-st="${n}"></span>${n}</td>` +
    `<td class="n">S ${p.ts.toFixed(1)} s · ${ROMAN(p.mmi)}</td></tr>`).join("");
  $("places").querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", () => selectPlace(tr.dataset.p, true)));
}

function selectPlace(name, fly) {
  st.place = name;
  document.querySelectorAll("#places tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.p === name));
  if (fly) st.map.flyTo({ center: PLACES[name], zoom: 14.2, duration: 1800 });
  const p = st.data.meta.places[name], r = st.data.meta.routes[name];
  $("cardName").textContent = name;
  $("cardP").textContent = `${p.tp.toFixed(1)} s`;
  $("cardS").textContent = `${p.ts.toFixed(1)} s`;
  $("cardI").textContent = ROMAN(p.mmi);
  $("cardW").textContent = `${(p.ts - p.tp).toFixed(1)} s`;
  const b = r.before, a = r.after;
  $("cardHosp").innerHTML = b.min === null ? "No hospital reachable by road." :
    `Nearest hospital: <b>${b.hospital}</b>, ${b.min.toFixed(1)} min by road. ` +
    (a.min === null ? `<b style="color:#ff6b55">Afterwards it cannot be reached: the streets around are blocked.</b>`
      : a.min > b.min + 0.05 ? `Afterwards <b>${a.min.toFixed(1)} min</b>${a.hospital !== b.hospital ? ` (${a.hospital})` : ""}: detours around debris.`
        : "Still reachable afterwards.");
  render();
}

function marks() {
  const m = st.data.meta, f = m.places.Fatih, end = m.end_s;
  const ev = [[f.tp, "p", "P wave reaches Fatih"], [f.ts, "s", "S wave reaches Fatih"], [m.rupture_s, "r", "the fault stops breaking"]];
  $("marks").innerHTML = ev.map(([t, c, title]) => `<i class="${c}" style="left:${(100 * t) / end}%" title="${title} (${t.toFixed(1)} s)"></i>`).join("");
}

function events() {
  const m = st.data.meta, f = m.places.Fatih;
  return [
    [0, `Mw ${m.mag}: the fault starts to break ${DEPTH_KM} km below the sea floor, south of Istanbul.`],
    [0.3, `The rupture runs both ways along the fault at ${m.speeds_kms.rupture} km/s.`],
    [f.tp, `The first jolt (P wave) reaches Fatih. An early-warning system would have about ${(f.ts - f.tp).toFixed(0)} s before strong shaking.`],
    [f.ts, `Strong shaking reaches Fatih: intensity ${ROMAN(f.mmi)}. Buildings start to fail and debris fills the streets.`],
    [m.rupture_s, `The fault stops breaking after ${m.rupture_s.toFixed(0)} s: ${m.length_km.toFixed(0)} km have ruptured.`],
    [f.ts + f.dur, "Shaking in central Istanbul dies down. Further out it is still going on."],
    [m.end_s - 3, `The shaking is over. ${short(m.roads.cut_off_people)} residents are now more than 30 minutes from a hospital by road.`],
  ].sort((a, b) => a[0] - b[0]);
}

function updatePanels() {
  const D = st.data, t = st.t, C = D.C;
  $("clock").textContent = `${t.toFixed(1)} s`;
  $("scrub").value = t;
  const k = upperIndex(D.B.time, t), kr = upperIndex(D.R.time, t);
  $("sCollapsed").textContent = fmt(D.B.cumC[k]);
  $("sHeavy").textContent = fmt(D.B.cumE[k]);
  $("sStreets").textContent = `${fmt(D.R.cumKm[kr])} km`;
  $("sMajor").textContent = `${D.R.cumMajor[kr].toFixed(1)} km`;
  let shaking = 0, deaths = 0;
  for (let i = 0; i < C.ts.length; i++) {
    const x = (t - C.ts[i]) / C.dur[i];
    if (x >= 0 && x < 1 && C.mmi[i] >= 6) shaking += st.cells.pop[i];
    deaths += C.deaths[i] * Math.min(1, Math.max(0, x));
  }
  $("sShaking").textContent = short(shaking);
  $("sDeaths").textContent = fmt(deaths);
  const ev = events().filter(([et]) => et <= t + 1e-6).pop();
  $("phase").textContent = ev ? ev[1] : "The fault is locked.";
  for (const [n, p] of Object.entries(D.meta.places)) {
    const el = document.querySelector(`[data-st="${n}"]`);
    if (el) el.className = `st ${placeState(p, t)}`;
  }
  const ps = placeState(D.meta.places[st.place], t);
  $("cardState").className = `qtag ${ps}`;
  $("cardState").textContent = STATE_TEXT[ps];
  drawSeis();
}

function summary() {
  const m = st.data.meta, e = m.expected, r = m.roads;
  $("sumTitle").textContent = `After the shaking: Mw ${m.mag}`;
  $("sumText").textContent = `In this simulated earthquake ${fmt(m.drawn.complete)} buildings collapse or are damaged beyond repair ` +
    `and ${fmt(m.drawn.extensive)} more are heavily damaged. Debris blocks ${fmt(r.km)} km of streets, almost all of them ` +
    `narrow residential streets (${r.major_km.toFixed(1)} km on main roads), and leaves ${fmt(r.cut_off_people)} residents more than ` +
    `30 minutes from a hospital by road. The model expects about ${fmt(e.deaths)} deaths at night (90 % of simulated earthquakes: ` +
    `${fmt(e.deaths_p5_p95[0])}–${fmt(e.deaths_p5_p95[1])}). It is one plausible outcome; the same magnitude can shake harder or softer.`;
  $("sumStats").innerHTML = [
    [fmt(m.drawn.complete), "buildings collapsed"], [`${fmt(r.km)} km`, "of streets blocked"],
    [short(r.cut_off_people), "residents cut off from hospitals (> 30 min)"], [fmt(e.deaths), "deaths (model, night)"],
  ].map(([n, l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  $("magTable").innerHTML = `<tr><th>Magnitude</th><th>Rupture</th><th class="n">Collapsed</th><th class="n">Heavily damaged</th>` +
    `<th class="n">Streets blocked</th><th class="n">Cut off</th><th class="n">Deaths</th><th class="n">Warning, Fatih</th></tr>` +
    st.idx.magnitudes.map((x) => `<tr><td>Mw ${x.mag}</td><td>${x.length_km.toFixed(0)} km</td><td class="n">${fmt(x.complete)}</td>` +
      `<td class="n">${fmt(x.extensive)}</td><td class="n">${fmt(x.roads_km)} km</td><td class="n">${short(x.cut_off_people)}</td>` +
      `<td class="n">${fmt(x.deaths)}</td><td class="n">${x.warning_s.toFixed(1)} s</td></tr>`).join("");
}

/* ------------------------------------------------------------------ main */

let render = () => {};

(async function main() {
  const [idx, cells, sum, hosp] = await Promise.all([
    getJSON("data/anim/index.json"), getJSON("data/scenario/cells.json"), getJSON("data/scenario/summary.json"), getJSON("data/anim/hospitals.json"),
  ]);
  Object.assign(st, { idx, cells, fault: sum.scenario.trace, hospitals: hosp });
  const map = new maplibregl.Map({
    container: "quakemap", style: "https://tiles.openfreemap.org/styles/dark",
    bounds: VIEW, fitBoundsOptions: { padding: 20 }, attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  st.map = map;
  const overlay = new MapboxOverlay({
    interleaved: false, layers: [],
    getTooltip: ({ object, layer }) => (layer && layer.id === "hospitals" && object ? { text: object[2] || "Hospital" } : null),
  });
  map.addControl(overlay);
  render = () => { overlay.setProps({ layers: layers() }); updatePanels(); };
  map.on("zoomend", () => { if (!st.playing) render(); });

  $("magSeg").innerHTML = idx.magnitudes.map((m) => `<button data-v="${m.tag}" ${m.mag === 7.5 ? 'class="on"' : ""}>${m.mag}</button>`).join("");
  const choose = async (tag) => {
    st.playing = false;
    $("play").textContent = "▶ Play";
    st.mag = tag;
    await loadMag(tag);
    st.t = 0;
    selectPlace(st.place, false);
  };
  $("magSeg").querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    $("magSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    await choose(b.dataset.v);
  }));
  $("speedSeg").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $("speedSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    st.speed = +b.dataset.v;
  }));
  const toggle = () => {
    if (!st.data) return;
    if (st.t >= st.data.meta.end_s) st.t = 0;
    st.playing = !st.playing;
    $("play").textContent = st.playing ? "❚❚ Pause" : "▶ Play";
    st.last = performance.now();
  };
  $("play").addEventListener("click", toggle);
  document.addEventListener("keydown", (e) => { if (e.code === "Space" && e.target.tagName !== "INPUT") { e.preventDefault(); toggle(); } });
  $("restart").addEventListener("click", () => { st.t = 0; render(); });
  $("overview").addEventListener("click", () => map.fitBounds(VIEW, { duration: 1500, padding: 20 }));
  $("scrub").addEventListener("input", (e) => { st.t = +e.target.value; render(); });
  $("layers").querySelectorAll("input").forEach((cb) => cb.addEventListener("change", () => { st.show[cb.dataset.l] = cb.checked; render(); }));
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

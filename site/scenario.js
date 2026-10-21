/* Istanbul earthquake scenario page. Data: site/data/scenario/*.json from scripts/istanbul_scenario.py. */
"use strict";

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const comma = d3.format(",.0f");
const pct = (x, d = 0) => `${(100 * x).toFixed(d)} %`;
const short = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}K` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : `${Math.round(n)}`);
const getJSON = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(u); return r.json(); };
const q = (arr, p) => d3.quantile([...arr].sort(d3.ascending), p);
const ROMAN = ["V–VI", "VI–VII", "VII–VIII", "VIII–IX", "IX+"];
const GSIM = { AkkarEtAlRjb2014: "Akkar et al. 2014", BooreEtAl2014: "Boore et al. 2014" };
const DS = ["none", "slight", "moderate", "extensive", "complete"];
const DSC = () => [css("--ds0"), css("--ds1"), css("--ds2"), css("--ds3"), css("--ds4")];
const WARM = ["#f1e6d8", "#e9c9ad", "#dc9a72", "#b04a30", "#6e2019"];

/* ------------------------------------------------------------------ tooltip and frames */

const tipEl = () => $("tip");
function tip(e, html) {
  const t = tipEl();
  t.innerHTML = html;
  t.style.display = "block";
  const x = Math.min(e.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  t.style.left = `${x}px`;
  t.style.top = `${e.clientY + 14}px`;
}
const untip = () => { tipEl().style.display = "none"; };

function frame(el, height, m = { t: 10, r: 16, b: 34, l: 48 }) {
  el.innerHTML = "";
  const W = el.clientWidth;
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${height}`).attr("height", height);
  const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
  return { svg, g, w: W - m.l - m.r, h: height - m.t - m.b, m };
}

function keys(el, items) {
  el.innerHTML = items.map(([c, label, kind]) => `<span><i class="${kind || ""}" style="background:${c}"></i>${label}</span>`).join("");
}

const draws = [];
function chart(fn) { draws.push(fn); fn(); }
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => draws.forEach((f) => f()), 150); });

/* ------------------------------------------------------------------ headline */

function headline(S, C) {
  const heavy = C.hist_heavy;
  const deaths = C.hist_deaths;
  const kmBlocked = d3.sum(Object.values(S.roads.km_blocked), (d) => d.mean);
  const a = S.access;
  $("stats").innerHTML = [
    [short(d3.mean(heavy)), "buildings with extensive or complete damage", `90 %: ${short(q(heavy, 0.05))} – ${short(q(heavy, 0.95))}`],
    [short(d3.mean(deaths)), "deaths, earthquake at night", `90 %: ${short(q(deaths, 0.05))} – ${short(q(deaths, 0.95))}`],
    [pct(a.pop_over_30min_after.mean, 1), "of residents more than 30 min by road from a hospital", `before: ${pct(a.pop_over_30min_before, 1)}`],
    [`${comma(kmBlocked)} km`, "of streets blocked by debris", `of ${comma(S.roads.km_total)} km of road`],
  ].map(([n, l, r]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div><div class="r">${r}</div></div>`).join("");
  const d = S.damage;
  $("bLead").textContent = `On average ${short(d.complete.mean)} buildings reach complete damage and ` +
    `${short(d.extensive.mean)} extensive damage, out of ${comma(S.scenario.buildings)}. ` +
    `Most of them are reinforced-concrete frames with infill walls built before modern codes, 4 to 7 storeys high.`;
  const ibbH = S.ibb.very_heavy_plus_heavy;
  $("disagree").innerHTML = `<strong>This scenario is harsher than İBB's.</strong> İBB estimates ${short(ibbH)} very heavily or heavily ` +
    `damaged buildings and ${short(S.ibb.deaths)} deaths for the same magnitude; this one ${short(d3.mean(heavy))} and ${short(d3.mean(deaths))}. ` +
    `Where the damage falls agrees reasonably well (correlation ${S.validation.r_log_heavy_ratio.toFixed(2)} across mahalle), ` +
    `the level does not. The main reason is the fragility model: ESRM20's curves for low-code concrete frames go from slight ` +
    `to complete damage within a factor of two in shaking, so moderate shaking already brings many of them to "complete". ` +
    `Neither model has been tested against an Istanbul earthquake; the gap between them is part of the uncertainty.`;
}

/* ------------------------------------------------------------------ map */

function cellsGeo(Cc) {
  return {
    type: "FeatureCollection",
    features: Cc.corners.map((c, i) => ({
      type: "Feature",
      geometry: { type: "Polygon", coordinates: [[...c, c[0]]] },
      properties: {
        i, mmi: Cc.mmi[i], vs30: Cc.vs30[i], pga: Cc.pga[i], sa10: Cc.sa10[i], n: Cc.buildings[i],
        heavy: Cc.heavy[i], share: Cc.buildings[i] ? Cc.heavy[i] / Cc.buildings[i] : 0, pop: Cc.pop[i],
        t0: Cc.t_base[i], t1: Cc.t_post[i], cut: Cc.t_base[i] > 30 ? -1 : Cc.p_cut[i],
      },
    })),
  };
}

const LAYERS = {
  mmi: {
    color: ["interpolate", ["linear"], ["get", "mmi"], 5, WARM[0], 6, WARM[1], 6.7, WARM[2], 7.3, WARM[3], 8, WARM[4]],
    key: () => WARM.map((c, i) => [c, ["V", "VI", "VI½", "VII", "VIII"][i]]),
    note: "Modified Mercalli intensity from the mean peak ground acceleration over 200 simulated earthquakes (Worden et al. 2012).",
  },
  vs30: {
    color: ["interpolate", ["linear"], ["get", "vs30"], 200, "#104281", 300, "#2a78d6", 420, "#86b6ef", 600, "#cde2fb", 760, "#eef3f9"],
    key: () => [["#104281", "soft, 200 m/s"], ["#2a78d6", "300"], ["#86b6ef", "420"], ["#cde2fb", "600"], ["#eef3f9", "rock, 760"]],
    note: "Average shear-wave velocity of the top 30 m (Vs30), estimated from topographic slope. Soft ground (dark) amplifies shaking.",
  },
  heavy: {
    color: ["interpolate", ["linear"], ["get", "share"], 0, WARM[0], 0.05, WARM[1], 0.12, WARM[2], 0.22, WARM[3], 0.35, WARM[4]],
    key: () => WARM.map((c, i) => [c, ["0 %", "5 %", "12 %", "22 %", "35 %+"][i]]),
    note: "Share of buildings in each square expected to reach extensive or complete damage.",
  },
  roads: {
    color: ["interpolate", ["linear"], ["get", "mmi"], 5, "#f3efe7", 8, "#e7dccb"],
    key: () => [["#dc9a72", "blocked in 25–50 % of runs", "line"], ["#b04a30", "50–75 %", "line"], ["#6e2019", "75 %+", "line"]],
    note: "Main streets (tertiary and above) blocked by debris in at least a quarter of the simulated earthquakes. Narrow residential streets, where most blockages happen, are left out for legibility.",
  },
  access: {
    color: ["case", ["<", ["get", "cut"], 0], "#c9c3b8",
      ["interpolate", ["linear"], ["get", "cut"], 0, "#f3efe7", 0.1, WARM[1], 0.3, WARM[2], 0.6, WARM[3], 0.9, WARM[4]]],
    key: () => [["#f3efe7", "0 %"], [WARM[1], "10 %"], [WARM[2], "30 %"], [WARM[3], "60 %"], [WARM[4], "90 %"],
      ["#c9c3b8", "over 30 min even before"], ["#fff", "hospital", "dot"]],
    note: "Chance that a square, normally within 30 minutes of a hospital by road, is cut off beyond 30 minutes once blocked streets are taken out. Grey: rural squares already further away before the earthquake.",
  },
};

function setupMap(S, Cc, roads, hosp) {
  const map = new maplibregl.Map({
    container: "mapview", style: "https://tiles.openfreemap.org/styles/positron",
    bounds: [[28.0, 40.72], [29.95, 41.33]], fitBoundsOptions: { padding: 10 }, attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }));
  const roadFC = {
    type: "FeatureCollection",
    features: roads.map(([p, a, b, c, d]) => ({ type: "Feature", properties: { p }, geometry: { type: "LineString", coordinates: [[a, b], [c, d]] } })),
  };
  const hospFC = {
    type: "FeatureCollection",
    features: hosp.map(([lon, lat, name, pga, mmi]) => ({ type: "Feature", properties: { name, pga, mmi }, geometry: { type: "Point", coordinates: [lon, lat] } })),
  };
  map.on("load", () => {
    map.addSource("cells", { type: "geojson", data: cellsGeo(Cc) });
    map.addLayer({ id: "cells", type: "fill", source: "cells", paint: { "fill-color": LAYERS.mmi.color, "fill-opacity": 0.82 } });
    map.addSource("roads", { type: "geojson", data: roadFC });
    map.addLayer({
      id: "roads", type: "line", source: "roads", layout: { visibility: "none", "line-cap": "round" },
      paint: {
        "line-color": ["step", ["get", "p"], "#dc9a72", 0.5, "#b04a30", 0.75, "#6e2019"],
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.5, 14, 4],
      },
    });
    map.addSource("hosp", { type: "geojson", data: hospFC });
    map.addLayer({
      id: "hosp", type: "circle", source: "hosp", layout: { visibility: "none" },
      paint: { "circle-radius": 3.5, "circle-color": "#fff", "circle-stroke-color": "#1d1b18", "circle-stroke-width": 1.2 },
    });
    map.addSource("fault", { type: "geojson", data: { type: "Feature", geometry: { type: "LineString", coordinates: S.scenario.trace } } });
    map.addLayer({ id: "fault-halo", type: "line", source: "fault", paint: { "line-color": "#fff", "line-width": 6 } });
    map.addLayer({ id: "fault", type: "line", source: "fault", paint: { "line-color": "#1d1b18", "line-width": 3 } });
    map.on("click", "cells", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat).setHTML(
        `Intensity <b>${p.mmi.toFixed(1)}</b> · PGA <b>${p.pga.toFixed(2)} g</b><br>Vs30 ${p.vs30} m/s<br>` +
        `${comma(p.n)} buildings, ~${comma(p.heavy)} heavily damaged<br>Hospital: ${p.t0.toFixed(0)} min before, ` +
        `${p.t1 > 900 ? "cut off" : `${p.t1.toFixed(0)} min after`} (median)`).addTo(map);
    });
    map.on("click", "hosp", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat).setHTML(`<b>${p.name || "Hospital"}</b><br>Intensity ${p.mmi}, PGA ${p.pga} g`).addTo(map);
    });
    ["cells", "hosp"].forEach((l) => {
      map.on("mouseenter", l, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", l, () => { map.getCanvas().style.cursor = ""; });
    });
  });
  const show = (v) => {
    const L = LAYERS[v];
    keys($("mapKey"), L.key());
    $("mapNote").textContent = L.note;
    if (!map.getLayer("cells")) return;
    map.setPaintProperty("cells", "fill-color", L.color);
    map.setPaintProperty("cells", "fill-opacity", v === "roads" ? 0.6 : 0.82);
    map.setLayoutProperty("roads", "visibility", v === "roads" ? "visible" : "none");
    map.setLayoutProperty("hosp", "visibility", v === "access" ? "visible" : "none");
  };
  document.querySelectorAll("#mapSeg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#mapSeg button").forEach((x) => x.classList.toggle("on", x === b));
    show(b.dataset.v);
  }));
  show("mmi");
  map.on("load", () => show(document.querySelector("#mapSeg button.on").dataset.v));
}

/* ------------------------------------------------------------------ charts */

function attenuation(C) {
  const el = $("cAtten");
  const A = C.attenuation;
  const series = [["AkkarEtAlRjb2014", css("--s1")], ["BooreEtAl2014", css("--s2")]];
  const f = frame(el, 300);
  const x = d3.scaleLinear().domain([0, 90]).range([0, f.w]);
  const y = d3.scaleLinear().domain([0, 0.7]).range([f.h, 0]);
  f.g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(7).tickSize(-f.w).tickFormat(""));
  const sc = C.cells_scatter;
  f.g.append("g").selectAll("circle").data(sc.rjb.map((r, i) => [r, sc.pga[i]]).filter((d) => d[0] <= 90))
    .join("circle").attr("cx", (d) => x(d[0])).attr("cy", (d) => y(Math.min(d[1], 0.7))).attr("r", 1.6)
    .attr("fill", css("--muted")).attr("opacity", 0.25);
  for (const [g, c] of series) {
    const s = A[`${g}|PGA|300`];
    const pts = s.rjb.map((r, i) => ({ r, lo: s.p16[i], mid: s.p50[i], hi: s.p84[i] })).filter((d) => d.r <= 90);
    f.g.append("path").datum(pts).attr("fill", c).attr("opacity", 0.12)
      .attr("d", d3.area().x((d) => x(d.r)).y0((d) => y(d.lo)).y1((d) => y(Math.min(d.hi, 0.7))));
    f.g.append("path").datum(pts).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2).attr("stroke-linejoin", "round")
      .attr("d", d3.line().x((d) => x(d.r)).y((d) => y(d.mid)));
  }
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(9));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(7).tickFormat((d) => `${d} g`));
  f.g.append("text").attr("class", "lbl muted").attr("x", f.w).attr("y", f.h + 30).attr("text-anchor", "end").text("distance from the rupture (km)");
  // crosshair
  const hair = f.g.append("line").attr("stroke", css("--ink")).attr("stroke-width", 1).attr("y1", 0).attr("y2", f.h).style("display", "none");
  f.g.append("rect").attr("width", f.w).attr("height", f.h).attr("fill", "transparent")
    .on("mousemove", (e) => {
      const r = x.invert(d3.pointer(e)[0]);
      hair.style("display", null).attr("x1", x(r)).attr("x2", x(r));
      const rows = series.map(([g]) => {
        const s = A[`${g}|PGA|300`];
        const i = d3.minIndex(s.rjb, (v) => Math.abs(v - r));
        return `${GSIM[g]}: <b>${s.p50[i].toFixed(2)} g</b> (${s.p16[i].toFixed(2)}–${s.p84[i].toFixed(2)})`;
      });
      tip(e, `<b>${r.toFixed(0)} km</b><br>${rows.join("<br>")}`);
    })
    .on("mouseleave", () => { hair.style("display", "none"); untip(); });
  keys($("kAtten"), [[css("--s1"), GSIM.AkkarEtAlRjb2014, "line"], [css("--s2"), GSIM.BooreEtAl2014, "line"], [css("--muted"), "1 km cells (all site conditions)", "dot"]]);
  $("kAtten").querySelectorAll("i.dot").forEach((i) => { i.style.borderRadius = "50%"; i.style.opacity = 0.5; i.style.width = "8px"; i.style.height = "8px"; });
}

function hbars(el, rows, opt) {
  // rows: [{label, value, lo?, hi?, color, note?, marker?}]
  const bh = opt.bar || 16;
  const gap = opt.gap || 10;
  const height = rows.length * (bh + gap) + 34;
  const f = frame(el, height, { t: 4, r: opt.right || 70, b: 28, l: opt.left || 110 });
  const xmax = opt.max || d3.max(rows, (d) => Math.max(d.hi ?? d.value, d.marker ?? 0)) * 1.05;
  const x = d3.scaleLinear().domain([0, xmax]).range([0, f.w]).nice();
  f.g.append("g").attr("class", "grid").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(opt.ticks || 4).tickSize(-f.h).tickFormat(""));
  rows.forEach((d, i) => {
    const y = i * (bh + gap);
    const g = f.g.append("g").attr("transform", `translate(0,${y})`);
    g.append("text").attr("class", "lbl").attr("x", -8).attr("y", bh / 2 + 4).attr("text-anchor", "end").text(d.label);
    const w = Math.max(x(d.value), 1);
    g.append("path").attr("fill", d.color).attr("d", `M0,0 H${w - 4} Q${w},0 ${w},4 V${bh - 4} Q${w},${bh} ${w - 4},${bh} H0 Z`);
    if (d.lo != null) {
      g.append("line").attr("x1", x(d.lo)).attr("x2", x(d.hi)).attr("y1", bh / 2).attr("y2", bh / 2).attr("stroke", css("--ink")).attr("stroke-width", 1.2);
      [d.lo, d.hi].forEach((v) => g.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", bh / 2 - 4).attr("y2", bh / 2 + 4).attr("stroke", css("--ink")).attr("stroke-width", 1.2));
    }
    if (d.marker != null) {
      g.append("circle").attr("cx", x(d.marker)).attr("cy", bh / 2).attr("r", 4.5).attr("fill", css("--s2")).attr("stroke", css("--paper")).attr("stroke-width", 2);
    }
    g.append("text").attr("class", "lbl muted").attr("x", x(Math.max(d.hi ?? d.value, d.marker ?? 0)) + 8).attr("y", bh / 2 + 4).text(d.note ?? short(d.value));
    g.append("rect").attr("x", -opt.left || -110).attr("width", f.w + (opt.left || 110)).attr("height", bh + gap).attr("y", -gap / 2).attr("fill", "transparent")
      .on("mousemove", (e) => tip(e, d.tip || `<b>${d.label}</b><br>${short(d.value)}`)).on("mouseleave", untip);
  });
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(opt.ticks || 4).tickFormat(opt.fmt || short));
}

function exposure(S) {
  const E = S.exposure;
  const el = $("cExposure");
  el.innerHTML = "";
  const panels = [["Buildings", E.buildings], ["Residents (night)", E.population], ["Hospitals", E.hospitals]];
  for (const [title, vals] of panels) {
    const box = document.createElement("div");
    box.innerHTML = `<p class="chart-title" style="margin-top:6px">${title}</p><div class="chart"></div>`;
    el.appendChild(box);
    const rows = vals.slice(0, 4).map((v, i) => ({ label: ROMAN[i], value: v, color: WARM[i + 1],
      tip: `<b>${title}</b>, intensity ${ROMAN[i]}<br>${comma(v)} (${pct(v / d3.sum(vals), 1)})` }));
    hbars(box.querySelector(".chart"), rows, { left: 62, right: 46, bar: 18, gap: 10, ticks: 3 });
  }
}

function states(S) {
  const el = $("cStates");
  const d = S.damage;
  const vals = DS.map((k) => d[k].mean);
  const tot = d3.sum(vals);
  const f = frame(el, 70, { t: 6, r: 2, b: 30, l: 2 });
  const x = d3.scaleLinear().domain([0, tot]).range([0, f.w]);
  let acc = 0;
  const cols = DSC();
  vals.forEach((v, i) => {
    const x0 = x(acc);
    const w = Math.max(x(acc + v) - x0 - (i < 4 ? 2 : 0), 1);
    acc += v;
    const g = f.g.append("g");
    g.append("rect").attr("x", x0).attr("width", w).attr("height", 28).attr("rx", i === 4 ? 4 : 0).attr("fill", cols[i])
      .on("mousemove", (e) => tip(e, `<b>${DS[i]}</b><br>${comma(v)} buildings (${pct(v / tot, 1)})<br>90 %: ${short(d[DS[i]].p5)}–${short(d[DS[i]].p95)}`))
      .on("mouseleave", untip);
    if (w > 54) {
      g.append("text").attr("x", x0 + 6).attr("y", 46).attr("class", "lbl").text(pct(v / tot, 1));
    }
  });
  keys($("kStates"), DS.map((k, i) => [cols[i], `${k} ${short(vals[i])}`]));
}

function histogram(S, C) {
  const el = $("cHist");
  const v = C.hist_heavy;
  const g = C.gsim_of_r;
  const f = frame(el, 240);
  const x = d3.scaleLinear().domain([0, d3.max(v) * 1.02]).range([0, f.w]).nice();
  const bins = d3.bin().domain(x.domain()).thresholds(x.ticks(30));
  const b0 = bins(v.filter((_, i) => g[i] === 0));
  const b1 = bins(v.filter((_, i) => g[i] === 1));
  const y = d3.scaleLinear().domain([0, d3.max(b0, (b, i) => b.length + b1[i].length)]).range([f.h, 0]).nice();
  f.g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(4).tickSize(-f.w).tickFormat(""));
  const c0 = css("--s1");
  const c1 = css("--s2");
  b0.forEach((b, i) => {
    const bw = Math.max(x(b.x1) - x(b.x0) - 2, 1);
    const h0 = f.h - y(b.length);
    const h1 = f.h - y(b1[i].length);
    const gx = f.g.append("g").attr("transform", `translate(${x(b.x0) + 1},0)`);
    gx.append("rect").attr("y", y(b.length)).attr("width", bw).attr("height", Math.max(h0 - (h1 > 0 ? 1 : 0), 0)).attr("fill", c0);
    if (h1 > 0) gx.append("path").attr("fill", c1).attr("d", `M0,${y(b.length) - 1} V${y(b.length) - h1 + 3} Q0,${y(b.length) - h1 - 1} 4,${y(b.length) - h1 - 1} H${bw - 4} Q${bw},${y(b.length) - h1 - 1} ${bw},${y(b.length) - h1 + 3} V${y(b.length) - 1} Z`);
    gx.append("rect").attr("width", bw + 2).attr("x", -1).attr("height", f.h).attr("fill", "transparent")
      .on("mousemove", (e) => tip(e, `<b>${short(b.x0)}–${short(b.x1)}</b> heavily damaged<br>${GSIM.AkkarEtAlRjb2014}: ${b.length} runs<br>${GSIM.BooreEtAl2014}: ${b1[i].length} runs`))
      .on("mouseleave", untip);
  });
  const mark = (val, label, dy) => {
    f.g.append("line").attr("x1", x(val)).attr("x2", x(val)).attr("y1", 0).attr("y2", f.h).attr("stroke", css("--ink")).attr("stroke-width", 1.5);
    f.g.append("text").attr("class", "lbl").attr("x", x(val) + 6).attr("y", 12 + dy).text(label);
  };
  mark(S.ibb.very_heavy_plus_heavy, `İBB estimate ${short(S.ibb.very_heavy_plus_heavy)}`, 0);
  mark(d3.mean(v), `mean ${short(d3.mean(v))}`, 16);
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(6).tickFormat(short));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
  f.g.append("text").attr("class", "lbl muted").attr("x", f.w).attr("y", f.h + 30).attr("text-anchor", "end").text("buildings with extensive or complete damage");
  f.g.append("text").attr("class", "lbl muted").attr("x", -40).attr("y", -2).text("runs");
  keys($("kHist"), [[c0, GSIM.AkkarEtAlRjb2014], [c1, GSIM.BooreEtAl2014]]);
}

function districts(C) {
  const rows = [...C.districts].sort((a, b) => b.heavy - a.heavy).map((d) => ({
    label: d.d, value: d.heavy, lo: d.lo, hi: d.hi, marker: d.ibb_heavy, color: css("--ds3"),
    note: `${short(d.heavy)} · İBB ${short(d.ibb_heavy)}`,
    tip: `<b>${d.d}</b> (${comma(d.buildings)} buildings)<br>This scenario: <b>${comma(d.heavy)}</b> (90 %: ${comma(d.lo)}–${comma(d.hi)})<br>İBB: <b>${comma(d.ibb_heavy)}</b><br>Deaths: ${comma(d.deaths)}`,
  }));
  hbars($("cDistricts"), rows, { left: 110, right: 96, bar: 12, gap: 8, ticks: 5 });
  keys($("kDistricts"), [[css("--ds3"), "this scenario, mean (line: 90 % range)"], [css("--s2"), "İBB estimate", "dot"]]);
  $("kDistricts").querySelectorAll("i.dot").forEach((i) => { i.style.borderRadius = "50%"; });
}

function validation(C, names) {
  const el = $("cValid");
  const V = C.validation;
  const pts = V.ours.map((o, i) => ({ i, o: (1000 * o) / V.n[i], b: (1000 * V.ibb[i]) / V.n[i], n: V.n[i] }))
    .filter((d) => V.has[d.i] && d.n >= 50 && d.o > 0 && d.b > 0);
  const size = Math.min(el.clientWidth, 520);
  el.style.maxWidth = `${size}px`;
  const f = frame(el, size, { t: 10, r: 16, b: 40, l: 52 });
  const ext = [1, 1000];
  const x = d3.scaleLog().domain(ext).range([0, f.w]);
  const y = d3.scaleLog().domain(ext).range([f.h, 0]);
  const dec = [1, 10, 100, 1000];
  f.g.append("g").attr("class", "grid").call(d3.axisLeft(y).tickValues(dec).tickSize(-f.w).tickFormat(""));
  f.g.append("g").attr("class", "grid").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).tickValues(dec).tickSize(-f.h).tickFormat(""));
  f.g.append("line").attr("x1", x(1)).attr("y1", y(1)).attr("x2", x(1000)).attr("y2", y(1000)).attr("stroke", css("--muted")).attr("stroke-width", 1);
  f.g.append("text").attr("class", "lbl muted").attr("x", x(400)).attr("y", y(400) - 8).attr("text-anchor", "end").text("same estimate");
  f.g.append("g").selectAll("circle").data(pts).join("circle")
    .attr("cx", (d) => x(Math.min(Math.max(d.b, 1), 1000))).attr("cy", (d) => y(Math.min(Math.max(d.o, 1), 1000))).attr("r", 4)
    .attr("fill", css("--s1")).attr("opacity", 0.55).attr("stroke", css("--paper")).attr("stroke-width", 1)
    .on("mousemove", (e, d) => tip(e, `<b>${names[d.i]}</b><br>${comma(d.n)} buildings<br>This scenario: ${d.o.toFixed(0)} per 1,000<br>İBB: ${d.b.toFixed(0)} per 1,000`))
    .on("mouseleave", untip);
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).tickValues(dec).tickFormat(d3.format(",")));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickValues(dec).tickFormat(d3.format(",")));
  f.g.append("text").attr("class", "lbl muted").attr("x", f.w).attr("y", f.h + 34).attr("text-anchor", "end").text("İBB: heavily damaged per 1,000 buildings");
  f.g.append("text").attr("class", "lbl muted").attr("x", 6).attr("y", 4).text("this scenario, per 1,000");
}

function fragility(C) {
  const el = $("cFrag");
  el.innerHTML = "";
  const panels = [
    ["CR_LFINF-CDL-15_H5", "Concrete frame, low code, 5 storeys", "SA(1.0 s)"],
    ["MUR-CL99_LWAL-DNO_H2", "Unreinforced masonry, 2 storeys", "SA(0.3 s)"],
    ["CR_LWAL-DUH_H5", "Concrete walls, modern code, 5 storeys", "SA(0.3 s)"],
  ];
  const cols = DSC();
  for (const [k, title, imt] of panels) {
    const F = C.fragility[k];
    const box = document.createElement("div");
    box.innerHTML = `<p class="chart-title" style="margin-top:6px;font-size:14px;min-height:2.6em">${title}</p><div class="chart"></div>`;
    el.appendChild(box);
    const f = frame(box.querySelector(".chart"), 200, { t: 8, r: 10, b: 36, l: 40 });
    const x = d3.scaleLog().domain([0.02, 3]).range([0, f.w]);
    const y = d3.scaleLinear().domain([0, 1]).range([f.h, 0]);
    f.g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(4).tickSize(-f.w).tickFormat(""));
    F.exceed.forEach((curve, j) => {
      f.g.append("path").datum(curve).attr("fill", "none").attr("stroke", cols[j + 1]).attr("stroke-width", 2)
        .attr("d", d3.line().x((_, i) => x(F.im[i])).y((d) => y(d)));
    });
    f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).tickValues([0.05, 0.1, 0.3, 1, 3]).tickFormat((d) => d));
    f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat((d) => `${d * 100}%`));
    f.g.append("text").attr("class", "lbl muted").attr("x", f.w).attr("y", f.h + 32).attr("text-anchor", "end").text(`${imt}, g`);
    const hair = f.g.append("line").attr("stroke", css("--ink")).attr("y1", 0).attr("y2", f.h).style("display", "none");
    f.g.append("rect").attr("width", f.w).attr("height", f.h).attr("fill", "transparent")
      .on("mousemove", (e) => {
        const v = x.invert(d3.pointer(e)[0]);
        const i = d3.minIndex(F.im, (a) => Math.abs(Math.log(a / v)));
        hair.style("display", null).attr("x1", x(F.im[i])).attr("x2", x(F.im[i]));
        tip(e, `<b>${imt} = ${F.im[i].toFixed(2)} g</b><br>${DS.slice(1).map((n, j) => `${n} or worse: ${pct(F.exceed[j][i])}`).join("<br>")}`);
      })
      .on("mouseleave", () => { hair.style("display", "none"); untip(); });
  }
  keys($("kFrag"), DS.slice(1).map((n, j) => [cols[j + 1], `${n} or worse`, "line"]));
}

function roadsChart(S) {
  const K = S.roads.km_blocked;
  const order = ["residential", "service", "living_street", "unclassified", "tertiary", "secondary", "primary", "trunk", "motorway"];
  const names = { residential: "residential", service: "service", living_street: "living street", unclassified: "unclassified", tertiary: "tertiary", secondary: "secondary", primary: "primary", trunk: "trunk", motorway: "motorway" };
  const rows = order.filter((c) => K[c]).map((c) => ({
    label: names[c], value: K[c].mean, lo: K[c].p5, hi: K[c].p95, color: css("--ds3"),
    note: `${comma(K[c].mean)} of ${comma(K[c].km)} km (${pct(K[c].mean / K[c].km, 1)})`,
    tip: `<b>${names[c]}</b><br>${comma(K[c].mean)} km blocked of ${comma(K[c].km)} km<br>90 %: ${comma(K[c].p5)}–${comma(K[c].p95)} km`,
  }));
  hbars($("cRoads"), rows, { left: 100, right: 170, bar: 14, gap: 9, ticks: 4, fmt: (d) => `${comma(d)} km` });
}

function access(C, S) {
  const el = $("cAccess");
  const A = C.access_cdf;
  const f = frame(el, 280);
  const x = d3.scaleLinear().domain([0, 60]).range([0, f.w]);
  const y = d3.scaleLinear().domain([0.5, 1]).range([f.h, 0]);
  f.g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-f.w).tickFormat(""));
  const line = d3.line().x((_, i) => x(A.x[i])).y((d) => y(Math.max(d, 0.5)));
  A.after.forEach((c) => f.g.append("path").datum(c).attr("fill", "none").attr("stroke", css("--s2")).attr("stroke-width", 1).attr("opacity", 0.3).attr("d", line));
  f.g.append("path").datum(A.after_median).attr("fill", "none").attr("stroke", css("--s2")).attr("stroke-width", 2).attr("d", line);
  f.g.append("path").datum(A.before).attr("fill", "none").attr("stroke", css("--s1")).attr("stroke-width", 2).attr("d", line);
  f.g.append("line").attr("x1", x(30)).attr("x2", x(30)).attr("y1", 0).attr("y2", f.h).attr("stroke", css("--muted")).attr("stroke-width", 1);
  f.g.append("text").attr("class", "lbl muted").attr("x", x(30) + 6).attr("y", f.h - 8).text("30 min");
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(6).tickFormat((d) => `${d} min`));
  f.g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat((d) => `${Math.round(d * 100)}%`));
  const hair = f.g.append("line").attr("stroke", css("--ink")).attr("y1", 0).attr("y2", f.h).style("display", "none");
  f.g.append("rect").attr("width", f.w).attr("height", f.h).attr("fill", "transparent")
    .on("mousemove", (e) => {
      const i = Math.max(0, Math.min(60, Math.round(x.invert(d3.pointer(e)[0]))));
      hair.style("display", null).attr("x1", x(i)).attr("x2", x(i));
      tip(e, `<b>within ${i} min</b><br>before: ${pct(A.before[i], 1)}<br>after (median run): ${pct(A.after_median[i], 1)}`);
    })
    .on("mouseleave", () => { hair.style("display", "none"); untip(); });
  keys($("kAccess"), [[css("--s1"), "before the earthquake", "line"], [css("--s2"), "after, median run (thin: single runs)", "line"]]);
}

function rail(S) {
  const R = S.exposure.rail_km;
  const kinds = [["subway", "Metro"], ["rail", "Rail (Marmaray, mainline)"], ["tram", "Tram"]];
  const el = $("cRail");
  const f = frame(el, kinds.length * 34 + 30, { t: 4, r: 50, b: 26, l: 150 });
  const x = d3.scaleLinear().domain([0, d3.max(kinds, ([k]) => d3.sum(R[k] || [0]))]).range([0, f.w]).nice();
  kinds.forEach(([k, name], i) => {
    const vals = R[k] || [0, 0, 0, 0, 0];
    let acc = 0;
    const g = f.g.append("g").attr("transform", `translate(0,${i * 34})`);
    g.append("text").attr("class", "lbl").attr("x", -8).attr("y", 13).attr("text-anchor", "end").text(name);
    vals.forEach((v, j) => {
      if (v <= 0) return;
      const x0 = x(acc);
      acc += v;
      g.append("rect").attr("x", x0).attr("width", Math.max(x(acc) - x0 - 2, 1)).attr("height", 18).attr("fill", WARM[j + 1])
        .on("mousemove", (e) => tip(e, `<b>${name}</b>, intensity ${ROMAN[j]}<br>${v.toFixed(0)} km`)).on("mouseleave", untip);
    });
    g.append("text").attr("class", "lbl muted").attr("x", x(acc) + 6).attr("y", 13).text(`${comma(acc)} km`);
  });
  f.g.append("g").attr("class", "axis").attr("transform", `translate(0,${kinds.length * 34 - 8})`).call(d3.axisBottom(x).ticks(4).tickFormat((d) => `${d} km`));
  keys($("kRail"), ROMAN.slice(0, 4).map((r, j) => [WARM[j + 1], r]));
}

function bridges(S) {
  const B = S.exposure.named_bridges;
  const rows = Object.entries(B).sort((a, b) => b[1] - a[1]);
  const counts = S.exposure.bridges;
  $("tBridges").innerHTML = `<tr><th>Crossing</th><th style="text-align:right">Intensity</th></tr>` +
    rows.map(([n, m]) => `<tr><td>${n}</td><td class="n">${m.toFixed(1)}</td></tr>`).join("") +
    `<tr><td colspan="2" style="color:var(--muted);font-size:13px">All ${comma(d3.sum(counts))} road bridges and viaducts: ` +
    counts.map((c, j) => (c ? `${comma(c)} at ${ROMAN[j]}` : null)).filter(Boolean).join(", ") + `.</td></tr>`;
}

/* ------------------------------------------------------------------ main */

(async function main() {
  const [S, C, Cc, roads, hosp, ist] = await Promise.all([
    getJSON("data/scenario/summary.json"), getJSON("data/scenario/charts.json"), getJSON("data/scenario/cells.json"),
    getJSON("data/scenario/roads.json"), getJSON("data/scenario/hospitals.json"), getJSON("data/istanbul.json"),
  ]);
  const names = ist.features.map((f) => `${f.properties.name}, ${f.properties.district}`);
  headline(S, C);
  setupMap(S, Cc, roads, hosp);
  chart(() => attenuation(C));
  chart(() => exposure(S));
  chart(() => states(S));
  chart(() => histogram(S, C));
  chart(() => districts(C));
  chart(() => validation(C, names));
  chart(() => fragility(C));
  chart(() => roadsChart(S));
  chart(() => access(C, S));
  chart(() => rail(S));
  bridges(S);
})();

/* Istanbul readiness page: mahalle map of İBB's scenario estimates, building age and footprint
   coverage. Data: site/data/istanbul.json from scripts/istanbul_inventory.py. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString("en");
const NA = "#dcd6cb";

// one sequential ramp (paper -> ochre -> brick), as on the Antakya page
const RAMP = ["#efe4cf", "#e2c08c", "#d0902a", "#b8612a", "#a8352a", "#6e2019"];

const pctList = (s) => s.map((v) => `${Math.round(v * 100)}%`).join(" · ");
const LAYERS = {
  scenario: {
    prop: "heavy_all", stops: [0, 10, 25, 50, 100, 200], ramp: RAMP,
    legend: (s) => `very heavily or heavily damaged buildings in the scenario: ${s.join(" · ")}+`,
  },
  old: {
    prop: "old_share", stops: [0, 0.1, 0.25, 0.4, 0.55, 0.7], ramp: RAMP,
    legend: (s) => `share of buildings built before 1980: ${pctList(s)}+`,
  },
  cov: { // low coverage is the problem, so it gets the dark end
    prop: "coverage", stops: [0, 0.2, 0.4, 0.6, 0.8, 1], ramp: [...RAMP].reverse(),
    legend: (s) => `OpenStreetMap buildings ÷ İBB's count: ${pctList(s)}+`,
  },
};

function enrich(fc) {
  for (const f of fc.features) {
    const p = f.properties;
    p.heavy_all = p.very_heavy == null ? -1 : p.very_heavy + p.heavy;
    p.old_share = p.ibb_buildings_2017 ? p.pre1980 / p.ibb_buildings_2017 : -1;
    p.coverage = p.ibb_buildings_2017 ? p.osm_buildings / p.ibb_buildings_2017 : -1;
  }
  return fc;
}

function color(layer) {
  const { prop, stops, ramp } = LAYERS[layer];
  const step = ["step", ["get", prop], ramp[0], ...stops.slice(1).flatMap((s, i) => [s, ramp[i + 1]])];
  return ["case", ["<", ["get", prop], 0], NA, step];
}

function legend(layer) {
  const L = LAYERS[layer];
  $("legend").innerHTML = L.ramp.map((c) => `<i style="background:${c};border-radius:0;margin:0"></i>`).join("") +
    `<span>${L.legend(L.stops)}</span><span><i style="background:${NA};border-radius:0"></i>no data</span>`;
}

function stats(fc) {
  const P = fc.features.map((f) => f.properties);
  const sum = (k) => P.reduce((a, p) => a + (p[k] || 0), 0);
  const both = P.filter((p) => p.ibb_buildings_2017);
  const cov = both.map((p) => p.osm_buildings / p.ibb_buildings_2017).sort((a, b) => a - b);
  const median = cov[Math.floor(cov.length / 2)];
  $("stats").innerHTML = [
    [fmt(sum("osm_buildings")), "buildings with a footprint in OpenStreetMap, 39 districts"],
    [fmt(sum("very_heavy") + sum("heavy")), "very heavily or heavily damaged in İBB's Mw 7.5 scenario"],
    [fmt(sum("moderate")), "moderately damaged in the same scenario"],
    [`${Math.round(median * 100)}%`, "median footprint coverage per mahalle vs İBB's 2017 count"],
  ].map(([n, l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
}

function popup(p) {
  const row = (k, v) => (v == null ? "" : `${k}: <b>${typeof v === "number" ? fmt(v) : v}</b><br>`);
  return `<b>${p.name}</b>, ${p.district}<br>` +
    row("Mapped buildings (OSM)", p.osm_buildings) +
    row("Buildings (İBB 2017)", p.ibb_buildings_2017 || null) +
    row("Built before 1980", p.pre1980) +
    (p.very_heavy == null ? "No scenario row matched<br>" :
      row("Scenario: very heavy / heavy", `${fmt(p.very_heavy)} / ${fmt(p.heavy)}`) +
      row("Scenario: moderate", p.moderate) + row("Scenario: deaths (night)", p.deaths));
}

(async function main() {
  const fc = enrich(await (await fetch("data/istanbul.json")).json());
  stats(fc);
  const map = new maplibregl.Map({
    container: "mapview", style: "https://tiles.openfreemap.org/styles/positron",
    bounds: [[28.0, 40.8], [29.95, 41.35]], fitBoundsOptions: { padding: 10 },
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }));
  map.on("load", () => {
    map.addSource("m", { type: "geojson", data: fc });
    map.addLayer({ id: "fill", type: "fill", source: "m",
      paint: { "fill-color": color("scenario"), "fill-opacity": 0.78 } });
    map.addLayer({ id: "line", type: "line", source: "m",
      paint: { "line-color": "#fff", "line-width": ["interpolate", ["linear"], ["zoom"], 9, 0.2, 13, 1] } });
    map.on("click", "fill", (e) => new maplibregl.Popup({ closeButton: false })
      .setLngLat(e.lngLat).setHTML(popup(e.features[0].properties)).addTo(map));
    map.on("mouseenter", "fill", () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", "fill", () => { map.getCanvas().style.cursor = ""; });
  });
  legend("scenario");
  document.querySelectorAll("#mapSeg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#mapSeg button").forEach((x) => x.classList.toggle("on", x === b));
    map.setPaintProperty("fill", "fill-color", color(b.dataset.v));
    legend(b.dataset.v);
  }));
})();

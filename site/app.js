/* Enkaz web page: before/after comparison, Antakya map, results tables. Data comes from the
   evaluation notebook (runs/main/results.json and the site export). */
"use strict";

const CLASSES = ["intact", "damaged", "destroyed"];
const COLORS = { intact: "#8a9a8b", damaged: "#d0902a", destroyed: "#a8352a" };
const $ = (id) => document.getElementById(id);
const getJSON = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(u); return r.json(); };
const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;

/* ------------------------------------------------------------------ before / after */

function setupCompare(gallery) {
  const pre = $("cmpPre"), post = $("cmpPost"), fp = $("cmpFp"), slider = $("cmpSlider");
  const wrap = $("cmpPreWrap");
  const apply = () => { wrap.style.clipPath = `inset(0 ${100 - slider.value}% 0 0)`; };
  slider.addEventListener("input", apply);
  const show = (g, btn) => {
    pre.src = `data/gallery/${g.id}_pre.jpg`;
    post.src = `data/gallery/${g.id}_post.jpg`;
    fp.src = `data/gallery/${g.id}_fp.png`;
    const p = g.probs.map((v) => pct(v, 0)).join(" / ");
    $("cmpCaption").textContent = `Expert label: ${g.true}. Model: ${g.pred} ` +
      `(intact / damaged / destroyed = ${p}).`;
    document.querySelectorAll("#thumbs button").forEach((b) => b.classList.toggle("on", b === btn));
  };
  // prefer interesting examples first: correct destroyed, missed destroyed, false alarms
  const order = (g) =>
    g.true === "destroyed" && g.pred === "destroyed" ? 0 : g.true === "intact" && g.pred === "intact" ? 1
      : g.true === "destroyed" ? 2 : g.pred === "destroyed" ? 3 : 4;
  const items = [...gallery].sort((a, b) => order(a) - order(b));
  $("thumbs").innerHTML = "";
  items.forEach((g, i) => {
    const b = document.createElement("button");
    b.title = `label ${g.true}, model ${g.pred}`;
    b.innerHTML = `<img src="data/gallery/${g.id}_post.jpg" alt=""><i style="background:${COLORS[g.pred]}"></i>`;
    b.addEventListener("click", () => show(g, b));
    $("thumbs").appendChild(b);
    if (i === 0) show(g, b);
  });
  wrap.style.inset = "0"; wrap.style.width = "100%"; wrap.style.borderRight = "0";
  apply();
}

/* ------------------------------------------------------------------ map */

function predicted(row, t, bias) {
  const s = [3, 4, 5].map((k, c) => t * Math.log(Math.max(row[k], 1e-6)) + bias[c]);
  return s.indexOf(Math.max(...s));
}

function setupMap(buildings, t, bias) {
  const feats = buildings.map((r) => {
    const pred = predicted(r, t, bias), truth = r[2];
    return {
      type: "Feature",
      geometry: { type: "Point", coordinates: [r[0], r[1]] },
      properties: { pred, truth, err: pred !== truth ? Math.max(pred, truth) : -1, sev: r[4] + 2 * r[5] },
    };
  });
  const lon = buildings.map((r) => r[0]), lat = buildings.map((r) => r[1]);
  const map = new maplibregl.Map({
    container: "mapview", style: "https://tiles.openfreemap.org/styles/positron",
    bounds: [[Math.min(...lon), Math.min(...lat)], [Math.max(...lon), Math.max(...lat)]],
    fitBoundsOptions: { padding: 30 }, attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }));
  const colorBy = (prop) => ["match", ["get", prop], 0, COLORS.intact, 1, COLORS.damaged, 2, COLORS.destroyed, "rgba(0,0,0,0)"];
  map.on("load", () => {
    map.addSource("b", { type: "geojson", data: { type: "FeatureCollection", features: feats } });
    map.addLayer({
      id: "b", type: "circle", source: "b",
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 1.6, 16, 5],
        "circle-color": colorBy("pred"),
        "circle-opacity": ["case", ["==", ["get", "pred"], 0], 0.45, 0.9],
      },
    });
    map.on("click", "b", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat)
        .setHTML(`Model: <b>${CLASSES[p.pred]}</b><br>Expert label: <b>${CLASSES[p.truth]}</b>`).addTo(map);
    });
  });
  document.querySelectorAll("#mapSeg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#mapSeg button").forEach((x) => x.classList.toggle("on", x === b));
    const v = b.dataset.v;
    const prop = v === "pred" ? "pred" : v === "true" ? "truth" : "err";
    map.setPaintProperty("b", "circle-color", colorBy(prop));
    map.setPaintProperty("b", "circle-opacity",
      v === "err" ? ["case", ["==", ["get", "err"], -1], 0, 0.9] : ["case", ["==", ["get", prop], 0], 0.45, 0.9]);
  }));
}

/* ------------------------------------------------------------------ results */

function renderResults(res) {
  const rows = [
    ["", "Change score (no learning)", res.change_score.test],
    ["", "Post-only CNN", res.post_only.test],
    ["", "Post-only CNN + val-tuned thresholds", res.post_only.test_bias],
    ["", "Siamese pre/post CNN", res.siamese.test],
    ["ours", "Siamese pre/post CNN + val-tuned thresholds", res.siamese.test_bias],
  ].map((r) => (r[1].startsWith(res.map_model === "siamese" ? "Siamese" : "Post-only") && r[1].includes("+") ? ["ours", r[1], r[2]] : ["", r[1], r[2]]));
  const s = res[res.map_model].test_bias;
  $("resultsLead").textContent = `On ${s.n.toLocaleString("en")} buildings in Antakya, never seen in training or tuning. ` +
    `The highlighted row is the model chosen on the validation city (İslahiye), and the one on the map.`;
  const f = (x, d = 3) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
  $("resultsTable").innerHTML =
    `<tr><th>Method</th><th>Macro-F1</th><th>Intact</th><th>Damaged</th><th>Destroyed</th><th>AUROC</th><th>Top 10 %</th><th>ECE</th></tr>` +
    rows.map(([k, label, t]) => `<tr class="${k}"><td>${label}</td><td>${f(t.macro_f1)}</td>` +
      `<td>${f(t.f1.intact)}</td><td>${f(t.f1.damaged)}</td><td>${f(t.f1.destroyed)}</td>` +
      `<td>${f(t.auroc_damaged_or_destroyed)}</td><td>${t.destroyed_recall_top10pct == null ? "–" : pct(t.destroyed_recall_top10pct, 0)}</td>` +
      `<td>${t.ece == null ? "–" : f(t.ece)}</td></tr>`).join("");

  const cm = s.confusion;
  const cell = (v, rowSum, diag) => {
    const a = rowSum ? v / rowSum : 0;
    return `<div style="background:rgba(${diag ? "29,27,24" : "168,53,42"},${0.08 + 0.6 * a});color:${a > 0.5 ? "#fff" : "inherit"}">${v.toLocaleString("en")}<br><small>${pct(a, 0)}</small></div>`;
  };
  $("confusion").innerHTML = `<div class="cm"><div class="h">label ↓ / model →</div>${CLASSES.map((c) => `<div class="h">${c}</div>`).join("")}` +
    cm.map((r, i) => `<div class="h">${CLASSES[i]}</div>` + r.map((v, j) => cell(v, r.reduce((a, b) => a + b, 0), i === j)).join("")).join("") + `</div>`;

  $("stats").innerHTML = [
    [s.n.toLocaleString("en"), "buildings in the test city"],
    [f(s.macro_f1, 2), "macro-F1 (3 classes)"],
    [pct(s.destroyed_recall_top10pct, 0), "of destroyed buildings in the top 10 % of the ranking"],
    [f(s.auroc_damaged_or_destroyed, 2), "AUROC, damaged or destroyed vs intact"],
  ].map(([n, l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
}

(async function main() {
  const res = await getJSON("data/results.json");
  renderResults(res);
  const best = res[res.map_model].test_bias;
  const t = best.temperature, bias = best.bias || [0, 0, 0];
  const [buildings, gallery] = await Promise.all([getJSON("data/buildings.json"), getJSON("data/gallery.json")]);
  setupCompare(gallery);
  setupMap(buildings, t, bias);
})();

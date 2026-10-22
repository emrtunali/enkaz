/* "The list a field team would get": the response pipeline's ranking on Antakya (data/field.json,
   from scripts/improvements.py), coloured by the expert labels found there afterwards. */
"use strict";

(async function field() {
  const r = await fetch("data/field.json");
  if (!r.ok) { document.getElementById("field").style.display = "none"; return; }
  const F = await r.json();
  const COL = { "-1": "#c9c3b8", 0: "#8a9a8b", 1: "#d0902a", 2: "#a8352a" };
  const NAME = { "-1": "no expert label", 0: "intact", 1: "damaged", 2: "destroyed" };
  const fmt = (n) => n.toLocaleString("en");
  const pts = F.field;
  const feats = pts.map(([lon, lat, rank, sev, lab]) => ({
    type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: { rank, sev, lab },
  }));
  const lon = pts.map((p) => p[0]), lat = pts.map((p) => p[1]);
  const map = new maplibregl.Map({
    container: "fieldmap", style: "https://tiles.openfreemap.org/styles/positron",
    bounds: [[Math.min(...lon), Math.min(...lat)], [Math.max(...lon), Math.max(...lat)]],
    fitBoundsOptions: { padding: 30 }, attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }));
  let top = 500;
  const filter = () => ["<=", ["get", "rank"], top];
  map.on("load", () => {
    map.addSource("f", { type: "geojson", data: { type: "FeatureCollection", features: feats } });
    map.addLayer({
      id: "f", type: "circle", source: "f", filter: filter(),
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 3, 16, 7],
        "circle-color": ["match", ["get", "lab"], 0, COL[0], 1, COL[1], 2, COL[2], COL["-1"]],
        "circle-stroke-color": "#fff", "circle-stroke-width": 1,
      },
    });
    map.on("click", "f", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat)
        .setHTML(`Rank <b>${fmt(p.rank)}</b> · expected severity ${p.sev}<br>Expert label: <b>${NAME[p.lab]}</b>`).addTo(map);
    });
  });
  const note = () => {
    const s = F.field_summary[String(top)];
    const lab = s.intact + s.damaged + s.destroyed;
    document.getElementById("fieldNote").textContent =
      `Of the first ${fmt(top)}: ${fmt(s.destroyed)} destroyed, ${fmt(s.damaged)} damaged, ${fmt(s.intact)} intact ` +
      `according to the experts (${Math.round((100 * (s.destroyed + s.damaged)) / Math.max(lab, 1))} % of the labelled ones damaged or destroyed, ` +
      `against ${Math.round(100 * F.base_rate)} % in the whole city); ${fmt(s.unlabelled)} have no expert label.`;
  };
  document.getElementById("fieldLegend").innerHTML = ["2", "1", "0", "-1"]
    .map((k) => `<span><i style="background:${COL[k]}"></i>${NAME[k]}</span>`).join("");
  document.querySelectorAll("#fieldSeg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#fieldSeg button").forEach((x) => x.classList.toggle("on", x === b));
    top = +b.dataset.v;
    if (map.getLayer("f")) map.setFilter("f", filter());
    note();
  }));
  note();

  // inspection curve
  const C = F.curve;
  const el = document.getElementById("fieldCurve");
  const draw = () => {
    el.innerHTML = "";
    const W = el.clientWidth, H = 260, m = { t: 10, r: 16, b: 34, l: 44 };
    const w = W - m.l - m.r, h = H - m.t - m.b;
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("height", H);
    const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
    const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
    const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(""));
    const series = [["respond", "#2a78d6", "response pipeline (on the day)"], ["reference", "#eb6834", "evaluation pipeline"], ["random", "#8a9a8b", "random order"]];
    for (const [k, c] of series) {
      g.append("path").datum(C[k]).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2)
        .attr("d", d3.line().x((_, i) => x(C.fractions[i])).y((d) => y(d)));
    }
    g.append("line").attr("x1", x(0.1)).attr("x2", x(0.1)).attr("y1", 0).attr("y2", h).attr("stroke", "#6b655c");
    g.append("text").attr("class", "lbl muted").attr("x", x(0.1) + 6).attr("y", h - 8).text("top 10 %");
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickFormat((d) => `${d * 100}%`));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat((d) => `${d * 100}%`));
    g.append("text").attr("class", "lbl muted").attr("x", w).attr("y", h + 30).attr("text-anchor", "end").text("share of buildings inspected");
    const tipEl = document.getElementById("tip");
    const hair = g.append("line").attr("stroke", "#1d1b18").attr("y1", 0).attr("y2", h).style("display", "none");
    g.append("rect").attr("width", w).attr("height", h).attr("fill", "transparent")
      .on("mousemove", (e) => {
        const f = x.invert(d3.pointer(e)[0]);
        const i = d3.minIndex(C.fractions, (v) => Math.abs(v - f));
        hair.style("display", null).attr("x1", x(C.fractions[i])).attr("x2", x(C.fractions[i]));
        tipEl.innerHTML = `<b>checking the top ${Math.round(C.fractions[i] * 100)} %</b><br>` +
          series.map(([k, , n]) => `${n}: ${Math.round(C[k][i] * 100)} % of destroyed found`).join("<br>");
        tipEl.style.display = "block";
        tipEl.style.left = `${Math.min(e.clientX + 14, innerWidth - tipEl.offsetWidth - 8)}px`;
        tipEl.style.top = `${e.clientY + 14}px`;
      })
      .on("mouseleave", () => { hair.style("display", "none"); tipEl.style.display = "none"; });
    document.getElementById("fieldKeys").innerHTML = series.map(([, c, n]) => `<span><i class="line" style="background:${c}"></i>${n}</span>`).join("");
  };
  draw();
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 150); });
})();

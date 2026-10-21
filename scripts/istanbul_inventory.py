"""Build the Istanbul readiness data (needs osmium and shapely; a few minutes on a laptop).

    python scripts/istanbul_inventory.py turkey-latest.osm.pbf senaryo.csv bina.csv [ms/*.csv.gz]

Inputs: Geofabrik's Türkiye extract, from İBB's open data portal the datasets
"Deprem Senaryosu Analiz Sonuçları" and "Mahalle Bazlı Bina Sayıları" (2017), and optionally
Microsoft's Global ML Building Footprints for the quadkeys of ``istanbul.quadkeys(BBOX)``
(listed in ``istanbul.MS_LINKS``). Machine footprints are only added where OSM has no building.

Writes data/istanbul/buildings.npz (every footprint, for respond.py; not in git),
data/istanbul/mahalle.geojson (full boundaries) and site/data/istanbul.json (simplified
mahalle with per-mahalle numbers, for the web page).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely

from enkaz import istanbul as I

ROOT = Path(__file__).resolve().parents[1]


def main(pbf: str, scenario_csv: str, stock_csv: str, *ms_files: str):
    t0 = time.time()
    scenario = I.read_ibb_csv(Path(scenario_csv))
    stock = I.read_ibb_csv(Path(stock_csv))
    districts = {I.norm(r["ilce_adi"]) for r in scenario}

    admin = I.read_admin(Path(pbf))
    dist, mah = I.istanbul_units(admin, districts)
    print(f"{len(dist)} districts, {len(mah)} mahalle  ({time.time() - t0:.0f} s)")

    b = I.read_buildings(Path(pbf))
    n_osm_box = len(b["lon"])
    if ms_files:
        ms = I.read_ms_footprints([Path(f) for f in ms_files])
        b = I.merge_footprints(b, ms)
        print(
            f"Microsoft: {len(ms['lon']):,} in the box, {len(b['lon']) - n_osm_box:,} fill OSM gaps"
        )
    else:
        b["source"] = np.zeros(len(b["lon"]), np.int8)
    where = I.assign(b["lon"], b["lat"], [m["geom"] for m in mah])
    keep = where >= 0
    print(f"{keep.sum():,} of {len(keep):,} buildings in the box fall in a mahalle")

    # keep only Istanbul's buildings; re-pack their outlines
    idx = np.flatnonzero(keep)
    off = b["ring_off"]
    rings = [b["ring"][off[i] : off[i + 1]] for i in idx]
    out = {k: b[k][idx] for k in ("osm_id", "lon", "lat", "area_m2", "levels", "kind", "source")}
    out |= {
        "kind_names": b["kind_names"],
        "mahalle": where[idx],
        "ring": np.concatenate(rings),
        "ring_off": np.concatenate([[0], np.cumsum([len(r) for r in rings])]),
    }
    data = ROOT / "data" / "istanbul"
    data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data / "buildings.npz", **out)

    sc = I.join_ibb(mah, scenario)
    st = I.join_ibb(mah, stock)
    print(f"İBB scenario matched {len(sc)}/{len(scenario)}, building stock {len(st)}/{len(stock)}")
    n_osm = np.bincount(out["mahalle"][out["source"] == 0], minlength=len(mah))
    n_ms = np.bincount(out["mahalle"][out["source"] == 1], minlength=len(mah))

    full, small = [], []
    for i, m in enumerate(mah):
        s, k = sc.get(i), st.get(i)
        props = {"district": m["district"], "name": m["name"], "osm_id": m["osm_id"]}
        props["osm_buildings"] = int(n_osm[i])
        props["ms_added"] = int(n_ms[i])
        props["buildings"] = int(n_osm[i] + n_ms[i])
        if s:
            props |= {
                "uavt": s["mahalle_koy_uavt"],
                "very_heavy": s["cok_agir_hasarli_bina_sayisi"],
                "heavy": s["agir_hasarli_bina_sayisi"],
                "moderate": s["orta_hasarli_bina_sayisi"],
                "light": s["hafif_hasarli_bina_sayisi"],
                "deaths": s["can_kaybi_sayisi"],
                "shelter": s["gecici_barinma"],
            }
        if k:
            total = k["1980_oncesi"] + k["1980-2000_arasi"] + k["2000_sonrasi"]
            props |= {"ibb_buildings_2017": total, "pre1980": k["1980_oncesi"]}
        full.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(m["geom"]),
                "properties": props,
            }
        )
        g = shapely.simplify(m["geom"], 0.0004, preserve_topology=True)
        g = shapely.set_precision(g, 1e-5)
        small.append(
            {"type": "Feature", "geometry": shapely.geometry.mapping(g), "properties": props}
        )
    fc = lambda f: {"type": "FeatureCollection", "features": f}  # noqa: E731
    (data / "mahalle.geojson").write_text(json.dumps(fc(full)))
    site = ROOT / "site" / "data" / "istanbul.json"
    site.write_text(json.dumps(fc(small), separators=(",", ":"), ensure_ascii=False))

    summary = {
        "districts": len(dist),
        "mahalle": len(mah),
        "buildings": int(keep.sum()),
        "osm_buildings": int((out["source"] == 0).sum()),
        "ms_added": int((out["source"] == 1).sum()),
        "ibb_scenario_rows": len(scenario),
        "scenario_matched": len(sc),
        "stock_rows": len(stock),
        "stock_matched": len(st),
        "seconds": round(time.time() - t0),
    }
    (data / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), f"site file {site.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])

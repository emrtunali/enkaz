"""Data for the animated earthquake (site/earthquake.html), one file pair per magnitude.

For each magnitude: the rupture (length and width from Wells & Coppersmith 1994, centred on the
Kumburgaz segment south of Istanbul, bilateral), when the P and S waves reach every 1 km cell,
how long and how hard it shakes there, and which buildings collapse, and when.

Shaking and damage use the same models as the scenario page (``istanbul_scenario.py``): two GMPEs
with correlated residuals, ESRM20 fragility. Each building's chance of complete or extensive damage
is averaged over the simulated earthquakes, and one plausible outcome is drawn from those chances;
a drawn building collapses at a random moment during the strong shaking at its location.

Damage level. Uncalibrated, ESRM20's fragility curves give about four times İBB's estimate of very
heavily and heavily damaged buildings (see the scenario page). Here one factor stretches all curves
along the shaking axis so that Mw 7.5 matches İBB's total; the other magnitudes keep that factor.
Where damage falls still comes from this model; how much is anchored to İBB's.

Needs the OpenQuake environment and the caches written by ``istanbul_scenario.py``:

    python scripts/istanbul_animation.py --faults gem_active_faults_harmonized.geojson \\
        --stock bina.csv --esrm esrm/ [--realizations 40]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from enkaz import scenario as S

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data" / "anim"
MAGS = (6.5, 7.0, 7.2, 7.5, 7.7)
CENTER_LON = 28.65  # Kumburgaz basin, the segment closest to the European side
LON0, LAT0 = 27.9, 40.7  # origin of the integer coordinates in the .bin files
PLACES = {
    "Fatih": (28.95, 41.015),
    "Kadıköy": (29.03, 40.99),
    "Avcılar": (28.72, 40.98),
    "Beylikdüzü": (28.64, 41.0),
    "Üsküdar": (29.02, 41.025),
    "Bağcılar": (28.85, 41.04),
    "Sarıyer": (29.05, 41.17),
    "Silivri": (28.25, 41.075),
    "Pendik": (29.24, 40.88),
}


def load_scenario_module():
    spec = importlib.util.spec_from_file_location(
        "istanbul_scenario", ROOT / "scripts" / "istanbul_scenario.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fault_line(path: Path) -> np.ndarray:
    """The North Anatolian Fault under the Sea of Marmara (GEM GAF): the main segment south of
    Istanbul joined to the western segment that ends where it begins, lon/lat."""
    feats = json.loads(path.read_text())["features"]

    def coords(f):
        c = f["geometry"]["coordinates"]
        c = [p for line in c for p in line] if f["geometry"]["type"] == "MultiLineString" else c
        return [tuple(p[:2]) for p in c]

    main = next(coords(f) for f in feats if f["properties"].get("catalog_id") == "ME_TRCS013")
    west = next(
        c
        for c in map(coords, feats)
        if c and abs(c[-1][0] - 28.026) < 0.01 and abs(c[-1][1] - 40.802) < 0.01
    )
    west = [p for p in west if p[0] >= 27.3]
    return np.array(west + main)


ROAD_CLASSES = list(S.ROAD_KMH)
MAJOR = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
}


def road_network(sc, inv, exp_storeys):
    """Road graph, building-street pairs a collapse could block, hospitals (scenario caches)."""
    import pickle

    import shapely
    from scipy.spatial import cKDTree

    net = pickle.loads((sc.CACHE / "network.pkl").read_bytes())
    feats = json.loads((sc.DATA / "mahalle.geojson").read_text())["features"]
    tree = shapely.STRtree([shapely.geometry.shape(f["geometry"]) for f in feats])

    def inside(lon, lat):
        pi, gi = tree.query(shapely.points(lon, lat), predicate="within")
        out = np.full(len(lon), -1)
        out[pi] = gi
        return out

    g = sc.build_graph(net["edges"], inside)
    pairs = sc.building_edge_pairs(inv, exp_storeys * S.STOREY_M, g)
    h = np.array([(x, y) for x, y, _ in net["hosp"]], float)
    hin = inside(h[:, 0], h[:, 1]) >= 0
    hosp = [(float(x), float(y), n) for (x, y, n), k in zip(net["hosp"], hin) if k]
    main_nodes = np.flatnonzero(g["node_main"])
    kd = cKDTree(np.column_stack([g["nx"][main_nodes], g["ny"][main_nodes]]))
    hx, hy = sc.to_utm(np.array([x for x, _, _ in hosp]), np.array([y for _, y, _ in hosp]))
    hnode = main_nodes[kd.query(np.column_stack([hx, hy]))[1]]
    a_lon, a_lat = sc.to_lonlat(g["ax"], g["ay"])
    b_lon, b_lat = sc.to_lonlat(g["bx"], g["by"])
    g["lonlat"] = np.column_stack([a_lon, a_lat, b_lon, b_lat])
    g["cls_code"] = np.array([ROAD_CLASSES.index(c) for c in g["cls"]], np.uint8)
    return g, pairs, hosp, hnode, kd, main_nodes


def csr(g, blocked):
    from scipy.sparse import coo_matrix

    ok = ~blocked
    n = len(g["nx"])
    w = g["time"][ok]
    u, v = g["iu"][ok], g["iv"][ok]
    return coo_matrix(
        (np.concatenate([w, w]), (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n)
    ).tocsr()


def route(sc, graph, src, hnode, hosp, g):
    """Fastest route from node ``src`` to the nearest hospital: minutes, name, lon/lat path."""
    from scipy.sparse.csgraph import dijkstra

    dist, pred = dijkstra(graph, indices=src, return_predecessors=True)
    k = int(np.argmin(dist[hnode]))
    if not np.isfinite(dist[hnode[k]]):
        return {"min": None, "hospital": None, "path": []}
    path, n = [], int(hnode[k])
    while n >= 0 and n != src:
        path.append(n)
        n = int(pred[n])
    path.append(src)
    lon, lat = sc.to_lonlat(g["nx"][path[::-1]], g["ny"][path[::-1]])
    return {
        "min": round(float(dist[hnode[k]]) / 60, 1),
        "hospital": hosp[k][2] or "hospital",
        "path": [[round(float(x), 5), round(float(y), 5)] for x, y in zip(lon, lat)],
    }


def write_edges(path, g, idx, t=None):
    """Road segments as a small binary file: n, 4 x int32 coordinates, uint16 length (m),
    uint16 time (0.1 s, optional), uint8 road class."""
    c = g["lonlat"][idx]
    with open(path, "wb") as f:
        f.write(np.uint32(len(idx)).tobytes())
        for k, o in ((0, LON0), (1, LAT0), (2, LON0), (3, LAT0)):
            f.write(np.round((c[:, k] - o) * 1e5).astype(np.int32).tobytes())
        f.write(np.minimum(g["length"][idx], 65535).astype(np.uint16).tobytes())
        if t is not None:
            f.write(np.round(t * 10).astype(np.uint16).tobytes())
        f.write(g["cls_code"][idx].tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", type=Path, required=True)
    ap.add_argument("--stock", type=Path, required=True)
    ap.add_argument("--esrm", type=Path, required=True)
    ap.add_argument("--realizations", type=int, default=40, help="per GMPE")
    a = ap.parse_args()
    sc = load_scenario_module()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    inv = dict(np.load(sc.DATA / "buildings.npz"))
    props = [
        f["properties"] for f in json.loads((sc.DATA / "mahalle.geojson").read_text())["features"]
    ]
    cells = dict(np.load(sc.CACHE / "cells.npz"))
    vs30 = dict(np.load(sc.CACHE / "site.npz"))["vs30"]
    nc = len(cells["n"])
    rows, night, table, lol = sc.esrm_tables(a.esrm)
    mix = S.class_mix(rows)
    age, hgt, _ = sc.stock_shares(props, a.stock)
    P = sc.profiles(inv, age, hgt, mix, table)
    keys, W, prof = P["keys"], P["W"], P["prof"]
    floor = inv["area_m2"] * P["exp_storeys"][prof]
    occ = night * floor / floor.sum()
    pair_key = prof.astype(np.int64) * nc + cells["cell"]
    pk, pair = np.unique(pair_key, return_inverse=True)
    pair = pair.ravel()
    p_prof, p_cell = pk // nc, pk % nc
    p_n = np.bincount(pair).astype(np.float32)
    p_occ = np.bincount(pair, occ).astype(np.float32)
    imt_ix = {"PGA": 0, "SA(0.3)": 1, "SA(0.6)": 2, "SA(1.0)": 3}
    sheet = {"PGA": "PGA", "SA(0.3)": "SA(0.3s)", "SA(0.6)": "SA(0.6s)", "SA(1.0)": "SA(1.0s)"}
    kinfo = []
    for fam, h in keys:
        imt, med, beta = table[f"{fam}_H{h}"]
        sh = lol[sheet[imt]]
        kinfo.append(
            (
                imt_ix[imt],
                med,
                beta,
                sh.iloc[:, 0].to_numpy(float),
                sh[f"{fam}_H{h}"].to_numpy(float),
            )
        )

    def pair_damage(gm_r, scale=1.0):
        """``scale`` stretches every fragility and loss-of-life curve along the shaking axis."""
        exc = np.zeros((len(pk), 4), np.float32)
        fr = np.zeros(len(pk), np.float32)
        for k, (ii, med, beta, grid, frc) in enumerate(kinfo):
            im = gm_r[ii] / scale
            w = W[p_prof, k]
            exc += w[:, None] * S.lognormal_exceed(im, med, beta).astype(np.float32)[p_cell]
            fr += (
                w
                * np.interp(np.log(np.maximum(im, grid[0])), np.log(grid), frc).astype(np.float32)[
                    p_cell
                ]
            )
        return exc, fr

    from scipy.sparse.csgraph import dijkstra

    g, pairs, hosp, hnode, kd, main_nodes = road_network(sc, inv, P["exp_storeys"][prof])
    print(
        f"roads: {len(g['iu']):,} segments, {len(pairs['b']):,} building-street pairs, "
        f"{len(hosp)} hospitals",
        flush=True,
    )
    (OUT / "hospitals.json").write_text(
        json.dumps([[round(x, 5), round(y, 5), n] for x, y, n in hosp], ensure_ascii=False)
    )
    ccx, ccy = sc.to_utm(cells["lon"], cells["lat"])
    cell_node = main_nodes[kd.query(np.column_stack([ccx, ccy]))[1]]
    base_graph = csr(g, np.zeros(len(g["iu"]), bool))
    t_base = dijkstra(base_graph, indices=np.unique(hnode), min_only=True)[cell_node] / 60
    cell_pop = np.bincount(cells["cell"], occ, minlength=nc)
    px, py = sc.to_utm(
        np.array([v[0] for v in PLACES.values()]), np.array([v[1] for v in PLACES.values()])
    )
    place_node = dict(zip(PLACES, main_nodes[kd.query(np.column_stack([px, py]))[1]]))
    line = fault_line(a.faults)
    lx, ly = sc.to_utm(line[:, 0], line[:, 1])
    line_km = np.column_stack([np.asarray(lx), np.asarray(ly)]) / 1000
    s_line = S.along(line_km)
    center = float(s_line[np.argmin(np.abs(line[:, 0] - CENTER_LON))])
    cx, cy = sc.to_utm(cells["lon"], cells["lat"])
    cell_km = np.column_stack([np.asarray(cx), np.asarray(cy)]) / 1000
    bx, by = inv["lon"], inv["lat"]
    rng = np.random.default_rng(2023)
    place_cell = {
        n: int(np.argmin(np.hypot(cells["lon"] - lo, cells["lat"] - la)))
        for n, (lo, la) in PLACES.items()
    }

    def run(gm, scale):
        R = gm.shape[2]
        exc_m = np.zeros((len(pk), 4))
        fr_m = np.zeros(len(pk))
        tot = np.zeros((R, 3))
        for r in range(R):
            exc, fr = pair_damage(gm[:, :, r], scale)
            exc_m += exc / R
            fr_m += fr / R
            pds = S.ds_probabilities(exc)
            tot[r] = [(pds[:, 4] * p_n).sum(), (pds[:, 3] * p_n).sum(), (p_occ * fr).sum()]
        return exc_m, fr_m, tot

    ruptures = {}
    for M in MAGS:
        L, Wd = S.rupture_length_km(M), S.rupture_width_km(M)
        rup_km, _ = S.sub_trace(line_km, center, L)
        rlon, rlat = sc.to_lonlat(rup_km[:, 0] * 1000, rup_km[:, 1] * 1000)
        trace = list(
            zip(
                np.asarray(rlon)[::5].tolist() + [float(np.asarray(rlon)[-1])],
                np.asarray(rlat)[::5].tolist() + [float(np.asarray(rlat)[-1])],
            )
        )
        rup = sc.rupture(trace=trace, mag=M, lower_km=min(20.0, Wd))
        gm = sc.ground_motion(cells, vs30, a.realizations, seed=int(M * 100), rup=rup)["gm"]
        ruptures[M] = (L, Wd, rup_km, rlon, rlat, gm)

    # one factor on all fragility curves so that M7.5 matches İBB's very heavy + heavy estimate
    target = sum((p.get("very_heavy") or 0) + (p.get("heavy") or 0) for p in props)
    lo, hi = 1.0, 6.0
    for _ in range(14):
        mid = (lo * hi) ** 0.5
        e, _, _ = run(ruptures[7.5][5], mid)
        got = (S.ds_probabilities(e)[:, 3:] * p_n[:, None]).sum()
        lo, hi = (mid, hi) if got > target else (lo, mid)
    scale = (lo * hi) ** 0.5
    print(f"calibration factor {scale:.2f} (İBB target {target:,.0f})", flush=True)

    index = []
    for M in MAGS:
        L, Wd, rup_km, rlon, rlat, gm = ruptures[M]
        exc_m, fr_m, tot = run(gm, scale)
        raw, _, _ = run(gm, 1.0)
        raw_heavy = float((S.ds_probabilities(raw)[:, 3:] * p_n[:, None]).sum())
        pds = S.ds_probabilities(exc_m)
        hypo = len(rup_km) // 2
        tp, ts, rmin = S.arrival_times(cell_km, rup_km, hypo)
        dur = S.strong_duration(L, rmin)
        pga = np.exp(np.log(np.maximum(gm[0], 1e-4)).mean(1))
        mmi = S.mmi_from_pga(pga)
        # one plausible outcome, drawn building by building from the averaged chances
        u = rng.random(len(prof))
        p4, p3 = pds[pair, 4], pds[pair, 3]
        cls = np.where(u < p4, 4, np.where(u < p4 + p3, 3, 0)).astype(np.uint8)
        hit = np.flatnonzero(cls > 0)
        cb = cells["cell"][hit]
        t_hit = ts[cb] + rng.uniform(0.15, 0.8, len(hit)) * dur[cb]
        order = np.argsort(t_hit)
        hit, t_hit = hit[order], t_hit[order]
        n = len(hit)
        tag = f"m{int(round(M * 10))}"
        with open(OUT / f"{tag}.bin", "wb") as f:
            f.write(np.uint32(n).tobytes())
            f.write(np.round((bx[hit] - LON0) * 1e5).astype(np.int32).tobytes())
            f.write(np.round((by[hit] - LAT0) * 1e5).astype(np.int32).tobytes())
            f.write(np.round(t_hit * 10).astype(np.uint16).tobytes())
            f.write(cls[hit].tobytes())
        deaths_cell = np.bincount(p_cell, p_occ * fr_m, minlength=nc)
        # streets: a segment is blocked from the moment its first neighbour collapses onto it
        t_col = np.full(len(prof), np.inf)
        t_col[hit[cls[hit] == 4]] = t_hit[cls[hit] == 4]
        t_edge = np.full(len(g["iu"]), np.inf)
        np.minimum.at(t_edge, pairs["e"], t_col[pairs["b"]])
        blocked = np.isfinite(t_edge)
        bidx = np.flatnonzero(blocked)
        bidx = bidx[np.argsort(t_edge[bidx])]
        write_edges(OUT / f"{tag}_roads.bin", g, bidx, t_edge[bidx])
        after_graph = csr(g, blocked)
        t_after = dijkstra(after_graph, indices=np.unique(hnode), min_only=True)[cell_node] / 60
        cut = (t_base <= 30) & (t_after > 30)
        place_routes = {}
        for pname in place_cell:
            src = int(place_node[pname])
            place_routes[pname] = {
                "before": route(sc, base_graph, src, hnode, hosp, g),
                "after": route(sc, after_graph, src, hnode, hosp, g),
            }
        is_major = np.isin(g["cls"][bidx], list(MAJOR))
        roads = {
            "segments": int(len(bidx)),
            "km": round(float(g["length"][bidx].sum()) / 1000, 1),
            "major_km": round(float(g["length"][bidx][is_major].sum()) / 1000, 1),
            "major_segments": int(is_major.sum()),
            "network_km": round(float(g["length"].sum()) / 1000),
            "cut_off_people": round(float(cell_pop[cut].sum())),
            "cut_off_share": round(float(cell_pop[cut].sum() / cell_pop.sum()), 4),
        }
        q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
        meta = {
            "mag": M,
            "length_km": round(L, 1),
            "width_km": round(min(20.0, Wd), 1),
            "trace": [[round(float(x), 4), round(float(y), 4)] for x, y in zip(rlon, rlat)],
            "hypo": [
                round(float(np.asarray(rlon)[hypo]), 4),
                round(float(np.asarray(rlat)[hypo]), 4),
            ],
            "rupture_s": round(L / 2 / S.VR_KMS, 1),
            "speeds_kms": {"P": S.VP_KMS, "S": S.VS_KMS, "rupture": S.VR_KMS},
            "drawn": {"complete": int((cls == 4).sum()), "extensive": int((cls == 3).sum())},
            "calibration": {
                "factor": round(scale, 3),
                "target_ibb_heavy_m75": target,
                "uncalibrated_esrm20_heavy": round(raw_heavy),
            },
            "expected": {
                "complete": float((pds[:, 4] * p_n).sum()),
                "extensive": float((pds[:, 3] * p_n).sum()),
                "deaths": float((p_occ * fr_m).sum()),
                "complete_p5_p95": [q(tot[:, 0], 5), q(tot[:, 0], 95)],
                "deaths_p5_p95": [q(tot[:, 2], 5), q(tot[:, 2], 95)],
            },
            "cells": {
                "tp": np.round(tp, 1).tolist(),
                "ts": np.round(ts, 1).tolist(),
                "dur": np.round(dur, 1).tolist(),
                "mmi": np.round(mmi, 1).tolist(),
                "deaths": np.round(deaths_cell, 2).tolist(),
                "t0": np.round(np.minimum(t_base, 999), 1).tolist(),
                "t1": np.round(np.minimum(t_after, 999), 1).tolist(),
            },
            "places": {
                p: {
                    "tp": round(float(tp[c]), 1),
                    "ts": round(float(ts[c]), 1),
                    "mmi": round(float(mmi[c]), 1),
                    "dur": round(float(dur[c]), 1),
                }
                for p, c in place_cell.items()
            },
            "end_s": round(float((ts + dur).max()) + 3, 1),
            "roads": roads,
            "routes": place_routes,
        }
        (OUT / f"{tag}.json").write_text(
            json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
        )
        f = meta["places"]["Fatih"]
        index.append(
            {
                "mag": M,
                "tag": tag,
                "length_km": meta["length_km"],
                "complete": meta["expected"]["complete"],
                "extensive": meta["expected"]["extensive"],
                "deaths": meta["expected"]["deaths"],
                "fatih_mmi": f["mmi"],
                "warning_s": round(f["ts"] - f["tp"], 1),
                "roads_km": roads["km"],
                "major_km": roads["major_km"],
                "cut_off_people": roads["cut_off_people"],
            }
        )
        print(
            f"M{M}: L {L:.0f} km, {n:,} buildings drawn ({meta['drawn']}), expected complete "
            f"{meta['expected']['complete']:,.0f}, deaths {meta['expected']['deaths']:,.0f}, "
            f"Fatih P {meta['places']['Fatih']['tp']} s / S {meta['places']['Fatih']['ts']} s, "
            f"MMI {meta['places']['Fatih']['mmi']} ({time.time() - t0:.0f} s)",
            flush=True,
        )
    (OUT / "index.json").write_text(
        json.dumps(
            {
                "magnitudes": index,
                "origin": [LON0, LAT0],
                "calibration": round(scale, 3),
                "road_classes": ROAD_CLASSES,
                "major": sorted(MAJOR),
            }
        )
    )


if __name__ == "__main__":
    main()

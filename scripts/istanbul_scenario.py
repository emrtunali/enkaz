"""Run the Istanbul earthquake scenario (``enkaz.scenario``) and write the web page's data.

Needs an environment with OpenQuake (``openquake.engine==3.16.*`` with ``numpy<2``), rasterio,
osmium, shapely, scipy, pandas and openpyxl, plus the outputs of ``istanbul_inventory.py``:

    python scripts/istanbul_scenario.py --pbf turkey-latest.osm.pbf --stock bina.csv \\
        --dem dem/ --esrm esrm/ [--realizations 100]

``dem/`` holds the Copernicus GLO-90 tiles N40–N41 × E027–E029; ``esrm/`` the ESRM20 files
Exposure_Model_Turkey_Res.csv, esrm20_fragility_various_IM_lognormal.xlsx (as fragility.xlsx)
and esrm20_vulnerability_loss-of-life.xlsx (as lol.xlsx). Heavy stages are cached in
data/istanbul/scenario/, so a rerun only repeats what changed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from enkaz import istanbul as I
from enkaz import scenario as S

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "istanbul"
CACHE = DATA / "scenario"
SITE = ROOT / "site" / "data" / "scenario"
UTM = "EPSG:32635"
CELL_M = 1000.0
GSIMS = ("AkkarEtAlRjb2014", "BooreEtAl2014")
RAIL = {"rail", "subway", "light_rail", "tram", "monorail", "funicular", "narrow_gauge"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cached(name, fn):
    p = CACHE / f"{name}.npz"
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    out = fn()
    np.savez_compressed(p, **out)
    return out


# --------------------------------------------------------------------------- cells and site


def to_utm(lon, lat):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(lon, lat)


def to_lonlat(x, y):
    from pyproj import Transformer

    return Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(x, y)


def build_cells(inv):
    x, y = to_utm(inv["lon"], inv["lat"])
    key = np.column_stack(
        [np.floor(np.asarray(x) / CELL_M), np.floor(np.asarray(y) / CELL_M)]
    ).astype(np.int64)
    keys, cell = np.unique(key, axis=0, return_inverse=True)
    cell = cell.ravel()
    n = np.bincount(cell)
    lon = np.bincount(cell, inv["lon"]) / n
    lat = np.bincount(cell, inv["lat"]) / n
    return {"keys": keys, "cell": cell, "lon": lon, "lat": lat, "n": n}


def vs30_at(lon, lat, dem_dir: Path):
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in sorted(dem_dir.glob("*.tif"))]
    dem, tr = merge(srcs, bounds=(27.80, 40.60, 30.00, 41.70))
    dem = dem[0].astype(float)
    dem[dem < -100] = 0  # nodata over the sea
    f = 10  # 3" -> 30" (Wald & Allen use a 30 arc-second DEM)
    h, w = (dem.shape[0] // f) * f, (dem.shape[1] // f) * f
    d30 = dem[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    step = abs(tr.a) * f
    slope = S.slope_m_per_m(d30, 41.1, step, step)
    vs30 = S.vs30_from_slope(slope)
    r = np.clip(((tr.f - np.asarray(lat)) / step).astype(int), 0, d30.shape[0] - 1)
    c = np.clip(((np.asarray(lon) - tr.c) / step).astype(int), 0, d30.shape[1] - 1)
    return vs30[r, c]


# --------------------------------------------------------------------------- ground motion


def rupture():
    from openquake.hazardlib.const import TRT
    from openquake.hazardlib.geo import Line, Point
    from openquake.hazardlib.geo.surface import SimpleFaultSurface
    from openquake.hazardlib.source.rupture import BaseRupture

    surf = SimpleFaultSurface.from_fault_data(
        Line([Point(x, y) for x, y in S.FAULT_TRACE]), S.UPPER_KM, S.LOWER_KM, 90.0, 2.0
    )
    return BaseRupture(S.MAG, S.RAKE, TRT.ACTIVE_SHALLOW_CRUST, surf.get_middle_point(), surf)


def sitecol(lon, lat, vs30):
    from openquake.hazardlib.geo import Point
    from openquake.hazardlib.site import Site, SiteCollection

    # basin depths from the generic Vs30 relations used by the two GMPEs' authors
    z1 = np.exp(-7.15 / 4 * np.log((vs30**4 + 570.94**4) / (1360**4 + 570.94**4)))
    return SiteCollection(
        [
            Site(Point(a, b), vs30=float(v), vs30measured=False, z1pt0=float(z), z2pt5=1.0)
            for a, b, v, z in zip(lon, lat, vs30, z1)
        ]
    )


def gsim(name):
    from openquake.hazardlib import gsim as G

    return {
        "AkkarEtAlRjb2014": G.akkar_2014.AkkarEtAlRjb2014,
        "BooreEtAl2014": G.boore_2014.BooreEtAl2014,
    }[name]()


def ground_motion(cells, vs30, realizations, seed=42):
    """Correlated ground-motion fields, sampled here rather than with OpenQuake's own sampler.

    OpenQuake supplies each GMPE's mean, between-event (tau) and within-event (phi) standard
    deviations and the spatial correlation (Jayaram & Baker, 2009). Residuals are correlated across
    periods too (Baker & Jayaram, 2008): one between-event draw per earthquake shared by all
    periods, and within-event fields mixed with the same cross-period matrix. Without that, a
    realization could have weak PGA and strong 1-second shaking at the same time, which buildings
    designed for one would not see.
    """
    from openquake.hazardlib.contexts import ContextMaker
    from openquake.hazardlib.correlation import JB2009CorrelationModel
    from openquake.hazardlib.cross_correlation import BakerJayaram2008
    from openquake.hazardlib.imt import from_string

    rup, sc = rupture(), sitecol(cells["lon"], cells["lat"], vs30)
    imts = [from_string(i) for i in S.IMTS]
    bj = BakerJayaram2008()
    C = np.array([[bj.get_correlation(a, b) for b in imts] for a in imts])
    Lc = np.linalg.cholesky(C)
    jb = JB2009CorrelationModel(False)
    Ls = [jb.get_lower_triangle_correlation_matrix(sc.complete, imt) for imt in imts]
    rng = np.random.default_rng(seed)
    n, M = len(vs30), len(imts)
    gm, med = [], []
    for name in GSIMS:
        log(f"  GMF {name}: {n:,} sites x {realizations} realizations")
        cm = ContextMaker(
            "Active Shallow Crust",
            [gsim(name)],
            {"imtls": {i: [0] for i in S.IMTS}, "truncation_level": 3},
        )
        [ctx] = list(cm.get_ctx_iter([rup], sc))
        assert (np.asarray(ctx.sids) == np.arange(n)).all()
        mean, _, tau, phi = cm.get_mean_stds([ctx])[:, 0]  # each (M, N)
        eta = Lc @ rng.standard_normal((M, realizations))  # between-event, (M, R)
        u = np.stack(
            [Ls[m] @ rng.standard_normal((n, realizations)) for m in range(M)]
        )  # (M, N, R)
        z = np.einsum("mj,jnr->mnr", Lc, u)  # within-event, correlated across periods
        ln = mean[:, :, None] + tau[:, :, None] * eta[:, None, :] + phi[:, :, None] * z
        gm.append(np.exp(ln).astype(np.float32))
        med.append(np.exp(mean).astype(np.float32))
    rjb = rup.surface.get_joyner_boore_distance(sc.mesh)
    return {
        "gm": np.concatenate(gm, axis=2),
        "median": np.stack(med),
        "rjb": rjb.astype(np.float32),
        "gsim_of_r": np.repeat(np.arange(len(GSIMS)), realizations),
    }


def attenuation_profile():
    """Median and median x/÷ one total sigma of PGA and SA(1.0) against distance, for two site
    classes and both GMPEs (for the chart), straight from the models."""
    from openquake.hazardlib.contexts import ContextMaker

    rup = rupture()
    lat = np.linspace(40.87, 41.75, 60)
    lon = np.full_like(lat, 28.80)
    out = {}
    for vs in (300.0, 760.0):
        sc = sitecol(lon, lat, np.full_like(lat, vs))
        rjb = rup.surface.get_joyner_boore_distance(sc.mesh)
        for name in GSIMS:
            cm = ContextMaker(
                "Active Shallow Crust",
                [gsim(name)],
                {"imtls": {"PGA": [0], "SA(1.0)": [0]}, "truncation_level": 3},
            )
            [ctx] = list(cm.get_ctx_iter([rup], sc))
            mean, sig, _, _ = cm.get_mean_stds([ctx])[:, 0]
            for m, imt in enumerate(("PGA", "SA(1.0)")):
                out[f"{name}|{imt}|{int(vs)}"] = {
                    "rjb": np.round(rjb, 2).tolist(),
                    "p16": np.round(np.exp(mean[m] - sig[m]), 4).tolist(),
                    "p50": np.round(np.exp(mean[m]), 4).tolist(),
                    "p84": np.round(np.exp(mean[m] + sig[m]), 4).tolist(),
                }
    return out


# --------------------------------------------------------------------------- exposure


def esrm_tables(esrm: Path):
    import pandas as pd

    res = pd.read_csv(esrm / "Exposure_Model_Turkey_Res.csv")
    ist = res[res.NAME_1.str.contains("stanbul")]
    rows = list(ist.groupby("TAXONOMY").BUILDINGS.sum().items())
    night = float(ist.OCCUPANTS_PER_ASSET_NIGHT.sum())
    frag = pd.read_excel(esrm / "fragility.xlsx", "Final")
    table = {
        t: (imt, np.array([a, b, c, d], float), float(beta))
        for t, imt, a, b, c, d, beta in frag.itertuples(index=False)
    }
    lol = {
        s: pd.read_excel(esrm / "lol.xlsx", s) for s in ("PGA", "SA(0.3s)", "SA(0.6s)", "SA(1.0s)")
    }
    return rows, night, table, lol


def stock_shares(props, stock_csv: Path):
    """Per-mahalle shares of İBB's age classes and storey bands (district, then city fallback)."""
    rows = I.read_ibb_csv(stock_csv)
    mah = [{"district": p["district"], "name": p["name"]} for p in props]
    st = I.join_ibb(mah, rows)
    age_cols = ("1980_oncesi", "1980-2000_arasi", "2000_sonrasi")
    h_cols = ("1-4 kat_arasi", "5-9 kat_arasi", "9-19 kat_arasi")
    n = len(props)
    age, hgt, known = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n, bool)
    for i, r in st.items():
        age[i] = [r[c] for c in age_cols]
        hgt[i] = [r[c] for c in h_cols]
        known[i] = age[i].sum() > 0 and hgt[i].sum() > 0
    dist = np.array([p["district"] for p in props])
    for d in np.unique(dist):
        m = (dist == d) & known
        for arr in (age, hgt):
            fill = arr[m].sum(0) if m.any() else arr[known].sum(0)
            arr[(dist == d) & ~known] = fill
    age /= age.sum(1, keepdims=True)
    hgt /= hgt.sum(1, keepdims=True)
    return age, hgt, known


def profiles(inv, age, hgt, mix, table):
    """Building-class weights per (mahalle, known storeys) profile over (family, storeys) keys."""
    hmax = {}
    for t in table:
        fam, h = t.rsplit("_H", 1)
        hmax[fam] = max(hmax.get(fam, 0), int(h))
    lv = inv["levels"]
    storeys = np.where(np.isfinite(lv), np.clip(np.round(lv), 1, 40), 0).astype(int)
    storeys[(storeys == 0) & (inv["area_m2"] < S.SMALL_M2)] = -1  # small, untagged: low-rise
    pkey = np.column_stack([inv["mahalle"], storeys])
    uniq, prof = np.unique(pkey, axis=0, return_inverse=True)
    keys, kidx = [], {}
    rows = []
    exp_storeys = np.zeros(len(uniq))
    for p, (m, s) in enumerate(uniq):
        w = {}
        if s == -1:
            bands = {0: 1.0}
            st = {0: S.SMALL_STOREYS}
            exp_storeys[p] = sum(k * v for k, v in S.SMALL_STOREYS.items())
        elif s > 0:
            bands = {int(S.height_band(s)): 1.0}
            st = {b: {int(s): 1.0} for b in bands}
            exp_storeys[p] = s
        else:
            bands = {b: hgt[m, b] for b in range(3)}
            st = {b: S.BAND_STOREYS[b] for b in range(3)}
            exp_storeys[p] = sum(
                hgt[m, b] * sum(k * v for k, v in S.BAND_STOREYS[b].items()) for b in range(3)
            )
        for a in range(3):
            for b, pb in bands.items():
                for fam, pf in mix[(a, b)].items():
                    for sv, ps in st[b].items():
                        k = (fam, min(sv, hmax[fam]))
                        w[k] = w.get(k, 0.0) + age[m, a] * pb * pf * ps
        rows.append(w)
        for k in w:
            if k not in kidx:
                kidx[k] = len(keys)
                keys.append(k)
    W = np.zeros((len(uniq), len(keys)), np.float32)
    for p, w in enumerate(rows):
        for k, v in w.items():
            W[p, kidx[k]] = v
    return {
        "prof": prof.ravel(),
        "W": W,
        "keys": keys,
        "pmah": uniq[:, 0],
        "exp_storeys": exp_storeys,
    }


# --------------------------------------------------------------------------- OSM network


def read_network(pbf: Path):
    import osmium

    edges, hosp, fire, rail, bridges = [], [], [], [], []
    fp = (
        osmium.FileProcessor(str(pbf))
        .with_locations()
        .with_filter(osmium.filter.KeyFilter("highway", "railway", "amenity"))
    )
    for o in fp:
        tags = o.tags
        if o.is_node():
            if tags.get("amenity") in ("hospital", "fire_station"):
                loc = o.location
                if I._in_box(loc.lon, loc.lat):
                    (hosp if tags["amenity"] == "hospital" else fire).append(
                        (loc.lon, loc.lat, tags.get("name", ""))
                    )
            continue
        if not o.is_way():
            continue
        try:
            pts = [(n.ref, n.lon, n.lat) for n in o.nodes]
        except osmium.InvalidLocationError:
            continue
        lon0, lat0 = pts[0][1], pts[0][2]
        if not I._in_box(lon0, lat0):
            continue
        am = tags.get("amenity")
        if am in ("hospital", "fire_station"):
            cx, cy = np.mean([p[1] for p in pts]), np.mean([p[2] for p in pts])
            (hosp if am == "hospital" else fire).append((cx, cy, tags.get("name", "")))
        hw, rw = tags.get("highway"), tags.get("railway")
        if hw in S.ROAD_KMH:
            for a, b in zip(pts, pts[1:]):
                edges.append((a[0], b[0], a[1], a[2], b[1], b[2], hw))
            if tags.get("bridge") in ("yes", "viaduct"):
                bridges.append((o.id, tags.get("name", ""), hw, [(p[1], p[2]) for p in pts]))
        elif rw in RAIL:
            rail.append((rw, tags.get("name", ""), [(p[1], p[2]) for p in pts]))
    return edges, hosp, fire, rail, bridges


def build_graph(edges, inside):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u = np.array([e[0] for e in edges], np.int64)
    v = np.array([e[1] for e in edges], np.int64)
    ax, ay = to_utm(np.array([e[2] for e in edges]), np.array([e[3] for e in edges]))
    bx, by = to_utm(np.array([e[4] for e in edges]), np.array([e[5] for e in edges]))
    cls = np.array([e[6] for e in edges])
    mid_lon = np.array([(e[2] + e[4]) / 2 for e in edges])
    mid_lat = np.array([(e[3] + e[5]) / 2 for e in edges])
    keep = inside(mid_lon, mid_lat) >= 0
    u, v, cls = u[keep], v[keep], cls[keep]
    ax, ay, bx, by = (np.asarray(t)[keep] for t in (ax, ay, bx, by))
    ids, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    iu, iv = inv[: len(u)], inv[len(u) :]
    n = len(ids)
    length = np.hypot(bx - ax, by - ay)
    g = coo_matrix((np.ones(len(iu)), (iu, iv)), shape=(n, n)).tocsr()
    _, lab = connected_components(g, directed=False)
    main = np.argmax(np.bincount(lab))
    ok = lab[iu] == main
    nx = np.zeros(n)
    ny = np.zeros(n)
    nx[iu], ny[iu], nx[iv], ny[iv] = ax, ay, bx, by
    speed = np.array([S.ROAD_KMH[c] for c in cls]) / 3.6
    width = np.array([S.ROAD_WIDTH_M[c] for c in cls], float)
    return {
        "iu": iu[ok],
        "iv": iv[ok],
        "ax": ax[ok],
        "ay": ay[ok],
        "bx": bx[ok],
        "by": by[ok],
        "length": length[ok],
        "time": (length / speed)[ok],
        "width": width[ok],
        "cls": cls[ok],
        "nx": nx,
        "ny": ny,
        "node_main": lab == main,
    }


def building_edge_pairs(inv, exp_height, g, max_m=30.0):
    """Building–street pairs where a collapse could leave less than 3 m of the street free."""
    import shapely

    bx, by = to_utm(inv["lon"], inv["lat"])
    pts = shapely.points(np.asarray(bx), np.asarray(by))
    lines = shapely.linestrings(
        np.stack([np.column_stack([g["ax"], g["ay"]]), np.column_stack([g["bx"], g["by"]])], 1)
    )
    tree = shapely.STRtree(lines)
    b, e = tree.query(pts, predicate="dwithin", distance=max_m)
    dist = shapely.distance(pts[b], lines[e])
    facade = np.maximum(dist - np.sqrt(np.maximum(inv["area_m2"][b], 1.0)) / 2, 0.5)
    free = S.free_width(facade, g["width"][e], exp_height[b])
    keep = free < S.MIN_FREE_M
    return {"b": b[keep], "e": e[keep], "free": free[keep]}


def multi_source_time(g, blocked, sources):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    ok = ~blocked
    n = len(g["nx"])
    w = g["time"][ok]
    m = coo_matrix(
        (
            np.concatenate([w, w]),
            (
                np.concatenate([g["iu"][ok], g["iv"][ok]]),
                np.concatenate([g["iv"][ok], g["iu"][ok]]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    return dijkstra(m, directed=True, indices=sources, min_only=True)


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", type=Path, required=True)
    ap.add_argument("--stock", type=Path, required=True)
    ap.add_argument("--dem", type=Path, required=True)
    ap.add_argument("--esrm", type=Path, required=True)
    ap.add_argument("--realizations", type=int, default=100, help="per GMPE")
    ap.add_argument("--road-realizations", type=int, default=40)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    inv = dict(np.load(DATA / "buildings.npz"))
    props = [
        f["properties"] for f in json.loads((DATA / "mahalle.geojson").read_text())["features"]
    ]
    nm = len(props)
    log(f"{len(inv['lon']):,} buildings, {nm} mahalle")

    cells = cached("cells", lambda: build_cells(inv))
    nc = len(cells["n"])
    site = cached("site", lambda: {"vs30": vs30_at(cells["lon"], cells["lat"], a.dem)})
    vs30 = site["vs30"]
    log(f"{nc:,} cells of 1 km, Vs30 median {np.median(vs30):.0f} m/s")
    gmf = cached(f"gmf_{a.realizations}", lambda: ground_motion(cells, vs30, a.realizations))
    gm = gmf["gm"]  # (imt, cell, r) in g
    R = gm.shape[2]
    log(f"ground motion: {R} realizations; median PGA over cells {np.median(gm[0]):.3f} g")

    rows, night, table, lol = esrm_tables(a.esrm)
    mix = S.class_mix(rows)
    age, hgt, known = stock_shares(props, a.stock)
    P = profiles(inv, age, hgt, mix, table)
    keys, W, prof = P["keys"], P["W"], P["prof"]
    log(f"{len(keys)} building classes, {W.shape[0]:,} profiles")

    storeys_b = P["exp_storeys"][prof]
    floor = inv["area_m2"] * storeys_b
    occ = night * floor / floor.sum()

    # unique (profile, cell) pairs carry everything the damage model needs
    pair_key = prof.astype(np.int64) * nc + cells["cell"]
    pk, pair = np.unique(pair_key, return_inverse=True)
    pair = pair.ravel()
    p_prof, p_cell = pk // nc, pk % nc
    p_n = np.bincount(pair).astype(np.float32)
    p_occ = np.bincount(pair, occ).astype(np.float32)
    p_mah = P["pmah"][p_prof]
    log(f"{len(pk):,} profile-cell pairs")

    imt_ix = {"PGA": 0, "SA(0.3)": 1, "SA(0.6)": 2, "SA(1.0)": 3}
    lol_sheet = {"PGA": "PGA", "SA(0.3)": "SA(0.3s)", "SA(0.6)": "SA(0.6s)", "SA(1.0)": "SA(1.0s)"}
    kinfo = []
    for fam, h in keys:
        imt, med, beta = table[f"{fam}_H{h}"]
        sheet = lol[lol_sheet[imt]]
        kinfo.append(
            (
                imt_ix[imt],
                med,
                beta,
                sheet.iloc[:, 0].to_numpy(float),
                sheet[f"{fam}_H{h}"].to_numpy(float),
            )
        )

    def pair_damage(r):
        exc = np.zeros((len(pk), 4), np.float32)
        fr = np.zeros(len(pk), np.float32)
        for k, (ii, med, beta, imgrid, frc) in enumerate(kinfo):
            im = gm[ii, :, r]
            e_cell = S.lognormal_exceed(im, med, beta).astype(np.float32)
            f_cell = np.interp(np.log(np.maximum(im, imgrid[0])), np.log(imgrid), frc).astype(
                np.float32
            )
            w = W[p_prof, k]
            exc += w[:, None] * e_cell[p_cell]
            fr += w * f_cell[p_cell]
        return exc, fr

    ds_mah = np.zeros((R, nm, 5), np.float32)
    deaths_mah = np.zeros((R, nm), np.float32)
    ds_cell = np.zeros((nc, 5), np.float64)
    tot_ds = np.zeros((R, 5))
    for r in range(R):
        exc, fr = pair_damage(r)
        pds = S.ds_probabilities(exc)
        cnt = pds * p_n[:, None]
        for k in range(5):
            ds_mah[r, :, k] = np.bincount(p_mah, cnt[:, k], minlength=nm)
            ds_cell[:, k] += np.bincount(p_cell, cnt[:, k], minlength=nc) / R
        deaths_mah[r] = np.bincount(p_mah, p_occ * fr, minlength=nm)
        tot_ds[r] = cnt.sum(0)
        if r % 25 == 0:
            log(
                f"  damage realization {r}/{R}: complete {cnt[:, 4].sum():,.0f}, "
                f"extensive {cnt[:, 3].sum():,.0f}, deaths {deaths_mah[r].sum():,.0f}"
            )
    log(f"damage done ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------ roads and hospital access
    import pickle

    netp = CACHE / "network.pkl"
    if netp.exists():
        net = pickle.loads(netp.read_bytes())
    else:
        edges, hosp, fire, rail, bridges = read_network(a.pbf)
        net = {"edges": edges, "hosp": hosp, "fire": fire, "rail": rail, "bridges": bridges}
        netp.write_bytes(pickle.dumps(net))
    import shapely

    mpolys = [
        shapely.geometry.shape(f["geometry"])
        for f in json.loads((DATA / "mahalle.geojson").read_text())["features"]
    ]
    tree_m = shapely.STRtree(mpolys)

    def inside(lon, lat):
        pi, gi = tree_m.query(shapely.points(lon, lat), predicate="within")
        out = np.full(len(lon), -1)
        out[pi] = gi
        return out

    g = build_graph(net["edges"], inside)
    log(
        f"road graph: {len(g['nx']):,} nodes, {len(g['iu']):,} edges, "
        f"{g['length'].sum() / 1000:,.0f} km"
    )
    exp_h = storeys_b * S.STOREY_M
    pairs = building_edge_pairs(inv, exp_h, g)
    log(f"{len(pairs['b']):,} building-street pairs could block a street")

    from scipy.spatial import cKDTree

    main_nodes = np.flatnonzero(g["node_main"])
    kd = cKDTree(np.column_stack([g["nx"][main_nodes], g["ny"][main_nodes]]))
    hx, hy = to_utm(
        np.array([h[0] for h in net["hosp"]], float), np.array([h[1] for h in net["hosp"]], float)
    )
    hin = (
        inside(
            np.array([h[0] for h in net["hosp"]], float),
            np.array([h[1] for h in net["hosp"]], float),
        )
        >= 0
    )
    hosp_nodes = np.unique(main_nodes[kd.query(np.column_stack([hx, hy])[hin])[1]])
    cx, cy = to_utm(cells["lon"], cells["lat"])
    cell_node = main_nodes[kd.query(np.column_stack([cx, cy]))[1]]
    cell_pop = np.bincount(cells["cell"], occ, minlength=nc)
    t_base = multi_source_time(g, np.zeros(len(g["iu"]), bool), hosp_nodes)[cell_node] / 60
    log(f"{hin.sum()} hospitals in Istanbul; median baseline access {np.median(t_base):.1f} min")

    rng = np.random.default_rng(0)
    RR = min(a.road_realizations, R)
    pick = np.linspace(0, R - 1, RR).astype(int)
    blocked_p = np.zeros(len(g["iu"]))
    t_post = np.zeros((RR, nc), np.float32)
    blocked_km = {}
    for j, r in enumerate(pick):
        exc, _ = pair_damage(r)
        p4_b = exc[pair, 3]
        collapsed = rng.random(len(p4_b)) < p4_b
        bl = S.blocked_edges(len(g["iu"]), pairs["e"], pairs["free"], collapsed[pairs["b"]])
        blocked_p += bl / RR
        for c in np.unique(g["cls"]):
            blocked_km.setdefault(c, []).append(
                float(g["length"][bl & (g["cls"] == c)].sum() / 1000)
            )
        t_post[j] = multi_source_time(g, bl, hosp_nodes)[cell_node] / 60
        log(
            f"  roads {j + 1}/{RR}: {bl.sum():,} edges blocked "
            f"({g['length'][bl].sum() / 1000:,.0f} km)"
        )

    pp = CACHE / "profile.json"
    if not pp.exists():
        pp.write_text(json.dumps(attenuation_profile()))
    profile = json.loads(pp.read_text())
    write_outputs(locals())
    log(f"done in {(time.time() - t0) / 60:.1f} min")


def write_outputs(v):
    """Compact JSON for the web page (all numbers rounded; arrays aligned with istanbul.json)."""
    import shapely

    props, cells, gm, R, nm = v["props"], v["cells"], v["gm"], v["R"], v["nm"]
    tot_ds, ds_mah, deaths_mah = v["tot_ds"], v["ds_mah"], v["deaths_mah"]
    g, pick, t_base, t_post, cell_pop = v["g"], v["pick"], v["t_base"], v["t_post"], v["cell_pop"]
    pga = gm[0]
    pga_mean = np.exp(np.log(np.maximum(pga, 1e-4)).mean(1))
    mmi_mean = S.mmi_from_pga(pga_mean)
    q = lambda x, p: np.percentile(x, p, axis=0)  # noqa: E731

    # cells: 1 km squares with shaking, site and damage
    keys = cells["keys"]
    x0, y0 = keys[:, 0] * 1000.0, keys[:, 1] * 1000.0
    lon_c, lat_c = to_lonlat(
        np.concatenate([x0, x0 + 1000, x0 + 1000, x0]),
        np.concatenate([y0, y0, y0 + 1000, y0 + 1000]),
    )
    nc = len(x0)
    corners = np.stack(
        [np.asarray(lon_c).reshape(4, nc), np.asarray(lat_c).reshape(4, nc)], -1
    ).transpose(1, 0, 2)
    ds_cell = v["ds_cell"]
    blocked_p = v["blocked_p"]
    cell_json = {
        "corners": np.round(corners, 4).tolist(),
        "vs30": np.round(v["vs30"]).astype(int).tolist(),
        "pga": np.round(pga_mean, 3).tolist(),
        "sa10": np.round(np.exp(np.log(np.maximum(gm[3], 1e-4)).mean(1)), 3).tolist(),
        "mmi": np.round(mmi_mean, 1).tolist(),
        "buildings": cells["n"].astype(int).tolist(),
        "heavy": np.round(ds_cell[:, 3] + ds_cell[:, 4], 1).tolist(),
        "complete": np.round(ds_cell[:, 4], 1).tolist(),
        "pop": np.round(cell_pop).astype(int).tolist(),
        "t_base": np.round(np.minimum(t_base, 999), 1).tolist(),
        "t_post": np.round(np.minimum(np.median(t_post, 0), 999), 1).tolist(),
        "p_cut": np.round((t_post > 30).mean(0), 2).tolist(),
    }
    (SITE / "cells.json").write_text(json.dumps(cell_json, separators=(",", ":")))

    # mahalle: aligned with site/data/istanbul.json features
    cm_ = v["cells"]["cell"]
    mah_of_b = v["inv"]["mahalle"]
    pga_b = pga_mean[cm_]
    pga_m = np.bincount(mah_of_b, pga_b, minlength=nm) / np.maximum(
        np.bincount(mah_of_b, minlength=nm), 1
    )
    heavy = ds_mah[:, :, 3] + ds_mah[:, :, 4]
    mah_json = {
        "pga": np.round(pga_m, 3).tolist(),
        "mmi": np.round(S.mmi_from_pga(pga_m), 1).tolist(),
        "ds": np.round(ds_mah.mean(0), 1).tolist(),
        "heavy": np.round(heavy.mean(0), 1).tolist(),
        "heavy_lo": np.round(q(heavy, 5), 1).tolist(),
        "heavy_hi": np.round(q(heavy, 95), 1).tolist(),
        "deaths": np.round(deaths_mah.mean(0), 1).tolist(),
        "deaths_lo": np.round(q(deaths_mah, 5), 1).tolist(),
        "deaths_hi": np.round(q(deaths_mah, 95), 1).tolist(),
        "blocked_share": [],
    }
    # share of street length blocked per mahalle (mean over road realizations)
    mid_lon, mid_lat = to_lonlat((g["ax"] + g["bx"]) / 2, (g["ay"] + g["by"]) / 2)
    tree_m = shapely.STRtree(
        [
            shapely.geometry.shape(f["geometry"])
            for f in json.loads((DATA / "mahalle.geojson").read_text())["features"]
        ]
    )
    pi, gi = tree_m.query(shapely.points(mid_lon, mid_lat), predicate="within")
    em = np.full(len(g["iu"]), -1)
    em[pi] = gi
    ok = em >= 0
    L = np.bincount(em[ok], g["length"][ok], minlength=nm)
    LB = np.bincount(em[ok], (g["length"] * blocked_p)[ok], minlength=nm)
    mah_json["blocked_share"] = np.round(LB / np.maximum(L, 1), 3).tolist()
    (SITE / "mahalle.json").write_text(json.dumps(mah_json, separators=(",", ":")))

    # main streets likely to be blocked (as lines)
    main = np.isin(g["cls"], ["trunk", "primary", "secondary", "tertiary"]) & (blocked_p >= 0.25)
    a_lon, a_lat = to_lonlat(g["ax"][main], g["ay"][main])
    b_lon, b_lat = to_lonlat(g["bx"][main], g["by"][main])
    roads = [
        [
            round(float(p), 2),
            round(float(a), 5),
            round(float(b), 5),
            round(float(c), 5),
            round(float(d), 5),
        ]
        for p, a, b, c, d in zip(blocked_p[main], a_lon, a_lat, b_lon, b_lat)
    ]
    (SITE / "roads.json").write_text(json.dumps(roads, separators=(",", ":")))

    # hospitals, bridges, rail
    hosp = list(v["net"]["hosp"])
    hl = np.array([h[0] for h in hosp], float)
    ht = np.array([h[1] for h in hosp], float)
    hin = v["inside"](hl, ht) >= 0
    from scipy.spatial import cKDTree

    cx, cy = to_utm(cells["lon"], cells["lat"])
    kdc = cKDTree(np.column_stack([cx, cy]))
    hcell = kdc.query(np.column_stack(to_utm(hl[hin], ht[hin])))[1]
    hosp_json = [
        [
            round(float(a), 5),
            round(float(b), 5),
            str(n),
            round(float(pga_mean[c]), 3),
            round(float(mmi_mean[c]), 1),
        ]
        for a, b, n, c in zip(hl[hin], ht[hin], [h[2] for h, k in zip(hosp, hin) if k], hcell)
    ]
    (SITE / "hospitals.json").write_text(
        json.dumps(hosp_json, separators=(",", ":"), ensure_ascii=False)
    )

    def mmi_at(lon, lat):
        return mmi_mean[
            kdc.query(np.column_stack(to_utm(np.asarray(lon, float), np.asarray(lat, float))))[1]
        ]

    def length_m(pts):
        x, y = to_utm(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        return float(np.hypot(np.diff(x), np.diff(y)).sum())

    bands = [5, 6, 7, 8, 9, 10.1]
    rail_km = {}
    for kind, _name, pts in v["net"]["rail"]:
        if v["inside"](np.array([pts[0][0]]), np.array([pts[0][1]]))[0] < 0:
            continue
        m = float(mmi_at([np.mean([p[0] for p in pts])], [np.mean([p[1] for p in pts])])[0])
        rail_km.setdefault(kind, [0.0] * (len(bands) - 1))
        i = int(np.clip(np.searchsorted(bands, m, side="right") - 1, 0, len(bands) - 2))
        rail_km[kind][i] += length_m(pts) / 1000
    named_bridges = {}
    bridge_counts = [0] * (len(bands) - 1)
    for _wid, name, hw, pts in v["net"]["bridges"]:
        if v["inside"](np.array([pts[0][0]]), np.array([pts[0][1]]))[0] < 0:
            continue
        m = float(mmi_at([np.mean([p[0] for p in pts])], [np.mean([p[1] for p in pts])])[0])
        i = int(np.clip(np.searchsorted(bands, m, side="right") - 1, 0, len(bands) - 2))
        bridge_counts[i] += 1
        if name and ("Köprüsü" in name) and hw in ("motorway", "trunk") and length_m(pts) > 500:
            named_bridges[name] = max(named_bridges.get(name, 0), round(m, 1))

    # summaries and chart data
    district = np.array([p["district"] for p in props])
    dists = sorted(set(district))
    by_d = {d: np.flatnonzero(district == d) for d in dists}
    heavy_d = {d: heavy[:, ix].sum(1) for d, ix in by_d.items()}
    deaths_d = {d: deaths_mah[:, ix].sum(1) for d, ix in by_d.items()}
    ibb_heavy = np.array([(p.get("very_heavy") or 0) + (p.get("heavy") or 0) for p in props], float)
    ibb_has = np.array([p.get("very_heavy") is not None for p in props])
    nb = np.bincount(v["inv"]["mahalle"], minlength=nm)
    ours_heavy = heavy.mean(0)
    pop_w = cell_pop / cell_pop.sum()
    t_med = lambda t: float(np.interp(0.5, np.cumsum(pop_w[np.argsort(t)]), np.sort(t)))  # noqa: E731
    summary = {
        "scenario": {
            "magnitude": S.MAG,
            "fault": "Main Marmara Fault, Central Basin to Çınarcık (GEM GAF, EMME)",
            "fault_km": 104,
            "depth_km": [S.UPPER_KM, S.LOWER_KM],
            "rake": S.RAKE,
            "trace": S.FAULT_TRACE,
            "gsims": list(GSIMS),
            "realizations": int(R),
            "road_realizations": int(len(pick)),
            "cells": int(nc),
            "buildings": int(len(v["inv"]["lon"])),
            "night_population": round(v["night"]),
            "debris_k": S.DEBRIS_K,
            "min_free_m": S.MIN_FREE_M,
        },
        "damage": {
            name: {
                "mean": float(tot_ds[:, k].mean()),
                "p5": float(q(tot_ds[:, k], 5)),
                "p95": float(q(tot_ds[:, k], 95)),
            }
            for k, name in enumerate(["none", "slight", "moderate", "extensive", "complete"])
        },
        "deaths": {
            "mean": float(deaths_mah.sum(1).mean()),
            "p5": float(q(deaths_mah.sum(1), 5)),
            "p95": float(q(deaths_mah.sum(1), 95)),
        },
        "ibb": {
            "very_heavy_plus_heavy": float(ibb_heavy.sum()),
            "moderate": float(sum(p.get("moderate") or 0 for p in props)),
            "deaths": float(sum(p.get("deaths") or 0 for p in props)),
        },
        "validation": {
            "r_log_heavy_ratio": float(
                np.corrcoef(
                    np.log1p(ours_heavy[ibb_has] / np.maximum(nb[ibb_has], 1) * 1000),
                    np.log1p(ibb_heavy[ibb_has] / np.maximum(nb[ibb_has], 1) * 1000),
                )[0, 1]
            )
        },
        "roads": {
            "km_total": float(g["length"].sum() / 1000),
            "km_blocked": {
                c: {
                    "mean": float(np.mean(x)),
                    "p5": float(np.percentile(x, 5)),
                    "p95": float(np.percentile(x, 95)),
                    "km": float(g["length"][g["cls"] == c].sum() / 1000),
                }
                for c, x in v["blocked_km"].items()
            },
        },
        "access": {
            "hospitals": int(len(hosp_json)),
            "median_min_before": t_med(t_base),
            "median_min_after": float(np.median([t_med(t) for t in t_post])),
            "pop_over_30min_before": float(pop_w[t_base > 30].sum()),
            "pop_over_30min_after": {
                "mean": float(np.mean([pop_w[t > 30].sum() for t in t_post])),
                "p5": float(np.percentile([pop_w[t > 30].sum() for t in t_post], 5)),
                "p95": float(np.percentile([pop_w[t > 30].sum() for t in t_post], 95)),
            },
        },
        "exposure": {
            "mmi_bands": bands,
            "rail_km": {k: [round(x, 1) for x in vv] for k, vv in rail_km.items()},
            "bridges": bridge_counts,
            "named_bridges": named_bridges,
            "hospitals": np.histogram([h[4] for h in hosp_json], bins=bands)[0].tolist(),
            "buildings": np.histogram(mmi_mean[cm_], bins=bands)[0].tolist(),
            "population": np.histogram(mmi_mean, bins=bands, weights=cell_pop)[0].round().tolist(),
        },
    }
    # chart data
    hist_heavy = (tot_ds[:, 3] + tot_ds[:, 4]).round().tolist()
    cdf_x = np.linspace(0, 60, 61)
    cdf = lambda t: [float(pop_w[t <= x].sum()) for x in cdf_x]  # noqa: E731
    fam_curves = {}
    for fam, h in [
        ("CR_LFINF-CDN-0", 5),
        ("CR_LFINF-CDL-15", 5),
        ("CR_LWAL-DUH", 5),
        ("MUR-CL99_LWAL-DNO", 2),
        ("CR_LDUAL-DUH", 10),
    ]:
        imt, med, beta = v["table"][f"{fam}_H{h}"]
        grid = np.geomspace(0.02, 3.0, 60)
        fam_curves[f"{fam}_H{h}"] = {
            "imt": imt,
            "im": grid.round(4).tolist(),
            "exceed": S.lognormal_exceed(grid, med, beta).round(4).T.tolist(),
        }
    charts = {
        "attenuation": v["profile"],
        "cells_scatter": {
            "rjb": np.round(v["gmf"]["rjb"], 1).tolist(),
            "pga": np.round(pga_mean, 3).tolist(),
            "vs30": np.round(v["vs30"]).astype(int).tolist(),
        },
        "hist_heavy": hist_heavy,
        "hist_deaths": deaths_mah.sum(1).round().tolist(),
        "gsim_of_r": v["gmf"]["gsim_of_r"].tolist(),
        "districts": [
            {
                "d": d,
                "heavy": float(heavy_d[d].mean()),
                "lo": float(np.percentile(heavy_d[d], 5)),
                "hi": float(np.percentile(heavy_d[d], 95)),
                "deaths": float(deaths_d[d].mean()),
                "ibb_heavy": float(ibb_heavy[by_d[d]].sum()),
                "buildings": int(nb[by_d[d]].sum()),
            }
            for d in dists
        ],
        "validation": {
            "ours": np.round(ours_heavy, 1).tolist(),
            "ibb": ibb_heavy.tolist(),
            "has": ibb_has.tolist(),
            "n": nb.tolist(),
        },
        "access_cdf": {
            "x": cdf_x.tolist(),
            "before": cdf(t_base),
            "after": [cdf(t) for t in t_post[:: max(1, len(t_post) // 10)]],
            "after_median": cdf(np.median(t_post, 0)),
        },
        "fragility": fam_curves,
        "class_shares": {
            f"{f}_H{h}": float(w)
            for (f, h), w in zip(v["keys"], (v["W"][v["prof"]].sum(0) / len(v["prof"])))
        },
    }
    (SITE / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (SITE / "charts.json").write_text(json.dumps(charts, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()

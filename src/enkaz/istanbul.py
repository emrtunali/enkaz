"""Istanbul readiness: building inventory, mahalle boundaries and İBB's scenario estimates.

Istanbul has no post-earthquake imagery and no damage labels, so nothing here measures accuracy.
It prepares what the response pipeline (``respond.py``) needs on the day:

* every building footprint mapped in OpenStreetMap inside the 39 districts (Geofabrik's Türkiye
  extract; one pass over the file takes well under a minute),
* mahalle and district boundaries from OpenStreetMap,
* İBB's open data joined to those boundaries by district and normalised mahalle name: the loss
  estimates for a Mw 7.5 night-time scenario, and the 2017 building stock (age and height).

The scenario numbers are a planning estimate, not an observation. They are useful as a prior for
where to look first while imagery is still on its way, and to see where the footprint inventory
is thin (OSM count far below İBB's count), because the image model can only score mapped buildings.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

BBOX = (27.95, 40.75, 29.95, 41.60)  # lon/lat box around the province (includes neighbours)

_TR = str.maketrans("çğıiöşüâîûÇĞİIÖŞÜÂÎÛ", "cgiiosuaiuCGIIOSUAIU")
_SUFFIX = re.compile(r"\s+(MAHALLESI|MAH\.?|KOYU)$")  # whole words only: DURSUNKÖY keeps its KÖY
_ALIAS = {"EYUP": "EYUPSULTAN"}  # district renamed in 2017; İBB's tables use the old name


def norm(name: str) -> str:
    """Join key for Turkish place names: no diacritics, upper case, no 'Mahallesi'/'Köyü'.

    İBB writes ``HEYBELİADA``, OSM ``Heybeliada Mahallesi``; both become ``HEYBELIADA``.
    """
    s = _SUFFIX.sub("", name.translate(_TR).upper().strip())
    s = re.sub(r"[^A-Z0-9]", "", s)
    return _ALIAS.get(s, s)


def read_ibb_csv(path: Path) -> list[dict]:
    """İBB open-data CSV (';'-separated, Turkish code page) -> rows with int fields."""
    text = Path(path).read_bytes().decode("cp1254")
    rows = []
    for r in csv.DictReader(io.StringIO(text), delimiter=";"):
        r = {k: (v or "").strip() for k, v in r.items() if k}
        if not r.get("ilce_adi"):  # blank or footer line
            continue
        rows.append({k: int(v) if v.lstrip("-").isdigit() else v for k, v in r.items()})
    return rows


# --------------------------------------------------------------------------- OpenStreetMap


def _in_box(lon: float, lat: float, bbox=BBOX) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def ring_area_m2(lon: np.ndarray, lat: np.ndarray) -> float:
    """Planar area of a small lon/lat ring (local equirectangular projection)."""
    k = 111_320.0
    x = (lon - lon.mean()) * k * math.cos(math.radians(float(lat.mean())))
    y = (lat - lat.mean()) * k
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def read_buildings(pbf: Path, bbox=BBOX) -> dict[str, np.ndarray]:
    """Closed ``building=*`` ways inside ``bbox``: outline, centroid, area and storeys.

    Multipolygon buildings (a few per thousand, mostly large complexes) are skipped.
    Outlines are packed as one float32 lon/lat array plus offsets, like the crop files.
    """
    import osmium

    ids, lon, lat, area, levels, kinds, coords, offs = [], [], [], [], [], [], [], [0]
    fp = (
        osmium.FileProcessor(str(pbf))
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("building"))
    )
    for w in fp:
        try:
            pts = np.array([(n.lon, n.lat) for n in w.nodes], dtype=np.float64)
        except osmium.InvalidLocationError:
            continue
        if len(pts) < 4 or not np.array_equal(pts[0], pts[-1]):
            continue
        cx, cy = pts[:-1].mean(axis=0)
        if not _in_box(cx, cy, bbox):
            continue
        ids.append(w.id)
        lon.append(cx)
        lat.append(cy)
        area.append(ring_area_m2(pts[:-1, 0], pts[:-1, 1]))
        lv = w.tags.get("building:levels", "")
        levels.append(float(lv) if re.fullmatch(r"\d{1,3}(\.\d)?", lv) else np.nan)
        kinds.append(w.tags.get("building"))
        coords.append(pts.astype(np.float32))
        offs.append(offs[-1] + len(pts))
    names, code = np.unique(np.array(kinds, dtype=object).astype(str), return_inverse=True)
    return {
        "osm_id": np.asarray(ids, np.int64),
        "lon": np.asarray(lon),
        "lat": np.asarray(lat),
        "area_m2": np.asarray(area, np.float32),
        "levels": np.asarray(levels, np.float32),
        "kind": code.astype(np.int16),
        "kind_names": names,
        "ring": np.concatenate(coords) if coords else np.zeros((0, 2), np.float32),
        "ring_off": np.asarray(offs, np.int64),
    }


MS_LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"


def quadkeys(bbox, zoom: int = 9) -> set[str]:
    """Bing-map quadkeys of the tiles meeting a lon/lat box (Microsoft's footprints are split
    into one file per zoom-9 quadkey)."""

    def tile(lon, lat):
        s = math.sin(math.radians(lat))
        x = int((lon + 180) / 360 * 2**zoom)
        y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 2**zoom)
        return x, y

    (x0, y1), (x1, y0) = tile(bbox[0], bbox[1]), tile(bbox[2], bbox[3])
    out = set()
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            out.add(
                "".join(str(((x >> i) & 1) + 2 * ((y >> i) & 1)) for i in range(zoom - 1, -1, -1))
            )
    return out


def read_ms_footprints(paths: Sequence[Path], bbox=BBOX) -> dict[str, np.ndarray]:
    """Microsoft Global ML Building Footprints (``.csv.gz`` of GeoJSON lines, one file per zoom-9
    quadkey) in the same packed form as :func:`read_buildings`. Ids are line numbers (negative,
    so they never collide with OSM ids); storeys are unknown."""
    import gzip
    import json

    ids, lon, lat, area, coords, offs = [], [], [], [], [], [0]
    n = 0
    for path in paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                n += 1
                g = json.loads(line)["geometry"]
                if g["type"] != "Polygon":
                    continue
                pts = np.asarray(g["coordinates"][0], dtype=np.float64)
                cx, cy = pts[:-1].mean(axis=0)
                if not _in_box(cx, cy, bbox):
                    continue
                ids.append(-n)
                lon.append(cx)
                lat.append(cy)
                area.append(ring_area_m2(pts[:-1, 0], pts[:-1, 1]))
                coords.append(pts.astype(np.float32))
                offs.append(offs[-1] + len(pts))
    k = len(ids)
    return {
        "osm_id": np.asarray(ids, np.int64),
        "lon": np.asarray(lon),
        "lat": np.asarray(lat),
        "area_m2": np.asarray(area, np.float32),
        "levels": np.full(k, np.nan, np.float32),
        "kind": np.zeros(k, np.int16),
        "kind_names": np.array(["ms_ml"]),
        "ring": np.concatenate(coords) if coords else np.zeros((0, 2), np.float32),
        "ring_off": np.asarray(offs, np.int64),
    }


def read_admin(pbf: Path, bbox=BBOX, levels=("6", "8")) -> list[dict]:
    """Administrative areas (districts = 6, mahalle = 8) whose label point is inside ``bbox``."""
    import osmium
    import shapely

    wkb = osmium.geom.WKBFactory()
    out = []
    fp = (
        osmium.FileProcessor(str(pbf))
        .with_areas(osmium.filter.TagFilter(("boundary", "administrative")))
        .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
    )
    for a in fp:
        if a.from_way() or a.tags.get("admin_level") not in levels:
            continue
        try:
            geom = shapely.from_wkb(wkb.create_multipolygon(a))
        except RuntimeError:  # broken ring in the source data
            continue
        p = geom.representative_point()
        if not _in_box(p.x, p.y, bbox):
            continue
        out.append(
            {
                "osm_id": a.orig_id(),
                "level": int(a.tags["admin_level"]),
                "name": a.tags.get("name", ""),
                "geom": geom,
            }
        )
    return out


def istanbul_units(admin: list[dict], districts: set[str]) -> tuple[list[dict], list[dict]]:
    """Pick Istanbul's districts by name, then the mahalle whose label point lies in one of them.

    Most mahalle relations in OSM carry no province tag, so membership is decided geometrically.
    """
    import shapely

    dist = [a for a in admin if a["level"] == 6 and norm(a["name"]) in districts]
    tree = shapely.STRtree([d["geom"] for d in dist])
    mah = []
    for a in admin:
        if a["level"] != 8:
            continue
        hit = tree.query(a["geom"].representative_point(), predicate="within")
        if len(hit):
            mah.append(a | {"district": dist[int(hit[0])]["name"]})
    return dist, mah


def assign(lon: np.ndarray, lat: np.ndarray, polygons, max_m: float = 0.0) -> np.ndarray:
    """Index of the polygon containing each point, -1 if none. With ``max_m`` > 0, points outside
    every polygon take the nearest one within about that many metres (footprints from different
    sources are often offset by a few metres)."""
    import shapely

    tree = shapely.STRtree(list(polygons))
    pts = shapely.points(lon, lat)
    pi, gi = tree.query(pts, predicate="within")
    out = np.full(len(lon), -1, np.int64)
    out[pi] = gi
    if max_m > 0 and (out < 0).any():
        rest = np.flatnonzero(out < 0)
        pi, gi = tree.query_nearest(pts[rest], max_distance=max_m / 111_000, all_matches=False)
        out[rest[pi]] = gi
    return out


def join_ibb(mahalle: list[dict], rows: list[dict]) -> dict[int, dict]:
    """Match İBB rows (``ilce_adi``, ``mahalle_adi``) to mahalle polygons. Returns {i: row}."""
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, m in enumerate(mahalle):
        by_key.setdefault((norm(m["district"]), norm(m["name"])), []).append(i)
    out = {}
    for r in rows:
        hits = by_key.get((norm(r["ilce_adi"]), norm(r["mahalle_adi"])), [])
        if len(hits) == 1:  # ambiguous names stay unmatched rather than guessed
            out[hits[0]] = r
    return out

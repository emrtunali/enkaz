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


def assign(lon: np.ndarray, lat: np.ndarray, polygons) -> np.ndarray:
    """Index of the polygon containing each point, -1 if none."""
    import shapely

    tree = shapely.STRtree(list(polygons))
    pi, gi = tree.query(shapely.points(lon, lat), predicate="within")
    out = np.full(len(lon), -1, np.int64)
    out[pi] = gi
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

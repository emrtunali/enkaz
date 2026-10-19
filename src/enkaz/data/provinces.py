"""Türkiye's 81 provinces (ADM1) and what open data exists for each.

Boundaries: geoBoundaries gbOpen TUR ADM1 (derived from OpenStreetMap, CC BY-SA 2.0).
The coverage table is the honest answer to "does this work in province X?": the model can
run anywhere given pre/post imagery and footprints, but measured accuracy exists only where
labels exist.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from shapely.geometry import Point, box, shape
from shapely.strtree import STRtree

from ..geo import BBox
from .http import download, get_json

STATUS_LABELED = "labeled"  # building-level labels -> measured accuracy
STATUS_PAIRED = "pre+post imagery"  # model can run; no labels to measure it
STATUS_POST_ONLY = "post imagery only"  # only the post-only model can run
STATUS_NONE = "no post-event VHR imagery"  # pipeline ready, waiting for imagery


def fetch_adm1(api_url: str, dest_dir: Path) -> Path:
    meta = get_json(api_url)
    dest = Path(dest_dir) / "tur_adm1.geojson"
    download(meta["simplifiedGeometryGeoJSON"], dest)
    (Path(dest_dir) / "tur_adm1.meta.json").write_text(
        json.dumps(
            {k: meta.get(k) for k in ("boundaryLicense", "boundarySource", "sourceDataUpdateDate")},
            indent=1,
        )
    )
    return dest


class Provinces:
    def __init__(self, geojson_path: Path):
        fc = json.loads(Path(geojson_path).read_text())
        self.names = [f["properties"]["shapeName"] for f in fc["features"]]
        self.geoms = [shape(f["geometry"]) for f in fc["features"]]
        self._tree = STRtree(self.geoms)

    def __len__(self) -> int:
        return len(self.names)

    def at(self, lon: float, lat: float) -> str | None:
        pt = Point(lon, lat)
        for i in self._tree.query(pt):
            if self.geoms[i].contains(pt):
                return self.names[i]
        return None

    def for_bbox(self, bbox: BBox) -> str | None:
        """Province containing the bbox centre."""
        return self.at((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def intersecting(self, bbox: BBox) -> list[str]:
        b = box(*bbox)
        return [self.names[i] for i in self._tree.query(b) if self.geoms[i].intersects(b)]


def coverage_table(
    provinces: Provinces,
    items: Iterable[dict],
    event_date: str,
    labeled_points: Mapping[str, tuple[float, float]],
) -> list[dict]:
    """One row per province (all 81), sorted by status then post-event tile count."""
    rows = {
        n: {"province": n, "pre_tiles": 0, "post_tiles": 0, "labeled_cities": []}
        for n in provinces.names
    }
    for it in items:
        name = provinces.for_bbox(it["bbox"])
        if name is None:
            continue  # tiles over Syria or the sea
        rows[name]["pre_tiles" if it["date"] < event_date else "post_tiles"] += 1
    for city, (lat, lon) in labeled_points.items():
        name = provinces.at(lon, lat)
        if name is not None:
            rows[name]["labeled_cities"].append(city)
    order = {STATUS_LABELED: 0, STATUS_PAIRED: 1, STATUS_POST_ONLY: 2, STATUS_NONE: 3}
    for r in rows.values():
        if r["labeled_cities"]:
            r["status"] = STATUS_LABELED
        elif r["pre_tiles"] and r["post_tiles"]:
            r["status"] = STATUS_PAIRED
        elif r["post_tiles"]:
            r["status"] = STATUS_POST_ONLY
        else:
            r["status"] = STATUS_NONE
    return sorted(
        rows.values(), key=lambda r: (order[r["status"]], -r["post_tiles"], r["province"])
    )

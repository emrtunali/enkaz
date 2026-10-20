"""BRIGHT building-damage labels (Chen et al., 2025) for the Türkiye 2023 earthquake.

BRIGHT provides, per 1024x1024 tile at ~0.35 m: a pre-event Maxar optical image, a post-event
SAR image and a building damage mask (0 background, 1 intact, 2 damaged, 3 destroyed). Damage
attributes come from Copernicus EMS gradings transferred to individual building polygons and
checked by the authors. We use the optical pre image and the mask, and pair them with a
post-event *optical* Maxar scene ourselves (see ``enkaz.data.maxar``).

Only the Türkiye members are fetched from the multi-event archives (HTTP range requests).
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

from .remotezip import RemoteZip, ZipEntry

TURKEY_PREFIX = "turkey-earthquake_"
ARCHIVES = {"target": "target.zip", "pre": "pre-event.zip"}
# Per-building polygons (COCO format) for the train/val tiles of the CVPR 2026 MONTI release;
# BRIGHT's held-out test tiles have none and fall back to connected components of the mask.
INSTANCE_ARCHIVE = "cvprw26_trainval_instance_labels.zip"


def members(
    rz: RemoteZip, prefix: str = TURKEY_PREFIX, suffixes: tuple[str, ...] = (".tif", ".tiff")
) -> list[ZipEntry]:
    return [
        e
        for e in rz.entries()
        if Path(e.name).name.startswith(prefix) and e.name.lower().endswith(suffixes)
    ]


def tile_id(filename: str) -> str:
    """'turkey-earthquake_00000052_pre_disaster.tif' -> 'turkey-earthquake_00000052'."""
    stem = Path(filename).name
    parts = stem.split("_")
    return "_".join(parts[:2])


def fetch_members(
    rz: RemoteZip, entries: list[ZipEntry], out_dir: Path, workers: int = 16, log=print
) -> int:
    """Download members to ``out_dir`` (flat). Existing files of the right size are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        e
        for e in entries
        if not (
            (out_dir / Path(e.name).name).exists()
            and (out_dir / Path(e.name).name).stat().st_size == e.size
        )
    ]
    gb = sum(e.size for e in todo) / 1e9
    log(f"{len(entries) - len(todo)} present, {len(todo)} to fetch ({gb:.2f} GB)")

    def one(e: ZipEntry) -> None:
        dest = out_dir / Path(e.name).name
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(rz.read(e))
        tmp.replace(dest)

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 100 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)}")
    return len(todo)


def tile_index(target_dir: Path, city_centers: dict, max_km: float = 15.0) -> list[dict]:
    """Georeference and class-count every Türkiye label tile (needs rasterio + pyproj)."""
    import numpy as np
    import rasterio
    from pyproj import Transformer

    from ..geo import nearest_city

    rows = []
    for p in sorted(Path(target_dir).glob(f"{TURKEY_PREFIX}*_building_damage.tif")):
        with rasterio.open(p) as ds:
            m = ds.read(1)
            b = ds.bounds
            to_ll = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
            lon0, lat0 = to_ll.transform(b.left, b.bottom)
            lon1, lat1 = to_ll.transform(b.right, b.top)
            counts = np.bincount(m.ravel(), minlength=4)
            rows.append(
                {
                    "tile": tile_id(p.name),
                    "crs": ds.crs.to_string(),
                    "bounds": [b.left, b.bottom, b.right, b.top],
                    "bbox": [lon0, lat0, lon1, lat1],
                    "res": ds.res[0],
                    "city": nearest_city(
                        (lat0 + lat1) / 2, (lon0 + lon1) / 2, city_centers, max_km
                    ),
                    "px_bg": int(counts[0]),
                    "px_intact": int(counts[1]),
                    "px_damaged": int(counts[2]),
                    "px_destroyed": int(counts[3]),
                    "px_other": int(counts[4:].sum()),
                }
            )
    return rows

"""Tile-level preparation pipeline (runs on Colab; needs rasterio)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import numpy as np

from . import prepare as P
from .data import maxar

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def read_post(items: list[dict], crs, transform, shape: tuple[int, int]):
    """Resample the post-event visual COG(s) onto the given grid. Items are mosaicked in
    order; later items only fill pixels still empty. Returns (rgb uint8 HxWx3, valid bool)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    h, w = shape
    rgb = np.zeros((h, w, 3), np.uint8)
    valid = np.zeros((h, w), bool)
    with rasterio.Env(**GDAL_ENV):
        for it in items:
            with (
                rasterio.open(it["visual"]) as src,
                WarpedVRT(
                    src,
                    crs=crs,
                    transform=transform,
                    width=w,
                    height=h,
                    resampling=Resampling.bilinear,
                    src_nodata=0,
                    nodata=0,
                ) as vrt,
            ):
                arr = vrt.read(indexes=[1, 2, 3])  # 3xHxW
            px = np.moveaxis(arr, 0, -1)
            v = px.any(axis=-1) & ~valid
            rgb[v] = px[v]
            valid |= v
            if valid.all():
                break
    return rgb, valid


def ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    if m.sum() < 100:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def prepare_tile(
    tile: dict,
    items: list[dict],
    bright_dir: Path,
    instance_dir: Path,
    out_dir: Path,
    event_date: str,
    max_shift: int = 57,  # ~20 m at 0.35 m
) -> dict:
    """Align, list buildings and write crops for one tile. Skips work already on disk."""
    import rasterio

    tid = tile["tile"]
    meta_path = out_dir / "meta" / f"{tid}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    post_items = maxar.post_items_for(items, tile["bbox"], event_date)
    if not post_items:
        return {"tile": tid, "status": "no_post_scene"}

    with rasterio.open(bright_dir / "pre" / f"{tid}_pre_disaster.tif") as ds:
        pre = np.moveaxis(ds.read(indexes=[1, 2, 3]), 0, -1)
        crs, transform, shape = ds.crs, ds.transform, (ds.height, ds.width)
    with rasterio.open(bright_dir / "target" / f"{tid}_building_damage.tif") as ds:
        mask = ds.read(1)

    post, valid = read_post(post_items, crs, transform, shape)
    pre_g = P.gradient_magnitude(P.to_gray(pre))
    post_g = P.gradient_magnitude(P.to_gray(post))
    both = valid & pre.any(axis=-1)
    dy, dx, peak = P.phase_correlation(pre_g, post_g, max_shift=max_shift, valid=both)
    ncc_before = ncc(pre_g, post_g, both)
    applied = False
    if (dy, dx) != (0, 0):
        post2, valid2 = read_post(post_items, crs, P.shifted_transform(transform, dy, dx), shape)
        both2 = valid2 & pre.any(axis=-1)
        ncc_after = ncc(pre_g, P.gradient_magnitude(P.to_gray(post2)), both2)
        if ncc_after > ncc_before:
            post, valid, applied = post2, valid2, True
    else:
        ncc_after = ncc_before

    inst_path = instance_dir / f"{tid}_instance_damage.json"
    if inst_path.exists():
        buildings = P.buildings_from_instances(tid, P.load_instances(inst_path), shape)
    else:
        buildings = P.buildings_from_mask(tid, mask)

    crops = P.tile_crops(pre, post, valid, mask, buildings)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "crops" / f"{tid}.npz", **crops)
    (out_dir / "post").mkdir(parents=True, exist_ok=True)
    (out_dir / "post" / f"{tid}_post.jpg").write_bytes(P.jpeg_bytes(post, 90))

    rows = [b.row() for b in buildings]
    for r, v in zip(rows, crops["post_valid"]):
        r["post_valid"] = round(float(v), 3)
        r["city"] = tile["city"]
    meta = {
        "tile": tid,
        "status": "ok",
        "city": tile["city"],
        "post_items": [it["id"] for it in post_items],
        "post_date": post_items[0]["date"],
        "post_off_nadir": post_items[0]["off_nadir"],
        "shift_px": [dy, dx],
        "shift_applied": applied,
        "peak_ratio": round(peak, 2),
        "ncc_before": round(ncc_before, 4),
        "ncc_after": round(ncc_after, 4),
        "post_valid_frac": round(float(valid.mean()), 4),
        "n_buildings": len(rows),
        "label_source": "instance" if inst_path.exists() else "component",
        "buildings": rows,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta))
    tmp.replace(meta_path)
    return meta


def prepare_all(
    tiles: list[dict],
    items: list[dict],
    bright_dir: Path,
    instance_dir: Path,
    out_dir: Path,
    event_date: str,
    workers: int | None = None,
    log=print,
) -> list[dict]:
    workers = workers or min(8, (os.cpu_count() or 2) * 4)
    t0 = time.time()
    metas: list[dict] = []

    def one(t):
        try:
            return prepare_tile(t, items, bright_dir, instance_dir, out_dir, event_date)
        except Exception as e:  # keep going; failures are reported and retried on rerun
            return {"tile": t["tile"], "status": f"error: {type(e).__name__}: {e}"}

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, m in enumerate(ex.map(one, tiles), 1):
            metas.append(m)
            if i % 25 == 0 or i == len(tiles):
                rate = i / (time.time() - t0)
                eta = (len(tiles) - i) / rate / 60 if rate else 0
                bad = sum(x["status"] != "ok" for x in metas)
                log(f"  {i}/{len(tiles)} tiles  ({rate:.2f}/s, ~{eta:.0f} min left, {bad} not ok)")
    return metas

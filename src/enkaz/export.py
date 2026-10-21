"""Small, web-ready exports of the test-city predictions (runs on Colab, after evaluation).

* ``buildings.json`` - every test building as [lon, lat, true class, P(intact), P(damaged),
  P(destroyed)], rounded so the whole city fits in a few hundred KB.
* ``gallery/`` - pre/post crops of a fixed, stratified sample of buildings (correct and wrong
  predictions of every class) plus ``gallery.json`` describing them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import metrics as M
from . import prepare as P


def lonlat(rows, tiles: list[dict]) -> np.ndarray:
    """Building centres (pixel cy/cx in the tile grid) -> WGS84 lon/lat."""
    from pyproj import Transformer

    by_tile = {t["tile"]: t for t in tiles}
    out = np.zeros((len(rows), 2))
    cache: dict = {}
    for i, (tid, cy, cx) in enumerate(zip(rows["tile"], rows["cy"], rows["cx"])):
        t = by_tile[tid]
        left, _, _, top = t["bounds"]
        x, y = left + (cx + 0.5) * t["res"], top - (cy + 0.5) * t["res"]
        tr = cache.get(t["crs"])
        if tr is None:
            tr = cache[t["crs"]] = Transformer.from_crs(t["crs"], "EPSG:4326", always_xy=True)
        out[i] = tr.transform(x, y)
    return out


def export_city(
    rows,
    logits: np.ndarray,
    temperature: float,
    tiles: list[dict],
    store,
    out_dir: Path,
    per_cell: int = 4,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "gallery").mkdir(parents=True, exist_ok=True)
    probs = M.softmax(logits, temperature)
    y = rows["label"].to_numpy() - 1
    ll = lonlat(rows, tiles)
    data = [
        [round(float(a), 6), round(float(b), 6), int(c), *[round(float(p), 3) for p in pr]]
        for (a, b), c, pr in zip(ll, y, probs)
    ]
    (out_dir / "buildings.json").write_text(json.dumps(data, separators=(",", ":")))

    pred = probs.argmax(1)
    rng = random.Random(seed)
    items = []
    for t in range(3):
        for p in range(3):
            idx = np.flatnonzero((y == t) & (pred == p))
            for i in rng.sample(list(idx), min(per_cell, len(idx))):
                pre, post, fp = store.get(rows["tile"][i], int(rows["crop"][i]))
                name = f"{rows['tile'][i]}_{int(rows['crop'][i]):04d}"
                (out_dir / "gallery" / f"{name}_pre.jpg").write_bytes(P.jpeg_bytes(pre, 88))
                (out_dir / "gallery" / f"{name}_post.jpg").write_bytes(P.jpeg_bytes(post, 88))
                (out_dir / "gallery" / f"{name}_fp.png").write_bytes(P.png_bytes(fp))
                items.append(
                    {
                        "id": name,
                        "true": M.CLASSES[t],
                        "pred": M.CLASSES[p],
                        "probs": [round(float(v), 3) for v in probs[i]],
                        "lon": round(float(ll[i, 0]), 6),
                        "lat": round(float(ll[i, 1]), 6),
                    }
                )
    (out_dir / "gallery.json").write_text(json.dumps(items, indent=1))
    return {"buildings": len(data), "gallery": len(items)}

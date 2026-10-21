"""Rapid response: from post-event imagery to a ranked list of buildings to check first.

Built for a city without labels (Istanbul): footprints come from an inventory such as
``data/istanbul/buildings.npz`` (see ``istanbul.py``), imagery from whatever post-event scenes
are released (local GeoTIFFs or COG URLs), and the classifier from the Kahramanmaraş training run.

    python -m enkaz.respond --inventory data/istanbul/buildings.npz \\
        --post scene1.tif scene2.tif --checkpoint runs/main/post_only/best.pt \\
        --calibration runs/main/post_only/eval.json --out response/

Steps, chosen to match how the model was trained:

1. Buildings are grouped into square cells (600 m) in the local UTM zone. For each cell the post
   scene(s) are resampled onto a 0.35 m grid with a margin, so crops at the edge are complete.
2. If a pre-event reference image is given, the post image is shifted onto it by phase
   correlation (OSM footprints are drawn on older imagery, so the same few-metre offsets occur).
3. Footprints are rasterised and the same 128 px pre/post/footprint crops are cut as in
   training (``prepare.tile_crops``, JPEG round trip included).
4. Logits are calibrated with the temperature and class offsets fitted on the validation city.
   Buildings are ranked by expected severity (p_damaged + 2 p_destroyed): the ranking carried over
   to an unseen city far better than any class threshold did.

Nothing here has been validated on Istanbul; the output is a list for analysts to work through.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import geo
from . import metrics as M
from . import prepare as P

RES = 0.35  # metres per pixel, as in training
CELL_M = 600.0
MARGIN_M = 40.0  # a 1.5x crop of a 50 m building still fits
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, as in learn.py
STD = np.array([0.229, 0.224, 0.225], np.float32)

Predict = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (x_pre, x_post) -> logits


# --------------------------------------------------------------------------- inventory


def load_inventory(path: Path, bbox: Sequence[float] | None = None) -> dict:
    """Footprints from ``istanbul_inventory.py`` (or any npz with the same keys), optionally
    restricted to a lon/lat box."""
    with np.load(path, allow_pickle=False) as z:
        return select(dict(z), bbox)


def select(inv: dict, bbox: Sequence[float] | None = None) -> dict:
    """Packed inventory arrays (as written by ``istanbul.read_buildings``) -> one ring per
    building, optionally only the buildings whose centre is inside a lon/lat box."""
    keep = np.ones(len(inv["lon"]), bool)
    if bbox is not None:
        lon, lat = inv["lon"], inv["lat"]
        keep = (lon >= bbox[0]) & (lat >= bbox[1]) & (lon <= bbox[2]) & (lat <= bbox[3])
    off = inv["ring_off"]
    idx = np.flatnonzero(keep)
    rings = [inv["ring"][off[i] : off[i + 1]].astype(np.float64) for i in idx]
    out = {k: v[idx] for k, v in inv.items() if len(v) == len(keep) and k != "ring_off"}
    out["rings"] = rings
    return out


def project(rings: list[np.ndarray], epsg: int) -> list[np.ndarray]:
    """lon/lat rings -> metres in ``epsg`` (one pyproj call for all vertices)."""
    from pyproj import Transformer

    if not rings:
        return []
    allpts = np.concatenate(rings)
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(
        allpts[:, 0], allpts[:, 1]
    )
    xy = np.column_stack([x, y])
    cuts = np.cumsum([len(r) for r in rings])[:-1]
    return np.split(xy, cuts)


def cell_of(cx: np.ndarray, cy: np.ndarray, size: float = CELL_M) -> np.ndarray:
    """Integer (column, row) of the square cell holding each point."""
    return np.column_stack([np.floor(cx / size), np.floor(cy / size)]).astype(np.int64)


def cell_grid(key: Sequence[int], size: float = CELL_M, margin: float = MARGIN_M, res: float = RES):
    """(left, top, height_px, width_px) of a cell's raster, margin included."""
    left = key[0] * size - margin
    top = (key[1] + 1) * size + margin
    n = int(round((size + 2 * margin) / res))
    return left, top, n, n


def footprints(rings_xy: list[np.ndarray], left: float, top: float, res: float = RES, name="cell"):
    """Projected rings -> ``prepare.Building`` objects in the cell's pixel grid (label unknown)."""
    out = []
    for i, r in enumerate(rings_xy):
        col = (r[:, 0] - left) / res
        row = (top - r[:, 1]) / res
        area = 0.5 * abs(np.dot(col, np.roll(row, -1)) - np.dot(row, np.roll(col, -1)))
        poly = np.column_stack([col, row]).ravel().tolist()
        out.append(
            P.Building(
                name,
                i,
                0,
                float(row.mean()),
                float(col.mean()),
                int(np.floor(row.min())),
                int(np.floor(col.min())),
                int(np.ceil(row.max())),
                int(np.ceil(col.max())),
                int(area),
                "inventory",
                poly,
            )
        )
    return out


# --------------------------------------------------------------------------- model


def model_inputs(pre: np.ndarray, post: np.ndarray, fp: np.ndarray):
    """Batches of uint8 crops (N,128,128,3) and footprints (N,128,128) -> network inputs,
    normalised exactly like ``learn.BuildingSet`` at evaluation time."""
    f = fp.astype(np.float32)[..., None] * 2 - 1

    def one(img):
        x = (img.astype(np.float32) / 255 - MEAN) / STD
        return np.ascontiguousarray(np.concatenate([x, f], -1).transpose(0, 3, 1, 2))

    return one(pre), one(post)


def torch_predictor(kind: str, checkpoint: Path, backbone: str = "resnet34") -> Predict:
    import torch

    from .learn import MODELS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODELS[kind](backbone, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    @torch.no_grad()
    def predict(x_pre: np.ndarray, x_post: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(x_post), 256):
            a = torch.from_numpy(x_pre[s : s + 256]).to(device)
            b = torch.from_numpy(x_post[s : s + 256]).to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                out.append(model(a, b).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)

    return predict


def calibrate(logits: np.ndarray, temperature: float, bias: Sequence[float] | None):
    """Probabilities, class (arg-max after the validation offsets) and expected severity."""
    probs = M.softmax(logits, temperature)
    pred = (logits + (0 if bias is None else np.asarray(bias))).argmax(1)
    return probs, pred, M.damage_score(probs)


def rank(severity: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """1 = check first. Buildings without enough post-event pixels go last, unranked by score."""
    key = np.where(usable, -severity, np.inf)
    order = np.argsort(key, kind="stable")
    r = np.empty(len(order), np.int64)
    r[order] = np.arange(1, len(order) + 1)
    return r


def by_mahalle(mahalle: np.ndarray, pred: np.ndarray, severity: np.ndarray, usable: np.ndarray):
    """Per-mahalle counts of scored buildings and of predicted damaged / destroyed."""
    out: dict[int, dict] = {}
    for m, c, s, u in zip(mahalle.tolist(), pred.tolist(), severity.tolist(), usable.tolist()):
        d = out.setdefault(
            m, {"buildings": 0, "scored": 0, "damaged": 0, "destroyed": 0, "sev": 0.0}
        )
        d["buildings"] += 1
        if u:
            d["scored"] += 1
            d["damaged"] += c == 1
            d["destroyed"] += c == 2
            d["sev"] += s
    return out


# --------------------------------------------------------------------------- imagery


def _scene_bounds(paths: Sequence[str], epsg: int) -> list[tuple]:
    import rasterio
    from rasterio.warp import transform_bounds

    out = []
    for p in paths:
        with rasterio.open(p) as ds:
            out.append(transform_bounds(ds.crs, f"EPSG:{epsg}", *ds.bounds))
    return out


def _overlapping(paths: Sequence[str], bounds: list[tuple], left: float, top: float, size: float):
    """The scenes whose footprint meets the cell (opening every COG for every cell is slow)."""
    cell = (left, top - size, left + size, top)
    return [p for p, b in zip(paths, bounds) if geo.bbox_intersects(b, cell)]


def cell_images(post: Sequence[str], pre: Sequence[str] | None, crs, left, top, shape):
    """Post (and pre) image of one cell; post shifted onto pre when that improves the match."""
    from rasterio.transform import from_origin

    from .pipeline import ncc, read_post

    items = lambda paths: [{"visual": p} for p in paths]  # noqa: E731
    transform = from_origin(left, top, RES, RES)
    post_rgb, valid = read_post(items(post), crs, transform, shape)
    shift = (0, 0)
    if pre:
        pre_rgb, pre_valid = read_post(items(pre), crs, transform, shape)
        pg = P.gradient_magnitude(P.to_gray(pre_rgb))
        both = valid & pre_valid
        dy, dx, _ = P.phase_correlation(pg, P.gradient_magnitude(P.to_gray(post_rgb)), valid=both)
        if (dy, dx) != (0, 0):
            t2 = P.shifted_transform(transform, dy, dx)
            post2, valid2 = read_post(items(post), crs, t2, shape)
            before = ncc(pg, P.gradient_magnitude(P.to_gray(post_rgb)), both)
            after = ncc(pg, P.gradient_magnitude(P.to_gray(post2)), valid2 & pre_valid)
            if after > before:
                post_rgb, valid, shift = post2, valid2, (dy, dx)
    else:
        pre_rgb = post_rgb  # post-only model: the pre input is ignored
    return pre_rgb, post_rgb, valid, shift


# --------------------------------------------------------------------------- run


def _decode(crops: dict, name: str, decode) -> np.ndarray:
    buf, off = crops[f"{name}_buf"], crops[f"{name}_off"]
    return np.stack([decode(P.unpack(buf, off, j)) for j in range(len(off) - 1)])


def run(
    inv: dict,
    post: Sequence[str],
    predict: Predict,
    temperature: float = 1.0,
    bias: Sequence[float] | None = None,
    pre: Sequence[str] | None = None,
    images=cell_images,
    log=print,
) -> dict:
    """Score every inventory building covered by the post scenes. Returns per-building arrays."""
    t0 = time.time()
    n = len(inv["rings"])
    lon0, lat0 = float(np.mean(inv["lon"])), float(np.mean(inv["lat"]))
    epsg = geo.utm_zone_epsg(lon0, lat0)
    rings = project(inv["rings"], epsg)
    centre = np.array([r[:-1].mean(axis=0) if len(r) > 1 else r[0] for r in rings]).reshape(-1, 2)
    cells = cell_of(centre[:, 0], centre[:, 1])
    keys, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    real = images is cell_images
    post_b = _scene_bounds(post, epsg) if real else None
    pre_b = _scene_bounds(pre, epsg) if real and pre else None
    logits = np.zeros((n, 3), np.float32)
    usable = np.zeros(n, bool)
    post_valid = np.zeros(n, np.float32)
    shift_m = np.zeros(n, np.float32)
    done = 0
    for k, key in enumerate(keys):
        left, top, h, w = cell_grid(key)
        cell_post, cell_pre = post, pre
        if real:
            cell_post = _overlapping(post, post_b, left, top, h * RES)
            cell_pre = _overlapping(pre, pre_b, left, top, h * RES) if pre else None
            if not cell_post:
                continue
        idx = np.flatnonzero(inverse == k)
        pre_rgb, post_rgb, valid, shift = images(
            cell_post, cell_pre, f"EPSG:{epsg}", left, top, (h, w)
        )
        if not valid.any():
            continue
        shift_m[idx] = float(np.hypot(*shift)) * RES
        blds = footprints([rings[i] for i in idx], left, top)
        crops = P.tile_crops(pre_rgb, post_rgb, valid, np.zeros((h, w), np.uint8), blds)
        x_pre, x_post = model_inputs(
            _decode(crops, "pre", P.decode_jpeg),
            _decode(crops, "post", P.decode_jpeg),
            _decode(crops, "fp", P.decode_png),
        )
        logits[idx] = predict(x_pre, x_post)
        post_valid[idx] = crops["post_valid"]
        usable[idx] = crops["post_valid"] >= 0.5
        done += len(idx)
        if (k + 1) % 50 == 0:
            log(f"  {k + 1}/{len(keys)} cells, {done:,} buildings ({time.time() - t0:.0f} s)")
    probs, pred, sev = calibrate(logits, temperature, bias)
    return {
        "logits": logits,
        "shift_m": shift_m,
        "seconds": time.time() - t0,
        "probs": probs,
        "pred": pred,
        "severity": sev,
        "usable": usable,
        "post_valid": post_valid,
        "rank": rank(sev, usable),
        "epsg": epsg,
    }


def write_outputs(inv: dict, res: dict, out_dir: Path, top_geojson: int = 5000) -> None:
    """ranking.csv (every building, best first), top.geojson (first ``top_geojson``) and
    mahalle.csv (per-mahalle counts) when the inventory has a ``mahalle`` column."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(res["rank"])
    classes = ("intact", "damaged", "destroyed")
    with open(out_dir / "ranking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "osm_id", "lon", "lat", "p_intact", "p_damaged", "p_destroyed",
                    "class", "severity", "post_valid"])  # fmt: skip
        for i in order:
            p = res["probs"][i]
            w.writerow([
                int(res["rank"][i]), int(inv["osm_id"][i]), f"{inv['lon'][i]:.6f}",
                f"{inv['lat'][i]:.6f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}",
                classes[res["pred"][i]] if res["usable"][i] else "no_image",
                f"{res['severity'][i]:.3f}", f"{res['post_valid'][i]:.2f}",
            ])  # fmt: skip
    feats = []
    for i in order[:top_geojson]:
        if not res["usable"][i]:
            break
        ring = inv["rings"][i]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [np.round(ring, 6).tolist()]},
            "properties": {"rank": int(res["rank"][i]), "osm_id": int(inv["osm_id"][i]),
                           "class": classes[res["pred"][i]],
                           "severity": round(float(res["severity"][i]), 3)},
        })  # fmt: skip
    (out_dir / "top.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats})
    )
    if "mahalle" in inv:
        summ = by_mahalle(inv["mahalle"], res["pred"], res["severity"], res["usable"])
        with open(out_dir / "mahalle.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mahalle", "buildings", "scored", "damaged", "destroyed", "severity_sum"])
            for m, d in sorted(summ.items(), key=lambda kv: -kv[1]["sev"]):
                w.writerow([m, d["buildings"], d["scored"], d["damaged"], d["destroyed"],
                            round(d["sev"], 2)])  # fmt: skip


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--post", nargs="+", required=True, help="post-event GeoTIFF/COG path(s)")
    ap.add_argument("--pre", nargs="*", help="pre-event reference image(s), for alignment")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--kind", default="post_only", choices=["post_only", "siamese"])
    ap.add_argument("--calibration", type=Path, help="eval.json of the run (val_bias block)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--out", type=Path, default=Path("response"))
    a = ap.parse_args(argv)
    if a.kind == "siamese" and not a.pre:
        ap.error("the Siamese model needs --pre images")
    t, bias = 1.0, None
    if a.calibration:
        cal = json.loads(a.calibration.read_text())["val_bias"]
        t, bias = cal["temperature"], cal["bias"]
    inv = load_inventory(a.inventory, a.bbox)
    print(f"{len(inv['rings']):,} buildings in the inventory selection")
    res = run(inv, a.post, torch_predictor(a.kind, a.checkpoint), t, bias, a.pre)
    write_outputs(inv, res, a.out)
    n = int(res["usable"].sum())
    print(f"scored {n:,} buildings; ranking written to {a.out}/ranking.csv")


if __name__ == "__main__":
    main()

"""End-to-end check of ``enkaz.respond`` on Antakya, the held-out test city.

Runs the response pipeline the way it would run in Istanbul and scores it against expert labels:

* footprints made in advance by someone else: Microsoft's Global ML Building Footprints. (OSM is
  no option here: its 1 January 2023 snapshot has ~400 buildings in Antakya, and today's OSM has
  dropped most destroyed buildings.) Not BRIGHT's polygons, which were drawn on the pre image.
* one released post-event Maxar acquisition for the whole city, read straight from its COGs,
* the post-only checkpoint and the İslahiye calibration, unchanged.

Two variants: without alignment, and aligned to a mosaic of the latest pre-event Maxar scenes.
The main configuration, fixed before looking at any result, is aligned + test-time augmentation
(8 rotations/flips averaged); single-view scores are reported next to it.
Every labelled test building is given the score of the footprint that contains it (or the nearest
one within 4 m); the rest are reported as not covered. The reference is the evaluation pipeline
(BRIGHT polygons, co-registered crops) on the same buildings.

Inside the training notebook, after cells 0.1–0.3:

    exec(open("/content/enkaz/scripts/antakya_respond_check.py").read())
"""
# ruff: noqa: F821

import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import enkaz.istanbul as ist
import enkaz.metrics as M
import enkaz.respond as R
from enkaz import config
from enkaz.data import maxar

ist, M, R = (importlib.reload(m) for m in (ist, M, R))
CODE = Path("/content/enkaz")
MATCH_M = 4.0


def ms_footprints(bbox) -> list[Path]:
    import csv
    import io
    import urllib.request

    keys = ist.quadkeys(bbox)
    with urllib.request.urlopen(ist.MS_LINKS) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))
    out = []
    for row in rows:
        if row["Location"] == "Turkey" and row["QuadKey"] in keys:
            p = paths.work / f"ms_{row['QuadKey']}.csv.gz"
            if not p.exists():
                subprocess.run(["curl", "-sL", "-o", str(p), row["Url"]], check=True)
            out.append(p)
    return out


def labelled_buildings():
    """Test-city buildings in evaluation order: lon, lat, label (0-2)."""
    b = np.asarray(json.loads((CODE / "site/data/buildings.json").read_text()))
    return b[:, 0], b[:, 1], b[:, 2].astype(int)


def scenes(bbox):
    items = json.loads((paths.catalog / "maxar_items.json").read_text())
    post = maxar.post_items_for(items, bbox, config.EVENT_DATE)
    pre = [
        r
        for r in items
        if maxar.phase(r, config.EVENT_DATE) == "pre"
        and R.geo.bbox_intersects(r["bbox"], bbox)
        and (r["off_nadir"] or 0) <= 30
        and (r["cloud_percent"] or 0) <= 10
    ]
    pre.sort(key=lambda r: r["date"], reverse=True)  # latest first: fills the mosaic first
    return post, pre


def scores_for_labels(res, match, usable_only=True):
    ok = match >= 0
    if usable_only:
        ok &= np.where(ok, res["usable"][np.maximum(match, 0)], False)
    return ok, res["logits"][np.maximum(match, 0)]


def stage_check():
    out: dict = {}
    lon, lat, y = labelled_buildings()
    bbox = (lon.min() - 0.003, lat.min() - 0.003, lon.max() + 0.003, lat.max() + 0.003)
    t0 = time.time()
    inv = R.select(ist.read_ms_footprints(ms_footprints(bbox), bbox))
    import shapely

    polys = [shapely.Polygon(r) for r in inv["rings"]]
    inside = ist.assign(lon, lat, polys)
    match = ist.assign(lon, lat, polys, max_m=MATCH_M)
    shared = np.bincount(match[match >= 0], minlength=len(polys))
    out["inventory"] = {
        "source": "Microsoft Global ML Building Footprints",
        "footprints": len(polys),
        "labelled": int(len(y)),
        "labelled_inside_a_footprint": float((inside >= 0).mean()),
        "labelled_in_a_footprint": float((match >= 0).mean()),
        "by_class": {c: float((match[y == k] >= 0).mean()) for k, c in enumerate(M.CLASSES)},
        "footprints_with_several_labels": int((shared > 1).sum()),
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(out["inventory"]))

    cal = json.loads((runs / "post_only" / "eval.json").read_text())["val_bias"]
    t, bias = cal["temperature"], cal["bias"]
    ref = np.load(runs / "post_only" / "test_logits.npz")["logits"]
    tta = R.torch_predictor("post_only", runs / "post_only" / "best.pt", tta=True)
    plain = R.torch_predictor("post_only", runs / "post_only" / "best.pt")

    def predict(a, b):  # one pass over the imagery, both scorings: [TTA | single view]
        return np.hstack([tta(a, b), plain(a, b)])

    post, pre = scenes(bbox)
    out["scenes"] = {
        "post": sorted({r["acquisition"] for r in post}),
        "post_date": post[0]["date"],
        "post_off_nadir": post[0]["off_nadir"],
        "post_items": len(post),
        "pre_items": len(pre),
        "pre_dates": [pre[-1]["date"], pre[0]["date"]] if pre else None,
    }
    print(json.dumps(out["scenes"]))

    out["reference_all"] = M.report(ref, y, t, bias)
    for name, pre_paths in (("no_alignment", None), ("aligned_to_pre", pre)):
        if name == "aligned_to_pre" and not pre:
            continue
        res = R.run(
            inv,
            [r["visual"] for r in post],
            predict,
            t,
            bias,
            [r["visual"] for r in pre_paths] if pre_paths else None,
        )
        ok, lg = scores_for_labels(res, match)
        out[name] = {
            "seconds": round(res["seconds"]),
            "buildings_scored": int(res["usable"].sum()),
            "buildings_per_second": round(float(res["usable"].sum()) / res["seconds"], 1),
            "median_shift_m": float(np.median(res["shift_m"][res["usable"]])),
            "labelled_scored": int(ok.sum()),
            "metrics": M.report(lg[ok][:, :3], y[ok], t, bias),
            "metrics_without_tta": M.report(lg[ok][:, 3:], y[ok], t, bias),
            "reference_same_buildings": M.report(ref[ok], y[ok], t, bias),
        }
        print(name, json.dumps({k: v for k, v in out[name].items() if "metrics" not in k}))
        for k in ("metrics", "metrics_without_tta", "reference_same_buildings"):
            m = out[name][k]
            print(
                f"  {k:26s} macro-F1 {m['macro_f1']:.3f}"
                f"  AUROC {m['auroc_damaged_or_destroyed']:.3f}"
                f"  destroyed in top 10% {m['destroyed_recall_top10pct']:.0%}  ECE {m['ece']:.3f}"
            )
    (runs / "respond_check.json").write_text(json.dumps(out, indent=1))
    return {k: v for k, v in out.items() if k in ("inventory", "scenes")}


print(runner.run("respond_check", stage_check, version="2"))
print(json.dumps(json.loads((runs / "respond_check.json").read_text()), indent=1)[:4000])

"""Follow-ups to the Antakya results, in one Colab session (after cells 0.1–0.3 and ``runs``):

1. Confidence intervals for the evaluation pipeline, by cluster bootstrap over image tiles.
2. A Siamese + post-only ensemble (mean logits). It replaces the single models only if it wins on
   the validation city; the test city is reported either way.
3. The response pipeline on Antakya again (main configuration), keeping per-building logits, so
   that its difference to the evaluation pipeline gets an interval too, and a field-map export.

    exec(open("/content/enkaz/scripts/improvements.py").read())
"""
# ruff: noqa: F821

import json
import time

import numpy as np
import pandas as pd

import enkaz.learn as learn

DEFINE_ONLY = True
with open("/content/enkaz/scripts/antakya_respond_check.py") as f:
    exec(f.read())
N_BOOT = 1000


def test_rows():
    bdf = pd.read_parquet(paths.processed / "buildings.parquet")
    return {k: learn.select(bdf, v) for k, v in config.SPLITS["main"].items()}


def calib(kind):
    ev = json.loads((runs / kind / "eval.json").read_text())["val_bias"]
    return ev["temperature"], ev["bias"]


def stage_ci_ensemble():
    rows = test_rows()
    tiles = rows["test"]["tile"].to_numpy()
    z = {k: dict(np.load(runs / k / "test_logits.npz")) for k in ("siamese", "post_only")}
    assert (z["siamese"]["y"] == z["post_only"]["y"]).all()
    y, yv = z["post_only"]["y"], z["post_only"]["val_y"]
    out = {"metrics": ["macro_f1", "auroc", "destroyed_top10", "ece"], "n_boot": N_BOOT}
    for k in z:
        t, b = calib(k)
        s = M.cluster_bootstrap(
            tiles, lambda i, lg=z[k]["logits"], t=t, b=b: M.headline(lg[i], y[i], t, b), N_BOOT
        )
        out[k] = {"test": M.headline(z[k]["logits"], y, t, b).tolist(), "ci": M.interval(s)}

    lv = (z["siamese"]["val_logits"] + z["post_only"]["val_logits"]) / 2
    lt = (z["siamese"]["logits"] + z["post_only"]["logits"]) / 2
    t, b = M.fit_temperature(lv, yv), M.fit_bias(lv, yv)
    val = M.report(lv, yv, t, b)
    best_single = max(M.report(z[k]["val_logits"], yv, *calib(k))["macro_f1"] for k in z)
    s = M.cluster_bootstrap(tiles, lambda i: M.headline(lt[i], y[i], t, b), N_BOOT)
    out["ensemble"] = {
        "val_macro_f1": val["macro_f1"],
        "best_single_val_macro_f1": best_single,
        "adopted": bool(val["macro_f1"] > best_single),
        "temperature": t,
        "bias": [float(v) for v in b],
        "test_report": M.report(lt, y, t, b),
        "test": M.headline(lt, y, t, b).tolist(),
        "ci": M.interval(s),
    }
    (runs / "improvements.json").write_text(json.dumps(out, indent=1))
    return {
        k: (out[k]["test"], out[k]["ci"]["lo"], out[k]["ci"]["hi"])
        for k in ("siamese", "post_only", "ensemble")
    } | {"adopted": out["ensemble"]["adopted"]}


def inspection_curve(score, y, fractions):
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order] == 2) / max((y == 2).sum(), 1)
    return [float(hits[max(int(f * len(y)) - 1, 0)]) for f in fractions]


def stage_respond_ci():
    lon, lat, y = labelled_buildings()
    rows = test_rows()
    tiles = rows["test"]["tile"].to_numpy()
    bbox = (lon.min() - 0.003, lat.min() - 0.003, lon.max() + 0.003, lat.max() + 0.003)
    inv = R.select(ist.read_ms_footprints(ms_footprints(bbox), bbox))
    import shapely

    match = ist.assign(lon, lat, [shapely.Polygon(r) for r in inv["rings"]], max_m=MATCH_M)
    t, bias = calib("post_only")
    ref = np.load(runs / "post_only" / "test_logits.npz")["logits"]
    tta = R.torch_predictor("post_only", runs / "post_only" / "best.pt", tta=True)
    post, pre = scenes(bbox)
    t0 = time.time()
    res = R.run(inv, [r["visual"] for r in post], tta, t, bias, [r["visual"] for r in pre])
    np.savez(runs / "respond_antakya.npz", logits=res["logits"], usable=res["usable"],
             rank=res["rank"], severity=res["severity"], match=match)  # fmt: skip
    ok = (match >= 0) & res["usable"][np.maximum(match, 0)]
    lg = res["logits"][np.maximum(match, 0)]
    idx = np.flatnonzero(ok)
    diff = M.cluster_bootstrap(
        tiles[idx],
        lambda i: (
            M.headline(lg[idx][i], y[idx][i], t, bias) - M.headline(ref[idx][i], y[idx][i], t, bias)
        ),
        N_BOOT,
    )
    resp = M.cluster_bootstrap(
        tiles[idx], lambda i: M.headline(lg[idx][i], y[idx][i], t, bias), N_BOOT
    )
    fr = [round(f, 2) for f in np.linspace(0.01, 1, 100)]
    p_resp = M.softmax(lg[idx], t)
    p_ref = M.softmax(ref[idx], t)
    out = {
        "seconds": round(time.time() - t0),
        "n": int(len(idx)),
        "respond": M.headline(lg[idx], y[idx], t, bias).tolist(),
        "respond_ci": M.interval(resp),
        "difference_ci": M.interval(diff),
        "curve": {
            "fractions": fr,
            "respond": inspection_curve(M.damage_score(p_resp), y[idx], fr),
            "reference": inspection_curve(M.damage_score(p_ref), y[idx], fr),
            "random": fr,
        },
    }
    # field-map export: the first 1,500 buildings of the ranking, with the expert label if any
    label_of = np.full(len(inv["rings"]), -1)
    for j in np.flatnonzero(match >= 0):  # a footprint holding several labels shows the worst
        label_of[match[j]] = max(label_of[match[j]], y[j])
    top = np.argsort(res["rank"])[:1500]
    out["field"] = [
        [round(float(inv["lon"][i]), 5), round(float(inv["lat"][i]), 5), int(res["rank"][i]),
         round(float(res["severity"][i]), 2), int(label_of[i])]
        for i in top
    ]  # fmt: skip
    out["field_summary"] = {
        str(n): {c: int((label_of[top[:n]] == k).sum()) for k, c in enumerate(M.CLASSES)}
        | {"unlabelled": int((label_of[top[:n]] < 0).sum())}
        for n in (100, 500, 1500)
    }
    (runs / "respond_ci.json").write_text(json.dumps(out))
    return {k: out[k] for k in ("seconds", "n", "respond", "field_summary")} | {
        "diff": out["difference_ci"]
    }


print(runner.run("ci_ensemble", stage_ci_ensemble, version="1"))
print(runner.run("respond_ci", stage_respond_ci, version="1"))
imp = json.loads((runs / "improvements.json").read_text())
rci = json.loads((runs / "respond_ci.json").read_text())
print(
    "IMPROVEMENTS_JSON "
    + json.dumps(
        {k: v for k, v in imp.items() if k != "ensemble"}
        | {
            "ensemble": {k: v for k, v in imp["ensemble"].items() if k != "test_report"}
            | {"f1": imp["ensemble"]["test_report"]["f1"]}
        }
    )
)
print("RESPOND_CI_JSON " + json.dumps({k: v for k, v in rci.items() if k != "field"}))

"""A second held-out city: the post-only model trained without Kahramanmaraş and tested on it
(split ``alt_kahramanmaras``: Antakya moves into training, İslahiye stays the validation city).

One test city is one sample; a second one shows how much the numbers move from city to city.
Same recipe as the main run (8 epochs, validation-fitted temperature and offsets), ~1 h on a T4.

    exec(open("/content/enkaz/scripts/second_city.py").read())   # after cells 0.1–0.3
"""
# ruff: noqa: F821

import json

import numpy as np
import pandas as pd

import enkaz.learn as learn
import enkaz.metrics as M
from enkaz import config

SPLIT = config.SPLITS["alt_kahramanmaras"]
run2 = paths.runs / "alt_kahramanmaras"
bdf = pd.read_parquet(paths.processed / "buildings.parquet")
rows2 = {k: learn.select(bdf, v) for k, v in SPLIT.items()}
for k, r in rows2.items():
    print(k, SPLIT[k], len(r), r["label"].value_counts().sort_index().tolist())
store2 = learn.CropStore(paths.processed / "crops", pd.concat(rows2.values())["tile"])


def stage_train_alt():
    return learn.fit(
        "post_only", rows2["train"], rows2["val"], store2, run2 / "post_only", epochs=8
    )


def stage_eval_alt():
    ev = learn.evaluate("post_only", run2 / "post_only", rows2["val"], rows2["test"], store2)
    z = np.load(run2 / "post_only" / "test_logits.npz")
    t, b = ev["val_bias"]["temperature"], ev["val_bias"]["bias"]
    tiles = rows2["test"]["tile"].to_numpy()
    s = M.cluster_bootstrap(tiles, lambda i: M.headline(z["logits"][i], z["y"][i], t, b), 1000)
    out = {
        "split": SPLIT,
        "test": ev["test_bias"],
        "test_argmax": ev["test"],
        "headline": M.headline(z["logits"], z["y"], t, b).tolist(),
        "ci": M.interval(s),
    }
    (run2 / "second_city.json").write_text(json.dumps(out, indent=1))
    return {"headline": out["headline"], "lo": out["ci"]["lo"], "hi": out["ci"]["hi"]}


print(runner.run("train_alt_post_only", stage_train_alt, version="1"))
print(runner.run("evaluate_alt", stage_eval_alt, version="1"))
sc = json.loads((run2 / "second_city.json").read_text())
print(
    "SECOND_CITY_JSON "
    + json.dumps(
        {k: sc[k] for k in ("split", "headline", "ci")}
        | {"f1": sc["test"]["f1"], "n": sc["test"]["n"], "confusion": sc["test"]["confusion"]}
    )
)

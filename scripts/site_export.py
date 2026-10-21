"""Web export stage, runnable inside the training notebook's session.

    exec(open("/content/enkaz/scripts/site_export.py").read())

Uses the notebook's globals (``runs``, ``rows``, ``store``, ``runner``, ``paths``). The map
shows the model that is best on the *validation* city; the test city never chooses it.
"""
# ruff: noqa: F821

import importlib
import json
import shutil

import numpy as np

import enkaz.export as export

export = importlib.reload(export)
tiles = json.loads((paths.catalog / "bright_tiles.json").read_text())


def stage_site():
    evs = {k: json.loads((runs / k / "eval.json").read_text()) for k in ("siamese", "post_only")}
    kind = max(evs, key=lambda k: evs[k]["val_bias"]["macro_f1"])
    t, bias = evs[kind]["test_bias"]["temperature"], evs[kind]["test_bias"]["bias"]
    z = np.load(runs / kind / "test_logits.npz")
    out = runs / "site_export"
    info = export.export_city(rows["test"], z["logits"], t, tiles, store, out, bias=bias)
    res = json.loads((runs / "results.json").read_text())
    res["map_model"] = kind
    for k in evs:
        res[k]["val_bias"] = evs[k]["val_bias"]
    (out / "results.json").write_text(json.dumps(res, indent=1))
    shutil.make_archive(str(runs / "site_export"), "zip", out)
    return info | {"map_model": kind}


print(runner.run("export_site", stage_site, version="3"))
print(
    "zip:",
    runs / "site_export.zip",
    round((runs / "site_export.zip").stat().st_size / 1e6, 1),
    "MB",
)

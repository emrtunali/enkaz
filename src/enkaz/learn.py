"""Building damage classifiers and their training loop (PyTorch + timm; runs on Colab).

Every sample is one building: a pre-event crop, a post-event crop (same grid after alignment)
and the building footprint, all 128x128 at ~0.35 m.

Models
* ``siamese``   - one shared encoder sees [pre, footprint] and [post, footprint]; the head gets
                  both feature vectors and their difference. Our main model.
* ``post_only`` - the same encoder on [post, footprint] only. Asks: how much does the pre image add?

Training is resumable: a checkpoint is written to Drive after every epoch and picked up again
after a disconnect. The epoch with the best validation macro-F1 is kept as ``best.pt``.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import metrics as M
from . import prepare as P

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------- data


class CropStore:
    """Keeps every tile's packed crops in memory (compressed bytes, ~2 GB for all tiles).

    Files are read in parallel: Google Drive's FUSE mount has a high per-file latency, and reading
    the 1,100 tile files one after another runs at ~2 MB/s.
    """

    def __init__(self, crops_dir: Path, tiles, workers: int = 32, log=print):
        import concurrent.futures as cf
        import io

        names = sorted(set(tiles))

        def load(t: str):
            with np.load(io.BytesIO((Path(crops_dir) / f"{t}.npz").read_bytes())) as z:
                return t, {k: z[k] for k in z.files}

        self.tiles = {}
        t0 = time.time()
        with cf.ThreadPoolExecutor(workers) as ex:
            for i, (t, arrays) in enumerate(ex.map(load, names), 1):
                self.tiles[t] = arrays
                if i % 200 == 0 or i == len(names):
                    log(f"  loaded {i}/{len(names)} tiles in {time.time() - t0:.0f} s")

    def get(self, tile: str, idx: int):
        z = self.tiles[tile]
        pre = P.decode_jpeg(P.unpack(z["pre_buf"], z["pre_off"], idx))
        post = P.decode_jpeg(P.unpack(z["post_buf"], z["post_off"], idx))
        fp = P.decode_png(P.unpack(z["fp_buf"], z["fp_off"], idx))
        return pre, post, fp


def select(bdf, cities, min_post_valid: float = 0.5):
    """Rows of the building table used for a split (labelled, enough valid post pixels).

    ``crop`` is the building's position inside its tile's crop file. It is *not* ``idx``: the
    building table keeps the tiles' row order, but instance ids can skip numbers.
    """
    if "crop" not in bdf:
        bdf = bdf.assign(crop=bdf.groupby("tile", sort=False).cumcount())
    m = (
        bdf["city"].isin(cities)
        & bdf["label"].between(1, 3)
        & (bdf["post_valid"] >= min_post_valid)
    )
    return bdf[m].reset_index(drop=True)


def _jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Brightness/contrast/saturation jitter; pre and post come from different acquisitions."""
    x = img.astype(np.float32) / 255
    x = (x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.08, 0.08)
    gray = x.mean(axis=-1, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.8, 1.2)
    return np.clip(x, 0, 1)


def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(img, (dy, dx), axis=(0, 1))


class BuildingSet(Dataset):
    def __init__(self, rows, store: CropStore, train: bool):
        self.tile = rows["tile"].tolist()
        self.idx = rows["crop"].astype(int).tolist()
        self.y = (rows["label"].astype(int) - 1).to_numpy()
        self.store, self.train = store, train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        pre, post, fp = self.store.get(self.tile[i], self.idx[i])
        if self.train:
            rng = random.Random()
            k = rng.randrange(4)
            flip = rng.random() < 0.5
            ops = (lambda a: np.rot90(a, k)), (lambda a: a[:, ::-1] if flip else a)
            for op in ops:
                pre, post, fp = op(pre), op(post), op(fp)
            # residual misalignment of a few pixels survives co-registration; teach robustness
            post = _shift(post, rng.randint(-3, 3), rng.randint(-3, 3))
            pre, post = _jitter(pre, rng), _jitter(post, rng)
        else:
            pre, post = pre.astype(np.float32) / 255, post.astype(np.float32) / 255
        pre = (pre - MEAN) / STD
        post = (post - MEAN) / STD
        fp = fp.astype(np.float32)[..., None] * 2 - 1
        x_pre = np.concatenate([pre, fp], -1).transpose(2, 0, 1)
        x_post = np.concatenate([post, fp], -1).transpose(2, 0, 1)
        return (
            torch.from_numpy(np.ascontiguousarray(x_pre)),
            torch.from_numpy(np.ascontiguousarray(x_post)),
            int(self.y[i]),
        )


def balanced_sampler(y: np.ndarray, power: float = 0.5) -> WeightedRandomSampler:
    """Sample classes ∝ count**power (0.5: rare classes oversampled, but not to parity)."""
    counts = np.bincount(y, minlength=3).astype(float)
    w_class = np.where(counts > 0, counts**power / np.maximum(counts, 1), 0)
    w = w_class[y]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(y), replacement=True)


# --------------------------------------------------------------------------- models


def encoder(backbone: str, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(backbone, pretrained=pretrained, in_chans=4, num_classes=0)


class Siamese(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, n_classes: int = 3):
        super().__init__()
        self.enc = encoder(backbone, pretrained)
        d = self.enc.num_features
        self.head = nn.Sequential(
            nn.Linear(3 * d, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, n_classes)
        )

    def forward(self, x_pre, x_post):
        a, b = self.enc(x_pre), self.enc(x_post)
        return self.head(torch.cat([a, b, b - a], 1))


class PostOnly(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, n_classes: int = 3):
        super().__init__()
        self.enc = encoder(backbone, pretrained)
        d = self.enc.num_features
        self.head = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, n_classes)
        )

    def forward(self, x_pre, x_post):  # x_pre is ignored on purpose
        return self.head(self.enc(x_post))


MODELS = {"siamese": Siamese, "post_only": PostOnly}


# --------------------------------------------------------------------------- training


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    out, ys = [], []
    for x_pre, x_post, y in loader:
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(
                x_pre.to(device, non_blocking=True), x_post.to(device, non_blocking=True)
            )
        out.append(logits.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(out), np.concatenate(ys)


def fit(
    kind: str,
    train_rows,
    val_rows,
    store: CropStore,
    run_dir: Path,
    backbone: str = "resnet34",
    epochs: int = 8,
    batch: int = 128,
    lr: float = 3e-4,
    workers: int = 2,
    seed: int = 0,
    log=print,
) -> dict:
    """Train (or resume) one model; returns the summary of the best epoch."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tr, va = BuildingSet(train_rows, store, True), BuildingSet(val_rows, store, False)
    dl = DataLoader(
        tr,
        batch,
        sampler=balanced_sampler(tr.y),
        num_workers=workers,
        pin_memory=device == "cuda",
        drop_last=True,
        persistent_workers=workers > 0,
    )
    vdl = DataLoader(va, 2 * batch, num_workers=workers, pin_memory=device == "cuda")

    model = MODELS[kind](backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    start, history, best = 0, [], {"macro_f1": -1.0}
    ckpt = run_dir / "last.pt"
    if ckpt.exists():
        s = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        start, history, best = s["epoch"] + 1, s["history"], s["best"]
        log(f"resumed {kind} at epoch {start}/{epochs}")

    for ep in range(start, epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x_pre, x_post, y) in enumerate(dl, 1):
            x_pre, x_post, y = (t.to(device, non_blocking=True) for t in (x_pre, x_post, y))
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x_pre, x_post), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
            if i % 100 == 0:
                log(
                    f"  ep {ep} step {i}/{len(dl)} loss {tot / n:.3f} "
                    f"({(time.time() - t0) / i:.2f}s/step)"
                )
        logits, yv = predict(model, vdl, device)
        rep = M.report(logits, yv)
        rec = {
            "epoch": ep,
            "train_loss": tot / max(n, 1),
            "val_macro_f1": rep["macro_f1"],
            "val_f1": rep["f1"],
            "minutes": (time.time() - t0) / 60,
        }
        history.append(rec)
        log(
            f"epoch {ep}: loss {rec['train_loss']:.3f}  "
            f"val macro-F1 {rep['macro_f1']:.3f}  {rep['f1']}"
        )
        if rep["macro_f1"] > best["macro_f1"]:
            best = {"epoch": ep, "macro_f1": rep["macro_f1"]}
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "history": history,
                "best": best,
            },
            run_dir / "last.tmp",
        )
        (run_dir / "last.tmp").replace(ckpt)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {"kind": kind, "backbone": backbone, "best": best, "epochs": len(history)}


def evaluate(
    kind: str,
    run_dir: Path,
    val_rows,
    test_rows,
    store: CropStore,
    backbone: str = "resnet34",
    workers: int = 2,
) -> dict:
    """Best checkpoint -> temperature fitted on validation -> metrics on the unseen test city."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODELS[kind](backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=device))
    dl = lambda rows: DataLoader(BuildingSet(rows, store, False), 256, num_workers=workers)  # noqa: E731
    lv, yv = predict(model, dl(val_rows), device)
    lt, yt = predict(model, dl(test_rows), device)
    t = M.fit_temperature(lv, yv)
    out = {
        "val": M.report(lv, yv, t),
        "test": M.report(lt, yt, t),
        "test_uncalibrated_ece": M.ece(M.softmax(lt), yt),
    }
    np.savez(Path(run_dir) / "test_logits.npz", logits=lt, y=yt)
    (Path(run_dir) / "eval.json").write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- non-learned baseline


def change_baseline(rows, store: CropStore) -> np.ndarray:
    """Damage score without learning: 1 - correlation of pre/post gradients inside the footprint."""
    scores = np.empty(len(rows))
    for i, (t, k) in enumerate(zip(rows["tile"], rows["crop"])):
        pre, post, fp = store.get(t, int(k))
        a = P.gradient_magnitude(P.to_gray(pre))[fp]
        b = P.gradient_magnitude(P.to_gray(post))[fp]
        if a.size < 20:
            scores[i] = 0.0
            continue
        a, b = a - a.mean(), b - b.mean()
        scores[i] = 1 - float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
    return scores


def thresholds_from_val(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two cut points on the change score (intact | damaged | destroyed) maximising val macro-F1."""
    qs = np.quantile(score, np.linspace(0.02, 0.98, 49))
    best, cut = -1.0, (qs[0], qs[-1])
    for i, a in enumerate(qs):
        for b in qs[i + 1 :]:
            pred = np.digitize(score, [a, b])
            f = M.per_class_f1(M.confusion(y, pred)).mean()
            if f > best:
                best, cut = f, (float(a), float(b))
    return cut
